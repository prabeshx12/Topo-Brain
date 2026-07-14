"""
Train the FR-U-Net baseline (Acs & Zhuang, PLOS ONE 2025) on OUR data, OUR splits, OUR metrics.

This is the SOTA baseline reviewers R1.2 / R2.1 demanded. Citing their published 23.25 dB would
be meaningless -- it is measured under a protocol worth 17.15 dB of free score on our data
(revision/PROTOCOL_SENSITIVITY.md). The only defensible comparison is to run their architecture
ourselves under identical conditions.

FAIRNESS INVARIANTS (each one deliberate -- a rigged baseline is worse than no baseline):
  * SAME dataloader, SAME folds, SAME seed, SAME patch sampler as scripts/train_cascaded.py.
  * SAME ssim3d function, imported from train_cascaded -- not a reimplementation.
  * SAME tiled Tukey reassembly at inference (their blending rule is not stated).
  * THEIR recipe where they state it: MSE + 0.7*SSIM, Adam lr 2e-5, batch 8, sigmoid to [0,1].
  * Near-identical capacity: 11.67 M vs our 11.90 M (0.98x) -- so no result can be attributed
    to us simply having a bigger model.

RANGE CONVERSION. Their model ends in a sigmoid and their volumes are min-maxed to [0,1]; our
pipeline is [-1,1] (tanh). We train the baseline in ITS OWN native range -- targets are mapped
[-1,1] -> [0,1] for the loss -- and map the prediction back for evaluation, so it is scored on
exactly the same footing as ours. Bending their design into our range would not be reproducing it.

Usage (Kaggle):
  python scripts/train_frunet.py --pairs-csv <csv> --n-iters 40000 --batch-size 8 --lr 2e-5
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse the EXACT ssim3d the cascaded model trains against. Importing it (rather than
# reimplementing) is what guarantees both arms are scored by the same function.
_spec = importlib.util.spec_from_file_location("_tc", ROOT / "scripts" / "train_cascaded.py")
_tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tc)
ssim3d, log = _tc.ssim3d, _tc.log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/train_diffusion.yaml")
    ap.add_argument("--out-dir", type=str, default="/kaggle/working")
    ap.add_argument("--n-iters", type=int, default=40000)
    ap.add_argument("--batch-size", type=int, default=8)       # theirs
    ap.add_argument("--lr", type=float, default=2e-5)          # theirs
    ap.add_argument("--lam-ssim", type=float, default=0.7)     # theirs: "We set lambda = 0.7"
    ap.add_argument("--norm", choices=("layer", "group"), default="layer",
                    help="'layer normalization' is all the paper says; the axis is not given. "
                         "layer = Keras default (channel axis). group = conv-net convention, "
                         "for a sensitivity check.")
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--test-fold", type=int, default=1)
    ap.add_argument("--save-freq", type=int, default=5000)
    ap.add_argument("--log-freq", type=int, default=50)
    ap.add_argument("--max-hours", type=float, default=10.5)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)            # SAME seed as ours
    args = ap.parse_args()

    import yaml
    CFG = yaml.safe_load(open(args.config))
    D = CFG["dataset"]

    from src.model_frunet import FRUNet, frunet_loss
    from src.synthesis_dataset import (
        create_synthesis_dataloaders, load_pairs_manifest, PatchConfig, SplitConfig,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {dev}")
    if dev.type == "cuda":
        _ = (torch.randn(64, 64, device=dev) @ torch.randn(64, 64, device=dev)).sum().item()
        _c = torch.nn.Conv3d(1, 4, 3, padding=1).to(dev)
        _ = _c(torch.randn(1, 1, 8, 8, 8, device=dev)).sum().item()
        log("CUDA sanity check OK")

    # ---- data: IDENTICAL to train_cascaded.py -------------------------------------------
    pairs = load_pairs_manifest(args.pairs_csv)
    patch_cfg = PatchConfig(
        patch_size=tuple(D["patch_size"]),
        patches_per_volume=D["patches_per_volume"],
        min_brain_fraction=D["min_brain_fraction"],
        seed=args.seed,
    )
    split_cfg = SplitConfig(n_folds=D["n_folds"], val_fold=args.val_fold,
                            test_fold=args.test_fold, use_loocv=D["use_loocv"],
                            seed=args.seed)
    train_loader, _, _ = create_synthesis_dataloaders(
        pairs, config=patch_cfg, split_config=split_cfg,
        batch_size=args.batch_size, num_workers=2,
        val_fold=args.val_fold, test_fold=args.test_fold,
    )
    log(f"pairs={len(pairs)}  train batches/epoch={len(train_loader)}  "
        f"(val=fold{args.val_fold}, test=fold{args.test_fold} -- SAME split as ours)")

    model = FRUNet(in_channels=1, out_channels=1,
                   features=tuple(CFG["model"]["features"]), norm=args.norm).to(dev)
    log(f"FR-U-Net params: {model.n_params():,} ({model.n_params() / 1e6:.2f} M) "
        f"[the paper never states theirs]")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)      # theirs: Adam, 2e-5
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.n_iters)
    scaler = GradScaler(dev.type, enabled=(dev.type == "cuda"))

    start = 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = int(ck["step"])
        log(f"resumed from {args.resume} @ step {start}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    it = iter(train_loader)
    hist = []

    for step in range(start, args.n_iters):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        x3 = batch["input"].to(dev, non_blocking=True)
        x7 = batch["target"].to(dev, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with autocast(device_type=dev.type, enabled=(dev.type == "cuda")):
            pred01 = model(x3)                                  # sigmoid -> [0,1]

        pred01 = pred01.float()
        tgt01 = (x7.float() + 1.0) / 2.0                        # [-1,1] -> their [0,1]

        # their Eq(1). data_range=1.0 because both tensors are now in [0,1].
        loss, l_mse, l_ssim = frunet_loss(
            pred01, tgt01,
            ssim_fn=lambda a, b: ssim3d(a, b, data_range=1.0),
            lam_ssim=args.lam_ssim)

        if not torch.isfinite(loss):
            log(f"step {step}: non-finite loss (mse={float(l_mse):.4f}) -- SKIPPING")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % args.log_freq == 0:
            el = (time.time() - t0) / 60.0
            rec = {"step": step, "mse": round(float(l_mse), 5),
                   "ssim_loss": round(float(l_ssim), 4), "total": round(float(loss), 4),
                   "lr": sched.get_last_lr()[0], "min": round(el, 1)}
            hist.append(rec)
            log(f"step {step} | MSE {rec['mse']:.5f} | SSIM_loss {rec['ssim_loss']:.4f} | "
                f"tot {rec['total']:.4f} | {el:.1f}m")

        if step > start and step % args.save_freq == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "args": vars(args)}, out / f"frunet_{step}.pt")
            (out / "frunet_history.json").write_text(json.dumps(hist, indent=2))

        if (time.time() - t0) / 3600.0 > args.max_hours:
            log(f"hit --max-hours at step {step}; saving and exiting cleanly")
            break

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "args": vars(args)}, out / f"frunet_{step}.pt")
    (out / "frunet_history.json").write_text(json.dumps(hist, indent=2))
    log(f"done @ step {step}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
