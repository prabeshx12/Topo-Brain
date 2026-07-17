"""
Train the rebuilt 3T->7T model: deterministic regression U-Net with a CASCADED seg head.

    3T --> UNetGenerator --> synth 7T --> SegHead --> seg
                                 ^                     |
                                 +---- gradient -------+

Loss (Phase 3 = no topology term yet; Phase 4 adds it):

    L = L1(img, 7T)
      + lam_ssim * (1 - SSIM3D(img, 7T))
      + lam_seg  * ( CE(seg, gt) + Dice(seg, gt) )
      [+ lam_topo * topology(seg, gt)]        <-- Phase 4

Why this shape (see revision/STRATEGY_2026.md):
  * L1 + SSIM is the standard objective in this exact subfield (FS-RWKV; LiteMamba-Synth).
  * The VGG/ImageNet perceptual loss is DROPPED: it is 2D ImageNet features on ~8 slices of
    a 3D volume, it had zero gradient on 16-26% of voxels, and the regression baseline beat
    the diffusion model without it. YODA (IEEE TMI 2026) shows perceptual realism does not
    imply medical accuracy.
  * The seg loss now backpropagates THROUGH the generator (that is the whole point of the
    cascade), so it forces the synthesised image to be segmentable with correct anatomy.

Everything runs on the FIXED data pipeline (random patch centres -> ~147x more unique
patches; a real held-out test fold; idempotent LOOCV folds). See revision/AUDIT.md.
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- #
#  Losses
# --------------------------------------------------------------------------- #
def _gaussian_kernel3d(ws: int, sigma: float, device) -> torch.Tensor:
    c = torch.arange(ws, dtype=torch.float32, device=device) - (ws - 1) / 2.0
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return k[None, None]


def ssim3d(x: torch.Tensor, y: torch.Tensor, data_range: float = 2.0,
           ws: int = 7, sigma: float = 1.5,
           mask: torch.Tensor = None) -> torch.Tensor:
    """Differentiable 3D SSIM. x, y: [B,1,D,H,W] in [-1,1].

    `mask` ([B,1,D,H,W] or [B,D,H,W], bool/float): if given, the SSIM map is averaged ONLY
    over the masked (brain) voxels.

    WHY THE MASK MATTERS. The volumes are skull-stripped, so ~85% of each volume is a
    constant -1 background, and the patch sampler only requires >=10% brain. Measured over
    the real valid centres:

        brain fraction of accepted patches: median 0.475, mean 0.499
        52.7% of accepted patches are >50% background

    SSIM SATURATES to ~1.0 wherever prediction and target are both flat and equal, so that
    background contributes a perfect score and ZERO gradient. An unmasked SSIM term is
    therefore diluted ~2x -- the loss added specifically to sharpen anatomy was running at
    half strength. Unlike L1 (which self-corrects once the model learns to emit -1), SSIM's
    saturation does not wash out.

    We mask it so the TRAINING objective mirrors the EVALUATION metric (which is brain-only,
    see src/metrics_honest.masked_ssim). L1 is deliberately left unmasked so the background
    stays supervised for the whole-volume reporting convention.
    """
    k = _gaussian_kernel3d(ws, sigma, x.device).to(x.dtype)
    pad = ws // 2
    mu_x = F.conv3d(x, k, padding=pad)
    mu_y = F.conv3d(y, k, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sx = F.conv3d(x * x, k, padding=pad) - mu_x2
    sy = F.conv3d(y * y, k, padding=pad) - mu_y2
    sxy = F.conv3d(x * y, k, padding=pad) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    s = ((2 * mu_xy + c1) * (2 * sxy + c2)) / ((mu_x2 + mu_y2 + c1) * (sx + sy + c2))

    if mask is None:
        return s.mean()

    m = mask.to(s.dtype)
    if m.dim() == 4:
        m = m.unsqueeze(1)
    denom = m.sum().clamp(min=1.0)
    return (s * m).sum() / denom


def dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int,
              eps: float = 1.0) -> torch.Tensor:
    """Soft multi-class Dice, computed PER SAMPLE then averaged.

    (The old boundary-Dice summed globally over the batch, so one sample's gradient
    depended on the others and the loss was batch-composition dependent.)
    """
    p = torch.softmax(logits, dim=1)
    t = F.one_hot(target.long(), num_classes).permute(0, 4, 1, 2, 3).to(p.dtype)
    dims = (2, 3, 4)
    inter = (p * t).sum(dims)
    denom = p.sum(dims) + t.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)          # [B, C]
    return 1.0 - dice.mean()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/train_diffusion.yaml")
    ap.add_argument("--out-dir", type=str, default="/kaggle/working")
    ap.add_argument("--n-iters", type=int, default=40000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lam-ssim", type=float, default=0.5)
    ap.add_argument("--lam-seg", type=float, default=1.0)
    ap.add_argument("--lam-topo", type=float, default=0.0)   # Phase 4 turns this on
    ap.add_argument("--topo-kind", choices=("euler", "edge"), default="euler",
                    help="euler = REAL topological invariant (chi = V-E+F-C, exact vs gudhi). "
                         "edge = the PUBLISHED 'topology loss' (edge-weighted CE + Sobel Dice), "
                         "which is not topological at all -- kept only as an ablation arm.")
    ap.add_argument("--topo-sharpness", type=float, default=32.0,
                    help="p -> sigmoid(k*(p-0.5)) before chi. Concentrates the term on the "
                         "decision boundary. MUST be 32: at k=20 an uncertain background injects "
                         "a +8.6 spurious-chi floor bias (UNUSABLE now that chi_gt is exact and no "
                         "longer shares the cancelling bias); k=32 -> bias +0.02. See "
                         "src/topology_euler.py:126-134 and test_topology_euler.py level 4.")
    ap.add_argument("--topo-warmup", type=int, default=10000,
                    help="steps of L1+CE before the topology term ramps in (PH losses are "
                         "unstable on early garbage predictions)")
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--test-fold", type=int, default=1)
    ap.add_argument("--save-freq", type=int, default=5000)
    ap.add_argument("--log-freq", type=int, default=50)
    ap.add_argument("--max-hours", type=float, default=10.5)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--detach-seg", action="store_true",
                    help="ABLATION: sever the cascade (reproduces the old broken design)")
    ap.add_argument("--mask-ssim", type=int, default=1,
                    help="1 = average the SSIM term over brain voxels only (default). The "
                         "volumes are skull-stripped, so an unmasked SSIM is ~2x diluted by "
                         "flat background where it saturates to 1.0. Set 0 to ablate.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import yaml
    CFG = yaml.safe_load(open(args.config))
    D = CFG["dataset"]

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from src.model_cascaded import CascadedSynthesisNet
    from src.synthesis_dataset import (
        create_synthesis_dataloaders, load_pairs_manifest, PatchConfig, SplitConfig,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {dev}")
    if dev.type == "cuda":
        # fail fast if CUDA kernels cannot actually run (Kaggle P100 = sm_60)
        _ = (torch.randn(64, 64, device=dev) @ torch.randn(64, 64, device=dev)).sum().item()
        _c = torch.nn.Conv3d(1, 4, 3, padding=1).to(dev)
        _ = _c(torch.randn(1, 1, 8, 8, 8, device=dev)).sum().item()
        log("CUDA sanity check OK")

    nc = int(CFG["model"]["num_classes"])

    # ---- data (FIXED pipeline: random patch centres, real test fold) ----
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
    train_loader, val_loader, _ = create_synthesis_dataloaders(
        pairs, config=patch_cfg, split_config=split_cfg,
        batch_size=args.batch_size, num_workers=2,
        val_fold=args.val_fold, test_fold=args.test_fold,
    )
    log(f"pairs={len(pairs)}  train batches/epoch={len(train_loader)}")

    # ---- model ----
    model = CascadedSynthesisNet(
        in_channels=1, out_channels=1, num_classes=nc,
        features=tuple(CFG["model"]["features"]),
        use_attention=CFG["model"]["use_attention"],
        detach_seg_input=args.detach_seg,
    ).to(dev)
    log(f"params: {model.n_params()}  (detach_seg={args.detach_seg})")

    ema = CascadedSynthesisNet(
        in_channels=1, out_channels=1, num_classes=nc,
        features=tuple(CFG["model"]["features"]),
        use_attention=CFG["model"]["use_attention"],
        detach_seg_input=args.detach_seg,
    ).to(dev)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    EMA_DECAY, EMA_START = 0.999, 2000

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.n_iters, eta_min=args.lr * 0.05)
    scaler = GradScaler(enabled=True, init_scale=4096)
    CLS_W = torch.tensor([0.5, 2.0, 1.5, 1.0][:nc], device=dev)

    start = 0
    if args.resume:
        hits = glob.glob(args.resume) or glob.glob(f"/kaggle/input/**/{args.resume}", recursive=True)
        if hits:
            st = torch.load(hits[0], map_location=dev, weights_only=False)
            model.load_state_dict(st["model"])
            ema.load_state_dict(st["ema"])
            opt.load_state_dict(st["optimizer"])
            scaler.load_state_dict(st["scaler"])
            if "sched" in st:
                sched.load_state_dict(st["sched"])
            start = st["step"] + 1
            log(f"RESUMED {hits[0]} @ step {start}")

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    it = iter(train_loader)
    hist = []
    log(f"\n=== training: steps {start} -> {args.n_iters} ===")

    for step in range(start, args.n_iters):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        x3 = batch["input"].to(dev, non_blocking=True)
        x7 = batch["target"].to(dev, non_blocking=True)
        seg = batch["seg"].to(dev, non_blocking=True)
        if seg.dim() == 5:
            seg = seg.squeeze(1)
        seg = seg.long().clamp_(0, nc - 1)

        # topology term ramps in only after the model produces something sane
        if args.lam_topo > 0 and step >= args.topo_warmup:
            ramp = min(1.0, (step - args.topo_warmup) / max(1, args.topo_warmup))
            lam_topo = args.lam_topo * ramp
        else:
            lam_topo = 0.0

        opt.zero_grad(set_to_none=True)
        with autocast(device_type=dev.type, enabled=True):
            out = model(x3)

        img = out["image"].float()
        seg_logits = out["seg"].float().clamp(-50, 50)

        # Brain mask from the ground-truth tissue labels (class 0 = background).
        # Used to mask the SSIM term (see ssim3d docstring): the volumes are skull-stripped,
        # so ~50% of an average accepted patch is flat background where SSIM saturates to 1.0
        # and yields no gradient -- diluting the term ~2x.
        brain = (seg > 0)

        l_l1 = F.l1_loss(img, x7.float())                       # unmasked: keep bg supervised
        if args.mask_ssim and brain.any():
            l_ssim = 1.0 - ssim3d(img, x7.float(), mask=brain)  # brain-only, mirrors the metric
        else:
            l_ssim = 1.0 - ssim3d(img, x7.float())
        l_ce = F.cross_entropy(seg_logits, seg, weight=CLS_W)
        l_dice = dice_loss(seg_logits, seg, nc)
        loss = l_l1 + args.lam_ssim * l_ssim + args.lam_seg * (l_ce + l_dice)

        l_topo = torch.zeros((), device=dev)
        chi_gap = float("nan")
        if lam_topo > 0:
            if not hasattr(model, "_topo"):
                if args.topo_kind == "euler":
                    # A REAL topological invariant: chi = V - E + F - C on the cubical complex,
                    # exact vs gudhi's persistent homology (scripts/test_topology_euler.py).
                    from src.topology_euler import EulerTopologyLoss
                    model._topo = EulerTopologyLoss(
                        num_classes=nc, include_background=False,
                        sharpness=args.topo_sharpness).to(dev)
                elif args.topo_kind == "edge":
                    # The PUBLISHED "topology loss" -- edge-weighted CE + Sobel-edge Dice. It is
                    # NOT topological (no Betti number, no connectivity, no persistence). Kept
                    # ONLY so the paper can ablate real topology against what was published.
                    from src.topology_loss import create_topology_loss
                    model._topo = create_topology_loss(
                        num_classes=nc, use_multiscale=True).to(dev)
                else:
                    raise ValueError(f"unknown --topo-kind {args.topo_kind!r}")
            _t = model._topo(seg_logits, seg)
            l_topo = _t["loss"]
            loss = loss + lam_topo * l_topo
            # Log the HONEST topology monitor: chi on the BINARISED (argmax) prediction. Both the
            # soft loss AND the soft chi_pred are confounded with softmax confidence -- they fall
            # as the head sharpens even when the discrete topology never changes
            # (test_topology_euler.py). chi_err_hard is the ONLY quantity that reflects the real
            # topology, so it is what the paper reports. Falls back to the soft gap for the edge
            # ablation loss, which has no hard monitor.
            if "topo_monitor" in _t:
                chi_gap = float(_t["topo_monitor"].detach())
            elif "chi_pred" in _t:
                chi_gap = float((_t["chi_pred"] - _t["chi_gt"]).abs().mean().detach())

        if not torch.isfinite(loss):
            log(f"step {step}: non-finite loss (l1={l_l1.item():.4f} ssim={l_ssim.item():.4f} "
                f"ce={l_ce.item():.4f} dice={l_dice.item():.4f}) -- SKIPPING")
            # NO scaler.update() HERE. It used to be called, and it CRASHED the run it exists to
            # save: GradScaler.update() asserts "No inf checks were recorded prior to update"
            # unless scale()/step() ran this iteration, and on the skip path they did not.
            # Reproduced on torch 2.9. The FIRST non-finite loss took the process down -- exactly
            # the event the guard is for. A bare `continue` is correct: the scaler's state is
            # untouched because nothing was scaled.
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step >= EMA_START:
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(EMA_DECAY).add_(pm.detach(), alpha=1 - EMA_DECAY)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
        elif step == EMA_START - 1:
            ema.load_state_dict(model.state_dict())

        if step % args.log_freq == 0:
            el = (time.time() - t0) / 60
            rec = {"step": step, "l1": round(float(l_l1), 4), "ssim": round(1 - float(l_ssim), 4),
                   "ce": round(float(l_ce), 4), "dice": round(float(l_dice), 4),
                   "topo": round(float(l_topo), 4), "total": round(float(loss), 4),
                   "chi_gap": None if chi_gap != chi_gap else round(chi_gap, 2),
                   "lr": round(sched.get_last_lr()[0], 6)}
            hist.append(rec)
            _chi = "" if rec["chi_gap"] is None else f" | dchi {rec['chi_gap']:.1f}"
            log(f"step {step:6d} | L1 {rec['l1']:.4f} | SSIM {rec['ssim']:.4f} | "
                f"CE {rec['ce']:.4f} | Dice {rec['dice']:.4f} | topo {rec['topo']:.4f}{_chi} | "
                f"tot {rec['total']:.4f} | {el:.1f}m | {(time.time()-t0)/max(step-start+1,1):.2f}s/it")
            # flush the structured curve data EVERY log step (not just at 5k checkpoints), so a
            # crash never loses the tail of the training curves the paper needs.
            with open(os.path.join(args.out_dir, "train_history.json"), "w") as _hf:
                json.dump(hist, _hf, indent=2)

        def save(tag):
            p = os.path.join(args.out_dir, f"cascaded_{tag}.pt")
            torch.save({"step": step, "model": model.state_dict(), "ema": ema.state_dict(),
                        "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                        "sched": sched.state_dict(), "config": CFG, "args": vars(args)}, p)
            with open(os.path.join(args.out_dir, "train_history.json"), "w") as f:
                json.dump(hist, f, indent=2)
            log(f"saved {p}")
            for old in sorted(glob.glob(os.path.join(args.out_dir, "cascaded_*.pt")))[:-2]:
                os.remove(old)

        if (step + 1) % args.save_freq == 0 or (step + 1) == args.n_iters:
            save(f"{step+1}")

        if (time.time() - t0) / 3600 > args.max_hours:
            save(f"{step+1}")
            log(f"\nTIME LIMIT at step {step}. Resume with --resume cascaded_{step+1}.pt")
            break

    log("done.")


if __name__ == "__main__":
    sys.exit(main())
