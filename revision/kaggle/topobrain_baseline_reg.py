"""
BASELINE B1 for reviewer R1.2 / R2.6 — deterministic 3D regression U-Net (no diffusion).

Fair-comparison protocol (this is what makes it credible):
  * SAME backbone: AnatomyGuidedUNet, identical config (features 32/64/128/256,
    attention, num_classes=4) -> controls for capacity, isolates the METHOD.
  * SAME data + SAME split: 10 paired subjects, diffusion-normalized to [-1,1],
    sub-06 held out (val_fold=0), identical patch sampler (synthesis_dataset).
  * SAME optimizer/AMP/EMA/grad-clip/batch/patch as the diffusion run.
  * DIFFERENT objective: no diffusion. Direct regression 3T -> 7T:
        pred = model(x=zeros, t=0, cond=3T)['prediction']
        loss = L1(pred, 7T) + lambda_seg * weighted-CE(seg_head, 7T-derived labels)
    No topology loss, no perceptual loss -> a genuine baseline, not our model.

The segmentation head is trained (plain CE) so the baseline also yields a tissue
map, enabling head-to-head Dice AND topology (Betti) comparison against TopoBrain.

Checkpoints every --save-freq to /kaggle/working so the run resumes across
Kaggle's session limit. Attach the previous run's output and pass --resume.
"""
import argparse
import glob
import importlib.util
import os
import subprocess
import sys
import time
import zipfile

def log(*a):
    print(*a, flush=True)


# --- GPU compat FIRST (before monai!). Kaggle may hand us a Pascal P100 (sm_60)
# that the preinstalled torch (sm_70+) cannot run. Install a cu121 torch that
# includes sm_60, re-exec once.
# CRITICAL: monai must be installed AFTER this, with --no-deps and pinned, or pip
# will pull torch back up to a version that drops sm_60 and silently break the GPU.
import torch as _t  # noqa: E402

_cap = _t.cuda.get_device_capability(0) if _t.cuda.is_available() else None
log(f"torch {_t.__version__} | cuda={_t.cuda.is_available()} | cap={_cap}")
if _t.cuda.is_available() and _cap and _cap[0] < 7 and not os.environ.get("TORCH_FIXED"):
    log(f"GPU cap {_cap} unsupported; installing cu121 torch and re-exec...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
                    "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchvision"], check=False)
    os.environ["TORCH_FIXED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

# monai pinned + --no-deps so it cannot upgrade torch out from under the P100
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "monai==1.4.0"],
               check=False)
_cap2 = _t.cuda.get_device_capability(0) if _t.cuda.is_available() else None
log(f"post-install torch {_t.__version__} | cap={_cap2} (must be usable)")

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from torch.amp import autocast, GradScaler  # noqa: E402

# --- locate code bundle ---------------------------------------------------
hits = glob.glob("/kaggle/input/**/src/model.py", recursive=True)
if hits:
    CODE = os.path.dirname(os.path.dirname(hits[0]))
else:
    z = glob.glob("/kaggle/input/**/*.zip", recursive=True)[0]
    CODE = "/kaggle/working/repo"
    os.makedirs(CODE, exist_ok=True)
    zipfile.ZipFile(z).extractall(CODE)
sys.path.insert(0, CODE)
log("code root:", CODE)

from src.model import AnatomyGuidedUNet  # noqa: E402
from src.synthesis_dataset import (  # noqa: E402
    create_synthesis_dataloaders,
    load_pairs_manifest,
    PatchConfig,
    SplitConfig,
)

CFG = yaml.safe_load(open(f"{CODE}/configs/train_diffusion.yaml"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log("device:", DEV, "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

# Fail fast if CUDA kernels can't actually run on this GPU (sm_60 vs torch build).
# Without this we'd waste minutes on data prep, then die (or silently crawl on CPU).
if DEV.type == "cuda":
    try:
        _ = (torch.randn(64, 64, device=DEV) @ torch.randn(64, 64, device=DEV)).sum().item()
        _c = torch.nn.Conv3d(1, 4, 3, padding=1).to(DEV)
        _ = _c(torch.randn(1, 1, 8, 8, 8, device=DEV)).sum().item()
        log("CUDA sanity check OK (matmul + conv3d ran on GPU)")
    except Exception as e:
        log(f"FATAL: CUDA present but kernels cannot run on this GPU -> {e}")
        raise SystemExit(1)

ap = argparse.ArgumentParser()
ap.add_argument("--n-iters", type=int, default=60000)
ap.add_argument("--save-freq", type=int, default=5000)
ap.add_argument("--lambda-seg", type=float, default=1.0)
ap.add_argument("--resume", type=str, default=None)
ap.add_argument("--max-hours", type=float, default=10.5)  # stop before Kaggle kills the session
args, _ = ap.parse_known_args()

# --- diffusion normalization (identical to src/preprocessing.py method="diffusion") ---
def diffusion_normalize(vol, clip_lo=0.5, clip_hi=99.5, p_lo=1.0, p_hi=99.0):
    v = vol.astype(np.float32).copy()
    roi = v[v > 0]
    if roi.size == 0:
        return v
    lo_c, hi_c = np.percentile(roi, clip_lo), np.percentile(roi, clip_hi)
    v = np.clip(v, lo_c, hi_c)
    roi = v[v > 0]
    lo, hi = np.percentile(roi, p_lo), np.percentile(roi, p_hi)
    if hi > lo:
        v = np.clip((v - lo) / (hi - lo), 0, 1) * 2.0 - 1.0
    bg = vol <= 0
    v[bg] = -1.0
    return v


# --- build normalized dataset + pairs csv (once per session) --------------
NORM = "/kaggle/working/norm"
os.makedirs(NORM, exist_ok=True)
PAIRS = "/kaggle/working/pairs_new.csv"

pre = glob.glob("/kaggle/input/**/preprocessed-mri-aligned", recursive=True)
pre = pre[0] if pre else None
segroot = glob.glob("/kaggle/input/**/seg-masks", recursive=True)
segroot = segroot[0] if segroot else None
log("preprocessed root:", pre, "| seg root:", segroot)

subs = sorted({os.path.basename(p) for p in glob.glob(f"{pre}/sub-*")})
log("subjects:", subs)

rows = ["subject,input_3t,target_7t,seg"]
for s in subs:
    i3 = f"{pre}/{s}/ses-1/anat/{s}_ses-1_desc-preproc_T1w_registered.nii"
    t7 = f"{pre}/{s}/ses-2/anat/{s}_ses-2_desc-preproc_T1w.nii"
    sg = f"{segroot}/{s}/ses-2/anat/{s}_ses-2_desc-preproc_T1w_seg.nii"
    if not (os.path.exists(i3) and os.path.exists(t7) and os.path.exists(sg)):
        log(f"  SKIP {s} (missing file)")
        continue
    oi, ot, os_ = f"{NORM}/{s}_3t.nii.gz", f"{NORM}/{s}_7t.nii.gz", f"{NORM}/{s}_seg.nii.gz"
    if not os.path.exists(oi):
        a = nib.load(i3)
        nib.save(nib.Nifti1Image(diffusion_normalize(a.get_fdata()), a.affine), oi)
        b = nib.load(t7)
        nib.save(nib.Nifti1Image(diffusion_normalize(b.get_fdata()), b.affine), ot)
        c = nib.load(sg)
        nib.save(nib.Nifti1Image(np.rint(c.get_fdata()).astype(np.uint8), c.affine), os_)
    rows.append(f"{s},{oi},{ot},{os_}")
open(PAIRS, "w").write("\n".join(rows) + "\n")
log(f"wrote {PAIRS} with {len(rows)-1} subjects")

# --- dataloader: SAME split as the diffusion run (val_fold=0 -> sub-06 held out) ---
D = CFG["dataset"]
pairs = load_pairs_manifest(PAIRS)
log(f"loaded {len(pairs)} pairs")
patch_cfg = PatchConfig(
    patch_size=tuple(D["patch_size"]),
    patches_per_volume=D["patches_per_volume"],
    min_brain_fraction=D["min_brain_fraction"],
    seed=D["seed"],
)
split_cfg = SplitConfig(
    n_folds=D["n_folds"],
    val_fold=D["val_fold"],
    test_fold=D["test_fold"],
    use_loocv=D["use_loocv"],
    seed=D["seed"],
)
train_loader, val_loader, _ = create_synthesis_dataloaders(
    pairs,
    config=patch_cfg,
    split_config=split_cfg,
    batch_size=D["batch_size"],
    num_workers=2,
    val_fold=D["val_fold"],
)
log(f"train batches/epoch: {len(train_loader)} | val: {len(val_loader)}")

# --- model (identical backbone) -------------------------------------------
model = AnatomyGuidedUNet(
    in_channels=CFG["model"]["in_channels"],
    cond_channels=CFG["model"].get("cond_channels", 1),
    out_channels=CFG["model"]["out_channels"],
    num_classes=CFG["model"]["num_classes"],
    features=CFG["model"]["features"],
    use_attention=CFG["model"]["use_attention"],
).to(DEV)
n_par = sum(p.numel() for p in model.parameters())
log(f"params: {n_par:,} ({n_par/1e6:.3f} M)  [matches TopoBrain backbone]")

ema = AnatomyGuidedUNet(
    in_channels=CFG["model"]["in_channels"],
    cond_channels=CFG["model"].get("cond_channels", 1),
    out_channels=CFG["model"]["out_channels"],
    num_classes=CFG["model"]["num_classes"],
    features=CFG["model"]["features"],
    use_attention=CFG["model"]["use_attention"],
).to(DEV)
ema.load_state_dict(model.state_dict())
for p in ema.parameters():
    p.requires_grad_(False)

opt = torch.optim.AdamW(model.parameters(), lr=float(CFG["training"]["lr"]),
                        weight_decay=float(CFG["training"]["weight_decay"]), eps=1e-5)
scaler = GradScaler(enabled=CFG["training"]["use_amp"], init_scale=4096)
CLS_W = torch.tensor([0.5, 2.0, 1.5, 1.0], device=DEV)  # same class weights as topology CE
EMA_DECAY = CFG["training"]["ema_decay"]
EMA_START = CFG["training"]["ema_start"]

start = 0
if args.resume:
    ck = glob.glob(args.resume) or glob.glob(f"/kaggle/input/**/{args.resume}", recursive=True)
    if ck:
        st = torch.load(ck[0], map_location=DEV, weights_only=False)
        model.load_state_dict(st["model"])
        ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["optimizer"])
        scaler.load_state_dict(st["scaler"])
        start = st["step"] + 1
        log(f"RESUMED from {ck[0]} at step {start}")

t0 = time.time()
it = iter(train_loader)
log(f"\n=== training regression baseline: steps {start} -> {args.n_iters} ===")
for step in range(start, args.n_iters):
    try:
        batch = next(it)
    except StopIteration:
        it = iter(train_loader)
        batch = next(it)

    x7 = batch["target"].to(DEV)      # 7T target
    x3 = batch["input"].to(DEV)       # 3T conditioning
    seg = batch["seg"].to(DEV)
    if seg.dim() == 5:
        seg = seg.squeeze(1)
    seg = seg.long().clamp_(0, CFG["model"]["num_classes"] - 1)

    opt.zero_grad(set_to_none=True)
    with autocast(device_type=DEV.type, enabled=CFG["training"]["use_amp"]):
        # NO diffusion: deterministic map 3T -> 7T. x=zeros, t=0, cond=3T.
        zeros = torch.zeros_like(x3)
        t0_b = torch.zeros(x3.shape[0], device=DEV, dtype=torch.long)
        out = model(zeros, t0_b, x3)
        pred = out["prediction"]
        seg_logits = out["segmentation"]

    pred = pred.float()
    seg_logits = seg_logits.float().clamp(-50, 50)
    loss_img = F.l1_loss(pred, x7.float())
    loss_seg = F.cross_entropy(seg_logits, seg, weight=CLS_W)
    loss = loss_img + args.lambda_seg * loss_seg

    if not torch.isfinite(loss):
        log(f"step {step}: non-finite loss, skipping")
        scaler.update()
        continue

    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["training"]["grad_clip"])
    scaler.step(opt)
    scaler.update()

    if step >= EMA_START:
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(EMA_DECAY).add_(pm.detach(), alpha=1 - EMA_DECAY)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
    elif step == EMA_START - 1:
        ema.load_state_dict(model.state_dict())

    if step % 50 == 0:
        el = (time.time() - t0) / 60
        log(f"step {step:6d} | L1 {loss_img.item():.4f} | segCE {loss_seg.item():.4f} "
            f"| total {loss.item():.4f} | {el:.1f} min | {(time.time()-t0)/max(step-start+1,1):.2f} s/it")

    if (step + 1) % args.save_freq == 0 or (step + 1) == args.n_iters:
        p = f"/kaggle/working/baseline_reg_{step+1}.pt"
        torch.save({"step": step, "model": model.state_dict(), "ema": ema.state_dict(),
                    "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "config": CFG, "baseline": "regression_unet_no_diffusion"}, p)
        log(f"saved {p}")
        for old in sorted(glob.glob("/kaggle/working/baseline_reg_*.pt"))[:-2]:
            os.remove(old)  # keep only last 2 to stay under output limits

    if (time.time() - t0) / 3600 > args.max_hours:
        p = f"/kaggle/working/baseline_reg_{step+1}.pt"
        torch.save({"step": step, "model": model.state_dict(), "ema": ema.state_dict(),
                    "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "config": CFG, "baseline": "regression_unet_no_diffusion"}, p)
        log(f"\nTIME LIMIT reached at step {step}. Saved {p}. Resume with --resume on next run.")
        break

log("done.")
