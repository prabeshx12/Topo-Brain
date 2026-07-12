"""
PHASE 3 â€” train the rebuilt model on the FIXED pipeline.

  Architecture : regression U-Net + CASCADED seg head (seg computed FROM the synth image,
                 so anatomy/topology gradient reaches the generator -- the old parallel
                 design gave the image decoder EXACTLY 0.00e+00 topology gradient).
  Data         : FIXED dataloader -- random patch centres (was a pure function of the index
                 -> only 288 unique patches ever, 0.3% of the brain).
  Split        : val = fold 0 (sub-06), test = fold 1 (sub-07) -- DISTINCT, so checkpoints
                 can no longer be selected on the test subject.
  Loss         : L1 (unmasked) + BRAIN-MASKED SSIM + (CE + Dice) on the cascaded seg.
                 NO topology term yet -- this run is the honest baseline Phase 4 must beat.
                 SSIM is brain-masked because the volumes are skull-stripped: an average
                 accepted patch is only ~50% brain (measured), and SSIM saturates to 1.0 on
                 the flat background, diluting the term ~2x.
  Metrics      : honest (brain-only SSIM/PSNR; segmentation Dice; surface-to-surface HD95).

WHERE WE ACTUALLY STAND (revision/BENCHMARK_REALITY.md): we are BEHIND the published
methods, not ahead. Inverting FS-RWKV's own RMSE column puts us at RMSE 0.0973 vs their
0.0898 and LiteMamba's 0.0920. The closest competitor -- Acs & Zhuang (PLOS ONE 2025), same
dataset, same 10-fold LOOCV -- reports PSNR 23.25 / SSIM 0.737, roughly 3 dB above us.
Their SSIM is also not comparable to ours (they do not skull-strip; the UNC release is only
defaced + FLIRT-registered, and they score 2D central slices in [0,1]).

The prior B1 baseline reached PSNR 20.24 dB while trained on only 288 unique patches for
17k steps. This run has ~147x more unique patches plus a cascaded seg head whose gradient
actually reaches the generator. That is where the headroom must come from.
"""
import glob
import os
import subprocess
import sys
import zipfile

# ---------------------------------------------------------------- GPU compat FIRST
# Kaggle may hand us a Pascal P100 (sm_60) that the preinstalled torch (sm_70+) cannot run.
# monai MUST be installed AFTER this, pinned and with --no-deps, or pip drags torch back up
# to a build that drops sm_60 and the GPU silently dies.
import torch as _t


def log(*a):
    print(*a, flush=True)


_cap = _t.cuda.get_device_capability(0) if _t.cuda.is_available() else None
log(f"torch {_t.__version__} | cuda={_t.cuda.is_available()} | cap={_cap}")
if _t.cuda.is_available() and _cap and _cap[0] < 7 and not os.environ.get("TORCH_FIXED"):
    log(f"GPU cap {_cap} unsupported -> installing cu121 torch, then re-exec")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
                    "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121"],
                   check=False)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchvision"],
                   check=False)
    os.environ["TORCH_FIXED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "monai==1.4.0"],
               check=False)
log(f"post-install torch {_t.__version__} | cap="
    f"{_t.cuda.get_device_capability(0) if _t.cuda.is_available() else None}")

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402

# ---------------------------------------------------------------- code bundle
hits = glob.glob("/kaggle/input/**/src/model_cascaded.py", recursive=True)
if hits:
    CODE = os.path.dirname(os.path.dirname(hits[0]))
else:
    z = glob.glob("/kaggle/input/**/*.zip", recursive=True)[0]
    CODE = "/kaggle/working/repo"
    os.makedirs(CODE, exist_ok=True)
    zipfile.ZipFile(z).extractall(CODE)
sys.path.insert(0, CODE)
log("code root:", CODE)

# ---------------------------------------------------------------- build [-1,1] dataset
PRE = glob.glob("/kaggle/input/**/preprocessed-mri-aligned", recursive=True)[0]
SEGR = glob.glob("/kaggle/input/**/seg-masks", recursive=True)[0]
NORM = "/kaggle/working/norm"
os.makedirs(NORM, exist_ok=True)
PAIRS = "/kaggle/working/pairs_new.csv"


def dnorm(vol, clip_lo=0.5, clip_hi=99.5, p_lo=1.0, p_hi=99.0):
    """The exact 'diffusion' normalization from src/preprocessing.py."""
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
    v[vol <= 0] = -1.0
    return v


subs = sorted({os.path.basename(p) for p in glob.glob(f"{PRE}/sub-*")})
rows = ["subject,input_3t,target_7t,seg"]
for s in subs:
    i3 = f"{PRE}/{s}/ses-1/anat/{s}_ses-1_desc-preproc_T1w_registered.nii"
    t7 = f"{PRE}/{s}/ses-2/anat/{s}_ses-2_desc-preproc_T1w.nii"
    sg = f"{SEGR}/{s}/ses-2/anat/{s}_ses-2_desc-preproc_T1w_seg.nii"
    if not all(os.path.exists(p) for p in (i3, t7, sg)):
        log(f"  SKIP {s}")
        continue
    oi, ot, osg = f"{NORM}/{s}_3t.nii.gz", f"{NORM}/{s}_7t.nii.gz", f"{NORM}/{s}_seg.nii.gz"
    if not os.path.exists(oi):
        a = nib.load(i3); nib.save(nib.Nifti1Image(dnorm(a.get_fdata()), a.affine), oi)
        b = nib.load(t7); nib.save(nib.Nifti1Image(dnorm(b.get_fdata()), b.affine), ot)
        c = nib.load(sg)
        nib.save(nib.Nifti1Image(np.rint(c.get_fdata()).astype(np.uint8), c.affine), osg)
    rows.append(f"{s},{oi},{ot},{osg}")
open(PAIRS, "w").write("\n".join(rows) + "\n")
log(f"wrote {PAIRS} with {len(rows)-1} subjects")

# ---------------------------------------------------------------- train
sys.argv = [
    "train_cascaded.py",
    "--pairs-csv", PAIRS,
    "--config", f"{CODE}/configs/train_diffusion.yaml",
    "--out-dir", "/kaggle/working",
    "--n-iters", "40000",
    "--batch-size", "8",
    "--lr", "2e-4",
    "--lam-ssim", "0.5",
    "--lam-seg", "1.0",
    "--lam-topo", "0.0",        # Phase 3: NO topology term -- this is the honest baseline
    "--mask-ssim", "1",         # brain-only SSIM: unmasked is ~2x diluted by flat background
    "--val-fold", "0",          # sub-06
    "--test-fold", "1",         # sub-07  (DISTINCT -- no selection on the test subject)
    "--save-freq", "5000",
    "--max-hours", "10.3",
]
resume = sorted(glob.glob("/kaggle/input/**/cascaded_*.pt", recursive=True),
                key=lambda p: int(p.split("_")[-1].split(".")[0]))
if resume:
    sys.argv += ["--resume", resume[-1]]
    log(f"resuming from {resume[-1]}")

runpy_path = f"{CODE}/scripts/train_cascaded.py"
log(f"\nrunning {runpy_path}\n" + "=" * 70)
exec(compile(open(runpy_path).read(), runpy_path, "exec"), {"__name__": "__main__", "__file__": runpy_path})

