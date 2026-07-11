"""
TopoBrain revision — re-evaluate sub-06 with the 110k EMA checkpoint.

Purpose (IET resubmission):
  1. DECIDE D8: does the surviving step-110,000 EMA checkpoint reproduce the
     paper's Table 1 (SSIM 0.8991, PSNR 20.00, fg Dice 0.7682, fg HD95 3.31,
     brain Dice 0.9436, brain HD95 2.05)? The paper attributes those numbers to
     a "105k" checkpoint that does not exist in the Kaggle account.
  2. Produce TRUE TOPOLOGICAL METRICS (Betti b0/b1/b2 + Euler characteristic)
     for the predicted vs ground-truth tissue segmentation  -> reviewer R2.3/R2.8.

Runs ONE tiled inference per sampler, then computes every metric off it.
Nothing is fabricated: whatever it prints is what the checkpoint produces.
"""
import glob
import json
import os
import subprocess
import sys
import zipfile

# --------------------------------------------------------------------------
# 0. Environment
# --------------------------------------------------------------------------
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gudhi"], check=False)

# --- GPU compatibility: Kaggle may assign a Pascal P100 (sm_60), which the
# preinstalled torch (sm_70+) cannot run. If so, install a cu121 torch that
# includes sm_60 and re-exec once. torchvision is not needed for inference,
# so we drop it to avoid an ABI mismatch.
import torch as _torch  # noqa: E402
_cap = _torch.cuda.get_device_capability(0) if _torch.cuda.is_available() else None
print(f"torch {_torch.__version__} | cuda={_torch.cuda.is_available()} | cap={_cap}")
if _torch.cuda.is_available() and _cap is not None and _cap[0] < 7 and not os.environ.get("TORCH_FIXED"):
    print(f"GPU capability {_cap} unsupported by torch {_torch.__version__}; installing cu121 build and re-exec...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
                    "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchvision"], check=False)
    os.environ["TORCH_FIXED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

# --- Diagnostics: what is actually mounted? -------------------------------
print("=== /kaggle/input contents ===")
for root in sorted(glob.glob("/kaggle/input/*")):
    print(" ", root)
    for sub in sorted(glob.glob(root + "/*"))[:12]:
        print("     ", sub)

# --- Robustly locate the code bundle anywhere under /kaggle/input ---------
CODE_ROOT = None
hits = glob.glob("/kaggle/input/**/src/model.py", recursive=True)
if hits:
    CODE_ROOT = os.path.dirname(os.path.dirname(hits[0]))  # .../<root>
else:
    zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
    if zips:
        REPO = "/kaggle/working/repo"
        os.makedirs(REPO, exist_ok=True)
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall(REPO)
        CODE_ROOT = REPO
assert CODE_ROOT and os.path.isfile(os.path.join(CODE_ROOT, "src", "model.py")), \
    f"could not locate code bundle (CODE_ROOT={CODE_ROOT})"
sys.path.insert(0, CODE_ROOT)
print("code root:", CODE_ROOT)

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from scipy import ndimage  # noqa: E402

from src.model import AnatomyGuidedUNet  # noqa: E402
from src.diffusion import GaussianDiffusion  # noqa: E402
import scripts.evaluate_full_volume as ev  # noqa: E402

try:
    from scripts.compute_betti import betti_3d
    HAVE_BETTI = True
except Exception as e:  # pragma: no cover
    print("WARN: could not import betti_3d:", e)
    HAVE_BETTI = False

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEV, "| gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

# --------------------------------------------------------------------------
# 1. Paths
# --------------------------------------------------------------------------
def find_one(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    return hits[0] if hits else None


IN_3T = find_one("sub-06_ses-1_desc-preproc_T1w_registered.nii*")
TGT_7T = find_one("sub-06_ses-2_desc-preproc_T1w.nii*")     # 7T target (not *_seg, not *_registered)
BMASK = find_one("sub-06_ses-2_desc-brainmask_T1w.nii*")
GT_SEG = find_one("sub-06_ses-2_desc-preproc_T1w_seg.nii*")
for nm, pth in [("IN_3T", IN_3T), ("TGT_7T", TGT_7T), ("GT_SEG", GT_SEG)]:
    assert pth, f"could not locate {nm} under /kaggle/input"

all_pt = glob.glob("/kaggle/input/**/*.pt", recursive=True)
ckpts = [p for p in all_pt if ("ckpt" in p.lower() or "checkpoint" in p.lower())] or all_pt
assert ckpts, f"no .pt checkpoint found under /kaggle/input (searched {len(all_pt)})"
CKPT = ckpts[0]
print("checkpoint:", CKPT)
for p in (IN_3T, TGT_7T, GT_SEG):
    print(f"  exists {os.path.exists(p)} : {p}")

CFG = yaml.safe_load(open(f"{CODE_ROOT}/configs/train_diffusion.yaml"))

# --------------------------------------------------------------------------
# 2. Model + EMA weights
# --------------------------------------------------------------------------
model = AnatomyGuidedUNet(
    in_channels=CFG["model"]["in_channels"],
    cond_channels=CFG["model"].get("cond_channels", 1),
    out_channels=CFG["model"]["out_channels"],
    num_classes=CFG["model"]["num_classes"],
    features=CFG["model"]["features"],
    use_attention=CFG["model"]["use_attention"],
).to(DEV)
diffusion = GaussianDiffusion(
    model=model,
    timesteps=CFG["diffusion"]["timesteps"],
    beta_schedule=CFG["diffusion"]["beta_schedule"],
).to(DEV)

ck = torch.load(CKPT, map_location=DEV, weights_only=False)
print("checkpoint step:", ck.get("step"))
assert "ema" in ck, "no EMA weights in checkpoint"
model.load_state_dict(ck["ema"])
model.eval()
n_par = sum(p.numel() for p in model.parameters())
print(f"loaded EMA weights | params = {n_par:,} ({n_par/1e6:.3f} M)")

# --------------------------------------------------------------------------
# 3. Volumes + masks
# --------------------------------------------------------------------------
inp_nii = nib.load(IN_3T)
input_vol = inp_nii.get_fdata().astype(np.float32)
affine = inp_nii.affine
target_vol = nib.load(TGT_7T).get_fdata().astype(np.float32)
gt_seg = np.rint(nib.load(GT_SEG).get_fdata()).astype(np.int32)


def diffusion_normalize(vol, mask, p_lo=1.0, p_hi=99.0, clip_lo=0.5, clip_hi=99.5):
    """Exact replica of src/preprocessing.py method='diffusion': clip to
    [p0.5,p99.5], percentile-scale [p1,p99] -> [0,1] -> [-1,1], background -> -1.
    The model REQUIRES inputs in [-1,1] (synthesis_dataset._robust_normalize)."""
    roi = vol[mask > 0] if mask is not None and mask.any() else vol[vol > vol.min()]
    lo_c, hi_c = np.percentile(roi, clip_lo), np.percentile(roi, clip_hi)
    v = np.clip(vol, lo_c, hi_c)
    roi = v[mask > 0] if mask is not None and mask.any() else v
    lo, hi = np.percentile(roi, p_lo), np.percentile(roi, p_hi)
    if hi > lo:
        v = np.clip((v - lo) / (hi - lo), 0, 1) * 2.0 - 1.0
    if mask is not None and mask.any():
        v[mask == 0] = -1.0
    return v.astype(np.float32)


# Brain mask for normalization ROI (prefer the provided brainmask NII).
_norm_mask = None
if BMASK and os.path.exists(BMASK):
    _bm = nib.load(BMASK).get_fdata() > 0.5
    if _bm.shape == input_vol.shape:
        _norm_mask = _bm
print("input range (raw):", float(input_vol.min()), float(input_vol.max()))
print("target range (raw):", float(target_vol.min()), float(target_vol.max()))
if input_vol.min() < -1.1 or input_vol.max() > 1.1:
    print(">>> input not in [-1,1]; applying diffusion normalization (model requirement)")
    input_vol = diffusion_normalize(input_vol, _norm_mask)
    target_vol = diffusion_normalize(target_vol, _norm_mask)

spacing = tuple(np.abs(np.diag(affine)[:3]))
print("shape:", input_vol.shape, "| spacing:", spacing)
print("input range (norm):", float(input_vol.min()), float(input_vol.max()))
print("target range (norm):", float(target_vol.min()), float(target_vol.max()))
print("gt seg labels:", np.unique(gt_seg))

masks = [("Threshold(-0.95)", ev.make_brain_mask(input_vol, -0.95))]
masks.append(("SegMaskGT(seg>0)", gt_seg > 0))
if BMASK and os.path.exists(BMASK):
    bm = nib.load(BMASK).get_fdata() > 0.5
    if bm.shape == input_vol.shape:
        masks.append(("BrainMaskNII", bm))
for lbl, m in masks:
    print(f"  mask {lbl}: {int(m.sum()):,} voxels")

# --------------------------------------------------------------------------
# 4. Topology helpers
# --------------------------------------------------------------------------
TISSUE = {1: "CSF", 2: "GM", 3: "WM"}


def dice_binary(a, b):
    inter = np.logical_and(a, b).sum()
    den = a.sum() + b.sum()
    return float(2.0 * inter / den) if den else float("nan")


def cc_stats(binary):
    lab, n = ndimage.label(binary)
    if n == 0:
        return 0, 0.0
    sizes = ndimage.sum(binary, lab, range(1, n + 1))
    return int(n), float(sizes.max() / binary.sum())


def topo_report(seg, tag):
    out = {}
    for lb, name in TISSUE.items():
        b = seg == lb
        if b.sum() == 0:
            continue
        n_cc, lcc = cc_stats(b)
        rec = {"voxels": int(b.sum()), "cc": n_cc, "largest_cc_frac": round(lcc, 4)}
        if HAVE_BETTI:
            try:
                b0, b1, b2 = betti_3d(b)
                rec.update({"betti0": int(b0), "betti1": int(b1), "betti2": int(b2),
                            "euler_chi": int(b0 - b1 + b2)})
            except Exception as e:
                rec["betti_error"] = str(e)
        out[name] = rec
        print(f"  [{tag}] {name}: {rec}")
    return out


# --------------------------------------------------------------------------
# 5. Run inference for each sampler, compute everything
# --------------------------------------------------------------------------
PAPER = {"ssim": 0.8991, "psnr": 20.00, "dice_fg": 0.7682, "hd95_fg": 3.31,
         "dice_brain": 0.9436, "hd95_brain": 2.05}

results = {"checkpoint_step": int(ck.get("step", -1)), "params": int(n_par),
           "paper_table1": PAPER, "runs": {}}

for sampler, steps in [("ddim", 50)]:  # DDIM-50 = the paper's inference sampler
    print("\n" + "=" * 78)
    print(f"TILED INFERENCE — sampler={sampler}" + (f" (ddim_steps={steps})" if steps else " (185 steps)"))
    print("=" * 78, flush=True)
    try:
        pred_vol, pred_seg = ev.tiled_inference(
            diffusion, input_vol, DEV,
            patch_size=CFG["dataset"]["patch_size"][0],
            overlap=32, sampler=sampler, ddim_steps=(steps or 50), n_samples=1,
        )
    except Exception as e:
        print(f"!! {sampler} inference failed: {e}")
        results["runs"][sampler] = {"error": str(e)}
        continue

    run = {"image_metrics": {}}
    for lbl, m in masks:
        ms = ev.compute_metric_set(lbl, m, pred_vol, target_vol, input_vol, spacing)
        run["image_metrics"][lbl] = {
            "ssim": round(float(ms["ssim_masked"]), 4),
            "psnr": round(float(ms["psnr_masked"]), 2),
            "dice": round(float(ms["dice"]), 4),
            "hd95_mm": round(float(ms["hd95_mm"]), 2),
        }
        print(f"  [{lbl}] SSIM={ms['ssim_masked']:.4f}  PSNR={ms['psnr_masked']:.2f}  "
              f"Dice={ms['dice']:.4f}  HD95={ms['hd95_mm']:.2f}mm")

    # per-tissue segmentation agreement
    print("\n  per-tissue Dice (pred_seg vs gt_seg):")
    seg_dice = {}
    for lb, name in TISSUE.items():
        d = dice_binary(pred_seg == lb, gt_seg == lb)
        seg_dice[name] = round(d, 4)
        print(f"    {name}: {d:.4f}")
    run["tissue_dice"] = seg_dice

    print("\n  TRUE TOPOLOGY — predicted seg:")
    run["topology_pred"] = topo_report(pred_seg, "pred")
    print("\n  TRUE TOPOLOGY — ground-truth seg:")
    run["topology_gt"] = topo_report(gt_seg, "gt")

    results["runs"][sampler] = run

    nib.save(nib.Nifti1Image(pred_vol.astype(np.float32), affine),
             f"/kaggle/working/sub-06_pred7T_{sampler}.nii.gz")
    nib.save(nib.Nifti1Image(pred_seg.astype(np.uint8), affine),
             f"/kaggle/working/sub-06_predseg_{sampler}.nii.gz")

    with open("/kaggle/working/eval_110k_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved partial results after sampler={sampler}", flush=True)

# --------------------------------------------------------------------------
# 6. Verdict on D8
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("D8 VERDICT — does 110k reproduce the paper's Table 1?")
print("=" * 78)
for sampler, run in results["runs"].items():
    if "image_metrics" not in run:
        continue
    for lbl, m in run["image_metrics"].items():
        ds = abs(m["ssim"] - PAPER["ssim"])
        print(f"  {sampler:5s} {lbl:18s} SSIM={m['ssim']:.4f} (paper {PAPER['ssim']}, d={ds:.4f}) "
              f"PSNR={m['psnr']:.2f} Dice={m['dice']:.4f} HD95={m['hd95_mm']:.2f}")
print("\nIf no combination lands near SSIM 0.8991 / HD95 2.05, the reported")
print("numbers came from a checkpoint that no longer exists.")

with open("/kaggle/working/eval_110k_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nWrote /kaggle/working/eval_110k_results.json")
