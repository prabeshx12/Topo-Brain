"""
The protocol-sensitivity experiment on REAL data: our cascaded model, held-out subject sub-06.

The phantom version (test_metrics_protocols.py) showed the evaluation convention is worth ~9 dB.
This runs the same sweep on the actual prediction, so the number in the paper is measured, not
simulated.

Usage:
  python scripts/protocol_sensitivity_real.py --pred <pred7T.nii.gz> --gt <7T.nii> --seg <seg.nii>
"""
import argparse
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mp", ROOT / "src" / "metrics_protocols.py")
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)

def dnorm(vol, clip_lo=0.5, clip_hi=99.5, p_lo=1.0, p_hi=99.0):
    """The pipeline's normalisation -- IDENTICAL to scripts/eval_cascaded.py.

    The prediction comes out of a tanh in [-1,1]. The ground truth on disk is NOT: it is raw
    preprocessed intensity (sub-06 spans [-1.512, 1.931]). Scoring one against the other without
    this transform compares two different scales and produces garbage -- it read brain SSIM 0.079
    and PSNR 12.93 before this was applied. Same normalisation, or no comparison.
    """
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


ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True)
ap.add_argument("--gt", required=True)
ap.add_argument("--seg", required=True, help="GT tissue seg; class 0 = background -> brain mask")
ap.add_argument("--out", default=None)
a = ap.parse_args()

pred = nib.load(a.pred).get_fdata().astype(np.float32)          # already tanh -> [-1,1]
gt_raw = nib.load(a.gt).get_fdata().astype(np.float32)
gt = dnorm(gt_raw)                                              # SAME transform as training/eval
seg = np.rint(nib.load(a.seg).get_fdata()).astype(np.int32)
brain = seg > 0
print(f"  [norm] GT raw range [{gt_raw.min():.3f}, {gt_raw.max():.3f}] "
      f"-> dnorm -> [{gt.min():.3f}, {gt.max():.3f}]  (matches the prediction's tanh range)")

print("=" * 92)
print("PROTOCOL SENSITIVITY ON REAL DATA — cascaded model, held-out sub-06")
print("=" * 92)
print(f"  pred  {Path(a.pred).name}   {pred.shape}  range [{pred.min():.3f}, {pred.max():.3f}]")
print(f"  gt    {Path(a.gt).name}   {gt.shape}  range [{gt.min():.3f}, {gt.max():.3f}]")
print(f"  brain {100 * brain.mean():.1f} % of the volume "
      f"({int(brain.sum()):,} of {gt.size:,} voxels) -- so {100 * (1 - brain.mean()):.1f} % is "
      f"background the model does not have to synthesise")
assert pred.shape == gt.shape == seg.shape, "shape mismatch"

R = mp.score_all_protocols(pred, gt, brain)
base = R["A_brain_3d"]["psnr"]

rows = [
    ("A  brain-masked 3D   (HONEST)", "A_brain_3d", "what we advocate"),
    ("B  whole-volume 3D", "B_volume_3d", "field convention"),
    ("C  2D per-slice, axis0 (ALL)", "C_slice2d_axis0", "Acs & Zhuang"),
    ("C  2D per-slice, axis1 (ALL)", "C_slice2d_axis1", "Acs & Zhuang"),
    ("C  2D per-slice, axis2 (ALL)", "C_slice2d_axis2", "Acs & Zhuang"),
    ("D  101 central axial slices", "D_central_101", "FS-RWKV"),
    ("E  central 128x128 crop", "E_central_crop", "LiteMamba Tab.4"),
    ("F  GT pasted outside brain", "F_gt_pasted", "THE PUBLISHED BUG"),
]

print()
print(f"  {'convention':32} {'PSNR':>8} {'SSIM':>8} {'vs honest':>11} | "
      f"{'PSNR*':>7} {'SSIM*':>7} {'empty':>6}  source")
print("  " + "-" * 104)
finite = []
for nm, key, src in rows:
    d = R[key]
    p, s = d.get("psnr", float("nan")), d.get("ssim", float("nan"))
    if np.isfinite(p):
        finite.append(p)
    delta = "--" if key == "A_brain_3d" else f"{p - base:+.2f} dB"
    ss = "      --" if not np.isfinite(s) else f"{s:8.4f}"
    bp = d.get("psnr_brain", float("nan"))
    bs = d.get("ssim_brain", float("nan"))
    bps = "      -" if not np.isfinite(bp) else f"{bp:7.2f}"
    bss = "      -" if not np.isfinite(bs) else f"{bs:7.4f}"
    ne = d.get("n_empty_slices", None)
    nes = "     -" if ne is None else f"{ne:6d}"
    print(f"  {nm:32} {p:8.2f} {ss} {delta:>11} | {bps} {bss} {nes}  {src}")

spread = max(finite) - min(finite)
print()
print("  * = the SAME statistic restricted to slices that actually contain brain.")
print("    The gap between the two columns is pure empty-slice inflation.")
print()
print(f"  >>> SPREAD ACROSS CONVENTIONS: {spread:.2f} dB — ONE model, ONE prediction.")
print(f"  >>> The honest (brain-masked 3D) number is the LOWEST: {base:.2f} dB.")
print()
print("  A per-slice mean over ALL slices averages in the near-empty ones at the ends of the")
print("  volume, whose MSE is tiny and whose PSNR is therefore enormous. NO published paper on")
print("  this dataset states any slice-exclusion rule. We state ours, and we report both.")

out = Path(a.out) if a.out else Path(a.pred).parent / "protocol_sensitivity.json"
out.write_text(json.dumps({"spread_db": spread, "honest_psnr": base, "protocols": R}, indent=2))
print(f"\n  wrote {out}")
