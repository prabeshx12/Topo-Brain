"""
Does skull-stripping starve the training signal?

The volumes are skull-stripped, so ~85% of each volume is a constant -1 background.
Patches are accepted if they contain >= min_brain_fraction (0.1) brain. So a patch can be
90% trivial background.

  * L1 self-corrects (the model learns "output -1" fast, then the background term -> 0).
  * SSIM does NOT: it SATURATES to 1.0 in flat matching regions, so a mostly-background
    patch yields near-zero SSIM loss no matter how bad the brain is.

This measures the ACTUAL brain-fraction distribution over the valid patch centres, exactly
as src/synthesis_dataset._get_valid_centers computes them, and then quantifies how much of
the SSIM signal is being spent on background.
"""
from pathlib import Path

import numpy as np
import nibabel as nib

H = Path(r"C:\Users\Asus\AppData\Local\Temp\claude"
         r"\d--BCT-pcampus-Semester-VII-Major-Project-topobrain-Topo-Brain"
         r"\f08ee9d9-7f5a-4048-bba5-59d5196c35a6\scratchpad\honest")

seg = np.rint(nib.load(str(H / "sub-06_ses-2_desc-preproc_T1w_seg.nii")).get_fdata()).astype(np.int32)
mask = seg > 0
P = 64
half = P // 2
print(f"volume {mask.shape}  brain = {100*mask.mean():.1f}% of the volume")

# integral image -> brain fraction of every possible patch, fast
ii = np.pad(mask.astype(np.int64), ((1, 0), (1, 0), (1, 0))).cumsum(0).cumsum(1).cumsum(2)


def patch_sum(z, y, x):
    z0, y0, x0 = z - half, y - half, x - half
    z1, y1, x1 = z0 + P, y0 + P, x0 + P
    return (ii[z1, y1, x1] - ii[z0, y1, x1] - ii[z1, y0, x1] - ii[z1, y1, x0]
            + ii[z0, y0, x1] + ii[z0, y1, x0] + ii[z1, y0, x0] - ii[z0, y0, x0])


D, Hh, W = mask.shape
rng = np.random.default_rng(0)
# sample candidate centres the way the dataset does (10k random candidates)
zs = rng.integers(half, D - half, 20000)
ys = rng.integers(half, Hh - half, 20000)
xs = rng.integers(half, W - half, 20000)
frac = patch_sum(zs, ys, xs) / (P ** 3)

MINF = 0.10
valid = frac >= MINF
vf = frac[valid]
print(f"\ncandidates                 : {len(frac):,}")
print(f"valid (brain fraction>={MINF}) : {valid.sum():,}  ({100*valid.mean():.1f}% of candidates)")
print("\nbrain fraction OF THE ACCEPTED PATCHES:")
for q in (5, 10, 25, 50, 75, 90, 95):
    print(f"   p{q:<3d} = {np.percentile(vf, q):.3f}")
print(f"   mean = {vf.mean():.3f}")
print(f"\n   fraction of accepted patches that are >50% background : "
      f"{100*(vf < 0.5).mean():.1f}%")
print(f"   fraction of accepted patches that are >75% background : "
      f"{100*(vf < 0.25).mean():.1f}%")

print("\n" + "=" * 62)
print("What this means for the SSIM term")
print("=" * 62)
print("SSIM saturates to ~1.0 wherever pred == target and both are flat.")
print("In a patch with brain fraction b, roughly (1-b) of the SSIM map is")
print("locked at ~1.0 and contributes NO gradient.")
print(f"\n   mean usable SSIM support = {vf.mean():.1%} of each patch")
print(f"   => the SSIM loss is diluted by a factor of ~{1/vf.mean():.1f}x")
if vf.mean() < 0.5:
    print("\n   VERDICT: the SSIM term IS substantially diluted -> mask the image")
    print("            losses to the brain, or raise min_brain_fraction.")
else:
    print("\n   VERDICT: dilution is modest; the accepted patches are brain-rich")
    print("            because valid centres cluster inside the brain.")
