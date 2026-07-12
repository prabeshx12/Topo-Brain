"""
Proves src/metrics_honest.py is immune to the bugs that inflated the published numbers.

The four claims under test (see revision/AUDIT.md):
  1. Pasting ground truth into the background must NOT change the score.
     (Old code: pred[~mask] = target[~mask], then averaged over the whole volume ->
      +7.10 dB PSNR and 75.7% of SSIM windows scored exactly 1.0 for free.)
  2. A perfect prediction must score HD95 = 0.0 mm.
     (Old code used the outer dilation shell -> a perfect prediction scored 1.0 mm.)
  3. HD95 must be surface-to-SURFACE, so a predicted surface lying deep INSIDE the target
     is penalised. (Old code used distance-to-OBJECT, which is 0 inside the target.)
  4. Anatomy metrics must run on SEGMENTATIONS. Pure noise must NOT look competitive.
     (Old code thresholded intensity at >0; pure noise scored HD95 4.36 mm vs the
      published 3.31 mm.)

CPU, seconds.  Run:  python scripts/test_metrics_honest.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mh", ROOT / "src" / "metrics_honest.py")
mh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mh)

rng = np.random.default_rng(0)
D = 64
# a brain-like blob occupying a minority of the volume, as in the real data (~19.5%)
zz, yy, xx = np.mgrid[0:D, 0:D, 0:D]
r = np.sqrt((zz - D / 2) ** 2 + (yy - D / 2) ** 2 + (xx - D / 2) ** 2)
mask = r < 20
frac = mask.mean()

target = np.full((D, D, D), -1.0, dtype=np.float32)      # background = -1
target[mask] = rng.uniform(-0.5, 1.0, mask.sum())        # brain texture

pred = target.copy()
pred[mask] += rng.normal(0, 0.25, mask.sum())            # error INSIDE the brain only
pred = np.clip(pred, -1, 1)
pred[~mask] = rng.normal(0, 0.30, (~mask).sum())         # junk in the background

print("=" * 72)
print("1) Pasting ground truth into the background must NOT change the score")
print("=" * 72)
print(f"   brain fraction f = {frac:.3f}  ->  background is {100*(1-frac):.1f}% of the volume")

psnr_plain = mh.masked_psnr(pred, target, mask)
ssim_plain = mh.masked_ssim(pred, target, mask)

pred_pasted = pred.copy()
pred_pasted[~mask] = target[~mask]                       # THE OLD CHEAT
psnr_pasted = mh.masked_psnr(pred_pasted, target, mask)
ssim_pasted = mh.masked_ssim(pred_pasted, target, mask)

print(f"   honest  : PSNR {psnr_plain:6.2f} dB   SSIM {ssim_plain:.4f}")
print(f"   + paste : PSNR {psnr_pasted:6.2f} dB   SSIM {ssim_pasted:.4f}")
assert abs(psnr_plain - psnr_pasted) < 1e-6, "PSNR must be unaffected by background pasting"
assert abs(ssim_plain - ssim_pasted) < 5e-3, "SSIM must be (essentially) unaffected by pasting"
print("   [OK] pasting the GT into the background buys NOTHING\n")

# and show what the OLD whole-volume method would have reported
def old_whole_volume(p, t, m):
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    q = p.copy()
    q[~m] = t[~m]                                        # paste
    return psnr(t, q, data_range=2.0), ssim(t, q, data_range=2.0)

old_psnr, old_ssim = old_whole_volume(pred, target, mask)
inflation = 10 * np.log10(1.0 / frac)
print(f"   OLD (paste + whole-volume average): PSNR {old_psnr:.2f} dB   SSIM {old_ssim:.4f}")
print(f"   honest brain-only                 : PSNR {psnr_plain:.2f} dB   SSIM {ssim_plain:.4f}")
print(f"   predicted inflation 10*log10(1/f) = {inflation:.2f} dB   "
      f"| measured = {old_psnr - psnr_plain:.2f} dB")
assert old_psnr > psnr_plain + 3.0, "the old method really was inflated"
print("   [OK] the inflation is reproduced and eliminated\n")

print("=" * 72)
print("2) A perfect prediction must score HD95 = 0.0 mm")
print("=" * 72)
seg_gt = np.zeros((D, D, D), np.int32)
seg_gt[r < 20] = 3                      # WM core
seg_gt[(r >= 20) & (r < 24)] = 2        # GM ribbon
seg_gt[(r >= 24) & (r < 26)] = 1        # CSF rim

hd_perfect = mh.hd95_surface(seg_gt == 3, seg_gt == 3)
print(f"   HD95(perfect prediction) = {hd_perfect:.3f} mm")
assert hd_perfect == 0.0, "a perfect prediction must score exactly 0 mm"
print("   [OK] (the old outer-shell surface scored 1.0 mm here)\n")

print("=" * 72)
print("3) HD95 must be surface-to-SURFACE (distance-to-OBJECT is blind inside the target)")
print("=" * 72)


def old_hd95(pred_b, tgt_b, spacing=(1.0, 1.0, 1.0)):
    """verbatim reimplementation of the old surface->OBJECT logic"""
    ps = ndimage.binary_dilation(pred_b) ^ pred_b
    ts = ndimage.binary_dilation(tgt_b) ^ tgt_b
    d1 = ndimage.distance_transform_edt(~tgt_b, sampling=spacing)[ps]   # ~OBJECT, not ~surface
    d2 = ndimage.distance_transform_edt(~pred_b, sampling=spacing)[ts]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


# (a) the blind spot, isolated: a predicted-surface voxel deep INSIDE the target
wm = seg_gt == 3
pred_small = ndimage.binary_erosion(wm, iterations=5)
ps = ndimage.binary_dilation(pred_small) ^ pred_small          # the old "surface"
d_to_object = ndimage.distance_transform_edt(~wm)[ps]          # OLD: distance to the OBJECT
true_surf = mh._surface(wm)
d_to_surface = ndimage.distance_transform_edt(~true_surf)[ps]  # NEW: distance to the SURFACE
print("   prediction eroded by 5 voxels; its surface lies ~5 mm inside the target")
print(f"   OLD  distance-to-OBJECT  : mean {d_to_object.mean():.3f} mm  "
      f"(fraction exactly 0: {100*(d_to_object==0).mean():.1f}%)   <- BLIND")
print(f"   NEW  distance-to-SURFACE : mean {d_to_surface.mean():.3f} mm")
assert (d_to_object == 0).mean() > 0.99, "old metric should score ~0 for interior surfaces"
assert d_to_surface.mean() > 2.0, "the corrected metric must report a real, non-zero distance"
print("   [OK] the old code scores an interior surface as 0 mm regardless of its depth\n")

# (b) why it matters HERE: the real predicted segs are FRAGMENTED, so they have huge
#     interior surface -> a mass of spurious zeros -> the 95th percentile is DEFLATED.
holes = rng.random(wm.shape) < 0.35
frag = wm & ~holes                       # sponge-like prediction, entirely inside the target
hd_old_frag = old_hd95(frag, wm)
hd_new_frag = mh.hd95_surface(frag, wm)
print("   fragmented prediction (35% of target voxels dropped -> huge interior surface):")
print(f"   OLD HD95 = {hd_old_frag:.3f} mm      NEW HD95 = {hd_new_frag:.3f} mm")
assert hd_new_frag > hd_old_frag, "the corrected HD95 must penalise fragmentation more"
print("   [OK] fragmentation is no longer rewarded with spurious zero distances")
print("        (this is exactly the regime the real predicted segs are in)\n")

print("=" * 72)
print("4) Anatomy is scored on SEGMENTATIONS -- pure noise must not look competitive")
print("=" * 72)
noise_seg = rng.integers(0, 4, (D, D, D))
d_noise = mh.tissue_dice(noise_seg, seg_gt)
d_perfect = mh.tissue_dice(seg_gt, seg_gt)
print(f"   Dice(pure noise)  CSF {d_noise[1]:.3f}  GM {d_noise[2]:.3f}  WM {d_noise[3]:.3f}")
print(f"   Dice(perfect)     CSF {d_perfect[1]:.3f}  GM {d_perfect[2]:.3f}  WM {d_perfect[3]:.3f}")
assert all(v < 0.35 for v in d_noise.values()), "noise must score poorly on a real Dice"
assert all(abs(v - 1.0) < 1e-9 for v in d_perfect.values()), "perfect must be 1.0"
print("   [OK] (the old INTENSITY-threshold Dice gave pure noise HD95 4.36 mm vs the")
print("         published 3.31 mm -- i.e. it barely beat noise)\n")

print("=" * 72)
print("ALL HONEST-METRIC GUARANTEES VERIFIED")
print("=" * 72)
