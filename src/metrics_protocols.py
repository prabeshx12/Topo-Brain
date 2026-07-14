"""
Score ONE prediction under EVERY published evaluation convention on this benchmark.

WHY THIS EXISTS
---------------
The three papers on the UNC paired 3T-7T dataset do not measure the same quantity, and none of
them can be compared to the others:

  Acs & Zhuang (PLOS ONE 2025)   23.25 / 0.737   2D per-slice, ALL 308 transverse slices,
                                                 full-head, NO mask. (Their coronal is 22.48
                                                 and sagittal 22.05 -- 23.25 is their best plane.)
  FS-RWKV      (BIBM 2025)       21.00 / 0.726   2D, 101 CENTRAL axial slices, resized 256x256.
  LiteMamba    (Front.Neuroanat  20.82 / 0.711   2D, full 256x256 image.
                2026)            23.87 / 0.719   ...THE SAME MODEL, central 128x128 crop.

That last pair is from LiteMamba's own Table 4: +3.05 dB from the evaluation region ALONE, same
weights, same data. Acs & Zhuang's headline falls INSIDE that spread. Corroborating: SSIM (which
is far less sensitive to background than PSNR) agrees across all three papers to +/-0.02 while
PSNR diverges by 2.3 dB -- when one metric agrees and the other doesn't, the disagreement is in
the MEASUREMENT.

So: rather than argue about it, MEASURE it. Score one model under all of them and publish the
spread. That is a result the field does not currently have.

THE CONVENTIONS
---------------
  A  brain_3d        3D, restricted to the brain mask.          <- honest; what we advocate
  B  volume_3d       3D, whole volume, unmasked.
  C  slice_2d_*      2D per-slice, ALL slices, per orientation. <- Acs & Zhuang
  D  central_101     2D, 101 central axial slices.              <- FS-RWKV
  E  central_crop    2D, central axial, central 128x128 crop.   <- LiteMamba Table 4
  F  gt_pasted       the ORIGINAL broken harness (GT pasted outside the ROI). For reference only.

HONEST SCOPE. Our volumes are SKULL-STRIPPED; theirs are full-head. No scoring convention can
repair a data difference. This module therefore measures HOW MUCH THE CONVENTION MOVES THE
NUMBER on a fixed prediction -- it does NOT license a claim of parity with their numbers.

A DEGENERACY THE PAPERS DO NOT MENTION. Per-slice PSNR is undefined (MSE = 0 -> PSNR = inf) on a
slice where prediction and target are both constant -- e.g. the air-only end slices of a head
volume, and EVERY empty slice of a skull-stripped one. No paper states an exclusion rule. We
count these explicitly and report the mean over finite slices, because silently dropping them
(or letting them saturate) is precisely how a per-slice mean gets inflated.
"""
from typing import Dict, Optional

import numpy as np

__all__ = ["to_unit", "psnr_2d", "ssim_2d", "score_all_protocols"]

_EPS = 1e-12


def to_unit(x: np.ndarray, lo: Optional[float] = None, hi: Optional[float] = None) -> np.ndarray:
    """Per-volume min-max to [0,1] -- the normalisation all three papers use.

    Acs & Zhuang: "each MRI volume was normalized to a range of [0,1]".
    LiteMamba:    "min-max normalization ... to the interval [0, 1]".
    They then use PSNR data_range = 1.0.

    lo/hi let the CALLER pin the scaling to the ground truth, so prediction and target share one
    scale. Rescaling each independently would silently correct a global intensity error.
    """
    lo = float(np.min(x)) if lo is None else lo
    hi = float(np.max(x)) if hi is None else hi
    if hi - lo < _EPS:
        return np.zeros_like(x, dtype=np.float64)
    return ((x.astype(np.float64) - lo) / (hi - lo)).clip(0.0, 1.0)


def psnr_2d(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """PSNR of one 2D slice. Returns inf when MSE == 0 -- the caller MUST handle it."""
    mse = float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def ssim_2d(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """2D SSIM, Wang et al. settings (11x11 Gaussian, sigma=1.5) -- what tf.image.ssim does.

    Acs & Zhuang used "a custom TensorFlow implementation"; FS-RWKV/LiteMamba do not state the
    window at all. Gaussian-weighted is the standard and the only defensible default.
    """
    from skimage.metrics import structural_similarity as _ssim
    return float(_ssim(gt.astype(np.float64), pred.astype(np.float64),
                       data_range=data_range, gaussian_weights=True, sigma=1.5,
                       use_sample_covariance=False))


def _slice_stats(pred: np.ndarray, gt: np.ndarray, axis: int,
                 brain: Optional[np.ndarray] = None,
                 brain_slices_only: bool = False) -> Dict[str, float]:
    """Mean per-slice PSNR/SSIM along `axis`.

    THE DEGENERACY THAT NO PAPER ADDRESSES, AND THAT I FIRST GOT WRONG.
    An earlier version of this function guarded only against MSE == 0 exactly (PSNR = inf) and
    claimed that "counted and excluded" the empty slices. IT DID NOT. A real network's background
    is not bit-identical to the target's, so MSE is tiny-but-nonzero and the slice scores 60-70 dB
    instead of inf -- it sails past an is-finite check and SATURATES the mean. Measured on a
    synthetic skull-stripped volume with a 1e-3 background noise floor:

        brain-only 3D                 22.65 dB
        per-slice mean, all slices    56.42 dB   <- 40 empty slices average 69.0 dB
        per-slice mean, brain slices  31.2  dB

    A 1e-3 perturbation of the background noise floor moved the all-slices mean by 25 dB. The
    statistic is discontinuous exactly where a headline number would sit.

    So we now report BOTH, and state the rule:
      * `psnr` / `ssim`            -- every slice, i.e. LITERALLY what the papers describe;
      * `psnr_brain` / `ssim_brain` -- slices containing >=1 brain voxel. A STATED exclusion rule.
    No paper on this dataset states any exclusion rule at all.
    """
    pred = np.moveaxis(pred, axis, 0)
    gt = np.moveaxis(gt, axis, 0)
    br = None if brain is None else np.moveaxis(brain, axis, 0)

    psnrs, ssims, bp, bs, n_degen, n_brain = [], [], [], [], 0, 0
    for i in range(gt.shape[0]):
        p, g = pred[i], gt[i]
        has_brain = bool(br[i].any()) if br is not None else True
        n_brain += int(has_brain)
        if brain_slices_only and not has_brain:
            continue
        v = psnr_2d(p, g)
        if not np.isfinite(v):
            n_degen += 1        # pred == gt BITWISE: PSNR undefined. Rare -- see the note above.
            continue
        s = ssim_2d(p, g)
        psnrs.append(v)
        ssims.append(s)
        if has_brain:
            bp.append(v)
            bs.append(s)

    mean = lambda a: float(np.mean(a)) if a else float("nan")
    return {
        "psnr": mean(psnrs),
        "ssim": mean(ssims),
        "psnr_brain": mean(bp),          # the same statistic, restricted to brain-bearing slices
        "ssim_brain": mean(bs),
        "n_slices": int(gt.shape[0]),
        "n_scored": len(psnrs),
        "n_degenerate": n_degen,
        "n_brain_slices": n_brain,
        "n_empty_slices": int(gt.shape[0]) - n_brain,
    }


def score_all_protocols(pred: np.ndarray, gt: np.ndarray,
                        brain: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Score one prediction under every convention. All inputs are raw 3D volumes.

    Args:
        pred, gt: [D,H,W] float. Any scale -- both are min-maxed onto the GT's scale.
        brain:    [D,H,W] bool brain mask (from the ground-truth segmentation).
    """
    assert pred.shape == gt.shape == brain.shape, "pred/gt/brain must have the same shape"
    lo, hi = float(np.min(gt)), float(np.max(gt))       # ONE scale, pinned to the GT
    p = to_unit(pred, lo, hi)
    g = to_unit(gt, lo, hi)
    out: Dict[str, Dict[str, float]] = {}

    # ---- A: brain-masked 3D. The honest one. ----------------------------------------------
    mse_b = float(np.mean((p[brain] - g[brain]) ** 2))
    out["A_brain_3d"] = {
        "psnr": float(10 * np.log10(1.0 / max(mse_b, _EPS))),
        "n_voxels": int(brain.sum()),
        "frac_of_volume": float(brain.mean()),
    }

    # ---- B: whole volume, unmasked, 3D ----------------------------------------------------
    mse_v = float(np.mean((p - g) ** 2))
    out["B_volume_3d"] = {
        "psnr": float(10 * np.log10(1.0 / max(mse_v, _EPS))),
        "n_voxels": int(g.size),
        "frac_of_volume": 1.0,
    }

    # ---- C: 2D per-slice, ALL slices, each orientation (Acs & Zhuang) ----------------------
    # axis 0/1/2 of a [D,H,W] volume. We report all three because THEY do -- and because their
    # headline 23.25 is the BEST of the three planes (coronal 22.48, sagittal 22.05).
    for name, ax in (("C_slice2d_axis0", 0), ("C_slice2d_axis1", 1), ("C_slice2d_axis2", 2)):
        out[name] = _slice_stats(p, g, ax, brain)

    # ---- D: 101 CENTRAL AXIAL slices (FS-RWKV) ---------------------------------------------
    # THE AXIS MATTERS AND I FIRST GOT IT WRONG. Volumes are reoriented to RAS
    # (src/config.py, src/preprocessing.py), so the array axes are (R, A, S):
    #     axis 0 = R = SAGITTAL   (256)
    #     axis 1 = A = CORONAL    (304)
    #     axis 2 = S = AXIAL / TRANSVERSE (308)
    # Acs & Zhuang state "256 sagittal, 304 coronal, and 308 transverse slices", which confirms
    # it. An earlier version of this function sliced axis 0 and CALLED IT AXIAL, so protocols D
    # and E were measuring a plane no competitor ever used.
    AXIAL = 2
    n = g.shape[AXIAL]
    k = min(101, n)
    s0 = n // 2 - k // 2
    s1 = s0 + k
    sl = [slice(None)] * 3
    sl[AXIAL] = slice(s0, s1)
    sl = tuple(sl)
    out["D_central_101"] = _slice_stats(p[sl], g[sl], AXIAL, brain[sl])

    # ---- E: central axial slices, central 128x128 crop (LiteMamba Table 4) -----------------
    # The in-plane axes of an AXIAL slice are 0 (R) and 1 (A) -- not 1 and 2.
    h, w = g.shape[0], g.shape[1]
    ch, cw = h // 2, w // 2
    hh, ww = min(64, h // 2), min(64, w // 2)
    crop = (slice(ch - hh, ch + hh), slice(cw - ww, cw + ww), slice(s0, s1))
    out["E_central_crop"] = _slice_stats(p[crop], g[crop], AXIAL, brain[crop])

    # ---- F: the ORIGINAL BROKEN HARNESS -- GT pasted outside the brain. INVALID. -----------
    # Reproduced only to quantify what the published paper's number actually measured.
    pasted = g.copy()
    pasted[brain] = p[brain]
    mse_f = float(np.mean((pasted - g) ** 2))
    out["F_gt_pasted"] = {
        "psnr": float(10 * np.log10(1.0 / max(mse_f, _EPS))),
        "n_voxels": int(g.size),
        "frac_actually_predicted": float(brain.mean()),
    }
    return out
