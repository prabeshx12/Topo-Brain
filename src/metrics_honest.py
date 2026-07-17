"""
Honest evaluation metrics for 3T->7T synthesis.

REPLACES src/metrics.py and the metric code in scripts/evaluate_full_volume.py, both of
which produce inflated or meaningless numbers (see revision/AUDIT.md):

  * SSIM/PSNR: the old code pasted the GROUND TRUTH into the background
    (`pred[~brain_mask] = target[~brain_mask]`) and then averaged over the WHOLE volume.
    80.5% of the volume is background, so:
        PSNR_reported = PSNR_brain + 10*log10(1/0.195) = PSNR_brain + 7.10 dB
        75.7% of SSIM windows lay entirely inside pasted GT and scored exactly 1.0
    Published "SSIM 0.8991 / PSNR 20.00 dB" -> honest ~0.585 / ~12.90 dB.

  * Dice/HD95: the old code binarised the INTENSITY image at > 0.0. That is a level-set
    overlap, not anatomy. Pure Gaussian noise scored HD95 = 4.36 mm vs the published
    3.31 mm -- i.e. the metric barely beat noise. Anatomy must be scored on SEGMENTATIONS.

  * HD95: the old code measured surface->OBJECT distance (`distance_transform_edt(~target)`),
    which is 0 for any predicted-surface voxel lying inside the target, and used the outer
    dilation shell so a PERFECT prediction scored 1.0 mm instead of 0.0.

Everything here is computed on the brain region only, on real segmentations, with
surface-to-surface distances, and with the voxel connectivity stated explicitly
(Berger et al., "Pitfalls of topology-aware image segmentation", arXiv:2412.14619).
"""
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

__all__ = [
    "masked_psnr",
    "masked_ssim",
    "tissue_dice",
    "hd95_surface",
    "assd_surface",
    "betti_numbers",
    "euler_characteristic",
    "evaluate_synthesis",
]


# --------------------------------------------------------------------------- #
#  Image quality — brain region ONLY, no ground-truth pasting
# --------------------------------------------------------------------------- #
def masked_psnr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                data_range: float = 2.0) -> float:
    """PSNR over the masked (brain) voxels only.

    data_range=2.0 because volumes live in [-1, 1].
    The old code divided the brain SSE by the TOTAL voxel count, which inflates PSNR by
    10*log10(N_total/N_brain) = +7.10 dB for this data. Here the denominator is N_brain.
    """
    m = mask.astype(bool)
    if not m.any():
        return float("nan")
    mse = float(np.mean((pred[m].astype(np.float64) - target[m].astype(np.float64)) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def masked_ssim(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                data_range: float = 2.0, win_size: int = 7) -> float:
    """Mean SSIM over brain voxels whose ENTIRE SSIM window lies inside the brain.

    Two things matter here, and the second is subtle:

    1. The prediction is NOT modified outside the mask. (The old code copied `target`
       into the background, so 75.7% of windows scored exactly 1.0 for free.)

    2. SSIM is a *windowed* statistic (win_size^3). A brain voxel near the mask boundary
       has a window that reaches into the background, so whatever sits in the background
       leaks into its score. Averaging over the raw mask is therefore STILL contaminated
       by background content -- verified empirically: pasting GT into the background moved
       a raw-mask-averaged SSIM from 0.454 to 0.893.

       So we erode the mask by the window radius and average only over voxels whose full
       window is interior. This makes the metric provably independent of the background.
    """
    from skimage.metrics import structural_similarity as ssim

    _, ssim_map = ssim(
        target.astype(np.float64), pred.astype(np.float64),
        data_range=data_range, full=True, win_size=win_size,
    )
    m = mask.astype(bool)
    r = win_size // 2
    if r > 0:
        # The SSIM support is a CUBE of side win_size, so the erosion must be in the
        # Chebyshev metric: use the full 3x3x3 structuring element (connectivity 3).
        # A 6-connected erosion is NOT enough -- a voxel can survive it and still have
        # its cube window poke into the background (measured: leaves a 0.047 SSIM leak).
        cube = ndimage.generate_binary_structure(3, 3)
        m = ndimage.binary_erosion(m, cube, iterations=r)
    if not m.any():  # brain too thin for this window -- fall back to the raw mask
        m = mask.astype(bool)
        if not m.any():
            return float("nan")
    return float(ssim_map[m].mean())


# --------------------------------------------------------------------------- #
#  Anatomy — on SEGMENTATIONS, not intensity thresholds
# --------------------------------------------------------------------------- #
def tissue_dice(pred_seg: np.ndarray, gt_seg: np.ndarray,
                labels: Tuple[int, ...] = (1, 2, 3)) -> Dict[int, float]:
    """Per-class Dice between two label maps. This is an anatomy metric; the old
    `compute_dice_coefficient` thresholded the INTENSITY image at >0 instead."""
    out = {}
    for lb in labels:
        a = pred_seg == lb
        b = gt_seg == lb
        denom = a.sum() + b.sum()
        out[lb] = float(2.0 * np.logical_and(a, b).sum() / denom) if denom else float("nan")
    return out


def _surface(binary: np.ndarray) -> np.ndarray:
    """Inner surface: foreground voxels with at least one background 6-neighbour.
    (The old code used `binary_dilation(x) ^ x`, the OUTER shell, which makes a perfect
    prediction score 1.0 mm instead of 0.0.)"""
    er = ndimage.binary_erosion(binary, ndimage.generate_binary_structure(3, 1))
    return binary & ~er


def _surface_distances(a: np.ndarray, b: np.ndarray,
                       spacing: Tuple[float, float, float]) -> np.ndarray:
    """Symmetric surface-to-SURFACE distances (mm).

    The old code computed the distance transform of ~b (the OBJECT), which is 0 everywhere
    INSIDE b -- so a predicted surface voxel deep inside the target scored 0 no matter how
    far it was from the true surface. Here we transform the distance to b's SURFACE.
    """
    sa, sb = _surface(a), _surface(b)
    if not sa.any() or not sb.any():
        return np.array([np.nan])
    dt_to_b = ndimage.distance_transform_edt(~sb, sampling=spacing)
    dt_to_a = ndimage.distance_transform_edt(~sa, sampling=spacing)
    return np.concatenate([dt_to_b[sa], dt_to_a[sb]])


def hd95_surface(pred_bin: np.ndarray, gt_bin: np.ndarray,
                 spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """95th-percentile symmetric surface-to-surface Hausdorff distance (mm)."""
    d = _surface_distances(pred_bin.astype(bool), gt_bin.astype(bool), spacing)
    return float(np.percentile(d, 95)) if np.isfinite(d).all() and d.size else float("nan")


def assd_surface(pred_bin: np.ndarray, gt_bin: np.ndarray,
                 spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """Average symmetric surface distance (mm)."""
    d = _surface_distances(pred_bin.astype(bool), gt_bin.astype(bool), spacing)
    return float(np.mean(d)) if np.isfinite(d).all() and d.size else float("nan")


def _hd95_assd_from(d: np.ndarray) -> Tuple[float, float]:
    """(HD95, ASSD) from ONE precomputed surface-distance array.

    WHY THIS EXISTS. hd95_surface() and assd_surface() each call _surface_distances(),
    which runs TWO distance_transform_edt passes over the full volume. Calling both (as
    evaluate_synthesis did) therefore paid for FOUR EDTs per structure where two suffice --
    on a 256x304x308 volume across 3 tissues + whole brain that is 16 EDTs instead of 8.
    The distances are identical either way; only the reduction differs (percentile vs mean),
    so computing d once and deriving both is exactly equivalent and ~2x faster.
    The public hd95_surface/assd_surface are left untouched for callers/tests.
    """
    ok = bool(d.size) and bool(np.isfinite(d).all())
    if not ok:
        return float("nan"), float("nan")
    return float(np.percentile(d, 95)), float(np.mean(d))


# --------------------------------------------------------------------------- #
#  Topology — connectivity is REPORTED, not implicit
# --------------------------------------------------------------------------- #
def betti_numbers(binary: np.ndarray, connectivity: int = 26) -> Tuple[int, int, int]:
    """(b0, b1, b2) of the foreground via a cubical complex (gudhi).

    `connectivity` is recorded by the caller and MUST be reported alongside the numbers --
    Berger et al. (arXiv:2412.14619) show the 6/18/26 choice dominates inter-method
    variation in published topology comparisons.
    """
    import gudhi as gd

    b = binary.astype(bool)
    if not b.any():
        return 0, 0, 0
    # crop to the bounding box + a 2-voxel background border: topology is unchanged and
    # memory stays bounded (a full 24M-voxel complex OOMs on modest machines).
    c = np.argwhere(b)
    lo, hi = c.min(0), c.max(0) + 1
    sub = b[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sub = np.pad(sub, 2, mode="constant", constant_values=False)

    filt = np.where(sub, 0.0, 1.0).astype(np.float32)
    cc = gd.CubicalComplex(top_dimensional_cells=filt)
    cc.compute_persistence(homology_coeff_field=2, min_persistence=0.0)
    raw = cc.persistent_betti_numbers(from_value=0.0, to_value=0.5)
    b0, b1, b2 = (list(raw) + [0, 0, 0])[:3]
    return int(b0), int(b1), int(b2)


def euler_characteristic(binary: np.ndarray, connectivity: int = 26) -> int:
    """chi = b0 - b1 + b2."""
    b0, b1, b2 = betti_numbers(binary, connectivity)
    return int(b0 - b1 + b2)


def connected_components(binary: np.ndarray, connectivity: int = 26) -> Tuple[int, float]:
    """(number of components, largest-component fraction)."""
    st = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    lab, _ = ndimage.label(binary, structure=st)
    if not binary.any():
        return 0, 0.0
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return int((counts > 0).sum()), float(counts.max() / binary.sum())


# --------------------------------------------------------------------------- #
#  One call that produces everything, honestly
# --------------------------------------------------------------------------- #
TISSUE_NAMES = {1: "CSF", 2: "GM", 3: "WM"}


def evaluate_synthesis(
    pred_img: np.ndarray,
    target_img: np.ndarray,
    brain_mask: np.ndarray,
    pred_seg: Optional[np.ndarray] = None,
    gt_seg: Optional[np.ndarray] = None,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    connectivity: int = 26,
    with_topology: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict:
    """Full honest evaluation of one subject.

    Image quality is scored on the brain region only (no GT pasting). Anatomy is scored on
    the segmentations (never on intensity thresholds). HD95/ASSD are surface-to-surface.
    Topology reports Betti numbers, Euler characteristic and the connectivity used.

    progress: optional callable receiving a status line per stage. Default None = silent,
        so existing callers/tests are unchanged. Pass `progress=lambda m: print(m, flush=True)`
        for a live log. THIS MATTERS: on a 256x304x308 volume this call runs 8 full-volume
        distance transforms plus 6 gudhi persistence computations and can take tens of
        minutes, all CPU-bound. Without a log it looks indistinguishable from a hang.
    """
    t0 = time.perf_counter()

    def _p(msg: str) -> None:
        if progress is not None:
            progress(f"  [{time.perf_counter() - t0:7.1f}s] {msg}")

    n_vox = int(brain_mask.sum())
    _p(f"image metrics: masked SSIM/PSNR over {n_vox:,} brain voxels ...")
    res: Dict = {
        "connectivity": connectivity,
        "brain_voxels": n_vox,
        "brain_fraction": float(brain_mask.sum() / brain_mask.size),
        "image": {
            "ssim_brain": masked_ssim(pred_img, target_img, brain_mask),
            "psnr_brain": masked_psnr(pred_img, target_img, brain_mask),
        },
    }
    _p(f"image metrics DONE: SSIM {res['image']['ssim_brain']:.4f} "
       f"PSNR {res['image']['psnr_brain']:.2f} dB")

    if pred_seg is None or gt_seg is None:
        _p("no segmentations supplied -- skipping anatomy/topology")
        return res

    _p("tissue Dice ...")
    dice = tissue_dice(pred_seg, gt_seg)
    res["tissue"] = {}
    n_tis = len(TISSUE_NAMES)
    for i, (lb, name) in enumerate(TISSUE_NAMES.items(), start=1):
        tag = f"[{i}/{n_tis}] {name}"
        p, g = pred_seg == lb, gt_seg == lb

        _p(f"{tag}: surface distances (2x EDT over full volume -- SLOW) ...")
        d = _surface_distances(p, g, spacing)
        hd95, assd = _hd95_assd_from(d)
        entry = {"dice": dice[lb], "hd95_mm": hd95, "assd_mm": assd}
        _p(f"{tag}: Dice {dice[lb]:.4f}  HD95 {hd95:.2f} mm  ASSD {assd:.2f} mm")

        _p(f"{tag}: connected components ...")
        ncc, lcc = connected_components(p, connectivity)
        entry["n_components"] = ncc
        entry["largest_cc_frac"] = lcc
        _p(f"{tag}: {ncc} components, largest holds {100 * lcc:.2f}%")

        if with_topology:
            _p(f"{tag}: Betti of PREDICTION (gudhi cubical complex -- SLOWEST step) ...")
            b0, b1, b2 = betti_numbers(p, connectivity)
            _p(f"{tag}: Betti pred = ({b0}, {b1}, {b2})")
            _p(f"{tag}: Betti of GROUND TRUTH (gudhi) ...")
            g0, g1, g2 = betti_numbers(g, connectivity)
            _p(f"{tag}: Betti gt   = ({g0}, {g1}, {g2})")
            entry["betti_pred"] = [b0, b1, b2]
            entry["betti_gt"] = [g0, g1, g2]
            entry["euler_pred"] = b0 - b1 + b2
            entry["euler_gt"] = g0 - g1 + g2
            # Betti-number ERROR (absolute). Note Berger et al. recommend Betti MATCHING
            # error for spatial correspondence; this is the cheaper, standard fallback and
            # must be reported as such.
            entry["betti_err"] = [abs(b0 - g0), abs(b1 - g1), abs(b2 - g2)]
            _p(f"{tag}: Betti ERR  = {entry['betti_err']}  (chi {entry['euler_pred']} "
               f"vs gt {entry['euler_gt']})")
        res["tissue"][name] = entry

    # whole-brain foreground (any tissue)
    _p("whole-brain foreground: Dice + surface distances ...")
    p_fg, g_fg = pred_seg > 0, gt_seg > 0
    d_fg = _surface_distances(p_fg, g_fg, spacing)
    hd95_fg, assd_fg = _hd95_assd_from(d_fg)
    res["brain"] = {
        "dice": float(2 * np.logical_and(p_fg, g_fg).sum() / (p_fg.sum() + g_fg.sum())),
        "hd95_mm": hd95_fg,
        "assd_mm": assd_fg,
    }
    _p(f"whole-brain: Dice {res['brain']['dice']:.4f}  HD95 {hd95_fg:.2f} mm  "
       f"ASSD {assd_fg:.2f} mm")
    _p("ALL METRICS DONE")
    return res
