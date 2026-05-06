"""
Medical Image Evaluation Metrics for Topo-Brain.

Implements standard metrics for validating MRI synthesis quality:
- SSIM, PSNR: Image similarity
- Dice: Volumetric overlap
- HD95: Surface/boundary accuracy
"""
import numpy as np
from typing import Optional, Tuple


def compute_ssim_psnr(
    pred_np: np.ndarray,
    target_np: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Compute SSIM and PSNR. Both arrays expected in [-1, 1].

    When `mask` is provided (HD-BET brain mask, bool or 0/1 array of the
    same shape as pred/target), reported metrics are restricted to the brain:

      - PSNR: prediction outside the mask is replaced with the target's
        background. This zeros the error contribution from background voxels
        so PSNR reflects only the brain region. (Same convention used in
        scripts/evaluate_full_volume.py for the headline "masked" metric.)
      - SSIM: computed on the brain bounding box of both volumes — the
        global SSIM is dominated by huge identical-background regions and
        masks out genuine brain detail otherwise.

    Without a mask, both metrics are whole-volume.

    Args:
        pred_np:   Predicted volume in [-1, 1]
        target_np: Ground truth volume in [-1, 1]
        mask:      Optional bool/0-1 brain mask of the same shape

    Returns:
        (ssim, psnr) tuple
    """
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr

    pred_np = np.clip(pred_np, -1.0, 1.0)
    target_np = np.clip(target_np, -1.0, 1.0)
    pred_01 = (pred_np + 1.0) / 2.0
    tgt_01 = (target_np + 1.0) / 2.0

    if mask is None:
        ssim_val = ssim(tgt_01, pred_01, data_range=1.0)
        psnr_val = psnr(tgt_01, pred_01, data_range=1.0)
        return float(ssim_val), float(psnr_val)

    m = np.asarray(mask).astype(bool)
    if m.shape != pred_np.shape:
        raise ValueError(f"mask shape {m.shape} != pred/target shape {pred_np.shape}")
    if not m.any():
        return float("nan"), float("nan")

    # PSNR: replace background with target so background error is 0.
    pred_for_psnr = pred_01.copy()
    pred_for_psnr[~m] = tgt_01[~m]
    psnr_val = psnr(tgt_01, pred_for_psnr, data_range=1.0)

    # SSIM: crop to brain bounding box so window stats reflect tissue.
    coords = np.argwhere(m)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    crop = tuple(slice(int(lo[i]), int(hi[i])) for i in range(m.ndim))
    ssim_val = ssim(tgt_01[crop], pred_01[crop], data_range=1.0)

    return float(ssim_val), float(psnr_val)


def compute_dice(pred_np: np.ndarray, target_np: np.ndarray,
                 mask: Optional[np.ndarray] = None,
                 threshold: float = 0.0) -> float:
    """
    Compute Dice coefficient for volumetric overlap.

    The Dice coefficient measures the similarity between two volumes:
    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Range: [0, 1] where:
    - 1.0 = perfect overlap
    - 0.0 = no overlap

    Clinical interpretation:
    - > 0.9: Excellent agreement
    - > 0.8: Good agreement
    - > 0.7: Acceptable
    - < 0.7: Poor agreement

    Args:
        pred_np: Predicted volume in [-1, 1]
        target_np: Target volume in [-1, 1]
        mask: Optional binary mask to restrict computation to ROI
        threshold: Binarization threshold (0.0 maps to 0.5 in [0,1] space)

    Returns:
        Dice coefficient (0 to 1, higher is better)
    """
    # Binarize volumes at threshold
    pred_binary = (pred_np > threshold).astype(np.float32)
    target_binary = (target_np > threshold).astype(np.float32)

    # Apply mask if provided
    if mask is not None:
        pred_binary = pred_binary[mask]
        target_binary = target_binary[mask]

    # Compute Dice: 2 * |A ∩ B| / (|A| + |B|)
    intersection = np.sum(pred_binary * target_binary)
    denominator = np.sum(pred_binary) + np.sum(target_binary)

    if denominator == 0:
        # Both empty: perfect match
        return 1.0 if intersection == 0 else 0.0

    dice = (2.0 * intersection) / denominator
    return float(dice)


def compute_hd95(pred_np: np.ndarray, target_np: np.ndarray,
                 mask: Optional[np.ndarray] = None,
                 voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 threshold: float = 0.0) -> float:
    """
    Compute 95th percentile Hausdorff Distance (HD95).

    The Hausdorff distance measures the maximum surface distance between two volumes.
    HD95 uses the 95th percentile instead of maximum, making it robust to outliers.

    Lower is better. Measures boundary/surface accuracy in mm.

    Clinical interpretation (for brain MRI):
    - < 2mm: Excellent boundary accuracy
    - < 5mm: Good boundary accuracy
    - < 10mm: Acceptable
    - > 10mm: Poor boundary delineation

    Args:
        pred_np: Predicted volume in [-1, 1]
        target_np: Target volume in [-1, 1]
        mask: Optional binary mask to restrict computation to ROI
        voxel_spacing: Physical voxel spacing (mm) in (x, y, z)
        threshold: Binarization threshold

    Returns:
        HD95 in mm (lower is better). Returns NaN if scipy unavailable or volumes empty.
    """
    try:
        from scipy.ndimage import distance_transform_edt, binary_dilation
    except ImportError:
        return float('nan')

    # Binarize volumes
    pred_binary = (pred_np > threshold).astype(bool)
    target_binary = (target_np > threshold).astype(bool)

    # Apply mask if provided
    if mask is not None:
        pred_binary = pred_binary & mask
        target_binary = target_binary & mask

    # Check if volumes are non-empty
    if not pred_binary.any() or not target_binary.any():
        return float('nan')

    # Extract surface voxels (boundary detection via morphological gradient)
    pred_surface = binary_dilation(pred_binary) ^ pred_binary
    target_surface = binary_dilation(target_binary) ^ target_binary

    if not pred_surface.any() or not target_surface.any():
        return float('nan')

    # Euclidean distance transform with physical spacing
    # Distance from pred surface to nearest target voxel
    dist_pred_to_target = distance_transform_edt(~target_binary, sampling=voxel_spacing)
    distances_pred = dist_pred_to_target[pred_surface]

    # Distance from target surface to nearest pred voxel
    dist_target_to_pred = distance_transform_edt(~pred_binary, sampling=voxel_spacing)
    distances_target = dist_target_to_pred[target_surface]

    # Combine both directions (symmetric distance)
    all_distances = np.concatenate([distances_pred, distances_target])

    # 95th percentile (more robust than max HD100)
    hd95 = np.percentile(all_distances, 95)
    return float(hd95)


def compute_all_metrics(pred_np: np.ndarray, target_np: np.ndarray,
                       mask: Optional[np.ndarray] = None,
                       voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> dict:
    """
    Compute all standard metrics at once.

    Args:
        pred_np: Predicted volume in [-1, 1]
        target_np: Target volume in [-1, 1]
        mask: Optional binary mask
        voxel_spacing: Physical voxel spacing in mm

    Returns:
        Dictionary with all metrics
    """
    ssim_val, psnr_val = compute_ssim_psnr(pred_np, target_np, mask=mask)
    dice_val = compute_dice(pred_np, target_np, mask=mask)
    hd95_val = compute_hd95(pred_np, target_np, mask=mask, voxel_spacing=voxel_spacing)

    return {
        'ssim': ssim_val,
        'psnr': psnr_val,
        'dice': dice_val,
        'hd95_mm': hd95_val,
    }


def print_metrics(metrics: dict, label: str = "Metrics"):
    """
    Pretty-print metrics dictionary.

    Args:
        metrics: Dictionary of metric values
        label: Label for this metric set
    """
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")

    if 'ssim' in metrics:
        print(f"  SSIM:      {metrics['ssim']:.4f}")
    if 'psnr' in metrics:
        print(f"  PSNR:      {metrics['psnr']:.2f} dB")
    if 'dice' in metrics:
        quality = "Excellent" if metrics['dice'] > 0.9 else "Good" if metrics['dice'] > 0.8 else "Acceptable" if metrics['dice'] > 0.7 else "Poor"
        print(f"  Dice:      {metrics['dice']:.4f} ({quality})")
    if 'hd95_mm' in metrics and not np.isnan(metrics['hd95_mm']):
        quality = "Excellent" if metrics['hd95_mm'] < 2 else "Good" if metrics['hd95_mm'] < 5 else "Acceptable" if metrics['hd95_mm'] < 10 else "Poor"
        print(f"  HD95:      {metrics['hd95_mm']:.2f} mm ({quality})")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("Testing Topo-Brain Metrics Module\n")
    rng = np.random.default_rng(0)

    # Volume in [-1, 1] with a brain region in the center and noisy background.
    target = np.full((64, 64, 64), -1.0, dtype=np.float32)
    target[16:48, 16:48, 16:48] = rng.normal(0.0, 0.3, (32, 32, 32)).clip(-1, 1)

    # Prediction = target with brain noise + heavy background noise (sim of model
    # mispredicting outside the brain). Brain-masking should boost PSNR.
    brain_noise = rng.normal(0.0, 0.05, target.shape)
    bg_noise = rng.normal(0.0, 0.4, target.shape)  # large bg mismatch
    mask = np.zeros_like(target, dtype=bool)
    mask[16:48, 16:48, 16:48] = True
    pred = target + np.where(mask, brain_noise, bg_noise)
    pred = np.clip(pred, -1, 1)

    whole = compute_all_metrics(pred, target, mask=None)
    masked = compute_all_metrics(pred, target, mask=mask)

    print_metrics(whole,  label="Whole-volume (no mask)")
    print_metrics(masked, label="Brain-masked (HD-BET style)")

    assert masked["psnr"] > whole["psnr"], (
        f"masked PSNR {masked['psnr']:.2f} should exceed whole {whole['psnr']:.2f} "
        "when prediction has heavy background mismatch"
    )
    print("✓ Brain-masked PSNR is higher than whole-volume — fix verified.")
