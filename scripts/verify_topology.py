"""
Verify topology preservation: compare model's predicted segmentation
against ground truth tissue segmentation.

Evidence for topology preservation:
  1. Per-class Dice (GM, WM, CSF) between predicted seg and ground truth seg
  2. Per-class volume agreement
  3. Connected component count per tissue class
  4. Brain mask overlap (Dice, HD95)
  5. (Optional) Ablation comparison

Inputs:
  --predicted-seg  : Model's segmentation output (predicted_seg.nii.gz)
  --ground-truth-seg: Ground truth tissue labels (e.g. sub-06_ses-2_desc-preproc_T1w_seg.nii)
  --predicted-vol  : (optional) Predicted 7T volume for intensity-based checks
  --target-vol     : (optional) Real 7T volume for intensity-based checks

Label convention (must match your data):
  0 = Background, 1 = CSF, 2 = GM, 3 = WM

Usage:
    python scripts/verify_topology.py \
        --predicted-seg results/sub-06/predicted_seg.nii.gz \
        --ground-truth-seg /path/to/sub-06_ses-2_desc-preproc_T1w_seg.nii \
        --output results/sub-06/topology_report.json

    # With ablation:
    python scripts/verify_topology.py \
        --predicted-seg results/sub-06/predicted_seg.nii.gz \
        --ablation-seg results/sub-06-notopo/predicted_seg.nii.gz \
        --ground-truth-seg /path/to/sub-06_ses-2_desc-preproc_T1w_seg.nii \
        --output results/sub-06/topology_report.json
"""
import argparse
import json
import numpy as np
from pathlib import Path


LABEL_NAMES = {0: 'Background', 1: 'CSF', 2: 'GM', 3: 'WM'}


def load_nifti(path):
    import nibabel as nib
    img = nib.load(str(path))
    return img.get_fdata(), img.affine


def dice_per_class(pred_seg, gt_seg, label):
    """Dice coefficient for a single tissue class."""
    p = (pred_seg == label)
    g = (gt_seg == label)
    intersection = np.sum(p & g)
    denom = np.sum(p) + np.sum(g)
    if denom == 0:
        return float('nan')
    return float(2.0 * intersection / denom)


def volume_per_class(seg, label, voxel_vol_mm3=1.0):
    """Volume in mm^3 for a single tissue class."""
    return float(np.sum(seg == label) * voxel_vol_mm3)


def connected_components(seg, label):
    """Count connected components for a tissue class."""
    from scipy import ndimage
    binary = (seg == label).astype(np.uint8)
    if binary.sum() == 0:
        return 0, 0.0
    labeled, n = ndimage.label(binary)
    sizes = ndimage.sum(binary, labeled, range(1, n + 1))
    largest_frac = float(max(sizes) / binary.sum())
    return int(n), largest_frac


def hd95_masks(mask_a, mask_b, voxel_spacing=(1, 1, 1)):
    """HD95 between two binary masks."""
    from scipy.ndimage import distance_transform_edt, binary_erosion

    surface_a = mask_a & ~binary_erosion(mask_a)
    surface_b = mask_b & ~binary_erosion(mask_b)

    if surface_a.sum() == 0 or surface_b.sum() == 0:
        return float('nan')

    dist_b = distance_transform_edt(~mask_b, sampling=voxel_spacing)
    dist_a = distance_transform_edt(~mask_a, sampling=voxel_spacing)

    d_a2b = dist_b[surface_a]
    d_b2a = dist_a[surface_b]

    return float(np.percentile(np.concatenate([d_a2b, d_b2a]), 95))


def analyze_seg_pair(pred_seg, gt_seg, voxel_spacing, label_prefix=""):
    """Full topology analysis between predicted and ground truth segmentation."""
    voxel_vol = float(np.prod(voxel_spacing))
    results = {}

    # Per-class metrics
    for label_id, label_name in LABEL_NAMES.items():
        if label_id == 0:
            continue  # skip background

        d = dice_per_class(pred_seg, gt_seg, label_id)
        vol_pred = volume_per_class(pred_seg, label_id, voxel_vol)
        vol_gt = volume_per_class(gt_seg, label_id, voxel_vol)
        vol_diff_pct = ((vol_pred - vol_gt) / vol_gt * 100) if vol_gt > 0 else float('nan')
        cc_pred, frac_pred = connected_components(pred_seg, label_id)
        cc_gt, frac_gt = connected_components(gt_seg, label_id)

        results[label_name] = {
            'dice': d,
            'volume_pred_mm3': vol_pred,
            'volume_gt_mm3': vol_gt,
            'volume_diff_pct': vol_diff_pct,
            'components_pred': cc_pred,
            'components_gt': cc_gt,
            'largest_frac_pred': frac_pred,
            'largest_frac_gt': frac_gt,
        }

    # Brain mask (all non-background)
    brain_pred = pred_seg > 0
    brain_gt = gt_seg > 0
    brain_dice = dice_per_class(pred_seg > 0, gt_seg > 0, True)
    # Recompute properly: both are boolean, Dice = 2*intersection / (sum_a + sum_b)
    inter = np.sum(brain_pred & brain_gt)
    denom = np.sum(brain_pred) + np.sum(brain_gt)
    brain_dice = float(2 * inter / denom) if denom > 0 else 0.0
    brain_hd95 = hd95_masks(brain_pred, brain_gt, voxel_spacing)

    results['brain_mask'] = {
        'dice': brain_dice,
        'hd95_mm': brain_hd95,
        'volume_pred_mm3': float(np.sum(brain_pred) * voxel_vol),
        'volume_gt_mm3': float(np.sum(brain_gt) * voxel_vol),
    }

    return results


def print_results(results, title):
    """Print formatted topology results."""
    print(f'\n{"=" * 80}')
    print(f'  {title}')
    print(f'{"=" * 80}')

    # Per-class table
    print(f'\n  {"Tissue":<8} {"Dice":>8} {"Vol Pred":>12} {"Vol GT":>12} '
          f'{"Vol Diff":>10} {"CC Pred":>8} {"CC GT":>8} {"LargestF":>9}')
    print(f'  {"-" * 77}')

    for label_name in ['CSF', 'GM', 'WM']:
        r = results.get(label_name)
        if r is None:
            continue
        dice_str = f'{r["dice"]:.4f}' if not np.isnan(r['dice']) else 'N/A'
        diff_str = f'{r["volume_diff_pct"]:+.1f}%' if not np.isnan(r['volume_diff_pct']) else 'N/A'
        print(f'  {label_name:<8} {dice_str:>8} {r["volume_pred_mm3"]:>10.0f}  '
              f'{r["volume_gt_mm3"]:>10.0f}  {diff_str:>10} '
              f'{r["components_pred"]:>8} {r["components_gt"]:>8} '
              f'{r["largest_frac_pred"]:>9.3f}')

    # Brain mask
    bm = results['brain_mask']
    print(f'\n  Brain mask Dice:  {bm["dice"]:.4f}')
    print(f'  Brain mask HD95:  {bm["hd95_mm"]:.2f} mm')

    # Quality assessment
    gm_dice = results.get('GM', {}).get('dice', 0)
    wm_dice = results.get('WM', {}).get('dice', 0)
    brain_dice = bm['dice']

    print(f'\n  Assessment:')
    if gm_dice > 0.8 and wm_dice > 0.85 and brain_dice > 0.9:
        print(f'    Tissue segmentation: EXCELLENT')
    elif gm_dice > 0.7 and wm_dice > 0.75 and brain_dice > 0.85:
        print(f'    Tissue segmentation: GOOD')
    elif gm_dice > 0.5 and wm_dice > 0.6:
        print(f'    Tissue segmentation: MODERATE')
    else:
        print(f'    Tissue segmentation: POOR')


def main():
    parser = argparse.ArgumentParser(
        description='Verify topology preservation via segmentation comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--predicted-seg', required=True,
                        help='Model segmentation output (predicted_seg.nii.gz)')
    parser.add_argument('--ground-truth-seg', required=True,
                        help='Ground truth tissue segmentation (e.g. *_seg.nii)')
    parser.add_argument('--ablation-seg', default=None,
                        help='Ablation model segmentation (no topo loss)')
    parser.add_argument('--output', default='results/topology_report.json')
    args = parser.parse_args()

    # Validate files exist
    for name, path in [('predicted-seg', args.predicted_seg),
                       ('ground-truth-seg', args.ground_truth_seg)]:
        if not Path(path).exists():
            print(f'ERROR: {name} not found: {path}')
            return

    # Load
    print('Loading segmentations...')
    pred_seg, pred_affine = load_nifti(args.predicted_seg)
    gt_seg, gt_affine = load_nifti(args.ground_truth_seg)
    pred_seg = np.round(pred_seg).astype(int)
    gt_seg = np.round(gt_seg).astype(int)

    voxel_spacing = tuple(np.abs(np.diag(gt_affine)[:3]))
    print(f'  Predicted seg shape: {pred_seg.shape}')
    print(f'  Ground truth shape:  {gt_seg.shape}')
    print(f'  Voxel spacing:       {voxel_spacing} mm')

    # Check label distributions
    for name, seg in [('Predicted', pred_seg), ('Ground truth', gt_seg)]:
        labels, counts = np.unique(seg, return_counts=True)
        print(f'  {name} labels: ' + ', '.join(
            f'{LABEL_NAMES.get(l, f"label{l}")}={c}' for l, c in zip(labels, counts)
        ))

    # Shape check
    if pred_seg.shape != gt_seg.shape:
        print(f'ERROR: Shape mismatch: predicted {pred_seg.shape} vs GT {gt_seg.shape}')
        return

    # Main comparison
    results = analyze_seg_pair(pred_seg, gt_seg, voxel_spacing)
    print_results(results, 'TOPOLOGY: Model Segmentation vs Ground Truth')

    report = {'model_vs_gt': results}

    # Ablation comparison
    if args.ablation_seg and Path(args.ablation_seg).exists():
        print('\nLoading ablation segmentation...')
        abl_seg, _ = load_nifti(args.ablation_seg)
        abl_seg = np.round(abl_seg).astype(int)

        abl_results = analyze_seg_pair(abl_seg, gt_seg, voxel_spacing)
        print_results(abl_results, 'TOPOLOGY: Ablation (no topo loss) vs Ground Truth')

        # Side-by-side comparison
        print(f'\n{"=" * 80}')
        print(f'  ABLATION COMPARISON: Does topology loss help?')
        print(f'{"=" * 80}')
        print(f'\n  {"Tissue":<8} {"With Topo":>12} {"Without Topo":>12} {"Winner":>12}')
        print(f'  {"-" * 48}')

        topo_wins = 0
        total = 0
        for label_name in ['CSF', 'GM', 'WM']:
            d_full = results.get(label_name, {}).get('dice', 0)
            d_abl = abl_results.get(label_name, {}).get('dice', 0)
            if np.isnan(d_full) or np.isnan(d_abl):
                continue
            total += 1
            winner = 'Topo' if d_full > d_abl else 'Ablation'
            if d_full > d_abl:
                topo_wins += 1
            print(f'  {label_name:<8} {d_full:>12.4f} {d_abl:>12.4f} {winner:>12}')

        bm_full = results['brain_mask']['dice']
        bm_abl = abl_results['brain_mask']['dice']
        print(f'  {"Brain":<8} {bm_full:>12.4f} {bm_abl:>12.4f} '
              f'{"Topo" if bm_full > bm_abl else "Ablation":>12}')

        hd_full = results['brain_mask']['hd95_mm']
        hd_abl = abl_results['brain_mask']['hd95_mm']
        print(f'  {"HD95":<8} {hd_full:>11.2f}m {hd_abl:>11.2f}m '
              f'{"Topo" if hd_full < hd_abl else "Ablation":>12}')

        if topo_wins > total / 2:
            print(f'\n  CONCLUSION: Topology loss improves tissue segmentation fidelity.')
        elif topo_wins == total / 2:
            print(f'\n  CONCLUSION: Mixed results — topology loss has marginal effect.')
        else:
            print(f'\n  CONCLUSION: Topology loss did NOT improve segmentation fidelity.')

        report['ablation_vs_gt'] = abl_results

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, default=lambda x: None if np.isnan(x) else x)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
