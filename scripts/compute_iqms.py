"""
Compute MRI Image Quality Metrics (IQMs) directly in Python.

Computes the same metrics as MRIQC without needing the container:
  - CJV  : Coefficient of Joint Variation (lower = better WM/GM separation)
  - CNR  : Contrast-to-Noise Ratio (higher = better GM/WM contrast)
  - SNR  : Signal-to-Noise Ratio in WM (higher = better)
  - FBER : Foreground-Background Energy Ratio (higher = better)
  - EFC  : Entropy Focus Criterion (lower = less ghosting/motion)

Uses tissue segmentation from the model output (predicted_seg.nii.gz)
or falls back to Otsu thresholding if not available.

Usage:
    # Compare 3T, synthetic 7T, and real 7T for sub-06
    python scripts/compute_iqms.py \
        --subject sub-06 \
        --data-root /eos/user/p/ppokhrel/Untitled\ Folder\ 1/preprocessed-mri-aligned-diffusion \
        --pairs-csv pairs_new.csv \
        --results-dir results/ \
        --output results/mriqc/iqm_comparison.csv
"""
import csv
import json
import argparse
import sys
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# IQM formulas (matching MRIQC definitions)
# ---------------------------------------------------------------------------

def get_tissue_masks(vol: np.ndarray, seg: np.ndarray = None):
    """
    Return (wm_mask, gm_mask, bg_mask) boolean arrays.
    If seg is provided (label map), uses it directly.
    Falls back to Otsu-based thresholding.
    """
    if seg is not None:
        # Model segmentation: 0=BG, 1=CSF, 2=GM, 3=WM
        bg_mask = seg == 0
        gm_mask = seg == 2
        wm_mask = seg == 3
        if wm_mask.sum() > 100 and gm_mask.sum() > 100:
            return wm_mask, gm_mask, bg_mask

    # Fallback: Otsu thresholding
    try:
        from skimage.filters import threshold_otsu, threshold_multiotsu
        thresholds = threshold_multiotsu(vol[vol > 0], classes=4)
        bg_mask = vol < thresholds[0]
        gm_mask = (vol >= thresholds[0]) & (vol < thresholds[1])
        wm_mask = vol >= thresholds[2]
    except Exception:
        # Simple percentile fallback
        p20 = np.percentile(vol[vol > 0], 20)
        p50 = np.percentile(vol[vol > 0], 50)
        p80 = np.percentile(vol[vol > 0], 80)
        bg_mask = vol < p20
        gm_mask = (vol >= p20) & (vol < p50)
        wm_mask = vol >= p80

    return wm_mask, gm_mask, bg_mask


def compute_cjv(vol, wm_mask, gm_mask):
    """Coefficient of Joint Variation. Lower = better WM/GM separation."""
    mu_wm = vol[wm_mask].mean()
    mu_gm = vol[gm_mask].mean()
    sig_wm = vol[wm_mask].std()
    sig_gm = vol[gm_mask].std()
    denom = abs(mu_wm - mu_gm)
    if denom < 1e-6:
        return float('nan')
    return float((sig_wm + sig_gm) / denom)


def compute_cnr(vol, wm_mask, gm_mask, bg_mask):
    """Contrast-to-Noise Ratio. Higher = better."""
    mu_wm = vol[wm_mask].mean()
    mu_gm = vol[gm_mask].mean()
    sig_bg = vol[bg_mask].std() if bg_mask.sum() > 10 else vol[wm_mask].std()
    denom = sig_bg
    if denom < 1e-6:
        return float('nan')
    return float(abs(mu_wm - mu_gm) / denom)


def compute_snr(vol, wm_mask):
    """Signal-to-Noise Ratio in WM. Higher = better."""
    mu_wm = vol[wm_mask].mean()
    sig_wm = vol[wm_mask].std()
    if sig_wm < 1e-6:
        return float('nan')
    return float(mu_wm / sig_wm)


def compute_fber(vol, bg_mask):
    """Foreground-Background Energy Ratio. Higher = better."""
    fg_mask = ~bg_mask
    fg_energy = float(np.sum(vol[fg_mask] ** 2))
    bg_energy = float(np.sum(vol[bg_mask] ** 2))
    if bg_energy < 1e-6:
        return float('nan')
    return float(fg_energy / bg_energy)


def compute_efc(vol):
    """
    Entropy Focus Criterion. Lower = less ghosting/motion.
    Based on Shannon entropy of voxel intensities.
    """
    vmax = np.sqrt(np.sum(vol ** 2))
    if vmax < 1e-6:
        return float('nan')
    b_max = vmax / np.sqrt(vol.size)
    efc = np.sum((vol / b_max) * np.log((vol / b_max) + 1e-12))
    return float(-efc / vol.size)


def compute_iqms(vol: np.ndarray, seg: np.ndarray = None, label: str = "") -> dict:
    """Compute all IQMs for a volume."""
    vol = vol.astype(np.float32)
    vol = np.clip(vol, 0, None)  # remove negatives

    wm_mask, gm_mask, bg_mask = get_tissue_masks(vol, seg)

    iqms = {
        'label': label,
        'cjv':  compute_cjv(vol, wm_mask, gm_mask),
        'cnr':  compute_cnr(vol, wm_mask, gm_mask, bg_mask),
        'snr':  compute_snr(vol, wm_mask),
        'fber': compute_fber(vol, bg_mask),
        'efc':  compute_efc(vol),
        'wm_voxels': int(wm_mask.sum()),
        'gm_voxels': int(gm_mask.sum()),
    }
    return iqms


def print_iqms(results: dict):
    """Print IQM comparison table."""
    modes = [m for m in ['3t', 'synthetic7t', 'real7t'] if m in results]
    metrics = [
        ('cjv',  'CJV  (lower=better)',  'lower'),
        ('cnr',  'CNR  (higher=better)', 'higher'),
        ('snr',  'SNR  (higher=better)', 'higher'),
        ('fber', 'FBER (higher=better)', 'higher'),
        ('efc',  'EFC  (lower=better)',  'lower'),
    ]

    print('\n' + '=' * 65)
    print('IMAGE QUALITY METRICS (IQMs)')
    print('=' * 65)
    header = f'{"Metric":<25}' + ''.join(f'{m:>13}' for m in modes)
    print(header)
    print('-' * 65)

    for key, label, direction in metrics:
        row = f'{label:<25}'
        vals = [results[m].get(key, float('nan')) for m in modes]
        for v in vals:
            row += f'{v:>13.4f}' if not np.isnan(v) else f'{"N/A":>13}'
        print(row)

    print('=' * 65)
    print('\nInterpretation:')
    print('  CJV  < 0.5  : Excellent tissue separation')
    print('  CNR  > 3.0  : Good GM/WM contrast')
    print('  SNR  > 10.0 : Good signal quality')
    print('  FBER > 100  : Good foreground vs background')
    print('  EFC  closer to 0 : Less motion/ghosting')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_nifti(path):
    import nibabel as nib
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32)


def resolve_pairs(pairs_csv, subject, data_root):
    with open(pairs_csv) as f:
        for row in csv.DictReader(f):
            if row['subject'] == subject:
                return {
                    'input_3t':  data_root / row['input_3t'],
                    'target_7t': data_root / row['target_7t'],
                }
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Compute IQMs without MRIQC container'
    )
    parser.add_argument('--subject', default='sub-06')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pairs-csv', default='pairs_new.csv')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--output', default='results/mriqc/iqm_comparison.csv')
    args = parser.parse_args()

    data_root   = Path(args.data_root)
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = resolve_pairs(Path(args.pairs_csv), args.subject, data_root)
    if not pairs:
        print(f'ERROR: {args.subject} not found in {args.pairs_csv}')
        sys.exit(1)

    # Load segmentation from model output if available
    seg_path = results_dir / args.subject / 'predicted_seg.nii.gz'
    seg = load_nifti(seg_path) if seg_path.exists() else None
    if seg is not None:
        print(f'Using model segmentation: {seg_path}')
    else:
        print('No segmentation found — using Otsu thresholding')

    results = {}

    # 3T input
    if pairs['input_3t'].exists():
        print(f'\nComputing IQMs for 3T: {pairs["input_3t"]}')
        vol = load_nifti(pairs['input_3t'])
        results['3t'] = compute_iqms(vol, seg, label='3T Input')
    else:
        print(f'[WARN] 3T not found: {pairs["input_3t"]}')

    # Synthetic 7T
    synth_path = results_dir / args.subject / 'predicted_7T.nii.gz'
    if synth_path.exists():
        print(f'Computing IQMs for synthetic 7T: {synth_path}')
        vol = load_nifti(synth_path)
        results['synthetic7t'] = compute_iqms(vol, seg, label='Synthetic 7T')
    else:
        print(f'[WARN] Synthetic 7T not found: {synth_path}')
        print(f'       Run evaluate_full_volume.py first.')

    # Real 7T
    if pairs['target_7t'].exists():
        print(f'Computing IQMs for real 7T: {pairs["target_7t"]}')
        vol = load_nifti(pairs['target_7t'])
        results['real7t'] = compute_iqms(vol, seg, label='Real 7T')
    else:
        print(f'[WARN] Real 7T not found: {pairs["target_7t"]}')

    if not results:
        print('ERROR: No volumes found.')
        sys.exit(1)

    # Print table
    print_iqms(results)

    # Save JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nSaved: {json_path}')

    # Save CSV
    modes = list(results.keys())
    metric_keys = ['cjv', 'cnr', 'snr', 'fber', 'efc']
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['metric'] + modes
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in metric_keys:
            row = {'metric': key}
            for mode in modes:
                row[mode] = results[mode].get(key, '')
            writer.writerow(row)
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
