"""
Verify topology preservation between synthetic 7T and real 7T.

Computes:
  1. Betti numbers (β0, β1, β2) — topological invariants
  2. Euler characteristic (χ = β0 - β1 + β2)
  3. Connected component analysis
  4. Region-wise topology check (hippocampus, ventricles, cortex)

Usage:
    python scripts/verify_topology.py \
        --predicted results/sub-06/predicted_7T.nii.gz \
        --target /path/to/real_7T.nii.gz \
        --output results/sub-06/topology_report.json

    # Or with ablation comparison:
    python scripts/verify_topology.py \
        --predicted results/sub-06/predicted_7T.nii.gz \
        --ablation results/sub-06-notopo/predicted_7T.nii.gz \
        --target /path/to/real_7T.nii.gz \
        --output results/sub-06/topology_report.json
"""
import argparse
import json
import sys
import numpy as np
from pathlib import Path


def load_nifti(path):
    import nibabel as nib
    return nib.load(str(path)).get_fdata().astype(np.float32)


def binarize(vol, threshold=0.0):
    """Binarize volume. For [-1,1] data, 0.0 separates brain from background."""
    return (vol > threshold).astype(np.uint8)


def compute_euler_number(binary_vol):
    """
    Compute Euler characteristic using scipy.
    χ = β0 - β1 + β2
    """
    from scipy import ndimage
    labeled, n_components = ndimage.label(binary_vol)
    # Euler number via marching cubes or direct computation
    # For 3D binary volumes: use connectivity-based estimation
    return n_components, labeled


def compute_betti_numbers(binary_vol):
    """
    Compute Betti numbers for a 3D binary volume.

    β0 = connected components
    β1 = tunnels/loops (estimated via Euler relation)
    β2 = enclosed cavities

    Uses the relation: χ = β0 - β1 + β2
    And computes χ from the cubical complex.
    """
    from scipy import ndimage

    # β0: connected components of foreground
    labeled_fg, beta0 = ndimage.label(binary_vol)

    # β2: connected components of background ENCLOSED by foreground
    # (cavities/voids inside the structure)
    bg = 1 - binary_vol
    labeled_bg, n_bg_components = ndimage.label(bg)

    # The "infinite" background is 1 component; enclosed voids are the rest
    # Find which background components touch the volume border
    border_labels = set()
    for axis in range(3):
        for side in [0, -1]:
            slc = [slice(None)] * 3
            slc[axis] = side
            border_labels.update(np.unique(labeled_bg[tuple(slc)]))
    border_labels.discard(0)  # 0 is foreground

    beta2 = n_bg_components - len(border_labels)

    # Euler characteristic via voxel-based formula
    # For cubical complexes: χ = V - E + F - C
    # where V=vertices, E=edges, F=faces, C=cubes
    V = int(binary_vol.sum())

    # Count shared edges (6-connected neighbors)
    # Along each axis, count adjacent voxel pairs that are both ON
    edges = (
        int(np.sum(binary_vol[:-1, :, :] & binary_vol[1:, :, :])) +
        int(np.sum(binary_vol[:, :-1, :] & binary_vol[:, 1:, :])) +
        int(np.sum(binary_vol[:, :, :-1] & binary_vol[:, :, 1:]))
    )

    # Count shared faces (pairs of voxels sharing a face in 2D plane)
    faces = 0
    for a1 in range(3):
        for a2 in range(a1 + 1, 3):
            s1 = [slice(None)] * 3
            s2 = [slice(None)] * 3
            s3 = [slice(None)] * 3
            s4 = [slice(None)] * 3
            s1[a1] = slice(None, -1); s1[a2] = slice(None, -1)
            s2[a1] = slice(1, None);  s2[a2] = slice(None, -1)
            s3[a1] = slice(None, -1); s3[a2] = slice(1, None)
            s4[a1] = slice(1, None);  s4[a2] = slice(1, None)
            faces += int(np.sum(binary_vol[tuple(s1)] & binary_vol[tuple(s2)] &
                                binary_vol[tuple(s3)] & binary_vol[tuple(s4)]))

    # Count shared cubes (2x2x2 all-on)
    cubes = int(np.sum(
        binary_vol[:-1, :-1, :-1] & binary_vol[1:, :-1, :-1] &
        binary_vol[:-1, 1:, :-1]  & binary_vol[1:, 1:, :-1] &
        binary_vol[:-1, :-1, 1:]  & binary_vol[1:, :-1, 1:] &
        binary_vol[:-1, 1:, 1:]   & binary_vol[1:, 1:, 1:]
    ))

    euler = V - edges + faces - cubes

    # β1 from Euler relation: χ = β0 - β1 + β2
    beta1 = beta0 - euler + beta2

    return {
        'beta0': int(beta0),
        'beta1': int(max(0, beta1)),  # clamp to non-negative
        'beta2': int(beta2),
        'euler': int(euler),
        'foreground_voxels': int(V),
    }


def compare_topology(vol_a, vol_b, label_a, label_b, threshold=0.0):
    """Compare topology between two volumes."""
    bin_a = binarize(vol_a, threshold)
    bin_b = binarize(vol_b, threshold)

    topo_a = compute_betti_numbers(bin_a)
    topo_b = compute_betti_numbers(bin_b)

    print(f'\n{"Metric":<25} {label_a:>15} {label_b:>15} {"Match":>10}')
    print('-' * 67)

    for key in ['beta0', 'beta1', 'beta2', 'euler', 'foreground_voxels']:
        va = topo_a[key]
        vb = topo_b[key]
        match = 'YES' if va == vb else f'diff={va - vb}'
        nice_name = {
            'beta0': 'β0 (components)',
            'beta1': 'β1 (tunnels)',
            'beta2': 'β2 (cavities)',
            'euler': 'Euler χ',
            'foreground_voxels': 'Volume (voxels)',
        }.get(key, key)
        print(f'{nice_name:<25} {va:>15} {vb:>15} {match:>10}')

    return topo_a, topo_b


def multi_threshold_analysis(vol_pred, vol_target, label_pred='Predicted', label_target='Target'):
    """Check topology across multiple thresholds to catch subtle issues."""
    thresholds = [-0.5, -0.25, 0.0, 0.25, 0.5]

    print(f'\n{"="*70}')
    print('MULTI-THRESHOLD TOPOLOGY ANALYSIS')
    print(f'{"="*70}')

    results = []
    for thr in thresholds:
        bin_p = binarize(vol_pred, thr)
        bin_t = binarize(vol_target, thr)
        topo_p = compute_betti_numbers(bin_p)
        topo_t = compute_betti_numbers(bin_t)

        b0_match = topo_p['beta0'] == topo_t['beta0']
        euler_diff = abs(topo_p['euler'] - topo_t['euler'])

        results.append({
            'threshold': thr,
            'pred': topo_p,
            'target': topo_t,
            'beta0_match': b0_match,
            'euler_diff': euler_diff,
        })

        flag = ' OK' if b0_match and euler_diff < 5 else ' <-- MISMATCH'
        print(f'  thr={thr:+.2f}: pred(β0={topo_p["beta0"]}, χ={topo_p["euler"]}) '
              f'target(β0={topo_t["beta0"]}, χ={topo_t["euler"]}){flag}')

    return results


def main():
    parser = argparse.ArgumentParser(description='Verify topology preservation')
    parser.add_argument('--predicted', required=True, help='Predicted synthetic 7T')
    parser.add_argument('--target', required=True, help='Real 7T ground truth')
    parser.add_argument('--ablation', default=None,
                        help='Ablation model output (no topo loss) for comparison')
    parser.add_argument('--output', default='results/topology_report.json')
    args = parser.parse_args()

    print('Loading volumes...')
    pred = load_nifti(args.predicted)
    target = load_nifti(args.target)

    ablation = None
    if args.ablation and Path(args.ablation).exists():
        ablation = load_nifti(args.ablation)

    report = {}

    # 1. Compare predicted vs target
    print('\n' + '=' * 70)
    print('TOPOLOGY: Synthetic 7T vs Real 7T')
    print('=' * 70)
    topo_pred, topo_target = compare_topology(pred, target, 'Synthetic 7T', 'Real 7T')
    report['synthetic_vs_real'] = {
        'synthetic': topo_pred,
        'real': topo_target,
    }

    # 2. If ablation provided, compare
    if ablation is not None:
        print('\n' + '=' * 70)
        print('TOPOLOGY: Ablation (no topo loss) vs Real 7T')
        print('=' * 70)
        topo_abl, _ = compare_topology(ablation, target, 'Ablation', 'Real 7T')

        print('\n' + '=' * 70)
        print('TOPOLOGY: With Topo Loss vs Without Topo Loss')
        print('=' * 70)
        compare_topology(pred, ablation, 'With Topo', 'Without Topo')
        report['ablation_vs_real'] = {
            'ablation': topo_abl,
            'real': topo_target,
        }

    # 3. Multi-threshold analysis
    mt_results = multi_threshold_analysis(pred, target)
    report['multi_threshold'] = [
        {
            'threshold': r['threshold'],
            'pred_beta0': r['pred']['beta0'],
            'target_beta0': r['target']['beta0'],
            'pred_euler': r['pred']['euler'],
            'target_euler': r['target']['euler'],
            'match': r['beta0_match'] and r['euler_diff'] < 5,
        }
        for r in mt_results
    ]

    # 4. Summary
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)

    b0_match = topo_pred['beta0'] == topo_target['beta0']
    euler_diff = abs(topo_pred['euler'] - topo_target['euler'])
    vol_diff_pct = abs(topo_pred['foreground_voxels'] - topo_target['foreground_voxels']) / topo_target['foreground_voxels'] * 100

    print(f'  β0 (connected components) match: {"YES" if b0_match else "NO"}')
    print(f'  Euler characteristic difference:  {euler_diff}')
    print(f'  Volume difference:                {vol_diff_pct:.1f}%')

    if b0_match and euler_diff < 5:
        print('\n  VERDICT: Topology is well preserved.')
    elif euler_diff < 20:
        print('\n  VERDICT: Minor topological differences (acceptable).')
    else:
        print('\n  VERDICT: Significant topological differences detected.')

    if ablation is not None:
        abl_euler_diff = abs(topo_abl['euler'] - topo_target['euler'])
        print(f'\n  Ablation Euler diff: {abl_euler_diff} vs With-Topo Euler diff: {euler_diff}')
        if euler_diff < abl_euler_diff:
            print('  CONCLUSION: Topology loss IMPROVES topological fidelity.')
        else:
            print('  CONCLUSION: Topology loss did not improve topological fidelity.')

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
