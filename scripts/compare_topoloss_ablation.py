"""
Compare a topology-loss model against a no-topology ablation.

This script evaluates both predicted tissue segmentations against the same
4-class ground-truth tissue segmentation:

    0 = Background, 1 = CSF, 2 = GM, 3 = WM

It reports metrics that are useful for defending the topology/anatomy loss:

    - Dice and Jaccard overlap
    - HD95 and ASSD boundary distances
    - Tissue and brain-mask volume error
    - Connected-component fragmentation
    - Largest connected-component fraction
    - Small island count and volume percentage

Example:
    python scripts/compare_topoloss_ablation.py \
        --topo-seg results/sub-06-105k/predicted_seg.nii.gz \
        --notopo-seg results/sub-06-notopo-50k/predicted_seg.nii.gz \
        --ground-truth-seg data/sub-06/ground_truth_4class_seg.nii.gz \
        --out-dir results/sub-06/topoloss_ablation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


LABELS = {
    1: "CSF",
    2: "GM",
    3: "WM",
}


def load_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    import nibabel as nib

    img = nib.load(str(path))
    data = np.rint(img.get_fdata()).astype(np.int16)
    spacing = tuple(float(x) for x in np.abs(img.header.get_zooms()[:3]))
    return data, spacing


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    denom = int(mask_a.sum()) + int(mask_b.sum())
    if denom == 0:
        return float("nan")
    return float(2.0 * np.logical_and(mask_a, mask_b).sum() / denom)


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = int(np.logical_or(mask_a, mask_b).sum())
    if union == 0:
        return float("nan")
    return float(np.logical_and(mask_a, mask_b).sum() / union)


def surface_distances(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    from scipy.ndimage import binary_erosion, distance_transform_edt

    if not mask_a.any() or not mask_b.any():
        return np.array([], dtype=np.float64)

    surface_a = mask_a & ~binary_erosion(mask_a)
    surface_b = mask_b & ~binary_erosion(mask_b)

    if not surface_a.any() or not surface_b.any():
        return np.array([], dtype=np.float64)

    dist_to_b = distance_transform_edt(~surface_b, sampling=spacing)
    dist_to_a = distance_transform_edt(~surface_a, sampling=spacing)
    return np.concatenate([dist_to_b[surface_a], dist_to_a[surface_b]])


def hd95_and_assd(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[float, float]:
    distances = surface_distances(mask_a, mask_b, spacing)
    if distances.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(distances, 95)), float(distances.mean())


def component_metrics(mask: np.ndarray, small_island_voxels: int) -> dict[str, Any]:
    from scipy import ndimage

    total = int(mask.sum())
    if total == 0:
        return {
            "components": 0,
            "largest_component_fraction": float("nan"),
            "small_island_count": 0,
            "small_island_voxels": 0,
            "small_island_volume_fraction": float("nan"),
        }

    labeled, n_components = ndimage.label(mask)
    sizes = np.asarray(
        ndimage.sum(mask, labeled, index=np.arange(1, n_components + 1)),
        dtype=np.float64,
    )

    small = sizes < small_island_voxels
    small_voxels = int(sizes[small].sum()) if sizes.size else 0

    return {
        "components": int(n_components),
        "largest_component_fraction": float(sizes.max() / total),
        "small_island_count": int(small.sum()),
        "small_island_voxels": small_voxels,
        "small_island_volume_fraction": float(small_voxels / total),
    }


def evaluate_scope(
    pred: np.ndarray,
    gt: np.ndarray,
    mask_pred: np.ndarray,
    mask_gt: np.ndarray,
    spacing: tuple[float, float, float],
    small_island_voxels: int,
) -> dict[str, Any]:
    voxel_vol = float(np.prod(spacing))
    pred_voxels = int(mask_pred.sum())
    gt_voxels = int(mask_gt.sum())
    pred_volume = pred_voxels * voxel_vol
    gt_volume = gt_voxels * voxel_vol

    vol_diff_pct = (
        (pred_volume - gt_volume) / gt_volume * 100.0
        if gt_volume > 0
        else float("nan")
    )

    hd95, assd = hd95_and_assd(mask_pred, mask_gt, spacing)
    pred_components = component_metrics(mask_pred, small_island_voxels)
    gt_components = component_metrics(mask_gt, small_island_voxels)

    return {
        "dice": dice(mask_pred, mask_gt),
        "jaccard": jaccard(mask_pred, mask_gt),
        "hd95_mm": hd95,
        "assd_mm": assd,
        "pred_voxels": pred_voxels,
        "gt_voxels": gt_voxels,
        "pred_volume_mm3": pred_volume,
        "gt_volume_mm3": gt_volume,
        "volume_diff_pct": vol_diff_pct,
        "abs_volume_diff_pct": abs(vol_diff_pct) if not math.isnan(vol_diff_pct) else float("nan"),
        "pred_components": pred_components["components"],
        "gt_components": gt_components["components"],
        "component_diff": pred_components["components"] - gt_components["components"],
        "abs_component_diff": abs(pred_components["components"] - gt_components["components"]),
        "pred_largest_component_fraction": pred_components["largest_component_fraction"],
        "gt_largest_component_fraction": gt_components["largest_component_fraction"],
        "pred_small_island_count": pred_components["small_island_count"],
        "gt_small_island_count": gt_components["small_island_count"],
        "pred_small_island_voxels": pred_components["small_island_voxels"],
        "gt_small_island_voxels": gt_components["small_island_voxels"],
        "pred_small_island_volume_pct": pred_components["small_island_volume_fraction"] * 100.0,
        "gt_small_island_volume_pct": gt_components["small_island_volume_fraction"] * 100.0,
    }


def evaluate_segmentation(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, float, float],
    small_island_voxels: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}

    for label_id, label_name in LABELS.items():
        results[label_name] = evaluate_scope(
            pred,
            gt,
            pred == label_id,
            gt == label_id,
            spacing,
            small_island_voxels,
        )

    results["Brain"] = evaluate_scope(
        pred,
        gt,
        pred > 0,
        gt > 0,
        spacing,
        small_island_voxels,
    )
    return results


HIGHER_IS_BETTER = {
    "dice": True,
    "jaccard": True,
    "pred_largest_component_fraction": True,
}

LOWER_IS_BETTER = {
    "hd95_mm",
    "assd_mm",
    "abs_volume_diff_pct",
    "abs_component_diff",
    "pred_small_island_count",
    "pred_small_island_voxels",
    "pred_small_island_volume_pct",
}


def compare_metric(metric: str, topo_value: float, notopo_value: float) -> dict[str, Any]:
    if math.isnan(float(topo_value)) or math.isnan(float(notopo_value)):
        return {"delta": float("nan"), "relative_improvement_pct": float("nan"), "winner": "NA"}

    delta = float(topo_value - notopo_value)

    if metric in HIGHER_IS_BETTER:
        improvement = delta
        winner = "topo" if topo_value > notopo_value else "notopo" if topo_value < notopo_value else "tie"
    elif metric in LOWER_IS_BETTER:
        improvement = notopo_value - topo_value
        winner = "topo" if topo_value < notopo_value else "notopo" if topo_value > notopo_value else "tie"
    else:
        return {"delta": delta, "relative_improvement_pct": float("nan"), "winner": "NA"}

    denom = abs(float(notopo_value))
    rel = improvement / denom * 100.0 if denom > 1e-12 else float("nan")
    return {"delta": delta, "relative_improvement_pct": float(rel), "winner": winner}


def build_comparison_rows(
    topo: dict[str, dict[str, Any]],
    notopo: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    metric_names = [
        "dice",
        "jaccard",
        "hd95_mm",
        "assd_mm",
        "volume_diff_pct",
        "abs_volume_diff_pct",
        "pred_components",
        "gt_components",
        "component_diff",
        "abs_component_diff",
        "pred_largest_component_fraction",
        "pred_small_island_count",
        "pred_small_island_voxels",
        "pred_small_island_volume_pct",
    ]

    for scope in ["CSF", "GM", "WM", "Brain"]:
        for metric in metric_names:
            topo_value = topo[scope].get(metric, float("nan"))
            notopo_value = notopo[scope].get(metric, float("nan"))
            comparison = compare_metric(metric, float(topo_value), float(notopo_value))
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "notopo": notopo_value,
                    "topo": topo_value,
                    **comparison,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "scope",
        "metric",
        "notopo",
        "topo",
        "delta",
        "relative_improvement_pct",
        "winner",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_report(rows: list[dict[str, Any]]) -> None:
    key_metrics = {
        "dice",
        "hd95_mm",
        "assd_mm",
        "abs_volume_diff_pct",
        "abs_component_diff",
        "pred_largest_component_fraction",
        "pred_small_island_count",
        "pred_small_island_volume_pct",
    }
    print("\nTopology-loss ablation summary")
    print("Metric values compare no-topo vs topo, both against the same GT.\n")
    print(f"{'Scope':<7} {'Metric':<32} {'No-topo':>12} {'Topo':>12} {'Improvement':>13} {'Winner':>8}")
    print("-" * 91)
    for row in rows:
        if row["metric"] not in key_metrics:
            continue
        imp = row["relative_improvement_pct"]
        imp_str = "NA" if math.isnan(float(imp)) else f"{float(imp):+.1f}%"
        print(
            f"{row['scope']:<7} {row['metric']:<32} "
            f"{float(row['notopo']):>12.4f} {float(row['topo']):>12.4f} "
            f"{imp_str:>13} {row['winner']:>8}"
        )


def save_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    selected = [
        ("CSF", "dice", "CSF Dice"),
        ("GM", "dice", "GM Dice"),
        ("WM", "dice", "WM Dice"),
        ("Brain", "dice", "Brain Dice"),
        ("Brain", "hd95_mm", "Brain HD95 (mm)"),
        ("CSF", "abs_component_diff", "CSF component error"),
        ("GM", "abs_component_diff", "GM component error"),
        ("WM", "abs_component_diff", "WM component error"),
        ("CSF", "pred_small_island_count", "CSF small islands"),
        ("GM", "pred_small_island_count", "GM small islands"),
        ("WM", "pred_small_island_count", "WM small islands"),
    ]
    lookup = {(r["scope"], r["metric"]): r for r in rows}
    lines = [
        "| Metric | No-topo | Topo | Relative improvement | Winner |",
        "|---|---:|---:|---:|---|",
    ]
    for scope, metric, label in selected:
        row = lookup[(scope, metric)]
        imp = row["relative_improvement_pct"]
        imp_str = "NA" if math.isnan(float(imp)) else f"{float(imp):+.1f}%"
        lines.append(
            f"| {label} | {float(row['notopo']):.4f} | {float(row['topo']):.4f} | "
            f"{imp_str} | {row['winner']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assert_same_shape(name: str, data: np.ndarray, gt: np.ndarray) -> None:
    if data.shape != gt.shape:
        raise ValueError(
            f"{name} shape {data.shape} does not match ground truth shape {gt.shape}. "
            "Resample all segmentations to the same grid first using nearest-neighbor interpolation."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topo-seg", required=True, help="Predicted segmentation from topo-loss model.")
    parser.add_argument("--notopo-seg", required=True, help="Predicted segmentation from no-topology model.")
    parser.add_argument("--ground-truth-seg", required=True, help="Correct 4-class ground-truth tissue segmentation.")
    parser.add_argument("--out-dir", default="results/topoloss_ablation", help="Output directory.")
    parser.add_argument(
        "--small-island-voxels",
        type=int,
        default=100,
        help="Components smaller than this voxel count are counted as small islands.",
    )
    args = parser.parse_args()

    topo_path = Path(args.topo_seg)
    notopo_path = Path(args.notopo_seg)
    gt_path = Path(args.ground_truth_seg)
    out_dir = Path(args.out_dir)

    for path in [topo_path, notopo_path, gt_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("Loading NIfTI segmentations...")
    topo, topo_spacing = load_nifti(topo_path)
    notopo, notopo_spacing = load_nifti(notopo_path)
    gt, gt_spacing = load_nifti(gt_path)

    assert_same_shape("topo segmentation", topo, gt)
    assert_same_shape("no-topo segmentation", notopo, gt)

    if not np.allclose(topo_spacing, gt_spacing, rtol=1e-3, atol=1e-3):
        print(f"WARNING: topo spacing {topo_spacing} differs from GT spacing {gt_spacing}; using GT spacing.")
    if not np.allclose(notopo_spacing, gt_spacing, rtol=1e-3, atol=1e-3):
        print(f"WARNING: no-topo spacing {notopo_spacing} differs from GT spacing {gt_spacing}; using GT spacing.")

    print("Evaluating topo model...")
    topo_metrics = evaluate_segmentation(topo, gt, gt_spacing, args.small_island_voxels)
    print("Evaluating no-topo model...")
    notopo_metrics = evaluate_segmentation(notopo, gt, gt_spacing, args.small_island_voxels)

    rows = build_comparison_rows(topo_metrics, notopo_metrics)

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "topoloss_ablation_metrics.json"
    csv_path = out_dir / "topoloss_ablation_comparison.csv"
    md_path = out_dir / "topoloss_ablation_report_table.md"

    report = {
        "inputs": {
            "topo_seg": str(topo_path),
            "notopo_seg": str(notopo_path),
            "ground_truth_seg": str(gt_path),
            "spacing_mm": gt_spacing,
            "small_island_voxels": args.small_island_voxels,
        },
        "topo_vs_gt": topo_metrics,
        "notopo_vs_gt": notopo_metrics,
        "comparison": rows,
    }
    json_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    write_csv(csv_path, rows)
    save_markdown_table(md_path, rows)

    print_report(rows)
    print(f"\nSaved JSON:     {json_path}")
    print(f"Saved CSV:      {csv_path}")
    print(f"Saved Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
