"""
Region preservation test (formalization of the hippocampus check from session
2026-05-05; see HANDOFF.md headline finding).

For each subcortical region defined by FreeSurfer aparc+aseg, measure how well
the model's predicted tissue segmentation preserves it as a contiguous GM blob.
Per-region metrics:
  components       int, # connected components of (predicted_seg == GM) within the GT region
  largest_frac     largest CC / total predicted-GM voxels in the region (1.0 = single blob)
  coverage         |predicted_GM and GT_region| / |GT_region|     (1.0 = no under-segmentation)
  vol_err_pct      (predicted_GM_in_region - GT_region_volume) / GT_region_volume * 100
  gt_volume_mm3    voxel-volume * |GT_region|
  pred_volume_mm3  voxel-volume * |predicted_GM and GT_region|

The "GM" label in predicted_seg is configurable but defaults to 2 (matches
the project's preprocess_masks.py convention: 0=BG, 1=CSF, 2=GM, 3=WM).

The aparc+aseg input MUST be on the same voxel grid as predicted_seg. If not,
resample it ahead of time with nearest-neighbour interpolation. This script
errors loudly on shape mismatch rather than silently resampling.

Single-subject usage:
    python scripts/check_region_preservation.py \\
        --subject sub-06 \\
        --predicted-seg results/sub-06/predicted_seg.nii.gz \\
        --aparc-aseg /path/to/sub-06_ses-2_aparc+aseg.nii.gz \\
        --output results/sub-06/region_preservation.csv

Batch usage: provide a CSV with columns (subject, predicted_seg, aparc_aseg).
Any of those paths may be absolute or relative to --pairs-root.
    python scripts/check_region_preservation.py \\
        --pairs-csv pairs_region_check.csv \\
        --pairs-root . \\
        --output results/region_preservation_all.csv
"""
import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# FreeSurfer aparc+aseg label IDs for the regions reported in HANDOFF.md.
# (See $FREESURFER_HOME/FreeSurferColorLUT.txt for the canonical list.)
REGIONS = {
    "Hippo_L":   17,
    "Hippo_R":   53,
    "Amyg_L":    18,
    "Amyg_R":    54,
    "Thal_L":    10,
    "Thal_R":    49,
    "Caud_L":    11,
    "Caud_R":    50,
    "Putamen_L": 12,
    "Putamen_R": 51,
    "Pallid_L":  13,
    "Pallid_R":  52,
}


def _load_nifti(path: Path):
    import nibabel as nib
    img = nib.load(str(path))
    return img.get_fdata(), img.affine


def _connected_components(mask: np.ndarray) -> tuple:
    from scipy import ndimage
    if mask.sum() == 0:
        return 0, 0.0
    labeled, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    largest_frac = float(max(sizes) / mask.sum())
    return int(n), largest_frac


def analyze_subject(
    predicted_seg: np.ndarray,
    aparc_aseg: np.ndarray,
    voxel_vol_mm3: float,
    gm_label: int,
) -> dict:
    if predicted_seg.shape != aparc_aseg.shape:
        raise ValueError(
            f"Shape mismatch: predicted_seg {predicted_seg.shape} vs "
            f"aparc+aseg {aparc_aseg.shape}. Resample to a common grid first."
        )

    pred_gm = (predicted_seg == gm_label)

    out = {}
    for region_name, region_label in REGIONS.items():
        gt_region = (aparc_aseg == region_label)
        gt_count = int(gt_region.sum())

        if gt_count == 0:
            out[region_name] = {
                "components": 0,
                "largest_frac": float("nan"),
                "coverage": float("nan"),
                "vol_err_pct": float("nan"),
                "gt_volume_mm3": 0.0,
                "pred_volume_mm3": 0.0,
            }
            continue

        intersection = pred_gm & gt_region
        inter_count = int(intersection.sum())

        coverage = inter_count / gt_count
        vol_err_pct = (inter_count - gt_count) / gt_count * 100.0

        if inter_count == 0:
            n_cc, largest_frac = 0, float("nan")
        else:
            n_cc, largest_frac = _connected_components(intersection)

        out[region_name] = {
            "components": n_cc,
            "largest_frac": largest_frac,
            "coverage": coverage,
            "vol_err_pct": vol_err_pct,
            "gt_volume_mm3": gt_count * voxel_vol_mm3,
            "pred_volume_mm3": inter_count * voxel_vol_mm3,
        }
    return out


def _flatten_for_csv(subject: str, results: dict) -> dict:
    row = {"subject": subject}
    for region_name, m in results.items():
        for metric, value in m.items():
            row[f"{region_name}__{metric}"] = value
    return row


def _print_table(subject: str, results: dict) -> None:
    print(f"\nSubject: {subject}")
    print(f"  {'Region':<10} {'CC':>4} {'Largest':>9} {'Coverage':>10} {'VolErr':>10}")
    print(f"  {'-' * 50}")
    for region_name, m in results.items():
        cc = m["components"]
        lf = m["largest_frac"]
        cov = m["coverage"]
        ve = m["vol_err_pct"]
        lf_str = f"{lf*100:>7.1f}%" if not np.isnan(lf) else "      N/A"
        cov_str = f"{cov*100:>8.1f}%" if not np.isnan(cov) else "       N/A"
        ve_str = f"{ve:>+8.1f}%" if not np.isnan(ve) else "       N/A"
        print(f"  {region_name:<10} {cc:>4} {lf_str:>9} {cov_str:>10} {ve_str:>10}")


def _resolve(path_str: str, root: Optional[Path]) -> Path:
    p = Path(path_str)
    if p.is_absolute() or root is None:
        return p
    return root / p


def _process_one(
    subject: str,
    predicted_seg_path: Path,
    aparc_aseg_path: Path,
    gm_label: int,
) -> Optional[dict]:
    if not predicted_seg_path.exists():
        logging.error("[%s] predicted_seg not found: %s", subject, predicted_seg_path)
        return None
    if not aparc_aseg_path.exists():
        logging.error("[%s] aparc_aseg not found: %s", subject, aparc_aseg_path)
        return None

    pred, _ = _load_nifti(predicted_seg_path)
    aseg, aseg_affine = _load_nifti(aparc_aseg_path)
    pred = np.round(pred).astype(np.int32)
    aseg = np.round(aseg).astype(np.int32)
    voxel_vol = float(np.prod(np.abs(np.diag(aseg_affine)[:3])))

    try:
        results = analyze_subject(pred, aseg, voxel_vol, gm_label)
    except ValueError as e:
        logging.error("[%s] %s", subject, e)
        return None
    _print_table(subject, results)
    return _flatten_for_csv(subject, results)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subject", default=None,
                        help="Subject id (single-subject mode)")
    parser.add_argument("--predicted-seg", default=None,
                        help="Path to predicted_seg NIfTI (single-subject mode)")
    parser.add_argument("--aparc-aseg", default=None,
                        help="Path to aparc+aseg NIfTI (single-subject mode)")
    parser.add_argument("--pairs-csv", default=None,
                        help="Batch mode CSV with columns: subject, predicted_seg, aparc_aseg")
    parser.add_argument("--pairs-root", default=None,
                        help="Root for resolving relative paths in --pairs-csv")
    parser.add_argument("--gm-label", type=int, default=2,
                        help="Integer label of GM in predicted_seg (default 2)")
    parser.add_argument("--output", required=True,
                        help="Output CSV path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    rows = []
    if args.pairs_csv:
        pairs_root = Path(args.pairs_root).resolve() if args.pairs_root else None
        with open(args.pairs_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"subject", "predicted_seg", "aparc_aseg"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                logging.error("--pairs-csv missing columns: %s", sorted(missing))
                return 2
            for row in reader:
                subject = row["subject"]
                pred_path = _resolve(row["predicted_seg"], pairs_root)
                aseg_path = _resolve(row["aparc_aseg"], pairs_root)
                result = _process_one(subject, pred_path, aseg_path, args.gm_label)
                if result is not None:
                    rows.append(result)
    else:
        if not (args.subject and args.predicted_seg and args.aparc_aseg):
            logging.error("Single-subject mode requires --subject, --predicted-seg, --aparc-aseg")
            return 2
        result = _process_one(
            args.subject,
            Path(args.predicted_seg),
            Path(args.aparc_aseg),
            args.gm_label,
        )
        if result is not None:
            rows.append(result)

    if not rows:
        logging.error("No subjects processed successfully.")
        return 1

    fieldnames = ["subject"] + [
        f"{r}__{m}"
        for r in REGIONS.keys()
        for m in ("components", "largest_frac", "coverage", "vol_err_pct",
                  "gt_volume_mm3", "pred_volume_mm3")
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    logging.info("Wrote %d row(s) to %s", len(rows), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
