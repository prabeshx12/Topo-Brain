#!/usr/bin/env python3
"""
Verify whether training will use seg masks or fallback masks from pairs CSV.

This mirrors training path handling:
- input/target are resolved under --data-root
- seg is resolved under --masks-root when provided, else under --data-root
- fallback is triggered when seg path cannot be resolved

It also checks segmentation label validity against --num-classes.
"""

import argparse
import csv
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np


def resolve_with_alt(path: Path) -> Optional[Path]:
    """Resolve a .nii/.nii.gz path with extension fallback."""
    if path.exists():
        return path
    if path.suffix == ".gz":
        alt = path.with_suffix("").with_suffix(".nii")
        if alt.exists():
            return alt
    elif path.suffix == ".nii":
        alt = path.with_suffix(".nii.gz")
        if alt.exists():
            return alt
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify seg mask usage and fallback risk.")
    parser.add_argument("--pairs-csv", default="pairs_new.csv", help="Path to pairs CSV.")
    parser.add_argument("--data-root", required=True, help="Data root used for input/target paths.")
    parser.add_argument("--masks-root", default=None, help="Masks root used for seg paths.")
    parser.add_argument("--num-classes", type=int, default=4, help="Model segmentation class count.")
    parser.add_argument("--check-labels", action="store_true", help="Load seg masks and check label validity.")
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows checked (0 = all).")
    args = parser.parse_args()

    pairs_csv = Path(args.pairs_csv)
    data_root = Path(args.data_root)
    masks_root = Path(args.masks_root) if args.masks_root else None

    if not pairs_csv.exists():
        raise FileNotFoundError(f"pairs CSV not found: {pairs_csv}")

    rows = []
    with pairs_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append(row)
            if args.max_rows > 0 and (i + 1) >= args.max_rows:
                break

    n_rows = len(rows)
    if n_rows == 0:
        print("No rows found in CSV.")
        return

    used_seg = 0
    fallback = 0
    missing_input = 0
    missing_target = 0
    invalid_label_rows = 0

    print(f"pairs_csv={pairs_csv}")
    print(f"data_root={data_root}")
    print(f"masks_root={masks_root if masks_root else '(none, using data_root for seg)'}")
    print(f"num_rows={n_rows}")
    print("-" * 80)

    for row in rows:
        subject = row.get("subject", "unknown")
        input_rel = row.get("input_3t", "")
        target_rel = row.get("target_7t", "")
        seg_rel = row.get("seg", "")

        input_path = resolve_with_alt(data_root / input_rel) if input_rel else None
        target_path = resolve_with_alt(data_root / target_rel) if target_rel else None
        seg_base = masks_root if masks_root else data_root
        seg_path = resolve_with_alt(seg_base / seg_rel) if seg_rel else None

        if input_path is None:
            missing_input += 1
        if target_path is None:
            missing_target += 1

        if seg_path is None:
            fallback += 1
            print(f"[FALLBACK] {subject}: seg missing -> {seg_rel}")
            continue

        used_seg += 1

        if args.check_labels:
            try:
                seg_arr = nib.load(str(seg_path)).get_fdata().astype(np.int64)
                uniq = np.unique(seg_arr)
                bad = uniq[(uniq < 0) | (uniq >= args.num_classes)]
                if bad.size > 0:
                    invalid_label_rows += 1
                    print(
                        f"[INVALID_LABELS] {subject}: "
                        f"labels(min={int(uniq.min())}, max={int(uniq.max())}) "
                        f"bad={bad[:10].tolist()} seg={seg_path}"
                    )
            except Exception as exc:
                invalid_label_rows += 1
                print(f"[READ_ERROR] {subject}: could not read seg mask {seg_path} ({exc})")

    print("-" * 80)
    print(f"seg used: {used_seg}/{n_rows}")
    print(f"fallback needed: {fallback}/{n_rows}")
    print(f"missing input paths: {missing_input}/{n_rows}")
    print(f"missing target paths: {missing_target}/{n_rows}")
    if args.check_labels:
        print(f"rows with invalid/failed seg labels: {invalid_label_rows}/{used_seg}")

    if fallback == 0 and missing_input == 0 and missing_target == 0:
        print("Status: path resolution looks correct for training.")
    else:
        print("Status: path issues found. Fix paths before training.")


if __name__ == "__main__":
    main()
