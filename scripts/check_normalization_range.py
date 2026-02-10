#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import nibabel as nib


def resolve_nii_path(path_str: str) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists():
        return p
    # Try alternate extension
    if p.suffix == ".gz":
        alt = p.with_suffix("").with_suffix(".nii")
    elif p.suffix == ".nii":
        alt = p.with_suffix(".nii.gz")
    else:
        alt = None
    if alt and alt.exists():
        return alt
    return None


def load_stats(path: Path) -> Tuple[float, float, float, float]:
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    return float(data.min()), float(data.max()), float(data.mean()), float(data.std())


def main() -> None:
    parser = argparse.ArgumentParser(description="Check normalization range for NIfTI volumes in pairs CSV.")
    parser.add_argument("--pairs", type=str, default=None, help="Pairs CSV (default: pairs_new.csv or pairs.csv)")
    parser.add_argument("--num", type=int, default=3, help="Number of rows to inspect (default: 3)")
    args = parser.parse_args()

    if args.pairs:
        pairs_path = Path(args.pairs)
    else:
        pairs_path = Path("pairs_new.csv") if Path("pairs_new.csv").exists() else Path("pairs.csv")

    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs CSV not found: {pairs_path}")

    print(f"Using pairs file: {pairs_path.resolve()}")

    rows = []
    with pairs_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append(row)
            if i >= max(0, args.num - 1):
                break

    for row in rows:
        subj = row.get("subject", "unknown")
        inp = row.get("input_3t") or row.get("input")
        tgt = row.get("target_7t") or row.get("target")

        for label, path_str in [("input", inp), ("target", tgt)]:
            if not path_str:
                print(f"{subj} {label}: missing")
                continue
            p = resolve_nii_path(path_str)
            if not p:
                print(f"{subj} {label}: not found -> {path_str}")
                continue
            vmin, vmax, vmean, vstd = load_stats(p)
            print(f"{subj} {label}: {p}  min={vmin:.3f} max={vmax:.3f} mean={vmean:.3f} std={vstd:.3f}")


if __name__ == "__main__":
    main()
