"""
Extract per-subject feature vectors from FreeSurfer/FastSurfer aparc+aseg files.

Output is the feature CSV that scripts/train_ad_classifier.py consumes:
one row per subject with subcortical volumes (mm^3), ventricular volumes,
and total intracranial volume (TIV) as columns. Optionally normalises by
TIV to remove head-size confounding.

This script is segmentation-tool agnostic — anything that produces a NIfTI
or MGZ with FreeSurfer aseg label IDs works (FreeSurfer recon-all,
FastSurfer, SynthSeg, etc.). Default region panel matches the AD
biomarker literature: subcortical volumes + lateral/3rd/4th ventricles.

Standard FreeSurfer aseg label IDs used:
    17/53   Hippocampus L/R
    18/54   Amygdala L/R
    10/49   Thalamus L/R
    11/50   Caudate L/R
    12/51   Putamen L/R
    13/52   Pallidum L/R
    26/58   Accumbens-area L/R
     7/46   Cerebellum White Matter L/R
     8/47   Cerebellum Cortex L/R
     4/43   Lateral Ventricle L/R         <- AD-relevant (enlarged)
     5/44   Inf Lateral Ventricle L/R
       14   3rd Ventricle
       15   4th Ventricle
       16   Brain Stem

TIV proxy = total volume of all non-zero voxels in aparc+aseg.
This is NOT FreeSurfer's eTIV (which uses an atlas registration), but is
a stable proxy good enough for AD-classifier head-size normalization.

Usage:
    python scripts/extract_features_from_aseg.py \\
        --pairs-csv aseg_pairs.csv \\
        --output features.csv

    # Express volumes as fraction-of-TIV instead of raw mm^3:
    python scripts/extract_features_from_aseg.py \\
        --pairs-csv aseg_pairs.csv \\
        --output features.csv \\
        --normalize-by-tiv

    # Validate the voxel-counting math:
    python scripts/extract_features_from_aseg.py --self-test

aseg_pairs.csv columns:
    subject           — subject ID (will appear in output as `subject`)
    group             — AD or CN (passed through to output)
    aparc_aseg_path   — path to the aparc+aseg NIfTI/MGZ
    [age, sex]        — optional, passed through if present
"""
import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


# Region name -> (label_L, label_R or None for midline)
ASEG_LABELS: Dict[str, tuple] = {
    "hippo":            (17, 53),
    "amygdala":         (18, 54),
    "thalamus":         (10, 49),
    "caudate":          (11, 50),
    "putamen":          (12, 51),
    "pallidum":         (13, 52),
    "accumbens":        (26, 58),
    "cerebellum_wm":    (7, 46),
    "cerebellum_cx":    (8, 47),
    "lat_ventricle":    (4, 43),
    "inf_lat_ventricle": (5, 44),
    "third_ventricle":  (14, None),
    "fourth_ventricle": (15, None),
    "brainstem":        (16, None),
}

DEFAULT_REGIONS = list(ASEG_LABELS.keys())


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def voxel_volume_mm3(affine: np.ndarray) -> float:
    return float(abs(np.linalg.det(affine[:3, :3])))


def load_seg(path: Path):
    """Load a NIfTI / MGZ aparc+aseg. Returns (int seg array, voxel vol in mm^3)."""
    import nibabel as nib
    img = nib.load(str(path))
    arr = np.round(img.get_fdata()).astype(np.int32)
    return arr, voxel_volume_mm3(img.affine)


def count_label(seg: np.ndarray, label_id: int) -> int:
    return int(np.count_nonzero(seg == label_id))


def features_for_seg(seg: np.ndarray, voxel_vol_mm3: float,
                     regions: Sequence[str]) -> Dict[str, float]:
    """Per-region L/R volumes (mm^3) plus TIV (sum of non-zero voxels)."""
    out: Dict[str, float] = {}

    # Subcortical / ventricular volumes
    for region in regions:
        if region not in ASEG_LABELS:
            raise KeyError(f"Unknown region '{region}'. "
                           f"Known: {list(ASEG_LABELS)}")
        l_id, r_id = ASEG_LABELS[region]
        out[f"{region}_L_mm3"] = count_label(seg, l_id) * voxel_vol_mm3
        if r_id is not None:
            out[f"{region}_R_mm3"] = count_label(seg, r_id) * voxel_vol_mm3

    # Total intracranial volume proxy: every labelled voxel.
    out["tiv_mm3"] = int(np.count_nonzero(seg)) * voxel_vol_mm3
    return out


def normalize_by_tiv(row: Dict[str, float]) -> Dict[str, float]:
    """Convert *_mm3 columns into fractions of TIV. Keeps tiv_mm3 untouched."""
    tiv = row.get("tiv_mm3", 0.0) or 0.0
    if tiv <= 0:
        # Fall back to raw values if TIV is degenerate
        return row
    out = {"tiv_mm3": tiv}
    for k, v in row.items():
        if k == "tiv_mm3":
            continue
        if k.endswith("_mm3"):
            out[k.replace("_mm3", "_frac")] = float(v) / tiv
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    """Verify voxel counting + region extraction on synthetic data."""
    print("Running extract_features_from_aseg self-test...")

    # Build a 50x50x50 synthetic aparc+aseg.
    # Place a 10x10x10 hippocampus_L (label 17) at one corner: 1000 voxels.
    # Place a  5x 5x 5 hippocampus_R (label 53) elsewhere:      125 voxels.
    # Some background label-2 noise to test it's correctly excluded.
    seg = np.zeros((50, 50, 50), dtype=np.int32)
    seg[2:12,  2:12,  2:12] = 17     # 1000 vox
    seg[20:25, 20:25, 20:25] = 53    #  125 vox
    seg[30:40, 30:40, 30:40] = 7     # 1000 vox cerebellum WM
    seg[15:18, 15:18, 15:18] = 99    #   27 vox of an unknown label (ignored)

    # 1.5 mm isotropic spacing -> voxel volume = 3.375 mm^3
    affine = np.eye(4)
    affine[0, 0] = 1.5
    affine[1, 1] = 1.5
    affine[2, 2] = 1.5
    vox_mm3 = voxel_volume_mm3(affine)
    print(f"  voxel volume = {vox_mm3:.3f} mm^3 (expected 3.375)")
    assert abs(vox_mm3 - 3.375) < 1e-9

    feats = features_for_seg(seg, vox_mm3, DEFAULT_REGIONS)

    expected = {
        "hippo_L_mm3":         1000 * 3.375,
        "hippo_R_mm3":         125  * 3.375,
        "cerebellum_wm_L_mm3": 1000 * 3.375,
        "cerebellum_wm_R_mm3": 0.0,
        "amygdala_L_mm3":      0.0,
        "tiv_mm3":             (1000 + 125 + 1000 + 27) * 3.375,
    }
    for k, want in expected.items():
        got = feats[k]
        ok = abs(got - want) < 1e-6
        sym = "✓" if ok else "✗"
        print(f"  {sym} {k:<25} got={got:>9.2f}  want={want:>9.2f}")
        assert ok, f"{k}: got {got}, want {want}"

    # TIV-normalize and verify it preserves ratios
    norm = normalize_by_tiv(feats)
    tiv = expected["tiv_mm3"]
    assert abs(norm["hippo_L_frac"] - 1000 * 3.375 / tiv) < 1e-9
    assert abs(norm["hippo_R_frac"] - 125 * 3.375 / tiv) < 1e-9
    print(f"  ✓ tiv-normalization: hippo_L_frac = {norm['hippo_L_frac']:.4f}")

    # Empty seg -> zero TIV path stays well-behaved
    empty = features_for_seg(np.zeros((10, 10, 10), dtype=np.int32), 1.0, ["hippo"])
    assert empty["tiv_mm3"] == 0.0 and empty["hippo_L_mm3"] == 0.0
    print("  ✓ empty seg: returns zeros without crashing")

    print("✓ All extract_features_from_aseg self-tests passed.")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pairs-csv", type=Path, default=None,
                        help="Input CSV with cols: subject, group, aparc_aseg_path "
                             "(plus optional age, sex)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output feature CSV path")
    parser.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS,
                        choices=list(ASEG_LABELS),
                        help=f"Regions to extract (default: all {len(DEFAULT_REGIONS)})")
    parser.add_argument("--normalize-by-tiv", action="store_true",
                        help="Convert per-region volumes to fraction-of-TIV")
    parser.add_argument("--self-test", action="store_true",
                        help="Validate voxel counting against synthetic data")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    if args.self_test:
        return _self_test()

    if args.pairs_csv is None or args.output is None:
        logging.error("--pairs-csv and --output are required (or use --self-test)")
        return 2

    pairs_csv = args.pairs_csv.resolve()
    if not pairs_csv.exists():
        logging.error("pairs CSV not found: %s", pairs_csv)
        return 2

    rows: List[Dict] = []
    passthrough_cols = ("group", "age", "sex")

    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        required = {"subject", "aparc_aseg_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            logging.error("pairs CSV missing required cols: %s", sorted(missing))
            return 2

        for entry in reader:
            subject = entry["subject"]
            aseg_path = Path(entry["aparc_aseg_path"]).expanduser()
            if not aseg_path.exists():
                logging.warning("[%s] aparc_aseg not found: %s", subject, aseg_path)
                continue

            try:
                seg, vox_mm3 = load_seg(aseg_path)
                feats = features_for_seg(seg, vox_mm3, args.regions)
            except Exception as e:
                logging.error("[%s] failed: %s", subject, e)
                continue

            if args.normalize_by_tiv:
                feats = normalize_by_tiv(feats)

            row = {"subject": subject}
            for col in passthrough_cols:
                if col in entry and entry[col] != "":
                    row[col] = entry[col]
            row.update(feats)
            rows.append(row)
            logging.info("[%s] %d features extracted (TIV=%.0f mm^3)",
                         subject, len(feats), feats.get("tiv_mm3", 0))

    if not rows:
        logging.error("No subjects processed successfully")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject"]
    for col in passthrough_cols:
        if any(col in r for r in rows):
            fieldnames.append(col)
    feature_cols = sorted({k for r in rows for k in r if k not in fieldnames})
    fieldnames += feature_cols

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    logging.info("Wrote %s (%d subjects, %d feature cols)",
                 args.output, len(rows), len(feature_cols))

    # Sanity-print first 3 rows column-aligned
    print(f"\nFirst rows of {args.output.name}:")
    for r in rows[:3]:
        head = f"  {r['subject']:<14} group={r.get('group','?'):<3}"
        sample = "  ".join(f"{c}={r[c]:.0f}"
                           for c in ("hippo_L_mm3", "hippo_R_mm3", "tiv_mm3")
                           if c in r)
        print(f"{head}  {sample}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
