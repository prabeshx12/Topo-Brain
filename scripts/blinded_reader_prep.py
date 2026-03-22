"""
Prepare blinded reader study images for anatomical correctness assessment.

Generates a set of comparison panels (3T input / synthetic 7T / real 7T)
with subject labels randomized/blinded so a reader can score them without
knowing which is synthetic and which is real.

Outputs:
  results/blinded_study/
    panels/
      case_01_axial.png
      case_01_coronal.png
      ...
    key.csv          -- reveals which case maps to which subject/condition
    scoresheet.csv   -- blank scoring template for the reader

Usage:
  python scripts/blinded_reader_prep.py \\
      --subjects sub-09 sub-10 \\
      --data-root /path/to/data \\
      --pairs-csv pairs_new.csv \\
      --results-dir results/ \\
      --output-dir results/blinded_study/
"""
import csv
import json
import random
import argparse
import numpy as np
from pathlib import Path


def load_nifti(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
        return nib.load(str(path)).get_fdata().astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to load NIfTI: {path}") from e


def normalize_for_display(vol: np.ndarray, pct_low=1, pct_high=99) -> np.ndarray:
    """Percentile normalization for display."""
    lo = np.percentile(vol, pct_low)
    hi = np.percentile(vol, pct_high)
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo + 1e-8)
    return vol


def get_slices(vol: np.ndarray, orient: str, pos: float = 0.5) -> np.ndarray:
    """Extract a 2D slice from a 3D volume."""
    D, H, W = vol.shape
    idx = {
        "axial": int(D * pos),
        "coronal": int(H * pos),
        "sagittal": int(W * pos),
    }[orient]
    if orient == "axial":
        return np.rot90(vol[idx, :, :])
    elif orient == "coronal":
        return np.rot90(vol[:, idx, :])
    elif orient == "sagittal":
        return np.rot90(vol[:, :, idx])


def make_panel(images: dict, orient: str, pos: float, title: str, out_path: Path):
    """
    Create a blinded comparison panel.

    Args:
        images: {label: np.ndarray_3d}  -- label will NOT be shown (blinded)
        orient: 'axial' | 'coronal' | 'sagittal'
        pos: slice position as fraction [0, 1]
        title: panel title (just a case number, no subject info)
        out_path: where to save the PNG
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(images.keys())
    n = len(labels)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    # Use shared intensity range across all volumes
    all_slices = [get_slices(normalize_for_display(images[l]), orient, pos) for l in labels]
    vmin = min(s.min() for s in all_slices)
    vmax = max(s.max() for s in all_slices)

    for ax, label, sl in zip(axes, labels, all_slices):
        ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=13)
        ax.axis("off")

    plt.suptitle(f"{title} - {orient.capitalize()} view", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def resolve_pairs(pairs_csv: Path, subjects: list, data_root: Path) -> dict:
    result = {}
    with open(pairs_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            subj = row.get("subject", "")
            if subj in subjects:
                result[subj] = {
                    "input_3t": data_root / row["input_3t"],
                    "target_7t": data_root / row["target_7t"],
                }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Prepare blinded reader study comparison panels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/blinded_reader_prep.py \\
      --subjects sub-09 sub-10 \\
      --data-root /path/to/data \\
      --pairs-csv pairs_new.csv \\
      --results-dir results/ \\
      --output-dir results/blinded_study/
        """
    )
    parser.add_argument("--subjects", nargs="+", default=["sub-09", "sub-10"])
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--pairs-csv", type=str, default="pairs_new.csv")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results/blinded_study")
    parser.add_argument("--orientations", nargs="+", default=["axial", "coronal", "sagittal"],
                        choices=["axial", "coronal", "sagittal"])
    parser.add_argument("--positions", nargs="+", type=float, default=[0.35, 0.5, 0.65],
                        help="Slice positions as fractions 0-1 (default: 0.35 0.5 0.65)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for blinding order")
    parser.add_argument("--include-3t", action="store_true",
                        help="Also include 3T input as a reference (NOT blinded)")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    panels_dir = output_dir / "panels"

    pairs = resolve_pairs(Path(args.pairs_csv), args.subjects, Path(args.data_root))
    results_dir = Path(args.results_dir)

    # Collect all cases (subject x condition)
    # Each "case" is one subject shown as synthetic or real 7T
    cases = []
    for subject in args.subjects:
        if subject not in pairs:
            print(f"[WARN] {subject} not in pairs CSV, skipping.")
            continue

        synthetic_path = results_dir / subject / "predicted_7T.nii.gz"
        real_path = pairs[subject]["target_7t"]

        if not synthetic_path.exists():
            print(f"[WARN] Synthetic 7T not found for {subject}: {synthetic_path}")
            print(f"       Run evaluate_full_volume.py first.")
            continue

        if not real_path.exists():
            print(f"[WARN] Real 7T not found for {subject}: {real_path}")
            continue

        cases.append({
            "subject": subject,
            "condition": "synthetic7t",
            "path": synthetic_path,
        })
        cases.append({
            "subject": subject,
            "condition": "real7t",
            "path": real_path,
        })
        if args.include_3t:
            t3_path = pairs[subject]["input_3t"]
            if t3_path.exists():
                cases.append({
                    "subject": subject,
                    "condition": "input_3t",
                    "path": t3_path,
                })

    if not cases:
        print("ERROR: No valid cases found. Check paths and run evaluate_full_volume.py first.")
        return

    # Randomize case order for blinding
    random.shuffle(cases)

    # Assign case numbers
    key_rows = []
    for i, case in enumerate(cases):
        case["case_id"] = f"case_{i+1:02d}"
        key_rows.append({
            "case_id": case["case_id"],
            "subject": case["subject"],
            "condition": case["condition"],
            "path": str(case["path"]),
        })

    # Pair cases: group by subject for side-by-side comparison panels
    # For blinded study, each IMAGE is shown separately (single condition per panel)
    # but we also create paired comparison panels for the full panel view

    print(f"\nGenerating blinded panels for {len(cases)} cases ...")
    print(f"Output: {output_dir}\n")

    # --- SINGLE-IMAGE panels (truly blinded: reader scores each alone) ---
    single_dir = panels_dir / "single"
    for case in cases:
        try:
            vol = load_nifti(case["path"])
        except RuntimeError as e:
            print(f"[WARN] {e}")
            continue

        for orient in args.orientations:
            for pos in args.positions:
                pos_label = f"pos{int(pos*100):02d}"
                out_path = single_dir / f"{case['case_id']}_{orient}_{pos_label}.png"
                make_panel(
                    images={"": vol},  # No label shown (blinded)
                    orient=orient,
                    pos=pos,
                    title=case["case_id"],
                    out_path=out_path,
                )
        print(f"  Generated: {case['case_id']} ({case['subject']} / {case['condition']})")

    # --- PAIRED comparison panels (for reference, NOT blinded) ---
    compare_dir = panels_dir / "paired_comparison"
    subject_cases = {}
    for case in cases:
        subject_cases.setdefault(case["subject"], {})[case["condition"]] = case

    for subject, cond_cases in subject_cases.items():
        volumes = {}
        labels = []
        for cond, case in sorted(cond_cases.items()):
            try:
                vol = load_nifti(case["path"])
                label = {
                    "synthetic7t": "Synthetic 7T",
                    "real7t": "Real 7T",
                    "input_3t": "3T Input",
                }[cond]
                volumes[label] = vol
                labels.append(label)
            except RuntimeError as e:
                print(f"[WARN] {e}")

        if len(volumes) < 2:
            continue

        for orient in args.orientations:
            for pos in args.positions:
                pos_label = f"pos{int(pos*100):02d}"
                out_path = compare_dir / f"{subject}_{orient}_{pos_label}.png"
                make_panel(
                    images=volumes,
                    orient=orient,
                    pos=pos,
                    title=f"{subject} comparison",
                    out_path=out_path,
                )

    # --- Save key (reveals which case is which) ---
    key_path = output_dir / "key.csv"
    with open(key_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "subject", "condition", "path"])
        writer.writeheader()
        writer.writerows(key_rows)
    print(f"\nSaved key (DO NOT SHARE WITH READERS): {key_path}")

    # --- Save blank scoresheet ---
    scoresheet_path = output_dir / "scoresheet.csv"
    score_rows = []
    for case in cases:
        for orient in args.orientations:
            for pos in args.positions:
                pos_label = f"pos{int(pos*100):02d}"
                score_rows.append({
                    "case_id": case["case_id"],
                    "orientation": orient,
                    "position": pos_label,
                    "image_file": f"panels/single/{case['case_id']}_{orient}_{pos_label}.png",
                    "score_overall_quality": "",   # 1-5
                    "score_anatomical_accuracy": "",  # 1-5
                    "score_tissue_contrast": "",   # 1-5
                    "score_edge_sharpness": "",    # 1-5
                    "notes": "",
                })

    with open(scoresheet_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(score_rows[0].keys()))
        writer.writeheader()
        writer.writerows(score_rows)
    print(f"Saved blank scoresheet: {scoresheet_path}")

    print(f"\nDone! {len(cases)} cases prepared in: {output_dir}/")
    print("\nBlinded reader instructions:")
    print("  1. Share ONLY the 'panels/single/' folder with the reader")
    print("  2. Do NOT share key.csv")
    print("  3. Reader fills in scoresheet.csv (scoring 1-5 per criterion)")
    print("  4. After scoring, merge with key.csv to reveal which is synthetic vs real")
    print("\nScoring criteria (1=Poor, 5=Excellent):")
    print("  overall_quality:       Overall image quality impression")
    print("  anatomical_accuracy:   Is the anatomy correct and realistic?")
    print("  tissue_contrast:       Is WM/GM/CSF contrast as expected for 7T?")
    print("  edge_sharpness:        Are cortical edges sharp and well-defined?")


if __name__ == "__main__":
    main()
