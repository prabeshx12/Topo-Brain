"""
Aggregate evaluation metrics across multiple subjects and produce a summary table.

After running evaluate_full_volume.py on each test subject, point this script
at the results directory to get a combined CSV and summary statistics.

Usage:
    python scripts/aggregate_metrics.py \
        --results-dir results/ \
        --subjects sub-09 sub-10 \
        --output results/aggregate_metrics.csv

    # Or aggregate all subjects found under results/
    python scripts/aggregate_metrics.py --results-dir results/
"""
import json
import csv
import sys
import argparse
import numpy as np
from pathlib import Path


METRIC_COLS = ["ssim", "psnr", "dice", "hd95_mm"]
METRIC_LABELS = {
    "ssim": "SSIM",
    "psnr": "PSNR (dB)",
    "dice": "Dice",
    "hd95_mm": "HD95 (mm)",
}
HIGHER_BETTER = {"ssim": True, "psnr": True, "dice": True, "hd95_mm": False}


def load_subject_metrics(results_dir: Path, subject: str) -> dict:
    """
    Load metrics for a single subject from its results directory.

    Looks for:
      results/<subject>/metrics.json  (preferred — written by evaluate_full_volume.py)
    """
    subject_dir = results_dir / subject
    metrics_file = subject_dir / "metrics.json"

    if not metrics_file.exists():
        # Try common alternative names
        for alt in ["eval_metrics.json", "results.json"]:
            alt_file = subject_dir / alt
            if alt_file.exists():
                metrics_file = alt_file
                break

    if not metrics_file.exists():
        print(f"  [WARN] No metrics.json found for {subject} in {subject_dir}")
        print(f"         Run: python scripts/evaluate_full_volume.py --subject {subject} ...")
        return {}

    with open(metrics_file) as f:
        data = json.load(f)

    # evaluate_full_volume.py nests metrics under mask-region keys (e.g. "FreeSurfer", "SegMask")
    # Flatten: prefer "FreeSurfer" if present, else take first available, else treat as flat
    if isinstance(data, dict):
        # Check if it has nested region keys
        region_keys = [k for k in data if isinstance(data[k], dict) and "ssim" in data[k]]
        if region_keys:
            preferred = "FreeSurfer" if "FreeSurfer" in region_keys else region_keys[0]
            return data[preferred]
        # Already flat
        return {k: v for k, v in data.items() if k in METRIC_COLS}

    return {}


def aggregate(results_dir: Path, subjects: list) -> list:
    """Return list of dicts: [{subject, ssim, psnr, dice, hd95_mm}, ...]"""
    rows = []
    for subject in subjects:
        metrics = load_subject_metrics(results_dir, subject)
        row = {"subject": subject}
        for col in METRIC_COLS:
            row[col] = metrics.get(col, float("nan"))
        rows.append(row)
    return rows


def print_table(rows: list):
    """Pretty-print the metrics table."""
    header = f"{'Subject':<12}" + "".join(f"{METRIC_LABELS[c]:>14}" for c in METRIC_COLS)
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("AGGREGATE METRICS SUMMARY")
    print("=" * len(header))
    print(header)
    print(sep)

    valid_rows = []
    for row in rows:
        vals = [row.get(c, float("nan")) for c in METRIC_COLS]
        fmt_vals = []
        for c, v in zip(METRIC_COLS, vals):
            if np.isnan(v):
                fmt_vals.append(f"{'N/A':>14}")
            elif c == "hd95_mm":
                fmt_vals.append(f"{v:>13.2f} ")
            elif c == "psnr":
                fmt_vals.append(f"{v:>13.2f} ")
            else:
                fmt_vals.append(f"{v:>13.4f} ")
        print(f"{row['subject']:<12}" + "".join(fmt_vals))
        if not all(np.isnan(v) for v in vals):
            valid_rows.append(row)

    if len(valid_rows) > 1:
        print(sep)
        # Mean ± std
        means = {}
        stds = {}
        for c in METRIC_COLS:
            vals = [r[c] for r in valid_rows if not np.isnan(r[c])]
            means[c] = np.mean(vals) if vals else float("nan")
            stds[c] = np.std(vals) if vals else float("nan")

        mean_str = f"{'Mean':<12}"
        std_str = f"{'Std':<12}"
        for c in METRIC_COLS:
            if np.isnan(means[c]):
                mean_str += f"{'N/A':>14}"
                std_str += f"{'N/A':>14}"
            elif c == "hd95_mm" or c == "psnr":
                mean_str += f"{means[c]:>13.2f} "
                std_str += f"{stds[c]:>13.2f} "
            else:
                mean_str += f"{means[c]:>13.4f} "
                std_str += f"{stds[c]:>13.4f} "
        print(mean_str)
        print(std_str)

    print("=" * len(header))
    print()

    # Clinical thresholds
    print("Clinical thresholds:")
    print("  SSIM  >0.90 = Excellent | >0.85 = Good | >0.80 = Acceptable")
    print("  PSNR  >35dB = Excellent | >30dB = Good | >25dB = Acceptable")
    print("  Dice  >0.90 = Excellent | >0.85 = Good | >0.80 = Acceptable")
    print("  HD95  <3mm  = Excellent |  <5mm = Good |  <10mm = Acceptable")
    print()


def save_csv(rows: list, output_path: Path):
    """Save metrics table as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject"] + METRIC_COLS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output_path}")


def find_evaluated_subjects(results_dir: Path) -> list:
    """Auto-discover subjects that have been evaluated (have a metrics.json)."""
    subjects = []
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and (d / "metrics.json").exists():
            subjects.append(d.name)
    return subjects


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation metrics across subjects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate test subjects
  python scripts/aggregate_metrics.py \\
      --results-dir results/ \\
      --subjects sub-09 sub-10 \\
      --output results/test_metrics.csv

  # Auto-discover all evaluated subjects
  python scripts/aggregate_metrics.py --results-dir results/
        """
    )
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing per-subject result folders (default: results/)")
    parser.add_argument("--subjects", type=str, nargs="+",
                        help="Subject IDs to aggregate (default: auto-discover from results-dir)")
    parser.add_argument("--output", type=str,
                        help="Output CSV path (default: <results-dir>/aggregate_metrics.csv)")
    parser.add_argument("--test-split", type=str, default="test_split.json",
                        help="Path to test_split.json (used when --subjects not given)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        print("Run evaluate_full_volume.py first to generate results.")
        sys.exit(1)

    # Determine subjects
    if args.subjects:
        subjects = args.subjects
    else:
        # Try test_split.json first
        split_file = Path(args.test_split)
        if split_file.exists():
            with open(split_file) as f:
                split = json.load(f)
            subjects = split.get("test", [])
            print(f"Loaded test subjects from {split_file}: {subjects}")
        else:
            # Auto-discover
            subjects = find_evaluated_subjects(results_dir)
            if not subjects:
                print("ERROR: No evaluated subjects found. Run evaluate_full_volume.py first.")
                sys.exit(1)
            print(f"Auto-discovered evaluated subjects: {subjects}")

    if not subjects:
        print("ERROR: No subjects specified or found.")
        sys.exit(1)

    # Aggregate
    rows = aggregate(results_dir, subjects)

    # Print table
    print_table(rows)

    # Save CSV
    output_path = Path(args.output) if args.output else results_dir / "aggregate_metrics.csv"
    save_csv(rows, output_path)

    # Check if any metrics are missing (subject not yet evaluated)
    missing = [r["subject"] for r in rows if all(np.isnan(r[c]) for c in METRIC_COLS)]
    if missing:
        print(f"WARNING: No metrics found for: {missing}")
        print("Run evaluate_full_volume.py on these subjects first.")
