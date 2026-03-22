"""
Run MRIQC on 3T input, synthetic 7T, and real 7T volumes for comparison.

MRIQC computes Image Quality Metrics (IQMs) including:
  - CJV  (Coefficient of Joint Variation)   -- measures WM/GM separation; lower is better
  - CNR  (Contrast-to-Noise Ratio)          -- GM-WM contrast; higher is better
  - EFC  (Entropy Focus Criterion)          -- ghosting/motion; lower is better
  - FBER (Foreground-Background Energy Ratio) -- SNR proxy; higher is better
  - SNR  (Signal-to-Noise Ratio)            -- per-tissue SNR
  - QI   (Quality Index)                    -- motion artifacts; lower is better

INSTALLATION (run once on the server, or locally via pip):
  pip install mriqc

  If pip install fails (missing system deps), use Docker:
    docker pull nipreps/mriqc:latest

USAGE (after running evaluate_full_volume.py to generate synthetic 7T):

  # 1. Run on 3T input volumes:
  python scripts/run_mriqc.py --mode 3t \\
      --subjects sub-09 sub-10 \\
      --data-root /path/to/data \\
      --pairs-csv pairs_new.csv \\
      --output-dir results/mriqc/3t/

  # 2. Run on synthetic 7T volumes:
  python scripts/run_mriqc.py --mode synthetic7t \\
      --subjects sub-09 sub-10 \\
      --results-dir results/ \\
      --output-dir results/mriqc/synthetic7t/

  # 3. Run on real 7T volumes:
  python scripts/run_mriqc.py --mode real7t \\
      --subjects sub-09 sub-10 \\
      --data-root /path/to/data \\
      --pairs-csv pairs_new.csv \\
      --output-dir results/mriqc/real7t/

  # 4. Compare all three:
  python scripts/run_mriqc.py --mode compare \\
      --mriqc-dir results/mriqc/
"""
import os
import sys
import json
import argparse
import subprocess
import tempfile
import csv
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# MRIQC availability check
# ---------------------------------------------------------------------------

def check_mriqc_available() -> bool:
    """Return True if mriqc is on PATH."""
    result = subprocess.run(["mriqc", "--version"], capture_output=True)
    return result.returncode == 0


def check_docker_available() -> bool:
    result = subprocess.run(["docker", "--version"], capture_output=True)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# BIDS-like layout helpers
# (MRIQC expects BIDS-compliant directory structure)
# ---------------------------------------------------------------------------

def create_bids_layout(volumes: dict, bids_dir: Path):
    """
    Create a minimal BIDS layout from a dict of {subject_id: nifti_path}.

    BIDS structure:
        bids_dir/
          dataset_description.json
          sub-XX/
            anat/
              sub-XX_T1w.nii.gz

    Args:
        volumes: {subject_id: Path_to_nifti}
        bids_dir: Where to create the BIDS tree (uses symlinks to avoid copying)
    """
    import shutil

    bids_dir.mkdir(parents=True, exist_ok=True)

    # dataset_description.json (required by MRIQC)
    desc = {
        "Name": "TopoBrain MRIQC",
        "BIDSVersion": "1.6.0",
        "DatasetType": "raw",
    }
    with open(bids_dir / "dataset_description.json", "w") as f:
        json.dump(desc, f, indent=2)

    for subject, nifti_path in volumes.items():
        nifti_path = Path(nifti_path)
        if not nifti_path.exists():
            print(f"  [WARN] Volume not found, skipping: {nifti_path}")
            continue

        anat_dir = bids_dir / subject / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)

        # Use symlink (or copy on Windows where symlinks may not work)
        dest = anat_dir / f"{subject}_T1w.nii.gz"
        if dest.exists():
            dest.unlink()
        try:
            dest.symlink_to(nifti_path.resolve())
        except (OSError, NotImplementedError):
            shutil.copy2(nifti_path, dest)

    print(f"  BIDS layout created at: {bids_dir}")


def run_mriqc_on_bids(bids_dir: Path, output_dir: Path, n_procs: int = 2,
                      use_docker: bool = False, use_singularity: bool = False,
                      singularity_image: str = "mriqc.sif"):
    """Run MRIQC on a BIDS directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    if use_singularity:
        cmd = [
            "apptainer", "run", "--cleanenv",
            "--bind", f"{bids_dir.resolve()}:/data:ro",
            "--bind", f"{output_dir.resolve()}:/out",
            "--bind", f"{work_dir.resolve()}:/work",
            "--bind", "/dev/null:/proc/1/cgroup",
            singularity_image,
            "/data", "/out", "participant",
            "--work-dir", "/work",
            "--nprocs", str(n_procs),
            "--no-sub",
        ]
    elif use_docker:
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{bids_dir.resolve()}:/data:ro",
            "-v", f"{output_dir.resolve()}:/out",
            "-v", f"{work_dir.resolve()}:/work",
            "nipreps/mriqc:latest",
            "/data", "/out", "participant",
            "--work-dir", "/work",
            "--nprocs", str(n_procs),
            "--no-sub",
        ]
    else:
        cmd = [
            "mriqc",
            str(bids_dir), str(output_dir),
            "participant",
            "--work-dir", str(work_dir),
            "--nprocs", str(n_procs),
            "--no-sub",
        ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"ERROR: MRIQC failed with exit code {result.returncode}")
        return False
    return True


# ---------------------------------------------------------------------------
# IQM comparison
# ---------------------------------------------------------------------------

def load_iqms(mriqc_output_dir: Path) -> dict:
    """
    Load IQMs from MRIQC output JSON files.

    MRIQC writes one JSON per subject under:
      <output_dir>/sub-XX_T1w.json
    """
    iqms = {}
    for json_file in sorted(mriqc_output_dir.glob("sub-*_T1w.json")):
        subject = json_file.stem.replace("_T1w", "")
        with open(json_file) as f:
            data = json.load(f)
        iqms[subject] = data
    return iqms


def compare_iqms(mriqc_dir: Path):
    """Compare IQMs across 3T, synthetic 7T, and real 7T."""
    modes = ["3t", "synthetic7t", "real7t"]
    all_iqms = {}
    for mode in modes:
        mode_dir = mriqc_dir / mode
        if not mode_dir.exists():
            continue
        iqms = load_iqms(mode_dir)
        if iqms:
            all_iqms[mode] = iqms

    if not all_iqms:
        print("No MRIQC results found. Run MRIQC on 3T, synthetic 7T, and real 7T first.")
        return

    # Key IQMs to compare
    key_metrics = ["cjv", "cnr", "efc", "fber", "snr_wm", "qi_1"]
    metric_descriptions = {
        "cjv": "CJV (lower=better, WM/GM separation)",
        "cnr": "CNR (higher=better, GM-WM contrast)",
        "efc": "EFC (lower=better, ghosting/motion)",
        "fber": "FBER (higher=better, SNR proxy)",
        "snr_wm": "SNR-WM (higher=better, white matter SNR)",
        "qi_1": "QI1 (lower=better, motion artifacts)",
    }

    print("\n" + "=" * 70)
    print("MRIQC COMPARISON: 3T vs Synthetic 7T vs Real 7T")
    print("=" * 70)

    # Get all subjects across modes
    all_subjects = set()
    for iqms in all_iqms.values():
        all_subjects.update(iqms.keys())

    for metric in key_metrics:
        print(f"\n{metric_descriptions.get(metric, metric)}:")
        print(f"  {'Subject':<12}" + "".join(f"{m:>16}" for m in modes if m in all_iqms))
        for subject in sorted(all_subjects):
            row = f"  {subject:<12}"
            for mode in modes:
                if mode not in all_iqms:
                    continue
                val = all_iqms[mode].get(subject, {}).get(metric, None)
                row += f"{str(round(val, 4)) if val is not None else 'N/A':>16}"
            print(row)

    # Save comparison CSV
    csv_path = mriqc_dir / "iqm_comparison.csv"
    subjects = sorted(all_subjects)
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["subject"] + [f"{mode}_{m}" for mode in modes if mode in all_iqms for m in key_metrics]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for subject in subjects:
            row = {"subject": subject}
            for mode in modes:
                if mode not in all_iqms:
                    continue
                for m in key_metrics:
                    row[f"{mode}_{m}"] = all_iqms[mode].get(subject, {}).get(m, "")
            writer.writerow(row)
    print(f"\nSaved comparison CSV: {csv_path}")


# ---------------------------------------------------------------------------
# Resolve paths from pairs CSV
# ---------------------------------------------------------------------------

def resolve_pairs(pairs_csv: Path, subjects: list, data_root: Path) -> dict:
    """Return {subject: {'input_3t': Path, 'target_7t': Path}}."""
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
    missing = [s for s in subjects if s not in result]
    if missing:
        print(f"WARNING: Subjects not found in CSV: {missing}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run MRIQC on 3T, synthetic 7T, and real 7T MRI volumes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  3t          Run on 3T input volumes (from pairs_new.csv + data-root)
  synthetic7t Run on synthetic 7T output (from evaluate_full_volume.py results)
  real7t      Run on real 7T target volumes (from pairs_new.csv + data-root)
  compare     Compare IQMs across the three conditions (after running above)

Installation:
  pip install mriqc
  # or Docker: docker pull nipreps/mriqc:latest

Examples:
  python scripts/run_mriqc.py --mode 3t --subjects sub-09 sub-10 \\
      --data-root /path/to/data --pairs-csv pairs_new.csv
  python scripts/run_mriqc.py --mode compare --mriqc-dir results/mriqc/
        """
    )
    parser.add_argument("--mode", choices=["3t", "synthetic7t", "real7t", "compare"],
                        required=True, help="Which mode to run")
    parser.add_argument("--subjects", nargs="+", default=["sub-09", "sub-10"],
                        help="Subject IDs (default: sub-09 sub-10)")
    parser.add_argument("--data-root", type=str,
                        help="Base directory containing raw MRI data")
    parser.add_argument("--pairs-csv", type=str, default="pairs_new.csv",
                        help="Path to pairs CSV (default: pairs_new.csv)")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory with evaluate_full_volume.py outputs (for synthetic7t mode)")
    parser.add_argument("--output-dir", type=str,
                        help="Where to save MRIQC outputs")
    parser.add_argument("--mriqc-dir", type=str, default="results/mriqc",
                        help="Root MRIQC output directory (for compare mode)")
    parser.add_argument("--nprocs", type=int, default=2,
                        help="Number of parallel processes for MRIQC (default: 2)")
    parser.add_argument("--docker", action="store_true",
                        help="Use Docker instead of local mriqc installation")
    parser.add_argument("--singularity", action="store_true",
                        help="Use Singularity/Apptainer instead of local mriqc installation")
    parser.add_argument("--singularity-image", type=str, default="mriqc.sif",
                        help="Path to Singularity/Apptainer .sif image")
    args = parser.parse_args()

    if args.mode == "compare":
        compare_iqms(Path(args.mriqc_dir))
        return

    # Check MRIQC availability
    if args.singularity:
        if not Path(args.singularity_image).exists():
            print(f"ERROR: Singularity image not found: {args.singularity_image}")
            sys.exit(1)
    elif args.docker:
        if not check_docker_available():
            print("ERROR: Docker not found. Install Docker or use local mriqc.")
            sys.exit(1)
    else:
        if not check_mriqc_available():
            print("MRIQC not found. Install with:")
            print("  pip install mriqc")
            print("Or use Apptainer (CERN):")
            print("  apptainer pull mriqc.sif docker://nipreps/mriqc:latest")
            print("  Then re-run with --singularity --singularity-image mriqc.sif")
            sys.exit(1)

    # Resolve output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        mriqc_base = Path(args.mriqc_dir)
        output_dir = mriqc_base / args.mode

    with tempfile.TemporaryDirectory(prefix="topobrain_mriqc_") as tmpdir:
        bids_dir = Path(tmpdir) / "bids"

        if args.mode in ("3t", "real7t"):
            if not args.data_root:
                print("ERROR: --data-root required for modes '3t' and 'real7t'")
                sys.exit(1)
            pairs = resolve_pairs(
                Path(args.pairs_csv),
                args.subjects,
                Path(args.data_root),
            )
            key = "input_3t" if args.mode == "3t" else "target_7t"
            volumes = {subj: data[key] for subj, data in pairs.items()}

        elif args.mode == "synthetic7t":
            results_dir = Path(args.results_dir)
            volumes = {}
            for subject in args.subjects:
                pred_path = results_dir / subject / "predicted_7T.nii.gz"
                if not pred_path.exists():
                    print(f"  [WARN] Synthetic 7T not found for {subject}: {pred_path}")
                    print(f"         Run: python scripts/evaluate_full_volume.py --subject {subject} ...")
                else:
                    volumes[subject] = pred_path

        if not volumes:
            print("ERROR: No valid volumes found to process.")
            sys.exit(1)

        print(f"\nMode: {args.mode}")
        print(f"Subjects: {list(volumes.keys())}")
        print(f"Output: {output_dir}\n")

        create_bids_layout(volumes, bids_dir)
        success = run_mriqc_on_bids(
            bids_dir, output_dir, n_procs=args.nprocs,
            use_docker=args.docker,
            use_singularity=args.singularity,
            singularity_image=args.singularity_image,
        )

        if success:
            print(f"\nMRIQC complete. Results in: {output_dir}")
            print("\nNext: run compare mode to see all three conditions side by side:")
            print(f"  python scripts/run_mriqc.py --mode compare --mriqc-dir {Path(args.mriqc_dir)}")
        else:
            print("\nMRIQC failed. Check error messages above.")
            sys.exit(1)


if __name__ == "__main__":
    main()
