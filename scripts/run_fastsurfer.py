"""
Run FastSurfer for hippocampal/MTL segmentation and morphometry.

FastSurfer is a deep-learning based replacement for FreeSurfer's recon-all
that runs in ~1 hour per subject vs ~8 hours. It produces:
  - Hippocampal volume (left + right)
  - Entorhinal cortex thickness/volume
  - Parahippocampal gyrus volume
  - Full cortical parcellation + subcortical segmentation

We run FastSurfer on:
  1. Synthetic 7T (model output)
  2. Real 7T (ground truth)
Then compare hippocampal volumes and MTL metrics between them.

INSTALLATION:
  Option A - Singularity/Docker (recommended on HPC):
    singularity pull fastsurfer.sif docker://deepmi/fastsurfer:latest
    # OR
    docker pull deepmi/fastsurfer:latest

  Option B - Conda:
    conda create -n fastsurfer python=3.8
    conda activate fastsurfer
    git clone https://github.com/Deep-MI/FastSurfer.git
    cd FastSurfer && pip install -r requirements.txt

  NOTE: FastSurfer requires FreeSurfer license (free from freesurfer.net)

USAGE:
  # Run on synthetic 7T output
  python scripts/run_fastsurfer.py --mode synthetic7t \\
      --subjects sub-09 sub-10 \\
      --results-dir results/ \\
      --fastsurfer-dir results/fastsurfer/ \\
      --fs-license /path/to/freesurfer/license.txt

  # Run on real 7T
  python scripts/run_fastsurfer.py --mode real7t \\
      --subjects sub-09 sub-10 \\
      --data-root /path/to/data \\
      --pairs-csv pairs_new.csv \\
      --fastsurfer-dir results/fastsurfer/

  # Compare hippocampal volumes between synthetic and real 7T
  python scripts/run_fastsurfer.py --mode compare \\
      --fastsurfer-dir results/fastsurfer/
"""
import sys
import csv
import json
import argparse
import subprocess
import shutil
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# FastSurfer runner
# ---------------------------------------------------------------------------

def run_fastsurfer_on_volume(
    subject_id: str,
    t1_path: Path,
    output_dir: Path,
    fs_license: str,
    n_threads: int = 4,
    use_docker: bool = False,
    use_singularity: bool = False,
    singularity_image: str = "fastsurfer.sif",
    seg_only: bool = False,
) -> bool:
    """
    Run FastSurfer on a single T1w volume.

    Args:
        subject_id: Subject identifier (e.g. 'sub-09_synthetic7t')
        t1_path: Path to T1w NIfTI file
        output_dir: FastSurfer output root directory
        fs_license: Path to FreeSurfer license.txt
        n_threads: Number of OMP threads
        use_docker: Use Docker container
        use_singularity: Use Singularity container
        singularity_image: Path to .sif file
        seg_only: Only run segmentation (faster, no surface reconstruction)

    Returns:
        True on success
    """
    t1_path = t1_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    subj_out = output_dir / subject_id
    if subj_out.exists() and (subj_out / "mri" / "aparc.DKTatlas+aseg.deep.mgz").exists():
        print(f"  [SKIP] FastSurfer already completed for {subject_id}")
        return True

    if use_docker:
        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{t1_path.parent}:/input:ro",
            "-v", f"{output_dir}:/output",
            "-v", f"{Path(fs_license).parent}:/fs_license:ro",
            "deepmi/fastsurfer:latest",
            "--t1", f"/input/{t1_path.name}",
            "--sid", subject_id,
            "--sd", "/output",
            "--fs_license", f"/fs_license/{Path(fs_license).name}",
            "--parallel", "--threads", str(n_threads),
        ]
        if seg_only:
            cmd += ["--seg_only"]

    elif use_singularity:
        cmd = [
            "singularity", "exec", "--nv",
            "--bind", f"{t1_path.parent}:/input",
            "--bind", f"{output_dir}:/output",
            "--bind", f"{Path(fs_license).parent}:/fs_license",
            singularity_image,
            "/fastsurfer/run_fastsurfer.sh",
            "--t1", f"/input/{t1_path.name}",
            "--sid", subject_id,
            "--sd", "/output",
            "--fs_license", f"/fs_license/{Path(fs_license).name}",
            "--parallel", "--threads", str(n_threads),
        ]
        if seg_only:
            cmd += ["--seg_only"]

    else:
        # Local installation
        fastsurfer_home = shutil.which("run_fastsurfer.sh")
        if not fastsurfer_home:
            # Try common paths
            for candidate in ["run_fastsurfer.sh", "FastSurfer/run_fastsurfer.sh", "~/FastSurfer/run_fastsurfer.sh"]:
                if Path(candidate).exists():
                    fastsurfer_home = candidate
                    break
        if not fastsurfer_home:
            print("ERROR: run_fastsurfer.sh not found. Install FastSurfer or use --docker / --singularity")
            return False

        cmd = [
            "bash", fastsurfer_home,
            "--t1", str(t1_path),
            "--sid", subject_id,
            "--sd", str(output_dir),
            "--fs_license", str(fs_license),
            "--parallel", "--threads", str(n_threads),
        ]
        if seg_only:
            cmd += ["--seg_only"]

    print(f"\nRunning FastSurfer on {subject_id}:")
    print(f"  Input: {t1_path}")
    print(f"  Output: {output_dir / subject_id}")
    print(f"  Command: {' '.join(cmd[:6])} ...")

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"ERROR: FastSurfer failed for {subject_id}")
        return False
    return True


# ---------------------------------------------------------------------------
# Volume extraction from stats files
# ---------------------------------------------------------------------------

HIPPOCAMPUS_LABELS = {
    "Left-Hippocampus": 17,
    "Right-Hippocampus": 53,
    "Left-Entorhinal": None,   # From aparc stats
    "Right-Entorhinal": None,
}

MTL_STRUCTURES = [
    "Left-Hippocampus",
    "Right-Hippocampus",
    "Left-Amygdala",
    "Right-Amygdala",
]

MTL_CORTEX = [
    "entorhinal",
    "parahippocampal",
    "temporalpole",
    "fusiform",
    "inferiortemporal",
    "middletemporal",
]


def parse_aseg_stats(stats_file: Path) -> dict:
    """Parse FreeSurfer/FastSurfer aseg.stats file, return {structure: volume_mm3}."""
    volumes = {}
    if not stats_file.exists():
        return volumes

    with open(stats_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                # Format: Index SegId NVoxels Volume_mm3 StructName ...
                try:
                    struct_name = parts[4]
                    volume_mm3 = float(parts[3])
                    volumes[struct_name] = volume_mm3
                except (ValueError, IndexError):
                    continue
    return volumes


def parse_aparc_stats(stats_file: Path) -> dict:
    """Parse aparc.stats for cortical thickness and surface area."""
    regions = {}
    if not stats_file.exists():
        return regions

    with open(stats_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                # Format: StructName NumVert SurfArea GrayVol ThickAvg ThickStd ...
                try:
                    struct = parts[0]
                    regions[struct] = {
                        "surface_area_mm2": float(parts[2]),
                        "gray_vol_mm3": float(parts[3]),
                        "thickness_avg_mm": float(parts[4]),
                    }
                except (ValueError, IndexError):
                    continue
    return regions


def extract_subject_morphometry(fastsurfer_dir: Path, subject_id: str) -> dict:
    """Extract hippocampal volumes and MTL cortical metrics from FastSurfer output."""
    subj_dir = fastsurfer_dir / subject_id
    stats_dir = subj_dir / "stats"

    morphometry = {"subject": subject_id}

    # --- Subcortical volumes (aseg.stats) ---
    aseg_file = stats_dir / "aseg.stats"
    if not aseg_file.exists():
        # FastSurfer seg-only produces aseg.stats in mri/
        aseg_file = subj_dir / "mri" / "aseg.stats"

    aseg = parse_aseg_stats(aseg_file)

    for struct in MTL_STRUCTURES:
        morphometry[struct.replace("-", "_").lower() + "_mm3"] = aseg.get(struct, float("nan"))

    # Derived: total hippocampal volume
    lh = aseg.get("Left-Hippocampus", 0)
    rh = aseg.get("Right-Hippocampus", 0)
    morphometry["hippocampus_total_mm3"] = lh + rh if (lh or rh) else float("nan")

    # --- Cortical morphometry (aparc stats) ---
    for hemi in ["lh", "rh"]:
        aparc_file = stats_dir / f"{hemi}.aparc.DKTatlas.stats"
        if not aparc_file.exists():
            aparc_file = stats_dir / f"{hemi}.aparc.stats"
        aparc = parse_aparc_stats(aparc_file)

        for region in MTL_CORTEX:
            if region in aparc:
                prefix = f"{hemi}_{region}"
                morphometry[f"{prefix}_vol_mm3"] = aparc[region]["gray_vol_mm3"]
                morphometry[f"{prefix}_thick_mm"] = aparc[region]["thickness_avg_mm"]

    return morphometry


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_morphometry(fastsurfer_dir: Path, subjects: list):
    """Compare hippocampal/MTL volumes between synthetic and real 7T."""
    results = {"synthetic7t": {}, "real7t": {}}

    for mode in results:
        for subject in subjects:
            subject_id = f"{subject}_{mode}"
            morph = extract_subject_morphometry(fastsurfer_dir, subject_id)
            results[mode][subject] = morph

    print("\n" + "=" * 80)
    print("HIPPOCAMPAL / MTL MORPHOMETRY COMPARISON")
    print("Synthetic 7T vs Real 7T")
    print("=" * 80)

    key_metrics = [
        "hippocampus_total_mm3",
        "left_hippocampus_mm3",
        "right_hippocampus_mm3",
        "left_amygdala_mm3",
        "right_amygdala_mm3",
    ]

    header = f"{'Metric':<35}" + f"{'Synthetic 7T':>16}" + f"{'Real 7T':>16}" + f"{'Diff %':>10}"
    print(header)
    print("-" * len(header))

    all_morph = {"synthetic7t": [], "real7t": []}
    for subject in subjects:
        for mode in ["synthetic7t", "real7t"]:
            all_morph[mode].append(results[mode].get(subject, {}))

    for metric in key_metrics:
        synth_vals = [m.get(metric, float("nan")) for m in all_morph["synthetic7t"]]
        real_vals = [m.get(metric, float("nan")) for m in all_morph["real7t"]]

        synth_mean = np.nanmean(synth_vals) if synth_vals else float("nan")
        real_mean = np.nanmean(real_vals) if real_vals else float("nan")

        if not np.isnan(synth_mean) and not np.isnan(real_mean) and real_mean != 0:
            diff_pct = 100 * (synth_mean - real_mean) / real_mean
            diff_str = f"{diff_pct:>+10.1f}%"
        else:
            diff_str = f"{'N/A':>10}"

        label = metric.replace("_", " ").title()
        synth_str = f"{synth_mean:.1f}" if not np.isnan(synth_mean) else "N/A"
        real_str = f"{real_mean:.1f}" if not np.isnan(real_mean) else "N/A"
        print(f"{label:<35}{synth_str:>16}{real_str:>16}{diff_str}")

    print("\nInterpretation:")
    print("  < 5%  difference: Excellent - synthetic 7T preserves hippocampal volumes")
    print("  < 10% difference: Acceptable")
    print("  > 10% difference: Poor - check topo loss weight (lambda_topo)")

    # Save JSON
    output_file = fastsurfer_dir / "morphometry_comparison.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {output_file}")

    # Save CSV
    csv_file = fastsurfer_dir / "morphometry_comparison.csv"
    all_rows = []
    for mode in ["synthetic7t", "real7t"]:
        for subject in subjects:
            row = {"mode": mode, **results[mode].get(subject, {"subject": subject})}
            all_rows.append(row)
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved: {csv_file}")


# ---------------------------------------------------------------------------
# Resolve paths from CSV
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run FastSurfer for hippocampal/MTL segmentation and morphometry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Installation:
  Docker (recommended):
    docker pull deepmi/fastsurfer:latest
    # Then use --docker flag

  Singularity (HPC):
    singularity pull fastsurfer.sif docker://deepmi/fastsurfer:latest
    # Then use --singularity --singularity-image fastsurfer.sif

  FreeSurfer license (required, free):
    Register at: https://surfer.nmr.mgh.harvard.edu/registration.html

Examples:
  python scripts/run_fastsurfer.py --mode synthetic7t \\
      --subjects sub-09 sub-10 --results-dir results/ \\
      --fastsurfer-dir results/fastsurfer/ --fs-license ~/license.txt --docker

  python scripts/run_fastsurfer.py --mode compare \\
      --subjects sub-09 sub-10 --fastsurfer-dir results/fastsurfer/
        """
    )
    parser.add_argument("--mode", choices=["synthetic7t", "real7t", "compare"], required=True)
    parser.add_argument("--subjects", nargs="+", default=["sub-09", "sub-10"])
    parser.add_argument("--data-root", type=str, help="Data root for real 7T mode")
    parser.add_argument("--pairs-csv", type=str, default="pairs_new.csv")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory with evaluate_full_volume.py outputs (synthetic7t mode)")
    parser.add_argument("--fastsurfer-dir", type=str, default="results/fastsurfer",
                        help="Root output directory for FastSurfer")
    parser.add_argument("--fs-license", type=str, help="Path to FreeSurfer license.txt")
    parser.add_argument("--docker", action="store_true", help="Use Docker")
    parser.add_argument("--singularity", action="store_true", help="Use Singularity")
    parser.add_argument("--singularity-image", type=str, default="fastsurfer.sif")
    parser.add_argument("--nthreads", type=int, default=4)
    parser.add_argument("--seg-only", action="store_true",
                        help="Only run segmentation (faster, skips surface reconstruction)")
    args = parser.parse_args()

    fastsurfer_dir = Path(args.fastsurfer_dir)

    if args.mode == "compare":
        compare_morphometry(fastsurfer_dir, args.subjects)
        return

    if not args.fs_license:
        print("ERROR: --fs-license required (path to FreeSurfer license.txt)")
        print("Get a free license at: https://surfer.nmr.mgh.harvard.edu/registration.html")
        sys.exit(1)

    # Resolve volumes
    if args.mode == "synthetic7t":
        results_dir = Path(args.results_dir)
        volumes = {}
        for subject in args.subjects:
            pred_path = results_dir / subject / "predicted_7T.nii.gz"
            if not pred_path.exists():
                print(f"[WARN] Synthetic 7T not found: {pred_path}")
                print(f"       Run evaluate_full_volume.py for {subject} first.")
            else:
                volumes[subject] = pred_path

    elif args.mode == "real7t":
        if not args.data_root:
            print("ERROR: --data-root required for real7t mode")
            sys.exit(1)
        pairs = resolve_pairs(Path(args.pairs_csv), args.subjects, Path(args.data_root))
        volumes = {subj: data["target_7t"] for subj, data in pairs.items()}

    if not volumes:
        print("ERROR: No volumes found to process.")
        sys.exit(1)

    # Run FastSurfer
    success_count = 0
    for subject, t1_path in volumes.items():
        subject_id = f"{subject}_{args.mode}"
        ok = run_fastsurfer_on_volume(
            subject_id=subject_id,
            t1_path=Path(t1_path),
            output_dir=fastsurfer_dir,
            fs_license=args.fs_license,
            n_threads=args.nthreads,
            use_docker=args.docker,
            use_singularity=args.singularity,
            singularity_image=args.singularity_image,
            seg_only=args.seg_only,
        )
        if ok:
            success_count += 1

    print(f"\nCompleted {success_count}/{len(volumes)} subjects.")
    if success_count > 0:
        print("\nNext: compare synthetic vs real 7T morphometry:")
        print(f"  python scripts/run_fastsurfer.py --mode compare --subjects {' '.join(args.subjects)} \\")
        print(f"      --fastsurfer-dir {fastsurfer_dir}")


if __name__ == "__main__":
    main()
