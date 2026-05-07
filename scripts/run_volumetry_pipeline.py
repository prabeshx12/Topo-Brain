"""
End-to-end Week 7 volumetry-agreement pipeline (CERN side).

For a list of training subjects, this:
    1. Synthesizes predicted_7T.nii.gz via evaluate_full_volume.py (skip if exists)
    2. Runs FastSurfer on each synth volume via run_fastsurfer.py (skip if exists)
    3. Auto-discovers the matching real ses-2 aparc+aseg under your data root
       and the synth aparc+aseg under the FastSurfer output dir
       (build_volumetry_pairs.py logic, inlined)
    4. Runs hippocampal_volumetry.py to produce ICC / Bland-Altman / per-subject CSV

The "real" side is the aparc+aseg you already have under the data root (the
auxiliary supervision target used during training). The "synth" side is
FastSurfer's parcellation of the model's predicted_7T.nii.gz output. Note
that mixing FreeSurfer-recon-all (real) with FastSurfer (synth) introduces
some tool-vs-tool noise — see --rerun-fastsurfer-on-real for the
tool-consistent variant if you want to remove that confound.

Designed to run on lxplus with the apptainer image at
    /eos/user/p/ppokhrel/Untitled\\ Folder\\ 1/images/fastsurfer.sif
and the FreeSurfer license at
    /eos/user/p/ppokhrel/license.txt.

Usage on lxplus (default 5-fold LOOCV test subjects):

    python scripts/run_volumetry_pipeline.py \\
        --subjects sub-02 sub-04 sub-06 sub-08 sub-10 \\
        --checkpoint output/checkpoint_145000.pt \\
        --pairs-csv pairs.csv \\
        --data-root /eos/user/p/ppokhrel/Untitled\\ Folder\\ 1/preprocessed-mri-aligned-diffusion \\
        --fastsurfer-image /eos/user/p/ppokhrel/Untitled\\ Folder\\ 1/images/fastsurfer.sif \\
        --fs-license /eos/user/p/ppokhrel/license.txt \\
        --output-dir results/

Skip flags let you resume after a partial failure without redoing slow stages:
    --skip-synth        : assume predicted_7T.nii.gz already exists per subject
    --skip-fastsurfer   : assume FastSurfer outputs already exist per subject
    --skip-volumetry    : stop before the final volumetry stage
"""
import argparse
import csv
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


# Same glob priority as build_volumetry_pairs.py — keep them in sync.
REAL_GLOBS = (
    "{subject}/ses-2/anat/{subject}_ses-2*aparc+aseg*.nii.gz",
    "{subject}/ses-2/anat/{subject}_ses-2*aparc+aseg*.nii",
    "{subject}/ses-2/anat/*aparc+aseg*.nii.gz",
    "{subject}/ses-2/anat/*aparc+aseg*.nii",
    "{subject}/ses-2/**/aparc+aseg.mgz",
    "{subject}/ses-2/**/aparc.DKTatlas+aseg.mgz",
    "{subject}/**/aparc+aseg*.nii.gz",
    "{subject}/**/aparc+aseg*.nii",
    "{subject}/**/aparc+aseg.mgz",
    "{subject}/**/aparc.DKTatlas+aseg.mgz",
)
SYNTH_GLOBS = (
    "{subject}/mri/aparc.DKTatlas+aseg.mgz",
    "{subject}/mri/aparc+aseg.mgz",
    "{subject}/mri/aparc.DKTatlas+aseg.deep.mgz",
    "{subject}/mri/aparc+aseg.nii.gz",
    "{subject}/mri/aparc+aseg.nii",
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _first_match(root: Path, subject: str, patterns) -> Optional[Path]:
    for pat in patterns:
        for hit in sorted(root.glob(pat.format(subject=subject))):
            if hit.is_file():
                return hit.resolve()
    return None


def _synthesize_one(repo: Path, checkpoint: Path, subject: str, pairs_csv: Path,
                    data_root: Path, results_dir: Path, sampler: str,
                    ddim_steps: int, n_samples: int) -> Path:
    """Call evaluate_full_volume.py on one subject. Returns predicted_7T path."""
    out_subj = results_dir / subject
    pred = out_subj / "predicted_7T.nii.gz"
    if pred.exists():
        logging.info("[%s] synth already exists: %s", subject, pred)
        return pred

    out_subj.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/evaluate_full_volume.py",
        "--checkpoint", str(checkpoint),
        "--subject", subject,
        "--pairs_csv", str(pairs_csv),
        "--data-root", str(data_root),
        "--output_dir", str(out_subj),
        "--sampler", sampler,
        "--ddim-steps", str(ddim_steps),
        "--n-samples", str(n_samples),
    ]
    logging.info("[%s] synthesizing", subject)
    subprocess.check_call(cmd, cwd=str(repo))
    if not pred.exists():
        raise RuntimeError(f"[{subject}] expected {pred} after synthesis but it's missing")
    return pred


def _fastsurfer_one(repo: Path, subject: str, results_dir: Path,
                    fastsurfer_dir: Path, fs_license: Path,
                    fastsurfer_image: Path, nthreads: int,
                    seg_only: bool) -> None:
    """Call run_fastsurfer.py on one subject (synth volume already in results_dir)."""
    cmd = [
        sys.executable, "scripts/run_fastsurfer.py",
        "--mode", "synthetic7t",
        "--subjects", subject,
        "--results-dir", str(results_dir),
        "--fastsurfer-dir", str(fastsurfer_dir),
        "--fs-license", str(fs_license),
        "--singularity",
        "--singularity-image", str(fastsurfer_image),
        "--nthreads", str(nthreads),
    ]
    if seg_only:
        cmd.append("--seg-only")
    logging.info("[%s] running FastSurfer", subject)
    subprocess.check_call(cmd, cwd=str(repo))


def _build_pairs(subjects: List[str], data_root: Path,
                 fastsurfer_dir: Path, out_csv: Path) -> int:
    """Auto-discover real and synth aparc+aseg per subject, write pairs CSV."""
    rows = []
    for s in subjects:
        real = _first_match(data_root, s, REAL_GLOBS)
        synth = _first_match(fastsurfer_dir, s, SYNTH_GLOBS)
        if real is None:
            logging.warning("[%s] missing real aparc+aseg under %s", s, data_root)
            continue
        if synth is None:
            logging.warning("[%s] missing synth aparc+aseg under %s", s, fastsurfer_dir)
            continue
        rows.append({"subject": s,
                     "real_aparc_aseg": str(real),
                     "synth_aparc_aseg": str(synth)})
        logging.info("[%s] pair: real=%s | synth=%s",
                     s, real.name, synth.name)

    if not rows:
        return 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["subject", "real_aparc_aseg", "synth_aparc_aseg"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return len(rows)


def _run_volumetry(repo: Path, pairs_csv: Path, output_dir: Path,
                   regions: List[str]) -> None:
    cmd = [
        sys.executable, "scripts/hippocampal_volumetry.py",
        "--pairs-csv", str(pairs_csv),
        "--output-dir", str(output_dir),
        "--regions", *regions,
    ]
    logging.info("Running volumetry agreement: %s", " ".join(cmd[1:]))
    subprocess.check_call(cmd, cwd=str(repo))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subjects", nargs="+", required=True,
                        help="Training subject IDs (e.g. sub-02 sub-04 sub-06 sub-08 sub-10)")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Trained diffusion checkpoint (.pt)")
    parser.add_argument("--pairs-csv", type=Path, default=Path("pairs.csv"),
                        help="Training pairs CSV (default: pairs.csv)")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root of preprocessed training data (real ses-2 aparc+aseg lives here)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Where evaluate_full_volume.py writes per-subject outputs")
    parser.add_argument("--fastsurfer-dir", type=Path, default=None,
                        help="Where FastSurfer outputs go (default: <results-dir>/fastsurfer/)")
    parser.add_argument("--fastsurfer-image", type=Path, required=True,
                        help="apptainer/singularity .sif image for FastSurfer")
    parser.add_argument("--fs-license", type=Path, required=True,
                        help="FreeSurfer license.txt")
    parser.add_argument("--output-dir", type=Path, default=Path("results/volumetry"),
                        help="Where ICC/Bland-Altman outputs go")
    parser.add_argument("--regions", nargs="+", default=["hippocampus"],
                        help="Regions to evaluate (default: hippocampus)")
    parser.add_argument("--repo-dir", type=Path, default=None,
                        help="Topo-Brain repo location (default: parent of this script)")

    parser.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--fastsurfer-nthreads", type=int, default=4)
    parser.add_argument("--fastsurfer-seg-only", action="store_true",
                        help="FastSurfer seg-only mode (faster, skips surface reconstruction). "
                             "Aparc+aseg is still produced; only cortical surfaces are skipped.")

    parser.add_argument("--skip-synth", action="store_true",
                        help="Assume predicted_7T.nii.gz already exists per subject")
    parser.add_argument("--skip-fastsurfer", action="store_true",
                        help="Assume FastSurfer outputs already exist per subject")
    parser.add_argument("--skip-volumetry", action="store_true",
                        help="Stop before the volumetry-agreement stage")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    repo = (args.repo_dir.resolve() if args.repo_dir
            else Path(__file__).resolve().parent.parent)
    fastsurfer_dir = (args.fastsurfer_dir.resolve()
                      if args.fastsurfer_dir
                      else (args.results_dir.resolve() / "fastsurfer"))

    # ---- preflight ----
    for label, path in [("checkpoint", args.checkpoint),
                        ("pairs-csv", args.pairs_csv),
                        ("data-root", args.data_root),
                        ("fastsurfer-image", args.fastsurfer_image),
                        ("fs-license", args.fs_license)]:
        if not Path(path).exists():
            logging.error("--%s not found: %s", label, path)
            return 2

    args.results_dir.mkdir(parents=True, exist_ok=True)
    fastsurfer_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) Synthesize ----
    if not args.skip_synth:
        for s in args.subjects:
            try:
                _synthesize_one(repo=repo,
                                checkpoint=args.checkpoint.resolve(),
                                subject=s,
                                pairs_csv=args.pairs_csv.resolve(),
                                data_root=args.data_root.resolve(),
                                results_dir=args.results_dir.resolve(),
                                sampler=args.sampler,
                                ddim_steps=args.ddim_steps,
                                n_samples=args.n_samples)
            except subprocess.CalledProcessError as e:
                logging.error("[%s] synthesis failed (exit %d) — skipping subject", s, e.returncode)
                continue

    # ---- 2) FastSurfer ----
    if not args.skip_fastsurfer:
        for s in args.subjects:
            existing = _first_match(fastsurfer_dir, s, SYNTH_GLOBS)
            if existing is not None:
                logging.info("[%s] FastSurfer output exists at %s — skipping", s, existing)
                continue
            try:
                _fastsurfer_one(repo=repo, subject=s,
                                results_dir=args.results_dir.resolve(),
                                fastsurfer_dir=fastsurfer_dir,
                                fs_license=args.fs_license.resolve(),
                                fastsurfer_image=args.fastsurfer_image.resolve(),
                                nthreads=args.fastsurfer_nthreads,
                                seg_only=args.fastsurfer_seg_only)
            except subprocess.CalledProcessError as e:
                logging.error("[%s] FastSurfer failed (exit %d) — skipping subject", s, e.returncode)
                continue

    if args.skip_volumetry:
        logging.info("Stopping before volumetry stage (--skip-volumetry).")
        return 0

    # ---- 3) Build pairs CSV ----
    pairs_out = args.output_dir / "pairs_volumetry.csv"
    n = _build_pairs(subjects=args.subjects,
                     data_root=args.data_root.resolve(),
                     fastsurfer_dir=fastsurfer_dir,
                     out_csv=pairs_out)
    if n == 0:
        logging.error("No subjects have BOTH real and synth aparc+aseg — nothing to volumetry.")
        return 1
    logging.info("Built pairs CSV with %d / %d subjects", n, len(args.subjects))

    # ---- 4) Volumetry agreement ----
    _run_volumetry(repo=repo, pairs_csv=pairs_out,
                   output_dir=args.output_dir.resolve(),
                   regions=args.regions)

    logging.info("=" * 60)
    logging.info("Done. Outputs in %s/", args.output_dir)
    logging.info("  per_subject_volumes.csv")
    logging.info("  agreement.json")
    logging.info("  bland_altman_<region>_<L|R>.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
