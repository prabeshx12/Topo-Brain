"""
Build the pairs CSV that hippocampal_volumetry.py consumes, by auto-discovering
the matching `real` ses-2 aparc+aseg and `synth` FastSurfer-on-synth aparc+aseg
files for a list of subjects.

You produce the inputs separately:
    1. Real ses-2 aparc+aseg already exists under your data root (it was used
       as the auxiliary supervision target during training).
    2. Synth aparc+aseg comes from running scripts/run_fastsurfer.py
       --mode synthetic7t — FastSurfer parcellates the model's predicted_7T.nii.gz.

This script does the path glob and emits pairs.csv with the three columns
hippocampal_volumetry.py expects: subject, real_aparc_aseg, synth_aparc_aseg.

CERN workflow (training subjects):

    # 1. Synthesize predicted_7T.nii.gz per subject
    for s in sub-06 sub-07 ...; do
      python scripts/evaluate_full_volume.py \\
          --checkpoint output/checkpoint_145000.pt \\
          --subject $s --pairs_csv pairs.csv \\
          --data-root /eos/.../preprocessed-mri-aligned-diffusion \\
          --output_dir results/
    done

    # 2. FastSurfer on each predicted_7T.nii.gz -> aparc+aseg
    python scripts/run_fastsurfer.py --mode synthetic7t \\
        --subjects sub-06 sub-07 ... \\
        --results-dir results/ \\
        --fastsurfer-dir results/fastsurfer/ \\
        --fs-license /eos/.../license.txt --singularity \\
        --singularity-image /eos/.../images/fastsurfer.sif

    # 3. Build volumetry pairs CSV (this script)
    python scripts/build_volumetry_pairs.py \\
        --subjects sub-06 sub-07 ... \\
        --data-root /eos/.../preprocessed-mri-aligned-diffusion \\
        --fastsurfer-dir results/fastsurfer/ \\
        --output pairs_volumetry.csv

    # 4. Volumetry agreement
    python scripts/hippocampal_volumetry.py \\
        --pairs-csv pairs_volumetry.csv \\
        --output-dir results/volumetry/
"""
import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple


# Glob patterns (in priority order) for the real ses-2 aparc+aseg under data-root.
# preprocess_masks.py uses the same rule: prefer aparc+aseg over plain aseg,
# prefer .nii.gz over .nii.
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

# FastSurfer's standard output location for the parcellation.
SYNTH_GLOBS = (
    "{subject}/mri/aparc.DKTatlas+aseg.mgz",
    "{subject}/mri/aparc+aseg.mgz",
    "{subject}/mri/aparc.DKTatlas+aseg.deep.mgz",
    "{subject}/mri/aparc+aseg.nii.gz",
    "{subject}/mri/aparc+aseg.nii",
    "{subject}/aparc+aseg.nii.gz",
    "{subject}/aparc+aseg.nii",
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _first_match(root: Path, subject: str, patterns) -> Optional[Path]:
    """Return the first existing match across the patterns (in order)."""
    for pat in patterns:
        for hit in sorted(root.glob(pat.format(subject=subject))):
            if hit.is_file():
                return hit.resolve()
    return None


def discover_pair(subject: str, data_root: Path,
                  fastsurfer_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    real = _first_match(data_root, subject, REAL_GLOBS)
    synth = _first_match(fastsurfer_dir, subject, SYNTH_GLOBS)
    return real, synth


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subjects", nargs="+", required=True,
                        help="Subject IDs (e.g. sub-06 sub-07)")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Real-data root (where ses-2 aparc+aseg lives)")
    parser.add_argument("--fastsurfer-dir", type=Path, required=True,
                        help="FastSurfer output root (--fastsurfer-dir from run_fastsurfer.py)")
    parser.add_argument("--output", type=Path, default=Path("pairs_volumetry.csv"),
                        help="Output pairs CSV path")
    parser.add_argument("--strict", action="store_true",
                        help="Fail (exit 1) if any subject has missing real or synth aseg")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    data_root = args.data_root.resolve()
    fastsurfer_dir = args.fastsurfer_dir.resolve()
    if not data_root.exists():
        logging.error("data-root not found: %s", data_root)
        return 2
    if not fastsurfer_dir.exists():
        logging.error("fastsurfer-dir not found: %s", fastsurfer_dir)
        return 2

    rows: List[dict] = []
    missing: List[str] = []
    for subject in args.subjects:
        real, synth = discover_pair(subject, data_root, fastsurfer_dir)
        status = []
        if real is None:
            status.append("real?")
        if synth is None:
            status.append("synth?")
        if status:
            logging.warning("[%s] missing: %s", subject, ", ".join(status))
            missing.append(subject)
            continue
        logging.info("[%s] real=%s  synth=%s",
                     subject, real.relative_to(data_root) if data_root in real.parents else real,
                     synth.relative_to(fastsurfer_dir) if fastsurfer_dir in synth.parents else synth)
        rows.append({
            "subject": subject,
            "real_aparc_aseg": str(real),
            "synth_aparc_aseg": str(synth),
        })

    if not rows:
        logging.error("No pairs discovered — nothing to write.")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "real_aparc_aseg", "synth_aparc_aseg"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logging.info("Wrote %s (%d pairs, %d skipped)", args.output, len(rows), len(missing))

    if missing and args.strict:
        logging.error("--strict: %d subjects had missing inputs", len(missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
