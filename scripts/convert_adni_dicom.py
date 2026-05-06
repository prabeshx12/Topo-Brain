"""
Batch DICOM -> NIfTI conversion for the ADNI baseline 3T MPRAGE cohort.

Expected input layout (as produced by the ADNI 1-Click Download UI):
    <root>/<COHORT>/ADNI/<SUBJECT_ID>/MPRAGE/<DATE>/<IMAGE_ID>/*.dcm

Output layout:
    <out_root>/<COHORT>/<SUBJECT_ID>_T1w.nii.gz
    <out_root>/conversion_manifest.csv

Cohorts auto-discovered from immediate children of <root>: any directory whose
name starts with the cohort token (AD/CN/MCI) is included. Cosmetic suffixes
like " (1)" are tolerated.

Notes:
- If a subject has multiple acquisitions (rare for baseline), the EARLIEST
  date is used. That matches the "baseline" intent of the ADNI download.
- Existing output files are skipped unless --overwrite is set.
- Failures are recorded in the manifest with status="failed" and the captured
  stderr; the script continues to the next subject.

Smoke-test usage (5 AD + 5 CN, per HANDOFF.md):
    python scripts/convert_adni_dicom.py \\
        --root adni_dataset \\
        --out-root adni_nifti \\
        --limit-per-cohort 5 \\
        --cohorts AD CN
"""
import argparse
import csv
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


COHORT_TOKENS = ("AD", "CN", "MCI")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _resolve_dcm2niix(explicit: Optional[str]) -> str:
    if explicit:
        if not Path(explicit).exists():
            raise FileNotFoundError(f"--dcm2niix points at non-existent path: {explicit}")
        return explicit
    found = shutil.which("dcm2niix")
    if found is None:
        raise FileNotFoundError(
            "dcm2niix not found on PATH. Install via `winget install rordenlab.dcm2niix` "
            "or pass --dcm2niix /full/path/to/dcm2niix.exe"
        )
    return found


def _discover_cohorts(root: Path, requested: Optional[Iterable[str]]) -> List[tuple]:
    """Return [(cohort_label, cohort_dir), ...] sorted by label."""
    requested_set = {c.upper() for c in requested} if requested else None
    out = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        token = child.name.split("_", 1)[0].upper()
        if token not in COHORT_TOKENS:
            continue
        if requested_set is not None and token not in requested_set:
            continue
        # Subject dirs live one level deeper, under "ADNI/"
        adni_dir = child / "ADNI"
        if not adni_dir.is_dir():
            logging.warning("Skipping %s: no ADNI/ subdir (still zipped?)", child.name)
            continue
        out.append((token, adni_dir))
    return out


def _find_dicom_dir(subject_dir: Path) -> Optional[Path]:
    """
    Each subject dir layout:  <SUBJECT>/MPRAGE/<DATE>/<IMAGE_ID>/*.dcm
    Pick the earliest date (lexicographic on YYYY-MM-DD prefix is correct),
    then the lexicographically-smallest IMAGE_ID inside it.
    """
    mprage = subject_dir / "MPRAGE"
    if not mprage.is_dir():
        # Some downloads use other series names; fall back to any subdir with .dcm files.
        candidates = [p for p in subject_dir.rglob("*.dcm")]
        if not candidates:
            return None
        return candidates[0].parent

    date_dirs = sorted(d for d in mprage.iterdir() if d.is_dir())
    if not date_dirs:
        return None
    chosen_date = date_dirs[0]
    image_dirs = sorted(d for d in chosen_date.iterdir() if d.is_dir())
    if not image_dirs:
        return None
    return image_dirs[0]


def _convert_one(
    dcm2niix: str,
    subject_id: str,
    dicom_dir: Path,
    out_dir: Path,
    overwrite: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    nifti_path = out_dir / f"{subject_id}_T1w.nii.gz"

    if nifti_path.exists() and not overwrite:
        return {
            "subject_id": subject_id,
            "dicom_dir": str(dicom_dir),
            "nifti_path": str(nifti_path),
            "status": "skipped_exists",
            "message": "",
        }

    cmd = [
        dcm2niix,
        "-o", str(out_dir),
        "-z", "y",
        "-f", f"{subject_id}_T1w",
        "-w", "1" if overwrite else "0",
        str(dicom_dir),
    ]
    logging.debug("Running: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {
            "subject_id": subject_id,
            "dicom_dir": str(dicom_dir),
            "nifti_path": "",
            "status": "failed",
            "message": "timeout after 600s",
        }

    if proc.returncode != 0 or not nifti_path.exists():
        # dcm2niix sometimes appends suffixes (e.g. _Eq_1) for split series; pick up the first.
        siblings = sorted(out_dir.glob(f"{subject_id}_T1w*.nii.gz"))
        if siblings and proc.returncode == 0:
            chosen = siblings[0]
            chosen.rename(nifti_path) if chosen != nifti_path else None
            return {
                "subject_id": subject_id,
                "dicom_dir": str(dicom_dir),
                "nifti_path": str(nifti_path),
                "status": "ok_with_suffix",
                "message": f"selected {chosen.name} from {[s.name for s in siblings]}",
            }
        return {
            "subject_id": subject_id,
            "dicom_dir": str(dicom_dir),
            "nifti_path": "",
            "status": "failed",
            "message": (proc.stderr or proc.stdout).strip()[-500:],
        }

    return {
        "subject_id": subject_id,
        "dicom_dir": str(dicom_dir),
        "nifti_path": str(nifti_path),
        "status": "ok",
        "message": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", required=True,
                        help="adni_dataset/ folder (parent of cohort folders)")
    parser.add_argument("--out-root", required=True,
                        help="Output root for NIfTIs")
    parser.add_argument("--cohorts", nargs="*", default=None,
                        help="Subset of cohorts to process (AD CN MCI); default = all found")
    parser.add_argument("--limit-per-cohort", type=int, default=0,
                        help="Stop after N subjects per cohort (0 = no limit). For smoke tests.")
    parser.add_argument("--dcm2niix", default=None,
                        help="Path to dcm2niix executable (default: from PATH)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-convert subjects whose output already exists")
    parser.add_argument("--manifest", default=None,
                        help="Manifest CSV path (default: <out_root>/conversion_manifest.csv)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    root = Path(args.root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else (out_root / "conversion_manifest.csv")

    dcm2niix = _resolve_dcm2niix(args.dcm2niix)
    logging.info("Using dcm2niix at: %s", dcm2niix)

    cohorts = _discover_cohorts(root, args.cohorts)
    if not cohorts:
        logging.error("No cohort folders found under %s", root)
        return 2

    rows = []
    for cohort, adni_dir in cohorts:
        subject_dirs = sorted(d for d in adni_dir.iterdir() if d.is_dir())
        if args.limit_per_cohort > 0:
            subject_dirs = subject_dirs[: args.limit_per_cohort]
        logging.info("Cohort %s: %d subjects", cohort, len(subject_dirs))

        cohort_out = out_root / cohort
        for subj_dir in subject_dirs:
            subject_id = subj_dir.name
            dicom_dir = _find_dicom_dir(subj_dir)
            if dicom_dir is None:
                logging.warning("[%s/%s] no DICOM dir found, skipping", cohort, subject_id)
                rows.append({
                    "cohort": cohort,
                    "subject_id": subject_id,
                    "dicom_dir": "",
                    "nifti_path": "",
                    "status": "no_dicom_dir",
                    "message": "",
                })
                continue

            result = _convert_one(dcm2niix, subject_id, dicom_dir, cohort_out, args.overwrite)
            result["cohort"] = cohort
            rows.append(result)
            logging.info("[%s/%s] %s", cohort, subject_id, result["status"])

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["cohort", "subject_id", "dicom_dir", "nifti_path", "status", "message"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    n_ok = sum(1 for r in rows if r["status"] in ("ok", "ok_with_suffix"))
    n_skip = sum(1 for r in rows if r["status"] == "skipped_exists")
    n_fail = sum(1 for r in rows if r["status"] not in ("ok", "ok_with_suffix", "skipped_exists"))
    logging.info("Manifest written: %s", manifest_path)
    logging.info("Summary: ok=%d, skipped=%d, failed=%d", n_ok, n_skip, n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
