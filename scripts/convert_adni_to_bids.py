"""
Convert ADNI-like NIfTI folder structure to minimal BIDS layout.

Expected ADNI source structure (example):
  ADNI/
    002_S_1018/
      MPR__GradWarp__B1_Correction__N3__Scaled/
        2006-11-29_10_00_05.0/
          I40817/
            ADNI_002_S_1018_MR_... .nii

Output BIDS layout:
  <output_root>/
    dataset_description.json
    participants.tsv
    sub-002S1018/
      ses-20061129/
        anat/
          sub-002S1018_ses-20061129_T1w.nii
          sub-002S1018_ses-20061129_T1w.json

Usage:
  python scripts/convert_adni_to_bids.py \
    --input-root MRI-AD-Part-2/ADNI \
    --output-root MRI-AD-Part-2/ADNI_BIDS
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

_SUBJECT_RE = re.compile(r"^(\d{3})_S_(\d{4})$")
_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def to_bids_subject(adni_subject: str) -> str:
    m = _SUBJECT_RE.match(adni_subject)
    if not m:
        raise ValueError(f"Unexpected ADNI subject format: {adni_subject}")
    return f"sub-{m.group(1)}S{m.group(2)}"


def date_to_session(date_folder: str) -> str:
    m = _DATE_RE.search(date_folder)
    if m:
        return f"ses-{m.group(1)}{m.group(2)}{m.group(3)}"
    # fallback if date parsing fails
    clean = re.sub(r"[^A-Za-z0-9]", "", date_folder)[:12]
    return f"ses-{clean}" if clean else "ses-unknown"


def find_nii_files(input_root: Path) -> List[Path]:
    return sorted(list(input_root.rglob("*.nii")) + list(input_root.rglob("*.nii.gz")))


def convert(input_root: Path, output_root: Path, overwrite: bool = False) -> Tuple[int, int, Path]:
    output_root.mkdir(parents=True, exist_ok=True)

    participants = set()
    report_rows: List[Dict[str, str]] = []
    converted = 0
    skipped = 0

    nii_files = find_nii_files(input_root)
    for src in nii_files:
        try:
            rel = src.relative_to(input_root)
        except ValueError:
            skipped += 1
            continue

        # Need at least: <subject>/<sequence>/<date>/<instance>/<file>
        if len(rel.parts) < 5:
            skipped += 1
            continue

        adni_subject = rel.parts[0]
        date_folder = rel.parts[2]

        # Keep only MRI-like files that contain ADNI_<id>_MR in filename
        name_upper = src.name.upper()
        if "ADNI_" not in name_upper or "_MR_" not in name_upper:
            skipped += 1
            continue

        try:
            sub = to_bids_subject(adni_subject)
        except ValueError:
            skipped += 1
            continue

        ses = date_to_session(date_folder)
        participants.add(sub)

        anat_dir = output_root / sub / ses / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)

        ext = ".nii.gz" if src.name.endswith(".nii.gz") else ".nii"
        dst = anat_dir / f"{sub}_{ses}_T1w{ext}"

        if dst.exists() and not overwrite:
            report_rows.append(
                {
                    "source": str(src),
                    "output": str(dst),
                    "subject": sub,
                    "session": ses,
                    "status": "exists_skipped",
                }
            )
            continue

        shutil.copy2(src, dst)
        # Create minimal sidecar JSON
        sidecar = anat_dir / f"{sub}_{ses}_T1w.json"
        if overwrite or not sidecar.exists():
            sidecar.write_text(
                json.dumps(
                    {
                        "Modality": "MR",
                        "SequenceDescription": rel.parts[1],
                        "SourceFile": str(src),
                        "ConvertedBy": "convert_adni_to_bids.py",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        converted += 1
        report_rows.append(
            {
                "source": str(src),
                "output": str(dst),
                "subject": sub,
                "session": ses,
                "status": "converted",
            }
        )

    # dataset_description.json
    dataset_desc = {
        "Name": "ADNI Converted for Topo-Brain",
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw",
    }
    (output_root / "dataset_description.json").write_text(json.dumps(dataset_desc, indent=2), encoding="utf-8")

    # participants.tsv
    with (output_root / "participants.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["participant_id"] , delimiter='\t')
        writer.writeheader()
        for p in sorted(participants):
            writer.writerow({"participant_id": p})

    # conversion report
    report_path = output_root / "conversion_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "output", "subject", "session", "status"])
        writer.writeheader()
        writer.writerows(report_rows)

    return converted, skipped, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ADNI folder tree to BIDS layout")
    parser.add_argument("--input-root", required=True, help="Path to ADNI source root")
    parser.add_argument("--output-root", required=True, help="Path to output BIDS root")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing converted files")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    converted, skipped, report_path = convert(input_root, output_root, overwrite=args.overwrite)
    print(f"Done. Converted: {converted}, Skipped: {skipped}")
    print(f"BIDS root: {output_root}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
