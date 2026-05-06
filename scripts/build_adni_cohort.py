"""
Build the ADNI baseline 3T MPRAGE cohort manifest from study-data CSVs.

Joins:
  - DXSUM (diagnosis-of-record)
  - PTDEMOG (age, sex, education)
  - MRI3META (per-visit MRI conduct flags)
  - MRIQC (per-scan metadata; soft cross-reference)
  - conversion_manifest.csv from convert_adni_dicom.py (NIfTI paths) [optional]

Cohort source-of-truth: DICOMs we actually downloaded (under --dicom-root)
or rows in --conversion-manifest. The CSVs annotate, they don't define the
cohort. MRI3META.MMSMPRAGE turns out to be unreliable across phases (only
~2% of scans flagged), so it's used as an advisory annotation only.

Filters (logged at each step):
  1. DICOM/manifest source -> baseline cohort PTIDs
  2. DXSUM left-join -> require baseline AD/CN diagnosis-of-record
  3. PTDEMOG left-join -> require age and sex present
  4. Optional age/sex 1:1 matching (AD <-> CN within +-N years, same sex)

Output: pairs_adni.csv with columns:
    ptid, group, age, sex, education, viscode, examdate,
    image_id, dicom_dir, nifti_path, qc_status, notes

Diagnosis encoding (DXSUM.DIAGNOSIS, current ADNI unified scheme):
    1 = CN, 2 = MCI, 3 = AD/Dementia

Visit codes counted as "baseline":
    VISCODE2 in {bl, sc, scmri, init, v06}
    (ADNI1 used 'sc'/'bl', ADNI2/GO 'v06'/'bl', ADNI3 'init'/'bl'.)

Usage:
    python scripts/build_adni_cohort.py \\
        --study-data adni_dataset/study_data \\
        --dicom-root adni_dataset \\
        --conversion-manifest adni_nifti/conversion_manifest.csv \\
        --out adni_dataset/pairs_adni.csv

    # with 1:1 age/sex matching
    python scripts/build_adni_cohort.py ... --match --max-age-diff 5
"""
import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


BASELINE_VISCODES = {"bl", "sc", "scmri", "init", "v06"}
DIAG_CN, DIAG_MCI, DIAG_AD = 1, 2, 3
GROUP_LABEL = {DIAG_CN: "CN", DIAG_AD: "AD"}
SEX_LABEL = {1: "M", 2: "F"}
COHORT_TOKEN_TO_GROUP = {"AD": "AD", "CN": "CN", "MCI": "MCI"}


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _find_csv(study_data_dir: Path, prefix: str) -> Path:
    """Find a CSV by case-insensitive prefix; tolerates date-suffixed filenames."""
    matches = sorted(p for p in study_data_dir.iterdir()
                     if p.is_file() and p.suffix.lower() == ".csv"
                     and p.name.upper().startswith(prefix.upper()))
    if not matches:
        raise FileNotFoundError(
            f"No CSV starting with '{prefix}' in {study_data_dir}. "
            f"Found: {[p.name for p in study_data_dir.iterdir()]}"
        )
    if len(matches) > 1:
        logging.warning("Multiple %s files found; using newest by name: %s",
                        prefix, matches[-1].name)
    return matches[-1]


def _to_int(x) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _to_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _parse_examdate(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _extract_year(x) -> Optional[int]:
    """ADNI PTDOBYY can be a 4-digit int OR a date string like '1939-01-01'.
    PTDOB is sometimes 'MM/YYYY'. Pull the year out of any of those."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        v = int(float(s))
        if 1900 <= v <= 2100:
            return v
    except ValueError:
        pass
    for token in s.replace("/", "-").split("-"):
        try:
            v = int(token)
            if 1900 <= v <= 2100:
                return v
        except ValueError:
            continue
    return None


def load_dxsum(path: Path) -> pd.DataFrame:
    """One row per (PTID, visit). Keep baseline AD/CN, earliest per PTID."""
    df = pd.read_csv(path, low_memory=False)
    n0 = len(df)

    df["VISCODE2"] = df["VISCODE2"].astype(str).str.lower().str.strip()
    base = df[df["VISCODE2"].isin(BASELINE_VISCODES)].copy()
    logging.info("DXSUM: %d rows -> %d baseline-visit rows", n0, len(base))

    base["DIAGNOSIS"] = base["DIAGNOSIS"].apply(_to_int)
    keep = base["DIAGNOSIS"].isin([DIAG_CN, DIAG_AD])
    base = base[keep].copy()
    logging.info("DXSUM: %d rows after AD/CN filter (drop MCI/missing)", len(base))

    base["_examdate"] = base["EXAMDATE"].apply(lambda x: _parse_examdate(_to_str(x)))
    base = base.sort_values(["PTID", "_examdate"], na_position="last")
    earliest = base.drop_duplicates(subset=["PTID"], keep="first").copy()
    logging.info("DXSUM: %d subjects after earliest-baseline-per-PTID dedupe", len(earliest))

    earliest["group"] = earliest["DIAGNOSIS"].map(GROUP_LABEL)
    return earliest[["PTID", "VISCODE2", "EXAMDATE", "DIAGNOSIS", "group", "_examdate"]]


def load_ptdemog(path: Path) -> pd.DataFrame:
    """One row per PTID with sex, DOB year, education. Coalesce across rows."""
    df = pd.read_csv(path, low_memory=False)
    df["PTGENDER"] = df["PTGENDER"].apply(_to_int)
    df["PTEDUCAT"] = df["PTEDUCAT"].apply(_to_int)
    # DOB year may live in either column; PTDOB ("MM/YYYY") is older, PTDOBYY
    # newer (sometimes a full date string). Coalesce.
    df["_dob_yy"] = df.apply(
        lambda r: _extract_year(r.get("PTDOBYY")) or _extract_year(r.get("PTDOB")),
        axis=1,
    )

    def first_valid(s):
        s = s.dropna()
        return s.iloc[0] if len(s) else None

    grouped = df.groupby("PTID", dropna=False).agg({
        "PTGENDER": first_valid,
        "_dob_yy": first_valid,
        "PTEDUCAT": first_valid,
    }).reset_index()
    grouped = grouped.rename(columns={"_dob_yy": "PTDOBYY"})
    grouped["sex"] = grouped["PTGENDER"].map(SEX_LABEL)
    logging.info("PTDEMOG: %d unique PTIDs (%d with DOB year, %d with sex)",
                 len(grouped),
                 grouped["PTDOBYY"].notna().sum(),
                 grouped["sex"].notna().sum())
    return grouped[["PTID", "sex", "PTDOBYY", "PTEDUCAT"]]


def load_mri3meta(path: Path) -> pd.DataFrame:
    """Per-PTID baseline MRI conduct flags (advisory cross-check only)."""
    df = pd.read_csv(path, low_memory=False)
    df["VISCODE2"] = df["VISCODE2"].astype(str).str.lower().str.strip()
    df["MMCONDCT"] = df["MMCONDCT"].apply(_to_int)
    df["MMSMPRAGE"] = df["MMSMPRAGE"].apply(_to_int)

    base = df[df["VISCODE2"].isin(BASELINE_VISCODES)].copy()
    base = base.drop_duplicates(subset=["PTID"], keep="first")
    logging.info("MRI3META: %d rows -> %d unique baseline PTIDs (%d with MMSMPRAGE=1)",
                 len(df), len(base), (base["MMSMPRAGE"] == 1).sum())
    return base[["PTID", "MMCONDCT", "MMSMPRAGE"]]


def load_mriqc(path: Path) -> pd.DataFrame:
    """Per-scan metadata indexed by image_id (LONI ImageID)."""
    df = pd.read_csv(path, low_memory=False)
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["ParticipantID"] = df["ParticipantID"].astype(str).str.strip()
    return df[["image_id", "ParticipantID", "VISCODE2", "MagneticFieldStrength",
               "SeriesDescription", "AcquisitionType"]]


def load_conversion_manifest(path: Optional[Path]) -> pd.DataFrame:
    """conversion_manifest.csv from convert_adni_dicom.py (subject_id == PTID)."""
    if path is None or not path.exists():
        logging.info("No conversion manifest provided; nifti_path/dicom_dir will be empty.")
        return pd.DataFrame(columns=["PTID", "cohort_folder", "dicom_dir",
                                     "nifti_path", "image_id"])
    df = pd.read_csv(path)
    df = df.rename(columns={"subject_id": "PTID", "cohort": "cohort_folder"})
    # image_id is the last path component of dicom_dir (e.g., .../I346242)
    df["image_id"] = df["dicom_dir"].astype(str).apply(
        lambda p: Path(p).name if p and p != "nan" else "")
    logging.info("Conversion manifest: %d rows (%d ok, %d failed)",
                 len(df),
                 (df["status"].isin(["ok", "ok_with_suffix", "skipped_exists"])).sum(),
                 (~df["status"].isin(["ok", "ok_with_suffix", "skipped_exists"])).sum())
    return df[["PTID", "cohort_folder", "dicom_dir", "nifti_path", "image_id", "status"]]


def discover_dicom_subjects(dicom_root: Path) -> pd.DataFrame:
    """Fallback when no conversion manifest: list PTIDs present under cohort folders."""
    rows: List[Dict] = []
    if not dicom_root.exists():
        return pd.DataFrame(columns=["PTID", "cohort_folder", "dicom_dir"])
    for cohort_dir in sorted(p for p in dicom_root.iterdir() if p.is_dir()):
        token = cohort_dir.name.split("_", 1)[0].upper()
        if token not in COHORT_TOKEN_TO_GROUP:
            continue
        adni_dir = cohort_dir / "ADNI"
        if not adni_dir.is_dir():
            continue
        for subj in sorted(p for p in adni_dir.iterdir() if p.is_dir()):
            rows.append({
                "PTID": subj.name,
                "cohort_folder": token,
                "dicom_dir": str(subj),
            })
    df = pd.DataFrame(rows)
    logging.info("DICOM root scan: %d subjects across %d cohort folders",
                 len(df), df["cohort_folder"].nunique() if len(df) else 0)
    return df


def compute_age(examdate: Optional[datetime], dob_year: Optional[int]) -> Optional[float]:
    if examdate is None or dob_year is None:
        return None
    return round(examdate.year + examdate.month / 12.0 - dob_year, 1)


def match_ad_to_cn(df: pd.DataFrame, max_age_diff: float) -> pd.DataFrame:
    """1:1 greedy nearest-neighbor: each AD subject paired to nearest unmatched
    CN of same sex within +-max_age_diff years. Drops unmatched on both sides."""
    ad = df[df["group"] == "AD"].copy().sort_values("age", na_position="last")
    cn = df[df["group"] == "CN"].copy()
    cn["_used"] = False
    matched_ad: List[int] = []
    matched_cn: List[int] = []

    for ad_idx, ad_row in ad.iterrows():
        if pd.isna(ad_row["age"]) or not ad_row["sex"]:
            continue
        candidates = cn[(~cn["_used"]) & (cn["sex"] == ad_row["sex"])
                        & cn["age"].notna()
                        & ((cn["age"] - ad_row["age"]).abs() <= max_age_diff)]
        if candidates.empty:
            continue
        best = (candidates["age"] - ad_row["age"]).abs().idxmin()
        matched_ad.append(ad_idx)
        matched_cn.append(best)
        cn.at[best, "_used"] = True

    keep_idx = matched_ad + matched_cn
    out = df.loc[keep_idx].sort_values(["group", "PTID"]).reset_index(drop=True)
    logging.info("Age/sex matching: %d AD <-> %d CN pairs (dropped %d unmatched)",
                 len(matched_ad), len(matched_cn),
                 len(df) - len(out))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--study-data", required=True,
                        help="Folder with the 5 ADNI study-data CSVs")
    parser.add_argument("--dicom-root", default=None,
                        help="adni_dataset/ root (used to discover PTIDs if no manifest)")
    parser.add_argument("--conversion-manifest", default=None,
                        help="conversion_manifest.csv from convert_adni_dicom.py")
    parser.add_argument("--out", required=True, help="Output pairs_adni.csv path")
    parser.add_argument("--groups", nargs="+", default=["AD", "CN"],
                        choices=["AD", "CN"],
                        help="Diagnosis groups to keep")
    parser.add_argument("--match", action="store_true",
                        help="1:1 age/sex match AD to CN")
    parser.add_argument("--max-age-diff", type=float, default=5.0,
                        help="Max age difference (years) for matching")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    study_data = Path(args.study_data).resolve()
    dxsum = load_dxsum(_find_csv(study_data, "DXSUM"))
    ptdemog = load_ptdemog(_find_csv(study_data, "PTDEMOG"))
    mri3meta = load_mri3meta(_find_csv(study_data, "MRI3META"))
    mriqc = load_mriqc(_find_csv(study_data, "MRIQC"))

    manifest_path = Path(args.conversion_manifest).resolve() if args.conversion_manifest else None
    manifest = load_conversion_manifest(manifest_path)
    if manifest.empty and args.dicom_root:
        manifest = discover_dicom_subjects(Path(args.dicom_root).resolve())
    if manifest.empty:
        logging.error("No cohort source: provide --conversion-manifest or --dicom-root")
        return 2

    # ---- Join chain (DICOMs drive the cohort, CSVs annotate) ----
    df = manifest.copy()
    logging.info("Cohort start: %d subjects from DICOM/manifest", len(df))

    df = df.merge(dxsum, on="PTID", how="left")
    n_with_dx = df["DIAGNOSIS"].notna().sum()
    logging.info("After DXSUM left-join: %d / %d subjects have a baseline AD/CN diagnosis",
                 n_with_dx, len(df))

    before = len(df)
    df = df[df["DIAGNOSIS"].isin([DIAG_CN, DIAG_AD])].copy()
    logging.info("Drop subjects without baseline AD/CN dx: %d -> %d", before, len(df))

    df = df.merge(ptdemog, on="PTID", how="left")
    df = df.merge(mri3meta, on="PTID", how="left")

    # Cohort-folder vs DXSUM cross-check
    for _, row in df.iterrows():
        cf_raw = row.get("cohort_folder")
        if pd.isna(cf_raw) or not cf_raw:
            continue
        cf = str(cf_raw).upper()
        if cf in COHORT_TOKEN_TO_GROUP and cf != row["group"]:
            logging.warning("PTID %s is in folder %s but DXSUM baseline says %s",
                            row["PTID"], cf, row["group"])

    # MMSMPRAGE advisory: log subjects ADNI didn't flag, but keep them.
    if "MMSMPRAGE" in df.columns:
        n_unflagged = (df["MMSMPRAGE"].fillna(0) != 1).sum()
        if n_unflagged:
            logging.info("MRI3META: %d subjects don't have MMSMPRAGE=1 (kept anyway; "
                         "DICOM presence is ground truth)", n_unflagged)

    # MRIQC cross-reference (soft): warn if image_id from DICOM not in MRIQC table.
    if "image_id" in df.columns and len(df):
        mriqc_ids = set(mriqc["image_id"].dropna().astype(str).tolist())
        for _, row in df.iterrows():
            iid = row.get("image_id")
            if pd.isna(iid) or not iid:
                continue
            iid_str = str(iid)
            stripped = iid_str.lstrip("I")
            if iid_str not in mriqc_ids and stripped not in mriqc_ids:
                logging.debug("PTID %s image_id %s not in MRIQC metadata",
                              row["PTID"], iid_str)

    # Age computation
    df["age"] = df.apply(lambda r: compute_age(r["_examdate"], r.get("PTDOBYY")), axis=1)

    # Drop missing demographics
    before = len(df)
    df = df[df["age"].notna() & df["sex"].notna() & (df["sex"] != "")].copy()
    logging.info("After demographics drop-missing: %d (was %d)", len(df), before)

    # Group filter from CLI
    df = df[df["group"].isin(args.groups)].copy()
    logging.info("After group filter %s: %d", args.groups, len(df))

    # Optional age/sex matching
    if args.match:
        df = match_ad_to_cn(df, max_age_diff=args.max_age_diff)

    # ---- Final output ----
    out_cols = ["ptid", "group", "age", "sex", "education", "viscode",
                "examdate", "image_id", "dicom_dir", "nifti_path",
                "qc_status", "notes"]
    out = pd.DataFrame({
        "ptid": df["PTID"],
        "group": df["group"],
        "age": df["age"],
        "sex": df["sex"],
        "education": df["PTEDUCAT"],
        "viscode": df["VISCODE2"],
        "examdate": df["EXAMDATE"],
        "image_id": df.get("image_id", ""),
        "dicom_dir": df.get("dicom_dir", ""),
        "nifti_path": df.get("nifti_path", ""),
        "qc_status": df.get("status", ""),
        "notes": "",
    })[out_cols].sort_values(["group", "ptid"]).reset_index(drop=True)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logging.info("Wrote %d rows to %s", len(out), out_path)

    # Summary by group
    summary = (out.groupby("group")
               .agg(n=("ptid", "count"),
                    mean_age=("age", "mean"),
                    pct_male=("sex", lambda s: 100.0 * (s == "M").mean()))
               .round(1))
    print("\nCohort summary:")
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
