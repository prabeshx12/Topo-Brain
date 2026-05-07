"""
Cohort demographics table for the smoke-test ADNI subjects.

Cross-references the smoke-cohort PTIDs against PTDEMOG.csv from ADNI's
study-data download to produce a publication-style demographics table:

  Group  N    Age (mean ± SD)    Female %    Education (years)    APOE-ε4 (where avail.)

Required for any clinical paper. Produces both per-subject CSV and
group-summary text.

Usage:
    python scripts/cohort_demographics.py \\
        --cohort-csv adni_smoke_analysis/features_smoke.csv \\
        --ptdemog adni_dataset/study_data/PTDEMOG_05May2026.csv \\
        --dxsum   adni_dataset/study_data/DXSUM_05May2026.csv \\
        --out-dir adni_smoke_analysis
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _setup_logging():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _extract_year(x):
    """ADNI PTDOBYY may be a 4-digit int or a date string. Pull the year."""
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


def _parse_examdate(s):
    s = str(s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort-csv", type=Path, required=True,
                        help="Smoke cohort CSV (must have a PTID-like column)")
    parser.add_argument("--ptdemog", type=Path, required=True,
                        help="ADNI PTDEMOG_*.csv from study data")
    parser.add_argument("--dxsum", type=Path, default=None,
                        help="Optional ADNI DXSUM_*.csv for examdate")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    _setup_logging()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(args.cohort_csv)
    if "subject" in cohort.columns:
        cohort = cohort.rename(columns={"subject": "PTID"})
    elif "ptid" in cohort.columns:
        cohort = cohort.rename(columns={"ptid": "PTID"})
    cohort["PTID"] = cohort["PTID"].astype(str).str.strip()

    ptdemog = pd.read_csv(args.ptdemog, low_memory=False)
    ptdemog["PTID"] = ptdemog["PTID"].astype(str).str.strip()

    # Coalesce DOB year (PTDEMOG often has it across multiple rows per subject)
    ptdemog["_dob_yy"] = ptdemog.apply(
        lambda r: _extract_year(r.get("PTDOBYY")) or _extract_year(r.get("PTDOB")),
        axis=1,
    )

    def first_valid(s):
        s = s.dropna()
        return s.iloc[0] if len(s) else None

    demo = ptdemog.groupby("PTID", dropna=False).agg({
        "PTGENDER": first_valid,   # 1 = M, 2 = F
        "_dob_yy":  first_valid,
        "PTEDUCAT": first_valid,
    }).reset_index()
    demo["sex"] = demo["PTGENDER"].map({1: "M", 1.0: "M", 2: "F", 2.0: "F"})
    demo = demo.rename(columns={"_dob_yy": "dob_year", "PTEDUCAT": "education_yrs"})

    # Pull examdate from DXSUM (we already used it in cohort assembly)
    if args.dxsum and args.dxsum.exists():
        dx = pd.read_csv(args.dxsum, low_memory=False)
        dx["PTID"] = dx["PTID"].astype(str).str.strip()
        dx["VISCODE2"] = dx["VISCODE2"].astype(str).str.lower().str.strip()
        baseline = dx[dx["VISCODE2"].isin({"bl", "sc", "scmri", "init", "v06"})].copy()
        baseline["_examdate"] = baseline["EXAMDATE"].apply(_parse_examdate)
        baseline = baseline.sort_values(["PTID", "_examdate"], na_position="last")
        examdates = baseline.drop_duplicates(subset=["PTID"], keep="first")[
            ["PTID", "_examdate"]]
        demo = demo.merge(examdates, on="PTID", how="left")
    else:
        demo["_examdate"] = None

    # Compute age = examdate.year + month/12 - dob_year
    def compute_age(row):
        exam = row.get("_examdate"); dob = row.get("dob_year")
        if exam is None or dob is None or pd.isna(exam) or pd.isna(dob):
            return None
        try:
            return round(exam.year + exam.month / 12.0 - int(dob), 1)
        except Exception:
            return None
    demo["age"] = demo.apply(compute_age, axis=1)

    # Join to the cohort
    out = cohort.merge(demo[["PTID", "sex", "age", "education_yrs"]],
                       on="PTID", how="left")
    out_path = args.out_dir / "cohort_demographics_per_subject.csv"
    keep = ["PTID", "group", "sex", "age", "education_yrs"]
    out[keep].to_csv(out_path, index=False)

    # Group summaries
    def fmt_msd(values, fmt="{:.1f}"):
        v = pd.to_numeric(values, errors="coerce").dropna()
        if v.empty:
            return "n/a"
        return f"{fmt.format(v.mean())} ± {fmt.format(v.std(ddof=1))}"

    groups = sorted(out["group"].unique())
    lines = []
    lines.append("Cohort demographics — ADNI smoke test (5 AD + 5 CN)")
    lines.append("=" * 72)
    lines.append(f"  {'Variable':<24} " +
                 "  ".join(f"{g:>16}" for g in groups))
    lines.append("  " + "-" * 60)

    n_row = "  N (subjects)            " + "  ".join(
        f"{int((out['group'] == g).sum()):>16d}" for g in groups)
    lines.append(n_row)

    age_row = "  Age (years, mean ± SD) " + "  ".join(
        f"{fmt_msd(out[out['group']==g]['age']):>16}" for g in groups)
    lines.append(age_row)

    edu_row = "  Education (years)      " + "  ".join(
        f"{fmt_msd(out[out['group']==g]['education_yrs']):>16}" for g in groups)
    lines.append(edu_row)

    for g in groups:
        sub = out[out["group"] == g]
        n_f = int((sub["sex"] == "F").sum())
        n_m = int((sub["sex"] == "M").sum())
        total = n_f + n_m
        pct = (n_f / total * 100) if total else 0
        lines.append(f"  {g} sex (F / M, % female):  {n_f}/{n_m} ({pct:.0f}%)")

    lines.append("")
    lines.append("Per-subject details:")
    for _, r in out.sort_values(["group", "PTID"]).iterrows():
        age = r.get("age");
        age_str = f"{age:.1f}" if age == age else "n/a"
        lines.append(f"  {r['PTID']:<14} {r['group']:<3} "
                     f"sex={r.get('sex','?')} age={age_str} "
                     f"educ={r.get('education_yrs','?')}")
    lines.append("")
    lines.append("Source CSVs (gitignored, in adni_dataset/study_data/):")
    lines.append(f"  PTDEMOG: {args.ptdemog.name}")
    if args.dxsum:
        lines.append(f"  DXSUM:   {args.dxsum.name}")

    text = "\n".join(lines)
    print(text)
    (args.out_dir / "cohort_demographics_summary.txt").write_text(text, encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"Wrote {args.out_dir / 'cohort_demographics_summary.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
