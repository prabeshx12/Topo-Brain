"""
Merge two feature CSVs into a single fusion CSV for the AD classifier's
C3 = fusion configuration.

The C3 row of the thesis combines features extracted from real-3T-derived
aparc+aseg with features extracted from synthetic-7T-derived aparc+aseg.
Both feature CSVs come from extract_features_from_aseg.py and have
overlapping column names like `hippo_L_mm3` — without suffix renaming the
merge would collide. This script:

  1. Inner-joins the two CSVs on `subject`
  2. Verifies that label / age / sex passthrough columns agree between the
     two inputs (hard-fails on mismatch — that's a data-integrity bug)
  3. Suffixes feature columns from each side (`_3t` / `_7t` by default)
  4. Writes the fused CSV ready for train_ad_classifier.py

Pass-through columns ARE NOT suffixed: the merged CSV keeps a single
`subject`, `group`, `age`, `sex` (taken from A; B is checked for agreement).

Usage:
    python scripts/merge_features.py \\
        --a features_3t.csv \\
        --b features_synth7t.csv \\
        --suffix-a _3t \\
        --suffix-b _7t \\
        --output features_fusion.csv

    # Validate the merge logic on synthetic data:
    python scripts/merge_features.py --self-test

The fused CSV is then passed to train_ad_classifier.py to produce the
C3 predictions:

    python scripts/train_ad_classifier.py \\
        --features features_fusion.csv \\
        --output-dir results/ad/fusion
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# Columns that are subject metadata, not features. Pass through unchanged
# from the A side; require B to agree.
PASSTHROUGH_COLS = ("subject", "group", "age", "sex")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def merge_feature_csvs(
    df_a, df_b, suffix_a: str, suffix_b: str,
    passthrough_cols: Tuple[str, ...] = PASSTHROUGH_COLS,
):
    """Pure-pandas merge implementation, separated so --self-test can drive it."""
    import pandas as pd

    if "subject" not in df_a.columns or "subject" not in df_b.columns:
        raise ValueError("Both inputs must have a 'subject' column")

    # Identify which passthrough columns each side actually has.
    a_pass = [c for c in passthrough_cols if c in df_a.columns]
    b_pass = [c for c in passthrough_cols if c in df_b.columns]

    # Feature columns are everything else. Numeric or not, we keep them.
    a_feat = [c for c in df_a.columns if c not in a_pass]
    b_feat = [c for c in df_b.columns if c not in b_pass]

    # Inner join on subject so we only keep subjects present in BOTH.
    n_a, n_b = len(df_a), len(df_b)
    merged = df_a.merge(df_b, on="subject", how="inner",
                        suffixes=("__A", "__B"))
    n_inner = len(merged)
    logging.info("Inner-join on 'subject': %d (A) ∩ %d (B) -> %d subjects",
                 n_a, n_b, n_inner)

    # Verify agreement on shared passthrough columns.
    shared_pass = [c for c in passthrough_cols if c in a_pass and c in b_pass and c != "subject"]
    for col in shared_pass:
        a_col, b_col = f"{col}__A", f"{col}__B"
        # Treat NaN as equal to NaN (some CSVs have empty cells).
        a_vals = merged[a_col].astype(str).fillna("")
        b_vals = merged[b_col].astype(str).fillna("")
        bad = merged[a_vals != b_vals]
        if len(bad):
            mismatched = bad.head(5)[["subject", a_col, b_col]].to_dict("records")
            raise ValueError(
                f"'{col}' disagrees between A and B for {len(bad)} subjects. "
                f"First few: {mismatched}. Fix the upstream feature CSVs before merging."
            )

    # Build the output: keep passthrough columns from A, drop the duplicate __B copies.
    out = pd.DataFrame()
    out["subject"] = merged["subject"]
    for col in passthrough_cols:
        if col == "subject":
            continue
        a_col, b_col = f"{col}__A", f"{col}__B"
        if a_col in merged.columns and b_col in merged.columns:
            out[col] = merged[a_col]
        elif a_col in merged.columns:
            out[col] = merged[a_col]
        elif b_col in merged.columns:
            out[col] = merged[b_col]
        elif col in merged.columns:  # only one side had it, no suffix conflict
            out[col] = merged[col]

    # Suffix feature columns. Note: feature columns from A may have got
    # __A added by merge() if they collided with B's column names, OR may
    # be unsuffixed if there was no collision. Same on the B side.
    def _resolve(side_feat: List[str], side_marker: str, my_suffix: str):
        """For each feature col on this side, find its merged column name and
        write it back into `out` with the user-chosen suffix."""
        for col in side_feat:
            collided_name = f"{col}{side_marker}"
            if collided_name in merged.columns:
                src = collided_name
            elif col in merged.columns:
                src = col
            else:
                # Shouldn't happen since col came from df_<side>.columns
                logging.warning("feature %s missing post-merge — skipping", col)
                continue
            out[f"{col}{my_suffix}"] = merged[src]

    _resolve(a_feat, "__A", suffix_a)
    _resolve(b_feat, "__B", suffix_b)

    return out, n_inner, n_a, n_b


def _self_test() -> int:
    """Validate merge correctness on synthetic data."""
    import pandas as pd
    print("Running merge_features self-test...")

    # Two CSVs with overlapping subjects but different feature sets.
    a = pd.DataFrame({
        "subject":      ["sub-1", "sub-2", "sub-3"],
        "group":        ["AD",    "CN",    "AD"],
        "age":          [70.0,    65.0,    75.0],
        "hippo_L_mm3":  [3500.0,  4000.0,  3200.0],
        "amygdala_R_mm3": [1500.0, 1700.0, 1400.0],
    })
    b = pd.DataFrame({
        "subject":      ["sub-2", "sub-3", "sub-4"],
        "group":        ["CN",    "AD",    "CN"],
        "age":          [65.0,    75.0,    60.0],
        "hippo_L_mm3":  [4500.0,  3800.0,  4200.0],
        "thalamus_L_mm3": [7500.0, 7200.0, 7800.0],
    })

    out, n_inner, n_a, n_b = merge_feature_csvs(a, b, "_3t", "_7t")
    print(f"  inner-join size: {n_inner} (expected 2: sub-2 and sub-3)")
    assert n_inner == 2

    # Required columns: subject, group, age, hippo_L_mm3_3t, amygdala_R_mm3_3t,
    #                   hippo_L_mm3_7t, thalamus_L_mm3_7t
    expected_cols = {"subject", "group", "age",
                     "hippo_L_mm3_3t", "amygdala_R_mm3_3t",
                     "hippo_L_mm3_7t", "thalamus_L_mm3_7t"}
    got_cols = set(out.columns)
    missing = expected_cols - got_cols
    extra = got_cols - expected_cols
    print(f"  columns: {sorted(got_cols)}")
    assert not missing, f"missing columns: {missing}"
    assert not extra,   f"unexpected columns: {extra}"

    # Verify that the SAME column name (hippo_L_mm3) on both sides was
    # disambiguated correctly via the suffixes.
    sub2 = out[out["subject"] == "sub-2"].iloc[0]
    assert sub2["hippo_L_mm3_3t"] == 4000.0, f"3t value wrong: {sub2['hippo_L_mm3_3t']}"
    assert sub2["hippo_L_mm3_7t"] == 4500.0, f"7t value wrong: {sub2['hippo_L_mm3_7t']}"
    print(f"  ✓ sub-2: hippo_L_mm3_3t={sub2['hippo_L_mm3_3t']}  hippo_L_mm3_7t={sub2['hippo_L_mm3_7t']}")

    # Group/age agreed for both sub-2 and sub-3, so passthrough should be unchanged.
    assert sub2["group"] == "CN" and sub2["age"] == 65.0
    print(f"  ✓ passthrough preserved: group={sub2['group']}  age={sub2['age']}")

    # Group-disagreement test: same subject, different group label.
    a_bad = pd.DataFrame({"subject": ["sub-1"], "group": ["AD"], "x": [1.0]})
    b_bad = pd.DataFrame({"subject": ["sub-1"], "group": ["CN"], "y": [2.0]})
    try:
        merge_feature_csvs(a_bad, b_bad, "_a", "_b")
        print("  ✗ group-disagreement should have raised but didn't")
        return 1
    except ValueError as e:
        print(f"  ✓ group-disagreement raised ValueError: {str(e)[:80]}...")

    # Empty intersection test
    a_only = pd.DataFrame({"subject": ["sub-1", "sub-2"], "x": [1.0, 2.0]})
    b_only = pd.DataFrame({"subject": ["sub-3", "sub-4"], "y": [5.0, 6.0]})
    out_e, n_e, _, _ = merge_feature_csvs(a_only, b_only, "_a", "_b")
    print(f"  ✓ empty intersection: n={n_e} (expected 0)")
    assert n_e == 0

    print("✓ All merge_features self-tests passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--a", type=Path, default=None,
                        help="Feature CSV for side A (e.g. features_3t.csv)")
    parser.add_argument("--b", type=Path, default=None,
                        help="Feature CSV for side B (e.g. features_synth7t.csv)")
    parser.add_argument("--suffix-a", default="_3t",
                        help="Suffix to append to A's feature columns (default: _3t)")
    parser.add_argument("--suffix-b", default="_7t",
                        help="Suffix to append to B's feature columns (default: _7t)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output fused feature CSV path")
    parser.add_argument("--self-test", action="store_true",
                        help="Validate the merge logic on synthetic data")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    if args.self_test:
        return _self_test()

    if args.a is None or args.b is None or args.output is None:
        logging.error("--a, --b, and --output are required (or use --self-test)")
        return 2

    if args.suffix_a == args.suffix_b:
        logging.error("--suffix-a and --suffix-b must differ to disambiguate columns")
        return 2

    import pandas as pd
    df_a = pd.read_csv(args.a)
    df_b = pd.read_csv(args.b)
    logging.info("A: %s (%d rows, %d cols)", args.a, len(df_a), len(df_a.columns))
    logging.info("B: %s (%d rows, %d cols)", args.b, len(df_b), len(df_b.columns))

    try:
        out, n_inner, n_a, n_b = merge_feature_csvs(
            df_a, df_b, args.suffix_a, args.suffix_b)
    except ValueError as e:
        logging.error("Merge failed: %s", e)
        return 1

    if n_inner == 0:
        logging.error("Empty intersection — A and B share no subjects. Nothing to merge.")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    n_feat_a = sum(1 for c in out.columns if c.endswith(args.suffix_a))
    n_feat_b = sum(1 for c in out.columns if c.endswith(args.suffix_b))
    logging.info("Wrote %s | %d subjects, %d features (%d from A%s + %d from B%s)",
                 args.output, n_inner,
                 n_feat_a + n_feat_b, n_feat_a, args.suffix_a, n_feat_b, args.suffix_b)

    # Heads-up if too few subjects survived
    if n_inner < min(n_a, n_b) * 0.5:
        logging.warning(
            "Less than half of either side survived the inner join "
            "(%d / min(%d, %d)). Check that subject IDs match across A and B.",
            n_inner, n_a, n_b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
