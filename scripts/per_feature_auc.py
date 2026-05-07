"""
Per-feature univariate AD-vs-CN AUC analysis.

For each feature in the panel, computes:
    - Mann-Whitney U AUC  (the classification AUC if you used JUST this feature)
    - 95% bootstrap CI    (n_boot=2000)
    - Cohen's d           (already in other scripts; re-reported here for context)

Identifies the strongest single feature, complementing the Random-Forest
feature-importance ranking from train_ad_classifier.py.

Usage:
    python scripts/per_feature_auc.py \\
        --features adni_smoke_analysis/features_smoke.csv \\
        --out-dir adni_smoke_analysis
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Direction key: AD-low features should be sign-flipped before AUC computation.
PANEL_DIRECTION = {
    "gm_frac":      "low",
    "gm_mm3":       "low",
    "csf_gm_ratio": "high",
    "csf_mm3":      "high",
    "wm_mm3":       "low",
    "brain_mm3":    "low",
}


def auc_mw(y, scores):
    pos = scores[y == 1]; neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    a = pos[:, None] - neg[None, :]
    return ((a > 0).sum() + 0.5 * (a == 0).sum()) / (len(pos) * len(neg))


def cohen_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = math.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1))
                  / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / max(s, 1e-12)


def auc_bootstrap_ci(y, scores, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for _retry in range(8):
            if len(np.unique(y[idx])) >= 2:
                break
            idx = rng.integers(0, n, size=n)
        a = auc_mw(y[idx], scores[idx])
        if not math.isnan(a):
            aucs.append(a)
    if not aucs:
        return float("nan"), float("nan")
    aucs = np.asarray(aucs)
    return (float(np.quantile(aucs, alpha/2)),
            float(np.quantile(aucs, 1 - alpha/2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--positive-label", default="AD")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    if "subject" not in df.columns and "ptid" in df.columns:
        df = df.rename(columns={"ptid": "subject"})

    feats = [c for c in df.columns if c not in ("subject", "group", "PTID")
             and pd.api.types.is_numeric_dtype(df[c])]
    y = (df["group"] == args.positive_label).astype(int).to_numpy()
    ad = df[df["group"] == args.positive_label]
    cn = df[df["group"] != args.positive_label]

    rows = []
    for f in feats:
        # Direction-aligned scores: positive = more AD-like
        v = df[f].to_numpy(dtype=float)
        direction = PANEL_DIRECTION.get(f, "high")
        scores = -v if direction == "low" else v
        auc = auc_mw(y, scores)
        ci_lo, ci_hi = auc_bootstrap_ci(y, scores, n_boot=args.n_boot, seed=args.seed)
        d = cohen_d(ad[f].to_numpy(dtype=float), cn[f].to_numpy(dtype=float))
        rows.append({
            "feature":     f,
            "direction":   direction,
            "auc":         round(auc, 4),
            "auc_ci_lo":   round(ci_lo, 4),
            "auc_ci_hi":   round(ci_hi, 4),
            "cohen_d":     round(d, 4),
        })
    rows.sort(key=lambda r: -r["auc"])

    csv_path = args.out_dir / "per_feature_auc.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    lines = []
    lines.append("Per-feature univariate AD-vs-CN AUC (Mann-Whitney U)")
    lines.append("=" * 70)
    lines.append(f"  N: AD={int(y.sum())}, CN={int((1-y).sum())}, "
                 f"bootstrap iters: {args.n_boot}")
    lines.append("")
    lines.append(f"  {'feature':<14} {'dir':>5}  "
                 f"{'AUC':>6} {'95% CI':>17}  {'Cohen d':>9}")
    lines.append("  " + "-" * 58)
    for r in rows:
        ci = f"[{r['auc_ci_lo']:.2f}, {r['auc_ci_hi']:.2f}]"
        lines.append(f"  {r['feature']:<14} {r['direction']:>5}  "
                     f"{r['auc']:>6.3f} {ci:>17}  {r['cohen_d']:>+9.3f}")
    lines.append("")
    lines.append("Reading: AUC of the SINGLE feature treated as a classifier (positive =")
    lines.append("more AD-like). Direction-flipped where AD effect is decreasing. Compare")
    lines.append("to multivariate composite (ad_risk_summary.txt) and ML classifier")
    lines.append("(classifier_demo/auc_summary.json).")

    text = "\n".join(lines)
    print(text)
    (args.out_dir / "per_feature_auc.txt").write_text(text, encoding="utf-8")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {args.out_dir / 'per_feature_auc.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
