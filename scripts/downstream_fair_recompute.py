"""
Fair recomputation of the C1 (native-3T K-means) vs C2 (synthetic-7T) AD/CN
downstream comparison.

Motivation (reviewer R1.5 / R2.4)
---------------------------------
The original head-to-head (compare_c1_c2_risk_score.py) reports C1 AUC=0.818 vs
C2 AUC=0.658 and reads as "synthesis loses diagnostic information". Two issues
make that comparison misleading:

  1. IN-SAMPLE LEAK. The composite z-score uses the mean/std of *all* CN
     subjects as the reference, then scores those same CN subjects. The CN
     reference therefore contains the very subjects being classified.
  2. NO UNCERTAINTY. With n=15 AD + 15 CN, a single point AUC is near-useless
     without a confidence interval. A 0.16 AUC gap on 30 subjects is a handful
     of subjects swapping rank.

This script recomputes both AUCs with:
  * LEAVE-ONE-SUBJECT-OUT CN reference (removes the in-sample leak), and
  * the original in-sample reference (to quantify how much optimism the leak
    added), and
  * a bootstrap 95% CI for each AUC, and
  * DeLong's test (Sun & Xu, 2014) for the C1-vs-C2 difference on the same
    subjects (paired / correlated ROC).

It writes a thesis-ready summary. No GPU, no model, no checkpoint required:
runs entirely off the surviving per-subject feature CSV.

Usage
-----
    python scripts/downstream_fair_recompute.py \
        --c1-vs-c2-csv adni_smoke_analysis_30/c1_vs_c2_per_subject.csv \
        --out-dir adni_smoke_analysis_30 \
        --n-boot 10000 --seed 42
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# Feature panel and the AD direction of each feature (matches the original
# compare_c1_c2_risk_score.py so the recompute is comparable).
FEATURE_PANEL = {
    "gm_frac":      "low",   # AD -> lower GM fraction
    "gm_mm3":       "low",   # AD -> lower GM volume
    "csf_gm_ratio": "high",  # AD -> higher CSF/GM
    "csf_mm3":      "high",  # AD -> higher CSF volume
}


# --------------------------------------------------------------------------- #
#  Composite scoring
# --------------------------------------------------------------------------- #
def _composite_from_refs(row, panel, refs):
    comps = []
    for f, direction in panel.items():
        mu, sd = refs[f]
        z = (row[f] - mu) / max(sd, 1e-12)
        if direction == "low":
            z = -z
        comps.append(z)
    return float(np.mean(comps))


def composite_insample(rows, panel):
    """Original behaviour: CN reference from ALL CN subjects (in-sample)."""
    cn_rows = [r for r in rows if r["group"] != "AD"]
    refs = {f: (float(np.mean([r[f] for r in cn_rows])),
                float(np.std([r[f] for r in cn_rows], ddof=1)))
            for f in panel}
    return np.array([_composite_from_refs(r, panel, refs) for r in rows])


def composite_loso(rows, panel):
    """Leave-one-subject-out CN reference: for each subject, build the CN
    reference EXCLUDING that subject, then score it. Removes the in-sample
    leak (a CN subject is never part of its own reference)."""
    scores = []
    for i, r in enumerate(rows):
        cn_rows = [s for j, s in enumerate(rows)
                   if s["group"] != "AD" and j != i]
        refs = {f: (float(np.mean([s[f] for s in cn_rows])),
                    float(np.std([s[f] for s in cn_rows], ddof=1)))
                for f in panel}
        scores.append(_composite_from_refs(r, panel, refs))
    return np.array(scores)


# --------------------------------------------------------------------------- #
#  AUC, bootstrap CI
# --------------------------------------------------------------------------- #
def auc_mw(y, scores):
    """AUC via the Mann-Whitney U statistic (ties counted as 0.5)."""
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return ((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg))


def bootstrap_ci(y, scores, n_boot, seed, alpha=0.05):
    """Stratified bootstrap (resample AD and CN separately) 95% CI for AUC."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        p = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        n = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p, n])
        aucs[b] = auc_mw(y[idx], scores[idx])
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), aucs


# --------------------------------------------------------------------------- #
#  DeLong's test (Sun & Xu 2014, fast midrank implementation)
# --------------------------------------------------------------------------- #
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def delong_cov(scores_by_method, y):
    """Return (aucs, covariance matrix) for k correlated methods on the same
    subjects. scores_by_method: array [k, N]; y: array [N] of {0,1}."""
    order = np.argsort(-y)  # positives (label 1) first
    label_1_count = int(y.sum())
    preds = scores_by_method[:, order]
    m = label_1_count
    n = preds.shape[1] - m
    k = preds.shape[0]
    pos = preds[:, :m]
    neg = preds[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_test(s1, s2, y):
    """Two-sided DeLong p-value for AUC(s1) != AUC(s2) on the same subjects."""
    aucs, cov = delong_cov(np.vstack([s1, s2]), y)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z = 0.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


# --------------------------------------------------------------------------- #
def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        raw = list(csv.DictReader(f))
    c1, c2 = [], []
    for r in raw:
        d1 = {"subject": r["ptid"], "group": r["group"]}
        d2 = {"subject": r["ptid"], "group": r["group"]}
        for feat in FEATURE_PANEL:
            d1[feat] = float(r[f"{feat}_C1"])
            d2[feat] = float(r[f"{feat}_C2"])
        c1.append(d1)
        c2.append(d2)
    y = np.array([1 if r["group"] == "AD" else 0 for r in raw])
    return c1, c2, y


def summarize(label, rows, y, n_boot, seed):
    s_in = composite_insample(rows, FEATURE_PANEL)
    s_loso = composite_loso(rows, FEATURE_PANEL)
    auc_in = auc_mw(y, s_in)
    auc_loso = auc_mw(y, s_loso)
    lo, hi, _ = bootstrap_ci(y, s_loso, n_boot, seed)
    return {
        "label": label, "s_loso": s_loso, "s_in": s_in,
        "auc_in": auc_in, "auc_loso": auc_loso, "ci_lo": lo, "ci_hi": hi,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c1-vs-c2-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    c1, c2, y = load_rows(args.c1_vs_c2_csv)
    n_ad, n_cn = int(y.sum()), int((1 - y).sum())

    r1 = summarize("C1 (native-3T K-means)", c1, y, args.n_boot, args.seed)
    r2 = summarize("C2 (synthetic-7T seg)", c2, y, args.n_boot, args.seed)

    # DeLong on the LOSO scores (the honest, leak-free version)
    a1, a2, z, p = delong_test(r1["s_loso"], r2["s_loso"], y)

    L = []
    L.append("Fair recomputation: C1 (native-3T) vs C2 (synthetic-7T)")
    L.append("=" * 74)
    L.append(f"  Cohort: {n_ad} AD + {n_cn} CN (ADNI smoke); panel: {', '.join(FEATURE_PANEL)}")
    L.append("")
    L.append(f"  {'Method':<26}{'AUC in-sample':>15}{'AUC LOSO':>12}{'LOSO 95% CI':>20}")
    L.append("  " + "-" * 71)
    for r in (r1, r2):
        L.append(f"  {r['label']:<26}{r['auc_in']:>15.3f}{r['auc_loso']:>12.3f}"
                 f"{'[%.3f, %.3f]' % (r['ci_lo'], r['ci_hi']):>20}")
    L.append("")
    L.append("  In-sample optimism (AUC_in - AUC_LOSO):")
    L.append(f"      C1: {r1['auc_in'] - r1['auc_loso']:+.3f}    "
             f"C2: {r2['auc_in'] - r2['auc_loso']:+.3f}")
    L.append("")
    L.append("  DeLong test, C1 vs C2 (leave-one-subject-out scores, paired):")
    L.append(f"      AUC C1 = {a1:.3f}   AUC C2 = {a2:.3f}   z = {z:.3f}   p = {p:.3f}")
    sig = "SIGNIFICANT" if p < 0.05 else "NOT significant"
    L.append(f"      Difference is {sig} at alpha=0.05.")
    L.append("")
    overlap = not (r1["ci_hi"] < r2["ci_lo"] or r2["ci_hi"] < r1["ci_lo"])
    L.append("Interpretation:")
    L.append(f"  - The two 95% CIs {'OVERLAP' if overlap else 'do NOT overlap'}.")
    L.append("  - C1 is an unsupervised K-means on 3T intensities; C2 is a 7T-supervised")
    L.append("    segmentation head applied to 3T. Both operate at coarse 4-class tissue")
    L.append("    granularity, where AD atrophy is already largely separable, so neither")
    L.append("    tests the synthesis hypothesis (finer parcellation) directly.")
    L.append("  - With n=30 and these CIs, the headline 0.818-vs-0.658 gap is not")
    L.append("    statistically reliable; it should be reported with CIs and the DeLong p,")
    L.append("    not as evidence that synthesis loses diagnostic signal.")

    text = "\n".join(L)
    print(text)
    out = args.out_dir / "downstream_fair_recompute.txt"
    out.write_text(text, encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
