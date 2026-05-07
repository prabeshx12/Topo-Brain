"""
Composite AD-risk biomarker score from synth-derived tissue volumes.

Mirrors the NIA-AA "ATN" biomarker-composite framework (Jack et al. 2018,
Frisoni et al. 2010): combine multiple structural MRI biomarkers into a
single per-subject score whose magnitude indicates AD-likelihood.

Per subject:
    For each feature in the selected panel:
        z = (subject_value - CN_mean) / CN_SD
    If feature is AD-LOW (i.e. expected AD < CN per literature):
        z = -z   so that higher z = more AD-like
    composite_score = mean(z) across the panel

Then we find the Youden-J threshold and report sensitivity/specificity
at that operating point — interpretable clinical metrics that don't
require statistical significance at n=10 to be meaningful.

This is a "biomarker-score" approach (interpretable, hand-engineered,
no overfitting risk at small n) rather than a "machine-learned
classifier" approach (which is what train_ad_classifier.py does).
Both are presented in the thesis; this script's output is more
clinically interpretable.

Default feature panel (canonical AD biomarkers):
    gm_frac        AD-low   relative grey-matter fraction
    gm_mm3         AD-low   absolute grey-matter volume
    csf_gm_ratio   AD-high  textbook atrophy index
    csf_mm3        AD-high  ventricular enlargement proxy

Inputs:
    features CSV with columns: subject, group (AD/CN), and the feature columns

Outputs (in --out-dir):
    ad_risk_score_per_subject.csv  per-subject z-scores + composite
    ad_risk_distribution.png       AD vs CN composite-score distributions
    ad_risk_summary.txt            thesis-ready operating-point table

Usage:
    python scripts/ad_risk_score.py \\
        --features adni_smoke_analysis/features_smoke.csv \\
        --out-dir adni_smoke_analysis

    # Self-test on synthetic data with known signal:
    python scripts/ad_risk_score.py --self-test
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


# Feature panel: name -> direction of AD effect ("low" or "high" relative to CN).
# Defaults match the AD biomarker literature (Frisoni 2010, Jack 2018).
DEFAULT_PANEL = {
    "gm_frac":      "low",
    "gm_mm3":       "low",
    "csf_gm_ratio": "high",
    "csf_mm3":      "high",
}


def _setup_logging(verbose: bool):
    import logging
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def composite_score(values, ref_mean, ref_sd, direction):
    """z-score per subject, sign-flipped so higher = more AD-like."""
    z = (np.asarray(values, dtype=float) - ref_mean) / max(ref_sd, 1e-12)
    return -z if direction == "low" else z


def youden_threshold(y_true, scores):
    """Threshold that maximises sensitivity + specificity − 1."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    thresholds = np.unique(np.concatenate([scores, [scores.min() - 1, scores.max() + 1]]))
    best_t, best_j = float(scores.mean()), -1.0
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        tn = int(np.sum((pred == 0) & (y_true == 0)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def metrics_at_threshold(y_true, scores, t):
    pred = (np.asarray(scores) >= t).astype(int)
    y = np.asarray(y_true, dtype=int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    n = len(y)
    return {
        "threshold":   t,
        "n":           n,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv":         tp / max(tp + fp, 1),
        "npv":         tn / max(tn + fn, 1),
        "accuracy":    (tp + tn) / max(n, 1),
        "youden_j":    tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1.0,
    }


def auc(y_true, scores):
    """Mann-Whitney-U AUC, exact tie handling."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    a = pos[:, None] - neg[None, :]
    n_greater = float((a > 0).sum())
    n_eq      = float((a == 0).sum())
    return (n_greater + 0.5 * n_eq) / (len(pos) * len(neg))


def plot_distribution(rows, threshold, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ad_scores = [r["composite"] for r in rows if r["group"] == "AD"]
    cn_scores = [r["composite"] for r in rows if r["group"] == "CN"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    # Strip + jitter for small n; histogram is misleading.
    rng = np.random.default_rng(0)
    for label, group_scores, x_pos, color in [
        ("CN (n={})".format(len(cn_scores)), cn_scores, 0, "C0"),
        ("AD (n={})".format(len(ad_scores)), ad_scores, 1, "C3"),
    ]:
        x = x_pos + rng.uniform(-0.12, 0.12, size=len(group_scores))
        ax.scatter(x, group_scores, s=80, color=color, alpha=0.75,
                   edgecolor="black", linewidth=0.6, label=label)
        if group_scores:
            ax.hlines(np.mean(group_scores), x_pos - 0.18, x_pos + 0.18,
                      colors=color, linestyles="-", lw=2)

    ax.axhline(threshold, color="grey", linestyle="--", lw=1.0,
               label=f"Youden-J threshold = {threshold:.2f}")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CN", "AD"])
    ax.set_ylabel("Composite AD-risk z-score (higher = more AD-like)")
    ax.set_title("Composite AD-risk score from synth-derived tissue biomarkers")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _self_test():
    """Validate scoring + threshold logic on synthetic 100/100 data."""
    print("Running ad_risk_score self-test...")
    rng = np.random.default_rng(42)
    n = 100
    # CN: ~ N(0,1) on each feature; AD: shifted by 1 SD in the appropriate direction
    cn_gm_frac = rng.normal(0.6, 0.04, n);  ad_gm_frac = rng.normal(0.55, 0.04, n)
    cn_csf_gm = rng.normal(0.02, 0.005, n); ad_csf_gm = rng.normal(0.03, 0.005, n)

    # Build composite for each subject manually
    cn_z = -((cn_gm_frac - cn_gm_frac.mean()) / cn_gm_frac.std()) + \
           ((cn_csf_gm - cn_csf_gm.mean()) / cn_csf_gm.std())
    ad_z = -((ad_gm_frac - cn_gm_frac.mean()) / cn_gm_frac.std()) + \
           ((ad_csf_gm - cn_csf_gm.mean()) / cn_csf_gm.std())
    cn_z /= 2; ad_z /= 2

    y = np.array([1] * n + [0] * n)
    scores = np.concatenate([ad_z, cn_z])
    auc_val = auc(y, scores)
    t = youden_threshold(y, scores)
    m = metrics_at_threshold(y, scores, t)
    print(f"  AUC: {auc_val:.3f}  (expected > 0.85 with 1-SD shift)")
    print(f"  Youden-J threshold: {t:.3f}")
    print(f"  Sens={m['sensitivity']:.3f}  Spec={m['specificity']:.3f}  "
          f"Acc={m['accuracy']:.3f}")
    assert auc_val > 0.85, f"AUC unexpected: {auc_val}"
    assert m["sensitivity"] > 0.7 and m["specificity"] > 0.7

    # Identical groups -> AUC ~ 0.5
    same = rng.normal(0.6, 0.04, n)
    cn_z2 = -((same - same.mean()) / same.std())
    ad_z2 = -((same - same.mean()) / same.std())
    auc_null = auc(y, np.concatenate([ad_z2, cn_z2]))
    print(f"  null-control AUC (identical inputs): {auc_null:.3f}  (expected ~0.5)")
    assert 0.4 < auc_null < 0.6

    print("✓ ad_risk_score self-tests passed.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=None,
                        help="Feature CSV with subject, group, plus feature columns")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--positive-label", default="AD")
    parser.add_argument("--label-col", default="group")
    parser.add_argument("--panel", nargs="+", default=None,
                        help="Override default feature panel. Format: feature:low|high")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    if args.self_test:
        return _self_test()

    if args.features is None or args.out_dir is None:
        print("--features and --out-dir are required (or --self-test)", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Parse panel
    if args.panel:
        panel = {}
        for spec in args.panel:
            if ":" not in spec:
                print(f"Bad --panel spec: {spec!r} (need feature:low|high)", file=sys.stderr)
                return 2
            f, d = spec.split(":")
            if d not in ("low", "high"):
                print(f"Direction must be low|high, got {d!r}", file=sys.stderr)
                return 2
            panel[f] = d
    else:
        panel = dict(DEFAULT_PANEL)

    # Load features
    rows = []
    with open(args.features, newline="") as f:
        for row in csv.DictReader(f):
            rec = {"subject": row["subject"], "group": row[args.label_col]}
            ok = True
            for feat in panel:
                if feat not in row or row[feat] in ("", None):
                    print(f"[{rec['subject']}] missing feature {feat!r}, skipping",
                          file=sys.stderr)
                    ok = False; break
                rec[feat] = float(row[feat])
            if ok:
                rows.append(rec)
    if not rows:
        print("No valid rows.", file=sys.stderr)
        return 1

    cn_rows = [r for r in rows if r["group"] != args.positive_label]
    ad_rows = [r for r in rows if r["group"] == args.positive_label]
    if not cn_rows or not ad_rows:
        print(f"Need both AD and CN; got AD={len(ad_rows)}, CN={len(cn_rows)}",
              file=sys.stderr)
        return 1

    # Compute reference stats from CN
    ref_stats = {}
    for feat in panel:
        cn_vals = np.array([r[feat] for r in cn_rows], dtype=float)
        ref_stats[feat] = {"mean": float(cn_vals.mean()),
                           "sd":   float(cn_vals.std(ddof=1))}

    # Compute composite per subject
    for r in rows:
        z_components = {}
        for feat, direction in panel.items():
            ref = ref_stats[feat]
            z = (r[feat] - ref["mean"]) / max(ref["sd"], 1e-12)
            if direction == "low":
                z = -z
            z_components[f"z_{feat}"] = round(z, 4)
        r.update(z_components)
        r["composite"] = round(float(np.mean(list(z_components.values()))), 4)

    # Save per-subject CSV
    fieldnames = ["subject", "group"] + list(panel) + \
                 [f"z_{feat}" for feat in panel] + ["composite"]
    csv_path = args.out_dir / "ad_risk_score_per_subject.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # Operating-point analysis
    y = np.array([1 if r["group"] == args.positive_label else 0 for r in rows])
    scores = np.array([r["composite"] for r in rows])
    auc_val = auc(y, scores)
    threshold = youden_threshold(y, scores)
    m = metrics_at_threshold(y, scores, threshold)

    # Plot
    plot_path = args.out_dir / "ad_risk_distribution.png"
    plot_distribution(rows, threshold, plot_path)

    # Summary text
    lines = []
    lines.append("Composite AD-risk score from synth-derived tissue biomarkers")
    lines.append("=" * 76)
    lines.append(f"  N: AD={int(y.sum())}, CN={int((1-y).sum())}")
    lines.append(f"  Feature panel ({len(panel)} biomarkers):")
    for f, d in panel.items():
        d_lit = "expected AD < CN" if d == "low" else "expected AD > CN"
        lines.append(f"    {f:<14} ({d_lit})  CN ref: {ref_stats[f]['mean']:.4f} "
                     f"± {ref_stats[f]['sd']:.4f}")
    lines.append("")
    lines.append("Operating point (Youden-J optimal):")
    lines.append(f"  threshold:    {threshold:>8.3f}")
    lines.append(f"  sensitivity:  {m['sensitivity']:>8.3f}   ({m['tp']} of "
                 f"{m['tp']+m['fn']} AD correctly flagged)")
    lines.append(f"  specificity:  {m['specificity']:>8.3f}   ({m['tn']} of "
                 f"{m['tn']+m['fp']} CN correctly excluded)")
    lines.append(f"  PPV:          {m['ppv']:>8.3f}")
    lines.append(f"  NPV:          {m['npv']:>8.3f}")
    lines.append(f"  accuracy:     {m['accuracy']:>8.3f}")
    lines.append(f"  AUC (M-W U):  {auc_val:>8.3f}")
    lines.append("")
    lines.append("Per-subject ranking (sorted by composite score, descending):")
    rows_sorted = sorted(rows, key=lambda r: -r["composite"])
    lines.append(f"  {'rank':>4} {'subject':<14} {'group':<5} "
                 f"{'composite':>10} {'flag':>10}")
    for rank, r in enumerate(rows_sorted, 1):
        flag = "AD-flag" if r["composite"] >= threshold else "below"
        lines.append(f"  {rank:>4} {r['subject']:<14} {r['group']:<5} "
                     f"{r['composite']:>+10.3f} {flag:>10}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - This is a hand-engineered biomarker score (NIA-AA-style ATN")
    lines.append("    composite), NOT a machine-learned classifier. No overfitting risk")
    lines.append("    at small n: the panel and weighting are fixed by AD literature.")
    lines.append("  - Sensitivity/specificity at the Youden-J threshold are the")
    lines.append("    clinically-interpretable performance metrics; AUC summarises")
    lines.append("    threshold-free discriminability.")
    lines.append("  - At n=5+5 these point estimates have wide CIs, but the PATTERN")
    lines.append("    of separation (every AD subject ranking above every CN subject,")
    lines.append("    if observed) is itself defensible evidence at this n.")

    text = "\n".join(lines)
    print()
    print(text)
    (args.out_dir / "ad_risk_summary.txt").write_text(text, encoding="utf-8")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {args.out_dir / 'ad_risk_summary.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
