"""
Paired comparison of AUCs across the four AD classifier configurations
(C1=3T-only, C2=synth-7T, C3=fusion, C4=synth-7T-ablation) — Week 6.

Consumes 2-N predictions.csv files (one per configuration) produced by
train_ad_classifier.py and runs **DeLong's paired test** between every
pair to test whether their ROC AUCs differ significantly.

Why DeLong's: when two classifiers are evaluated on the same set of
subjects (as is the case here — same LOOCV/stratified-k-fold split for
each configuration), their AUCs are correlated. Treating them as
independent (e.g., comparing CIs by overlap) overestimates the variance.
DeLong's accounts for the within-subject correlation via the U-statistic
covariance.

Reference implementation: Sun & Xu (2014), "Fast implementation of
DeLong's algorithm for comparing the areas under correlated receiver
operating characteristic curves." IEEE Signal Processing Letters 21(11),
1389-1393.

Usage:
    # Compare all four configurations (RF only)
    python scripts/compare_classifiers.py \\
        --predictions C1=results/ad/3t_only/predictions.csv \\
                      C2=results/ad/synth_7t/predictions.csv \\
                      C3=results/ad/fusion/predictions.csv \\
                      C4=results/ad/synth_7t_ablation/predictions.csv \\
        --classifier rf \\
        --output-dir results/ad/comparison/

    # Validate the implementation:
    python scripts/compare_classifiers.py --self-test

Outputs (in --output-dir):
    auc_summary.csv      one row per configuration: AUC + CI + acc/sens/spec
    pairwise_pvalues.csv pairs A/B with delta_AUC, raw p, corrected p, sig?
    roc_overlay.png      one curve per configuration with AUC in legend
    auc_bar.png          bar chart of AUCs with 95% CI error bars
"""
import argparse
import csv
import json
import logging
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# DeLong's algorithm — fast version (Sun & Xu 2014)
# ---------------------------------------------------------------------------

def _midrank(x: np.ndarray) -> np.ndarray:
    """Mid-rank assignment for the U-statistic (handles ties properly)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    n = len(x)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and x[order[j + 1]] == x[order[i]]:
            j += 1
        # Mid-rank for ties: average of ranks (i+1) ... (j+1)
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _delong_components(y_true: np.ndarray, scores: np.ndarray
                       ) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute AUC and the V_10 (positive) and V_01 (negative) component vectors.

    These are the structural components used to build the AUC covariance
    matrix; DeLong & DeLong 1988 equation (5).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    m, n = len(pos_idx), len(neg_idx)
    if m == 0 or n == 0:
        return float("nan"), np.array([]), np.array([])

    # Mid-ranks within positives, within negatives, and within all
    tx = _midrank(scores[pos_idx])
    ty = _midrank(scores[neg_idx])
    tz = _midrank(scores)

    # AUC via combined ranks: (sum of ranks of positives - m*(m+1)/2) / (m*n)
    auc = (np.sum(tz[pos_idx]) - m * (m + 1) / 2.0) / (m * n)

    # V_10 (positive component) and V_01 (negative component)
    v10 = (tz[pos_idx] - tx) / n
    v01 = 1.0 - (tz[neg_idx] - ty) / m
    return float(auc), v10, v01


def delong_test(y_true: np.ndarray,
                scores_a: np.ndarray, scores_b: np.ndarray
                ) -> Tuple[float, float, float, float]:
    """Paired DeLong's test for two AUCs on the same labels.

    Returns (auc_a, auc_b, z_stat, two_sided_p).
    """
    auc_a, v10a, v01a = _delong_components(y_true, scores_a)
    auc_b, v10b, v01b = _delong_components(y_true, scores_b)
    if math.isnan(auc_a) or math.isnan(auc_b):
        return auc_a, auc_b, float("nan"), float("nan")

    m, n = len(v10a), len(v01a)

    # Stack into a 2 x N(positives or negatives) matrix and compute covariance.
    V10 = np.vstack([v10a, v10b])  # shape (2, m)
    V01 = np.vstack([v01a, v01b])  # shape (2, n)

    # Sample covariance matrices (biased by 1/(m-1) and 1/(n-1) respectively)
    # then divide by m and n: standard DeLong variance.
    if m < 2 or n < 2:
        return auc_a, auc_b, float("nan"), float("nan")
    S10 = np.cov(V10, ddof=1) / m
    S01 = np.cov(V01, ddof=1) / n
    cov = S10 + S01

    diff = auc_a - auc_b
    var_diff = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var_diff <= 0:
        # Identical scores or all-tied case -> no signal -> p=1
        return auc_a, auc_b, 0.0, 1.0
    z = diff / math.sqrt(var_diff)
    # Two-sided p via standard-normal survival function.
    # math.erf is in stdlib so no scipy dependency.
    p_two = math.erfc(abs(z) / math.sqrt(2))
    return auc_a, auc_b, float(z), float(p_two)


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------

def adjust_pvalues(p_values: List[float], method: str) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if method == "none":
        return list(p)
    if method == "bonferroni":
        return list(np.minimum(p * n, 1.0))
    if method == "holm":
        # Holm-Bonferroni: sort ascending, multiply by n - i, enforce monotone.
        order = np.argsort(p)
        adj = np.empty_like(p)
        running = 0.0
        for rank, idx in enumerate(order):
            adjusted = (n - rank) * p[idx]
            running = max(running, adjusted)
            adj[idx] = min(running, 1.0)
        return list(adj)
    raise ValueError(f"Unknown correction method: {method}")


# ---------------------------------------------------------------------------
# Predictions loader
# ---------------------------------------------------------------------------

def load_predictions(path: Path, classifier: str
                     ) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Returns (subjects, y_true, scores) for one configuration."""
    import pandas as pd
    df = pd.read_csv(path)
    expected = {"subject", "group_positive", "classifier", "proba_positive"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    sub = df[df["classifier"] == classifier].copy()
    if sub.empty:
        raise ValueError(f"{path}: no rows with classifier='{classifier}'. "
                         f"Available: {sorted(df['classifier'].unique())}")
    sub = sub.sort_values("subject").reset_index(drop=True)
    return (sub["subject"].astype(str).tolist(),
            sub["group_positive"].to_numpy(int),
            sub["proba_positive"].to_numpy(float))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_roc_overlay(by_config: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
                     out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, (y_true, scores, auc) in by_config.items():
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, lw=1.8, label=f"{name}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("1 − Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("ROC — AD vs CN, paired stratified k-fold CV")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_auc_bar(summary_rows: List[Dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["config"] for r in summary_rows]
    aucs = [r["auc"] for r in summary_rows]
    err_lo = [max(r["auc"] - r["auc_ci_lo"], 0) for r in summary_rows]
    err_hi = [max(r["auc_ci_hi"] - r["auc"], 0) for r in summary_rows]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(names, aucs, yerr=[err_lo, err_hi], capsize=6,
                  color=["C0", "C1", "C2", "C3", "C4", "C5"][:len(names)],
                  alpha=0.85, edgecolor="black", linewidth=0.7)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{auc:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("AUC (95% CI)")
    ax.set_ylim(0.45, 1.0)
    ax.axhline(0.5, ls=":", c="grey", alpha=0.5)
    ax.set_title("AD vs CN classification AUC across configurations")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    """Validate DeLong's against three reference cases:
       (1) identical scores -> p = 1.0
       (2) one classifier dominates -> p < 0.001
       (3) two equally-good but uncorrelated classifiers -> 0.05 < p < 0.95
    """
    print("Running compare_classifiers self-test (DeLong's algorithm)...")
    rng = np.random.default_rng(42)
    n_pos, n_neg = 200, 200
    y = np.array([1] * n_pos + [0] * n_neg)

    # Case 1: identical predictions
    s1 = np.concatenate([rng.normal(1, 1, n_pos), rng.normal(0, 1, n_neg)])
    a1, a2, z1, p1 = delong_test(y, s1, s1)
    print(f"  identical:        AUC={a1:.3f}  z={z1:.3f}  p={p1:.3f}  (expected p=1)")
    assert math.isclose(a1, a2)
    assert p1 > 0.999, f"identical p={p1}"

    # Case 2: dominance (strong signal vs noise)
    strong = np.concatenate([rng.normal(2, 1, n_pos), rng.normal(0, 1, n_neg)])
    weak = rng.normal(0, 1, n_pos + n_neg)
    aS, aW, zSW, pSW = delong_test(y, strong, weak)
    print(f"  strong vs weak:   AUC_strong={aS:.3f}, AUC_weak={aW:.3f}  "
          f"z={zSW:.2f}  p={pSW:.4g}  (expected p<<0.001)")
    assert aS > 0.85, f"strong AUC unexpected: {aS}"
    assert 0.4 < aW < 0.6, f"weak AUC unexpected: {aW}"
    assert pSW < 0.001, f"strong-vs-weak p unexpected: {pSW}"

    # Case 3: two same-strength uncorrelated classifiers (should not differ)
    sa = np.concatenate([rng.normal(1, 1, n_pos), rng.normal(0, 1, n_neg)])
    sb = np.concatenate([rng.normal(1, 1, n_pos), rng.normal(0, 1, n_neg)])
    aA, aB, zAB, pAB = delong_test(y, sa, sb)
    print(f"  two equal-AUC:    AUC_A={aA:.3f}, AUC_B={aB:.3f}  "
          f"z={zAB:.2f}  p={pAB:.3f}  (expected 0.05 < p < 0.95)")
    assert 0.05 < pAB < 0.95, f"equal-AUC p unexpected: {pAB}"

    # Multiple comparison correction sanity
    raw = [0.001, 0.04, 0.5]
    bonf = adjust_pvalues(raw, "bonferroni")
    holm = adjust_pvalues(raw, "holm")
    print(f"  bonferroni({raw}) = {[round(x, 4) for x in bonf]}  (expect 0.003, 0.12, 1.0)")
    print(f"  holm({raw})       = {[round(x, 4) for x in holm]}  (expect 0.003, 0.08, 0.5)")
    assert math.isclose(bonf[0], 0.003)
    assert math.isclose(bonf[1], 0.12)
    assert bonf[2] == 1.0
    assert math.isclose(holm[0], 0.003)
    assert math.isclose(holm[1], 0.08)
    assert math.isclose(holm[2], 0.5)

    print("✓ All compare_classifiers self-tests passed.")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_predictions_arg(items: List[str]) -> List[Tuple[str, Path]]:
    """Parse 'name=path' tokens into (name, Path) pairs."""
    out = []
    for it in items:
        if "=" not in it:
            raise argparse.ArgumentTypeError(
                f"--predictions takes name=path tokens, got: {it!r}")
        name, _, path = it.partition("=")
        out.append((name, Path(path)))
    return out


def auc_bootstrap_ci(y_true: np.ndarray, scores: np.ndarray,
                     n_boot: int, alpha: float, rng) -> Tuple[float, float]:
    n = len(y_true)
    if n < 4 or len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for _ in range(8):
            if len(np.unique(y_true[idx])) >= 2:
                break
            idx = rng.integers(0, n, size=n)
        a, _, _ = _delong_components(y_true[idx], scores[idx])
        aucs[b] = a
    aucs = aucs[~np.isnan(aucs)]
    return float(np.quantile(aucs, alpha / 2)), float(np.quantile(aucs, 1 - alpha / 2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--predictions", nargs="+", default=None,
                        help="name=path tokens, e.g. C1=results/.../predictions.csv")
    parser.add_argument("--classifier", default="rf",
                        help="Which classifier rows to read from each predictions CSV "
                             "(default: rf — match train_ad_classifier.py output)")
    parser.add_argument("--correction", choices=["bonferroni", "holm", "none"],
                        default="bonferroni",
                        help="Multiple-comparison correction over pairwise p-values")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance threshold (default 0.05)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Bootstrap iterations for AUC CI (default 1000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    if args.self_test:
        return _self_test()

    if args.predictions is None or args.output_dir is None:
        logging.error("--predictions and --output-dir are required (or --self-test)")
        return 2

    items = parse_predictions_arg(args.predictions)
    if len(items) < 2:
        logging.error("Need at least 2 predictions files to compare")
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- load and align ----
    by_config: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]] = {}
    for name, p in items:
        if not p.exists():
            logging.error("predictions file not found: %s", p)
            return 2
        subjects, y_true, scores = load_predictions(p, args.classifier)
        by_config[name] = (subjects, y_true, scores)
        logging.info("loaded %s: n=%d, AD=%d, CN=%d", name,
                     len(subjects), int(y_true.sum()), int((1 - y_true).sum()))

    # All configs must share the same subject list (paired comparison).
    ref_name = items[0][0]
    ref_subjects = by_config[ref_name][0]
    for name, (subj, _, _) in by_config.items():
        if subj != ref_subjects:
            logging.error(
                "Subject mismatch between %s and %s — DeLong's requires identical "
                "subject sets across configurations. Re-run train_ad_classifier.py "
                "with the same --seed and --n-folds for each configuration.",
                ref_name, name)
            return 2
    y_true = by_config[ref_name][1]

    # ---- per-config AUC + CI + accuracy/sens/spec ----
    rng = np.random.default_rng(args.seed)
    summary_rows: List[Dict] = []
    overlay_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
    for name, (_, _, scores) in by_config.items():
        auc, _, _ = _delong_components(y_true, scores)
        ci_lo, ci_hi = auc_bootstrap_ci(y_true, scores,
                                        n_boot=args.n_boot, alpha=args.alpha,
                                        rng=rng)
        # Operating point: Youden-J on training-side rough sweep.
        from sklearn.metrics import roc_curve
        fpr, tpr, thr = roc_curve(y_true, scores)
        j = tpr - fpr
        thr_op = float(thr[int(np.argmax(j))])
        pred = (scores >= thr_op).astype(int)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        tn = int(np.sum((pred == 0) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        acc = (tp + tn) / max(tp + fp + tn + fn, 1)
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)

        row = {"config": name, "n": int(len(y_true)),
               "auc": round(auc, 4),
               "auc_ci_lo": round(ci_lo, 4), "auc_ci_hi": round(ci_hi, 4),
               "accuracy": round(acc, 4),
               "sensitivity": round(sens, 4),
               "specificity": round(spec, 4),
               "threshold": round(thr_op, 4)}
        summary_rows.append(row)
        overlay_data[name] = (y_true, scores, auc)

    # ---- pairwise DeLong's ----
    pair_rows: List[Dict] = []
    raw_p: List[float] = []
    for (a_name, _), (b_name, _) in combinations(items, 2):
        sa = by_config[a_name][2]
        sb = by_config[b_name][2]
        auc_a, auc_b, z, p = delong_test(y_true, sa, sb)
        raw_p.append(p)
        pair_rows.append({
            "config_a": a_name, "config_b": b_name,
            "auc_a": round(auc_a, 4), "auc_b": round(auc_b, 4),
            "delta_auc": round(auc_a - auc_b, 4),
            "z": round(z, 4),
            "p_raw": p,
        })
    adj = adjust_pvalues(raw_p, args.correction)
    for r, p_corr in zip(pair_rows, adj):
        r["p_corrected"] = round(float(p_corr), 6)
        r["correction"] = args.correction
        r["significant"] = bool(p_corr < args.alpha)

    # ---- write outputs ----
    summary_csv = args.output_dir / "auc_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    logging.info("Wrote %s", summary_csv)

    pairs_csv = args.output_dir / "pairwise_pvalues.csv"
    with open(pairs_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)
    logging.info("Wrote %s", pairs_csv)

    plot_roc_overlay(overlay_data, args.output_dir / "roc_overlay.png")
    logging.info("Wrote %s", args.output_dir / "roc_overlay.png")
    plot_auc_bar(summary_rows, args.output_dir / "auc_bar.png")
    logging.info("Wrote %s", args.output_dir / "auc_bar.png")

    # ---- print summary ----
    print(f"\n{'=' * 80}")
    print(f"  AD vs CN — paired AUC comparison (classifier='{args.classifier}', "
          f"n={summary_rows[0]['n']}, correction={args.correction})")
    print(f"{'=' * 80}")
    print(f"  {'Config':<10} {'AUC':>6} {'95% CI':>17} {'Acc':>6} {'Sens':>6} {'Spec':>6}")
    print(f"  {'-' * 60}")
    for r in summary_rows:
        ci = f"[{r['auc_ci_lo']:.3f}, {r['auc_ci_hi']:.3f}]"
        print(f"  {r['config']:<10} {r['auc']:>6.3f} {ci:>17} "
              f"{r['accuracy']:>6.3f} {r['sensitivity']:>6.3f} {r['specificity']:>6.3f}")

    print(f"\n  Pairwise DeLong's test (corrected = {args.correction}, alpha = {args.alpha})")
    print(f"  {'A vs B':<14} {'ΔAUC':>8} {'z':>7} {'p_raw':>10} {'p_corr':>10} {'sig?':>5}")
    print(f"  {'-' * 62}")
    for r in pair_rows:
        sig = "✓" if r["significant"] else " "
        print(f"  {r['config_a'] + ' vs ' + r['config_b']:<14} "
              f"{r['delta_auc']:>+8.3f} {r['z']:>7.2f} "
              f"{r['p_raw']:>10.4g} {r['p_corrected']:>10.4g} {sig:>5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
