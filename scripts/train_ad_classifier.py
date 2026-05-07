"""
Train and evaluate AD vs CN classifiers from a feature CSV (Week 6).

The thesis runs four configurations:
    C1 = 3T-only           features from real 3T (FreeSurfer/FastSurfer on real 3T)
    C2 = synth-7T          features from synthetic 7T (FastSurfer on model output)
    C3 = fusion            concatenated C1 + C2 features
    C4 = synth-7T-ablation features from the no-topology-loss ablation model

This script trains and evaluates ONE configuration at a time. Run it four
times with different --features CSVs (or different --feature-cols subsets
of the same CSV), then feed the four `predictions.csv` outputs into
`compare_classifiers.py` (next iteration) for paired DeLong AUC tests.

Inputs (feature CSV columns):
    subject  : subject ID (e.g. 002_S_5018 or sub-06)
    group    : "AD" or "CN" (configurable via --label-col / --positive-label)
    [age, sex]    optional covariates; included only with --include-covariates
    <feature columns>  any number of numeric columns; subset with --feature-cols
                       or --feature-pattern (regex), else all numeric columns
                       except subject / label / covariates are used.

Outputs (in --output-dir):
    predictions.csv     subject, group, fold, classifier, proba_positive
    auc_summary.json    per-classifier AUC + bootstrap 95% CI + accuracy / sens / spec
    roc.png             ROC curve overlay across classifiers
    feature_importance.csv  (RF only) — Gini importance per feature

Reproducibility: --seed controls fold split, classifier RNG, and bootstrap
sampling. Identical seeds produce identical numbers across runs.

Usage:
    # Real run on a feature CSV with hippocampus, amygdala, etc. columns:
    python scripts/train_ad_classifier.py \\
        --features features_synth7t.csv \\
        --output-dir results/ad/synth7t

    # Restrict to subcortical volume features only:
    python scripts/train_ad_classifier.py \\
        --features features_synth7t.csv \\
        --feature-pattern '^(hippo|amygdala|thalamus|caudate|putamen|pallidum)_' \\
        --output-dir results/ad/synth7t_subcortical

    # Validate the pipeline against synthetic data:
    python scripts/train_ad_classifier.py --self-test
"""
import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class ClassifierResult:
    classifier: str
    n: int
    auc: float
    auc_ci_lo: float
    auc_ci_hi: float
    accuracy: float
    sensitivity: float
    specificity: float
    threshold: float


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC with safe handling of degenerate cases."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def bootstrap_auc_ci(y_true: np.ndarray, scores: np.ndarray,
                     n_boot: int = 1000, alpha: float = 0.05,
                     rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    """Percentile bootstrap 100*(1-alpha)% CI for ROC AUC."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(y_true)
    if n < 4 or len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")

    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # Resample until we have both classes (rejection)
        for _ in range(8):
            if len(np.unique(y_true[idx])) >= 2:
                break
            idx = rng.integers(0, n, size=n)
        aucs[b] = _safe_auc(y_true[idx], scores[idx])
    aucs = aucs[~np.isnan(aucs)]
    if aucs.size == 0:
        return float("nan"), float("nan")
    lo = float(np.quantile(aucs, alpha / 2))
    hi = float(np.quantile(aucs, 1 - alpha / 2))
    return lo, hi


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Return the threshold that maximises Youden's J = sensitivity + specificity − 1."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def confusion_metrics(y_true: np.ndarray, scores: np.ndarray,
                      threshold: float) -> Tuple[float, float, float]:
    """Returns (accuracy, sensitivity, specificity) at the given threshold."""
    pred = (scores >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return accuracy, sens, spec


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_features(
    features_csv: Path,
    label_col: str,
    positive_label: str,
    feature_cols: Optional[List[str]],
    feature_pattern: Optional[str],
    include_covariates: bool,
    covariate_cols: Sequence[str] = ("age", "sex"),
):
    """Load feature CSV and return (X, y, subjects, feature_names).

    - y is binary: 1 if `positive_label` else 0
    - X is numeric only; non-numeric / missing rows are dropped (with a warning)
    - feature_cols (explicit) takes precedence over feature_pattern (regex)
    """
    import pandas as pd

    df = pd.read_csv(features_csv)
    if "subject" not in df.columns:
        raise ValueError(f"features CSV missing 'subject' column: {features_csv}")
    if label_col not in df.columns:
        raise ValueError(f"features CSV missing label column '{label_col}'")

    # Drop unlabeled rows
    n_pre = len(df)
    df = df.dropna(subset=[label_col]).copy()
    if len(df) < n_pre:
        logging.info("Dropped %d rows with missing label", n_pre - len(df))

    # Pick feature columns
    excluded = {"subject", label_col} | set(covariate_cols)
    if feature_cols:
        for col in feature_cols:
            if col not in df.columns:
                raise ValueError(f"--feature-cols includes '{col}' which is not in CSV")
        feats = list(feature_cols)
    elif feature_pattern:
        rx = re.compile(feature_pattern)
        feats = [c for c in df.columns if c not in excluded and rx.search(c)]
        if not feats:
            raise ValueError(f"--feature-pattern '{feature_pattern}' matched no columns")
    else:
        # All numeric columns except excluded
        numeric = df.select_dtypes(include=[np.number]).columns
        feats = [c for c in numeric if c not in excluded]
        if not feats:
            raise ValueError(f"No numeric feature columns found in {features_csv}")

    if include_covariates:
        for c in covariate_cols:
            if c in df.columns and c not in feats:
                feats.append(c)

    # Encode sex if present in features
    if "sex" in feats and df["sex"].dtype == object:
        df["sex"] = df["sex"].map({"M": 1, "F": 0}).fillna(np.nan)

    # Drop rows with any NaN feature
    df_feat = df[feats]
    n_pre = len(df_feat)
    df = df.loc[~df_feat.isna().any(axis=1)].copy()
    n_dropped = n_pre - len(df)
    if n_dropped:
        logging.warning("Dropped %d rows with missing features", n_dropped)

    if len(df) == 0:
        raise ValueError("No usable rows after filtering")

    y = (df[label_col].astype(str) == positive_label).astype(int).to_numpy()
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Need both classes: got {n_pos} '{positive_label}' and {n_neg} other"
        )

    X = df[feats].to_numpy(dtype=np.float64)
    subjects = df["subject"].astype(str).tolist()
    logging.info("Loaded features: n=%d, AD=%d, CN=%d, features=%d",
                 len(df), n_pos, n_neg, len(feats))
    return X, y, subjects, feats


# ---------------------------------------------------------------------------
# Training & cross-validation
# ---------------------------------------------------------------------------

def build_classifier(name: str, seed: int):
    """Returns an sklearn Pipeline for the given classifier name."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                 max_iter=2000, random_state=seed)
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=500, max_depth=None,
                                     n_jobs=-1, random_state=seed)
        # No scaling needed for RF; keep StandardScaler off the pipeline
        return Pipeline([("clf", clf)])
    raise ValueError(f"Unknown classifier '{name}' (choose logreg or rf)")


def cross_validate_classifier(
    name: str, X: np.ndarray, y: np.ndarray, subjects: List[str],
    n_folds: int, seed: int,
) -> Tuple[List[dict], object]:
    """Stratified k-fold CV. Returns (per-fold predictions, last fitted model)."""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    last_model = None
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        clf = build_classifier(name, seed=seed + fold_idx)
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        for j, ti in enumerate(test_idx):
            rows.append({
                "subject":         subjects[ti],
                "group_positive":  int(y[ti]),
                "fold":            fold_idx,
                "classifier":      name,
                "proba_positive":  float(proba[j]),
            })
        last_model = clf
    return rows, last_model


def summarise(name: str, rows: List[dict], rng: np.random.Generator) -> ClassifierResult:
    y_true = np.array([r["group_positive"] for r in rows], dtype=int)
    scores = np.array([r["proba_positive"] for r in rows], dtype=float)
    auc = _safe_auc(y_true, scores)
    lo, hi = bootstrap_auc_ci(y_true, scores, rng=rng)
    thr = youden_threshold(y_true, scores) if not np.isnan(auc) else float("nan")
    acc, sens, spec = confusion_metrics(y_true, scores, thr)
    return ClassifierResult(
        classifier=name, n=int(len(y_true)),
        auc=round(auc, 4),
        auc_ci_lo=round(lo, 4), auc_ci_hi=round(hi, 4),
        accuracy=round(acc, 4), sensitivity=round(sens, 4),
        specificity=round(spec, 4),
        threshold=round(thr, 4),
    )


def feature_importance_for_rf(model, feature_names: List[str]) -> List[dict]:
    rf = model.named_steps.get("clf")
    if rf is None or not hasattr(rf, "feature_importances_"):
        return []
    imp = rf.feature_importances_
    pairs = sorted(zip(feature_names, imp), key=lambda p: -p[1])
    return [{"feature": f, "importance": round(float(v), 6)} for f, v in pairs]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_roc_overlay(rows_per_classifier: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, rows in rows_per_classifier.items():
        y_true = np.array([r["group_positive"] for r in rows], dtype=int)
        scores = np.array([r["proba_positive"] for r in rows], dtype=float)
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = _safe_auc(y_true, scores)
        ax.plot(fpr, tpr, lw=1.8, label=f"{name}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("1 − Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("ROC — AD vs CN, stratified k-fold CV")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    """Validate the pipeline on synthetic data with a known signal-to-noise ratio.

    200 subjects (100 AD, 100 CN). 10 signal features differ between groups
    by 1 SD; 40 noise features are pure noise. Both classifiers should
    achieve AUC > 0.85; pure-noise control should fail.
    """
    print("Running train_ad_classifier self-test...")
    rng = np.random.default_rng(42)
    n_ad, n_cn = 100, 100
    n_signal, n_noise = 10, 40
    n = n_ad + n_cn

    X = np.empty((n, n_signal + n_noise))
    X[:n_ad, :n_signal] = rng.normal(loc=1.0, scale=1.0, size=(n_ad, n_signal))
    X[n_ad:, :n_signal] = rng.normal(loc=0.0, scale=1.0, size=(n_cn, n_signal))
    X[:, n_signal:] = rng.normal(size=(n, n_noise))

    y = np.array([1] * n_ad + [0] * n_cn, dtype=int)
    subjects = [f"sub-{i:03d}" for i in range(n)]

    boot_rng = np.random.default_rng(0)
    print(f"  n=200 (100 AD, 100 CN), {n_signal} signal + {n_noise} noise features")
    for clf_name in ("logreg", "rf"):
        rows, _ = cross_validate_classifier(
            clf_name, X, y, subjects, n_folds=5, seed=42)
        result = summarise(clf_name, rows, rng=boot_rng)
        print(f"  {clf_name:>7}  AUC={result.auc:.3f} [{result.auc_ci_lo:.3f}, "
              f"{result.auc_ci_hi:.3f}]  acc={result.accuracy:.3f}  "
              f"sens={result.sensitivity:.3f}  spec={result.specificity:.3f}")
        assert result.auc > 0.85, f"{clf_name} AUC too low: {result.auc}"

    # Negative control: pure noise -> AUC ~ 0.5
    Xn = rng.normal(size=(n, n_noise))
    rows_n, _ = cross_validate_classifier("logreg", Xn, y, subjects, 5, 42)
    aucn = _safe_auc(np.array([r["group_positive"] for r in rows_n]),
                     np.array([r["proba_positive"] for r in rows_n]))
    print(f"  pure-noise control logreg  AUC={aucn:.3f}  (expected ~0.5)")
    assert 0.4 < aucn < 0.6, f"noise AUC unexpected: {aucn}"

    print("✓ All train_ad_classifier self-tests passed.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--features", type=Path,
                        help="Feature CSV with `subject`, label col, and feature cols")
    parser.add_argument("--label-col", default="group",
                        help="Column name with the AD/CN label (default: group)")
    parser.add_argument("--positive-label", default="AD",
                        help="String value treated as positive class (default: AD)")
    parser.add_argument("--feature-cols", nargs="+", default=None,
                        help="Explicit feature column list (overrides --feature-pattern)")
    parser.add_argument("--feature-pattern", default=None,
                        help="Regex matched against column names to select features")
    parser.add_argument("--include-covariates", action="store_true",
                        help="Include age and sex columns as features")
    parser.add_argument("--classifiers", nargs="+", default=["logreg", "rf"],
                        choices=["logreg", "rf"],
                        help="Classifiers to train (default: logreg rf)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write predictions, AUC, ROC, importances")
    parser.add_argument("--self-test", action="store_true",
                        help="Validate the pipeline on synthetic data")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    if args.self_test:
        return _self_test()

    if args.features is None or args.output_dir is None:
        logging.error("--features and --output-dir are required (or use --self-test)")
        return 2

    # Load and validate inputs
    X, y, subjects, feature_names = load_features(
        features_csv=args.features.resolve(),
        label_col=args.label_col,
        positive_label=args.positive_label,
        feature_cols=args.feature_cols,
        feature_pattern=args.feature_pattern,
        include_covariates=args.include_covariates,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    boot_rng = np.random.default_rng(args.seed)

    # Cross-validate each classifier
    all_rows: List[dict] = []
    summary: dict = {"n_subjects": int(len(y)),
                     "n_positive": int(y.sum()),
                     "n_negative": int(len(y) - y.sum()),
                     "feature_count": len(feature_names),
                     "n_folds": args.n_folds,
                     "seed": args.seed,
                     "feature_names": feature_names,
                     "classifiers": {}}
    rows_per_clf: dict = {}
    for clf_name in args.classifiers:
        logging.info("Training %s with %d-fold stratified CV", clf_name, args.n_folds)
        rows, last_model = cross_validate_classifier(
            clf_name, X, y, subjects, n_folds=args.n_folds, seed=args.seed)
        all_rows.extend(rows)
        rows_per_clf[clf_name] = rows
        result = summarise(clf_name, rows, rng=boot_rng)
        summary["classifiers"][clf_name] = asdict(result)
        logging.info("  %s: AUC=%.3f [%.3f, %.3f]  acc=%.3f  sens=%.3f  spec=%.3f",
                     clf_name, result.auc, result.auc_ci_lo, result.auc_ci_hi,
                     result.accuracy, result.sensitivity, result.specificity)

        if clf_name == "rf":
            imp = feature_importance_for_rf(last_model, feature_names)
            if imp:
                imp_path = args.output_dir / "feature_importance.csv"
                with open(imp_path, "w") as f:
                    f.write("feature,importance\n")
                    for r in imp:
                        f.write(f"{r['feature']},{r['importance']}\n")
                logging.info("Wrote %s (top: %s)", imp_path,
                             ", ".join(f"{r['feature']}={r['importance']:.3f}"
                                       for r in imp[:5]))

    # Save predictions
    pred_path = args.output_dir / "predictions.csv"
    with open(pred_path, "w") as f:
        f.write("subject,group_positive,fold,classifier,proba_positive\n")
        for r in all_rows:
            f.write(f"{r['subject']},{r['group_positive']},{r['fold']},"
                    f"{r['classifier']},{r['proba_positive']:.6f}\n")
    logging.info("Wrote %s", pred_path)

    # Save AUC summary
    summary_path = args.output_dir / "auc_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info("Wrote %s", summary_path)

    # ROC overlay
    plot_roc_overlay(rows_per_clf, args.output_dir / "roc.png")
    logging.info("Wrote %s", args.output_dir / "roc.png")

    # Print summary table
    print(f"\n{'=' * 68}")
    print(f"  AD vs CN classification — {len(y)} subjects "
          f"({int(y.sum())} AD, {int(len(y) - y.sum())} CN), "
          f"{len(feature_names)} features, {args.n_folds}-fold CV")
    print(f"{'=' * 68}")
    print(f"  {'Classifier':<12} {'AUC':>6} {'95% CI':>17} {'Acc':>6} {'Sens':>6} {'Spec':>6}")
    print(f"  {'-' * 60}")
    for clf, r in summary["classifiers"].items():
        ci_str = f"[{r['auc_ci_lo']:.3f}, {r['auc_ci_hi']:.3f}]"
        print(f"  {clf:<12} {r['auc']:>6.3f} {ci_str:>17} "
              f"{r['accuracy']:>6.3f} {r['sensitivity']:>6.3f} {r['specificity']:>6.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
