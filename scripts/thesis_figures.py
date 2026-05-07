"""
Generate thesis-ready figures from the existing smoke-test analysis outputs.

Bundles four figures into a single ~30-second offline run:

  1. roc_with_bootstrap_ci.png   — ROC curve with 95% bootstrap CI band
                                   for the composite AD-risk score.
  2. feature_boxplots.png        — per-feature AD vs CN box+strip plots
                                   for the four canonical biomarkers.
  3. cohen_d_bar.png             — bar chart of |Cohen's d| per feature
                                   (with effect-size magnitude annotations).
  4. ranking_compare.png         — side-by-side ranked composite scores
                                   for C1 (3T naive) vs C2 (synth) — visualises
                                   the 100%-sensitivity-operating-point claim.

All figures are matplotlib + Agg backend (headless-safe for CERN/Kaggle).

Usage:
    python scripts/thesis_figures.py \\
        --features-csv adni_smoke_analysis/features_smoke.csv \\
        --c1-vs-c2-csv adni_smoke_analysis/c1_vs_c2_per_subject.csv \\
        --out-dir adni_smoke_analysis/figures
"""
import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Same panel as ad_risk_score.py — keep in sync
PANEL = {
    "gm_frac":      "low",
    "gm_mm3":       "low",
    "csf_gm_ratio": "high",
    "csf_mm3":      "high",
}


def composite_score(df, panel, positive_label="AD"):
    """Return per-subject composite z-score (NIA-AA-style)."""
    cn = df[df["group"] != positive_label]
    refs = {f: (cn[f].mean(), cn[f].std(ddof=1)) for f in panel}
    z = pd.DataFrame(index=df.index)
    for f, direction in panel.items():
        mu, sd = refs[f]
        z[f] = (df[f] - mu) / max(sd, 1e-12)
        if direction == "low":
            z[f] = -z[f]
    return z.mean(axis=1)


def auc_mw(y, scores):
    pos = scores[y == 1]; neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    a = pos[:, None] - neg[None, :]
    return ((a > 0).sum() + 0.5 * (a == 0).sum()) / (len(pos) * len(neg))


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = math.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1))
                  / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / max(s, 1e-12)


# ---------------------------------------------------------------------------
# Figure 1 — ROC with bootstrap CI band
# ---------------------------------------------------------------------------

def fig_roc_with_ci(features_csv, out_path, n_boot=1000):
    df = pd.read_csv(features_csv)
    y = (df["group"] == "AD").astype(int).to_numpy()
    scores = composite_score(df, PANEL).to_numpy()

    # Single ROC curve
    sorted_thresh = np.sort(np.unique(scores))[::-1]
    fprs, tprs = [0.0], [0.0]
    for t in sorted_thresh:
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        fprs.append(fpr); tprs.append(tpr)
    fprs.append(1.0); tprs.append(1.0)

    # Bootstrap ROCs
    rng = np.random.default_rng(0)
    n = len(y)
    grid = np.linspace(0, 1, 101)
    boot_tpr = np.empty((n_boot, len(grid)))
    boot_aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for _ in range(8):
            if len(np.unique(y[idx])) >= 2:
                break
            idx = rng.integers(0, n, size=n)
        yb, sb = y[idx], scores[idx]
        boot_aucs[b] = auc_mw(yb, sb)

        # ROC for this bootstrap sample
        st = np.sort(np.unique(sb))[::-1]
        f_pts, t_pts = [0.0], [0.0]
        for t in st:
            pr = (sb >= t).astype(int)
            tp = int(((pr == 1) & (yb == 1)).sum())
            fn = int(((pr == 0) & (yb == 1)).sum())
            tn = int(((pr == 0) & (yb == 0)).sum())
            fp = int(((pr == 1) & (yb == 0)).sum())
            t_pts.append(tp / max(tp + fn, 1))
            f_pts.append(fp / max(fp + tn, 1))
        f_pts.append(1.0); t_pts.append(1.0)
        boot_tpr[b] = np.interp(grid, sorted(f_pts), [t for _, t in sorted(zip(f_pts, t_pts))])

    lo = np.percentile(boot_tpr, 2.5, axis=0)
    hi = np.percentile(boot_tpr, 97.5, axis=0)
    auc = auc_mw(y, scores)
    auc_lo = float(np.percentile(boot_aucs[~np.isnan(boot_aucs)], 2.5))
    auc_hi = float(np.percentile(boot_aucs[~np.isnan(boot_aucs)], 97.5))

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.fill_between(grid, lo, hi, alpha=0.25, color="C0", label="95% bootstrap CI")
    ax.plot(fprs, tprs, "C0", lw=2.0, label=f"AUC = {auc:.3f}  95% CI [{auc_lo:.2f}, {auc_hi:.2f}]")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("1 − Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("Composite AD-risk score ROC, n=10 ADNI smoke cohort\n(95% bootstrap CI from 1000 resamples)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return auc, auc_lo, auc_hi


# ---------------------------------------------------------------------------
# Figure 2 — per-feature box plots AD vs CN
# ---------------------------------------------------------------------------

def fig_feature_boxplots(features_csv, out_path):
    df = pd.read_csv(features_csv)
    feats = list(PANEL)
    fig, axes = plt.subplots(1, len(feats), figsize=(3.4 * len(feats), 4.0))
    rng = np.random.default_rng(0)
    for ax, f in zip(axes, feats):
        ad_v = df[df["group"] == "AD"][f].to_numpy()
        cn_v = df[df["group"] == "CN"][f].to_numpy()
        # box plot
        bp = ax.boxplot([cn_v, ad_v], positions=[0, 1], widths=0.5,
                        patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], ["C0", "C3"]):
            patch.set_facecolor(color); patch.set_alpha(0.4)
        # jittered points
        for x_pos, vals, color in [(0, cn_v, "C0"), (1, ad_v, "C3")]:
            xs = x_pos + rng.uniform(-0.13, 0.13, size=len(vals))
            ax.scatter(xs, vals, s=45, color=color, edgecolor="black", linewidth=0.5,
                       alpha=0.85, zorder=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["CN", "AD"])
        d = cohen_d(ad_v, cn_v)
        ax.set_title(f"{f}\nCohen's d = {d:+.2f}", fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Per-feature AD vs CN distributions (n = 5 + 5)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Cohen's d bar chart
# ---------------------------------------------------------------------------

def fig_cohen_d_bar(features_csv, out_path):
    df = pd.read_csv(features_csv)
    feats = ["gm_frac", "gm_mm3", "csf_gm_ratio", "wm_mm3", "csf_mm3", "brain_mm3"]
    ds, mags, colors = [], [], []
    for f in feats:
        if f not in df.columns:
            continue
        ad = df[df["group"] == "AD"][f].to_numpy()
        cn = df[df["group"] == "CN"][f].to_numpy()
        d = cohen_d(ad, cn)
        ds.append(abs(d))
        if abs(d) >= 0.8: mags.append("large"); colors.append("C3")
        elif abs(d) >= 0.5: mags.append("medium"); colors.append("C1")
        elif abs(d) >= 0.2: mags.append("small"); colors.append("C0")
        else: mags.append("trivial"); colors.append("grey")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(feats))
    bars = ax.bar(x, ds, color=colors, edgecolor="black", linewidth=0.7)
    for bar, d, mag in zip(bars, ds, mags):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"|d|={d:.2f}\n({mag})", ha="center", va="bottom", fontsize=8.5)
    ax.axhline(0.2, color="grey", lw=0.7, ls=":", alpha=0.7)
    ax.axhline(0.5, color="C1", lw=0.7, ls=":", alpha=0.7)
    ax.axhline(0.8, color="C3", lw=0.7, ls=":", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(feats, rotation=20, ha="right")
    ax.set_ylabel("|Cohen's d|  (AD vs CN)")
    ax.set_ylim(0, max(ds) * 1.25)
    ax.set_title("Per-feature effect-size magnitude on the smoke cohort\n"
                 "(Cohen's convention: 0.2/0.5/0.8 = small/medium/large)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — ranking comparison C1 vs C2
# ---------------------------------------------------------------------------

def fig_ranking_compare(c1_vs_c2_csv, out_path):
    df = pd.read_csv(c1_vs_c2_csv)
    feats = list(PANEL)

    def composite(prefix):
        cn = df[df["group"] == "CN"]
        z = pd.DataFrame(index=df.index)
        for f, direction in PANEL.items():
            col = f"{f}_{prefix}"
            mu, sd = cn[col].mean(), cn[col].std(ddof=1)
            z[f] = (df[col] - mu) / max(sd, 1e-12)
            if direction == "low":
                z[f] = -z[f]
        return z.mean(axis=1)

    df["composite_C1"] = composite("C1")
    df["composite_C2"] = composite("C2")

    df_sorted_c1 = df.sort_values("composite_C1", ascending=False).reset_index(drop=True)
    df_sorted_c2 = df.sort_values("composite_C2", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True)
    for ax, dfx, label, col in [(axes[0], df_sorted_c1, "C1: 3T naive",      "composite_C1"),
                                 (axes[1], df_sorted_c2, "C2: synth-derived", "composite_C2")]:
        cols = ["C3" if g == "AD" else "C0" for g in dfx["group"]]
        ax.barh(np.arange(len(dfx))[::-1], dfx[col],
                color=cols, edgecolor="black", linewidth=0.6)
        ax.set_yticks(np.arange(len(dfx))[::-1])
        ax.set_yticklabels([f"{g}  {pid[-4:]}" for g, pid in zip(dfx["group"], dfx["ptid"])],
                           fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.axvline(0, color="grey", lw=0.7, alpha=0.5)
        ax.set_xlabel("Composite AD-risk z-score")
        ax.grid(alpha=0.3, axis="x")

    # Legend
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="C3", label="AD"),
                        Patch(facecolor="C0", label="CN")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.suptitle("Per-subject composite AD-risk ranking — naive 3T vs synth-derived features",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-csv", type=Path, required=True,
                        help="features_smoke.csv (subject, group, plus features)")
    parser.add_argument("--c1-vs-c2-csv", type=Path, required=True,
                        help="c1_vs_c2_per_subject.csv from baseline_3t_vs_synth_features.py")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating thesis figures...")

    auc, lo, hi = fig_roc_with_ci(args.features_csv,
                                  args.out_dir / "roc_with_bootstrap_ci.png")
    print(f"  [1] roc_with_bootstrap_ci.png   AUC={auc:.3f}  95% CI [{lo:.2f}, {hi:.2f}]")

    fig_feature_boxplots(args.features_csv,
                         args.out_dir / "feature_boxplots.png")
    print("  [2] feature_boxplots.png")

    fig_cohen_d_bar(args.features_csv,
                    args.out_dir / "cohen_d_bar.png")
    print("  [3] cohen_d_bar.png")

    fig_ranking_compare(args.c1_vs_c2_csv,
                        args.out_dir / "ranking_compare.png")
    print("  [4] ranking_compare.png")

    print(f"\nAll figures written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
