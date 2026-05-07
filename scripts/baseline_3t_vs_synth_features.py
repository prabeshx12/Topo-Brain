"""
Empirical C1 (3T-only naive) vs C2 (synth-derived) comparison on smoke cohort.

This is the lead-strengthening analysis the thesis defends "synth helps" with:

    For each ADNI smoke subject we compute tissue volumes (CSF, GM, WM,
    plus gm_frac and csf_gm_ratio derived) by TWO methods:
      C1 = 3T-only naive: K-means(k=3) on brain-masked 3T intensities,
           cluster labels assigned by mean intensity (low=CSF, mid=GM,
           high=WM). Uses NO information from the model's predicted_seg.
      C2 = synth-derived: voxel counts on the model's predicted_seg
           (already trained with 7T-supervised cross-entropy).
    Brain mask is a simple intensity threshold on the preprocessed 3T
    (`inp > -0.9` in the [-1,1] diffusion-normalized space) — same mask
    for both methods so the comparison is fair.

Then for each method, AD vs CN Cohen's d is computed on five features.
A side-by-side delta-d table is the headline output:

    feature       d_C1 (3T)    d_C2 (synth)    Δ|d| = |d_C2| - |d_C1|

Positive Δ|d| on canonical AD features (gm_frac, csf_gm_ratio, gm_mm3)
is empirical evidence that the 7T-supervised segmentation captures more
AD-discriminative signal than naive 3T thresholding — the core
"better-than-3T" claim of the thesis.

Outputs (in --out-dir):
    c1_vs_c2_per_subject.csv   per-subject volumes from both methods
    c1_vs_c2_summary.txt       thesis-ready Cohen's d comparison table

Usage:
    python scripts/baseline_3t_vs_synth_features.py \\
        --results-dir adni_smoke_results/results_adni_smoke \\
        --preprocessed-dir adni_preprocessed/adni_preprocessed \\
        --out-dir adni_smoke_analysis
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def cohen_d(a, b):
    """Cohen's d (Hedges' pooled-SD formulation). Returns NaN if degenerate."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s_pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1)
                          + (len(b) - 1) * b.var(ddof=1))
                         / (len(a) + len(b) - 2))
    if s_pooled == 0:
        return float("nan")
    return (a.mean() - b.mean()) / s_pooled


def naive_3t_segmentation(inp, brain_mask, seed=42):
    """K-means(k=3) on brain-masked 3T intensities. Returns int seg array
    same shape as `inp` with labels {0=BG, 1=CSF, 2=GM, 3=WM}."""
    from sklearn.cluster import KMeans
    intensities = inp[brain_mask].reshape(-1, 1)
    km = KMeans(n_clusters=3, random_state=seed, n_init=10)
    clusters = km.fit_predict(intensities)
    # Sort cluster IDs by mean intensity → CSF (lowest) < GM (mid) < WM (highest)
    centers = km.cluster_centers_.flatten()
    order = np.argsort(centers)
    label_map = {int(order[0]): 1, int(order[1]): 2, int(order[2]): 3}
    seg = np.zeros_like(inp, dtype=np.int32)
    seg[brain_mask] = np.array([label_map[c] for c in clusters], dtype=np.int32)
    return seg


def features_from_seg(seg, brain_mask, vox_mm3):
    """Five tissue-volume features in mm^3 + derived ratios."""
    csf = int(np.count_nonzero((seg == 1) & brain_mask)) * vox_mm3
    gm  = int(np.count_nonzero((seg == 2) & brain_mask)) * vox_mm3
    wm  = int(np.count_nonzero((seg == 3) & brain_mask)) * vox_mm3
    brain = csf + gm + wm
    return {
        "csf_mm3":      csf,
        "gm_mm3":       gm,
        "wm_mm3":       wm,
        "gm_frac":      gm / brain if brain > 0 else 0.0,
        "csf_gm_ratio": csf / gm if gm > 0 else 0.0,
    }


FEATURES = ["gm_frac", "csf_gm_ratio", "gm_mm3", "wm_mm3", "csf_mm3"]
# Direction expected per AD literature (per feature, AD vs CN):
EXPECTED_SIGN = {
    "gm_frac":      "-",  # AD < CN
    "csf_gm_ratio": "+",  # AD > CN
    "gm_mm3":       "-",  # AD < CN
    "wm_mm3":       "-",  # AD ≤ CN
    "csf_mm3":      "+",  # AD > CN
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--preprocessed-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--brain-threshold", type=float, default=-0.9,
                        help="Intensity threshold for brain mask in [-1,1] space "
                             "(default: -0.9). Diffusion-normalized background = -1.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for subj_dir in sorted(args.results_dir.iterdir()):
        ptid = subj_dir.name
        seg_path = subj_dir / "predicted_seg.nii.gz"
        if not seg_path.exists():
            continue

        # Find input + group
        group, inp_path = "?", None
        for g in ("AD", "CN"):
            cand = args.preprocessed_dir / g / f"{ptid}_T1w_preprocessed.nii.gz"
            if cand.exists():
                group, inp_path = g, cand
                break
        if inp_path is None:
            print(f"[{ptid}] no preprocessed 3T input, skipping", file=sys.stderr)
            continue

        # Load
        seg_img = nib.load(str(seg_path))
        synth_seg = np.round(seg_img.get_fdata()).astype(np.int32)
        inp = nib.load(str(inp_path)).get_fdata().astype(np.float32)
        if inp.shape != synth_seg.shape:
            print(f"[{ptid}] shape mismatch, skipping", file=sys.stderr)
            continue

        vox_mm3 = float(abs(np.linalg.det(seg_img.affine[:3, :3])))

        # Same brain mask for BOTH methods (intensity threshold on 3T)
        brain_mask = inp > args.brain_threshold
        if brain_mask.sum() < 100_000:
            print(f"[{ptid}] brain mask too small ({brain_mask.sum()} vox), "
                  f"skipping", file=sys.stderr)
            continue

        # C1: naive 3T k-means
        naive_seg = naive_3t_segmentation(inp, brain_mask, seed=42)
        feats_c1 = features_from_seg(naive_seg, brain_mask, vox_mm3)

        # C2: synth predicted_seg, restricted to same brain mask
        feats_c2 = features_from_seg(synth_seg, brain_mask, vox_mm3)

        row = {"ptid": ptid, "group": group, "voxel_mm3": round(vox_mm3, 4)}
        for k, v in feats_c1.items():
            row[f"{k}_C1"] = round(v, 4)
        for k, v in feats_c2.items():
            row[f"{k}_C2"] = round(v, 4)
        rows.append(row)
        print(f"[{ptid}] {group}  C1 gm_frac={feats_c1['gm_frac']:.4f}  "
              f"C2 gm_frac={feats_c2['gm_frac']:.4f}  "
              f"C1 csf_gm={feats_c1['csf_gm_ratio']:.4f}  "
              f"C2 csf_gm={feats_c2['csf_gm_ratio']:.4f}")

    if not rows:
        print("No subjects processed.", file=sys.stderr)
        return 1

    csv_path = args.out_dir / "c1_vs_c2_per_subject.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Per-feature Cohen's d for AD vs CN, both methods
    ad_rows = [r for r in rows if r["group"] == "AD"]
    cn_rows = [r for r in rows if r["group"] == "CN"]
    n_ad, n_cn = len(ad_rows), len(cn_rows)

    lines = []
    lines.append("C1 (3T-only naive) vs C2 (synth-derived) — Cohen's d on AD vs CN")
    lines.append("=" * 84)
    lines.append(f"  N: AD={n_ad}, CN={n_cn}")
    lines.append(f"  C1 = naive 3T K-means(k=3) on brain-masked input intensities")
    lines.append(f"  C2 = model's predicted_seg (4-class, 7T-supervised seg head)")
    lines.append(f"  Same brain mask used for both (intensity > {args.brain_threshold})")
    lines.append("")
    lines.append(f"  {'feature':<14} {'expected':>8}   "
                 f"{'d (C1, 3T)':>11} {'d (C2, synth)':>14}   "
                 f"{'|d_C2| − |d_C1|':>16}   {'C2 wins?':>9}")
    lines.append("  " + "-" * 80)

    deltas = {}
    summary_dict = {}
    for k in FEATURES:
        ad_c1 = np.array([r[f"{k}_C1"] for r in ad_rows], float)
        cn_c1 = np.array([r[f"{k}_C1"] for r in cn_rows], float)
        ad_c2 = np.array([r[f"{k}_C2"] for r in ad_rows], float)
        cn_c2 = np.array([r[f"{k}_C2"] for r in cn_rows], float)
        d_c1 = cohen_d(ad_c1, cn_c1)
        d_c2 = cohen_d(ad_c2, cn_c2)
        delta = abs(d_c2) - abs(d_c1)
        deltas[k] = delta
        summary_dict[k] = {"d_c1": d_c1, "d_c2": d_c2, "delta_abs_d": delta}
        c2_wins = "yes" if abs(d_c2) > abs(d_c1) else "no"
        # Also check direction matches AD literature
        sign_c2 = "+" if d_c2 >= 0 else "-"
        sign_match_c2 = "✓" if sign_c2 == EXPECTED_SIGN[k] else "✗"
        lines.append(f"  {k:<14} {EXPECTED_SIGN[k]:>8}   "
                     f"{d_c1:>+11.3f} {d_c2:>+14.3f}   "
                     f"{delta:>+16.3f}   {c2_wins:>9}  ({sign_match_c2} dir)")

    lines.append("  " + "-" * 80)
    n_c2_wins = sum(1 for v in deltas.values() if v > 0)
    mean_delta = np.mean(list(deltas.values()))
    lines.append(f"  C2 wins on {n_c2_wins} of {len(FEATURES)} features  "
                 f"(mean Δ|d| = {mean_delta:+.3f})")
    lines.append("")
    lines.append("Reading the table:")
    lines.append("  - 'expected' is AD-vs-CN direction per literature (e.g. gm_frac < 0)")
    lines.append("  - d_C1 / d_C2 are AD-minus-CN Cohen's d on each method's features")
    lines.append("  - Δ|d| > 0 means C2 (synth-derived) has LARGER effect-size magnitude")
    lines.append("    than C1 (naive 3T) — i.e. the 7T-supervised seg captures more AD signal")
    lines.append("  - 'C2 wins?' = does synth-derived have larger |d| than naive 3T?")
    lines.append("  - '(✓/✗ dir)' = does d_C2 sign match AD literature direction?")
    lines.append("")
    lines.append("Caveats:")
    lines.append(f"  - n=AD:{n_ad}/CN:{n_cn} → effect sizes have wide SE; magnitude robust")
    lines.append("    but absolute values uncertain. Direction + ordering most reliable.")
    lines.append("  - C1 baseline is K-means k=3 thresholding, NOT FreeSurfer-on-3T.")
    lines.append("    A FreeSurfer-on-3T baseline would be stronger but requires CERN compute.")
    lines.append("  - This is one cohort; reproduce on full 200-subject ADNI for")
    lines.append("    statistical power and external validity.")

    text = "\n".join(lines)
    print()
    print(text)
    (args.out_dir / "c1_vs_c2_summary.txt").write_text(text, encoding="utf-8")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {args.out_dir / 'c1_vs_c2_summary.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
