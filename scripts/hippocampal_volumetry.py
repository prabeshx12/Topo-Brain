"""
Hippocampal (and other subcortical) volumetry agreement: real vs synthetic 7T.

Week 7 of HANDOFF.md — clinical safety net for the AD downstream claim.
For each subject we have two aparc+aseg segmentations:
    real:  FreeSurfer/FastSurfer on the real 7T (ses-2) — ground truth.
    synth: FastSurfer on the synthetic 7T from the diffusion model — proxy.

For each region (hippocampus by default), this computes:
    - per-subject volume (mm^3) on each side
    - per-region agreement statistics across subjects:
        * ICC(2,1)  — two-way random absolute agreement, single rater
        * Bland-Altman bias + 95% limits of agreement
        * MAE% / RMSE%, Pearson r

Plus a Bland-Altman PNG per region for the thesis.

Usage:
    python scripts/hippocampal_volumetry.py \\
        --pairs-csv pairs.csv \\
        --output-dir results/volumetry

    # Wider regional sweep (matches check_region_preservation.py's region list):
    python scripts/hippocampal_volumetry.py \\
        --pairs-csv pairs.csv \\
        --regions hippocampus amygdala thalamus caudate \\
        --output-dir results/volumetry

    # Validate the stats implementation:
    python scripts/hippocampal_volumetry.py --self-test

Pairs CSV columns:
    subject               — identifier used in outputs
    real_aparc_aseg       — path to FreeSurfer/FastSurfer aparc+aseg on real 7T
    synth_aparc_aseg      — path to FastSurfer aparc+aseg on synthetic 7T
"""
import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# FreeSurfer aseg label IDs. Reference: FreeSurfer's FreeSurferColorLUT.txt
REGION_LABELS = {
    "hippocampus": {"L": 17, "R": 53},
    "amygdala":    {"L": 18, "R": 54},
    "thalamus":    {"L": 10, "R": 49},   # "Thalamus" / "Thalamus-Proper" in newer FS
    "caudate":     {"L": 11, "R": 50},
    "putamen":     {"L": 12, "R": 51},
    "pallidum":    {"L": 13, "R": 52},
}

DEFAULT_REGIONS = ("hippocampus",)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Volumetry from aparc+aseg
# ---------------------------------------------------------------------------

def voxel_volume_mm3(affine: np.ndarray) -> float:
    """Voxel volume in mm^3 = |det(rotation+scale part of affine)|."""
    return float(abs(np.linalg.det(affine[:3, :3])))


def region_volume(seg: np.ndarray, label_id: int, voxel_vol_mm3: float) -> float:
    """Volume of voxels matching `label_id` in mm^3."""
    return float(int(np.count_nonzero(seg == label_id)) * voxel_vol_mm3)


def load_aseg(path: Path) -> Tuple[np.ndarray, float]:
    """Load an aparc+aseg NIfTI; return (int seg, voxel volume mm^3)."""
    import nibabel as nib
    img = nib.load(str(path))
    arr = np.round(img.get_fdata()).astype(np.int32)
    return arr, voxel_volume_mm3(img.affine)


# ---------------------------------------------------------------------------
# Agreement statistics
# ---------------------------------------------------------------------------

@dataclass
class AgreementStats:
    n: int
    icc_2_1: float            # two-way random, absolute agreement, single rater
    bias: float               # mean(synth - real)
    sd_diff: float            # SD of differences
    loa_lower: float          # bias - 1.96 * SD
    loa_upper: float          # bias + 1.96 * SD
    mae_pct: float
    rmse_pct: float
    pearson_r: float


def icc_2_1(real: np.ndarray, synth: np.ndarray) -> float:
    """ICC(2,1) — two-way random effects, absolute agreement, single rater.

    Formula (Shrout & Fleiss 1979, McGraw & Wong 1996):
        n  = number of subjects
        k  = number of raters (here 2)
        MSR = between-subjects mean square
        MSC = between-raters mean square
        MSE = residual mean square

        ICC(2,1) = (MSR - MSE) / (MSR + (k-1)*MSE + k*(MSC - MSE)/n)

    Returns NaN if the design is degenerate (n<2 or zero between-subjects var).
    """
    real = np.asarray(real, dtype=float)
    synth = np.asarray(synth, dtype=float)
    if real.shape != synth.shape:
        raise ValueError(f"shape mismatch: {real.shape} vs {synth.shape}")
    n = real.size
    if n < 2:
        return float("nan")

    # Build the n x 2 ratings matrix.
    X = np.stack([real, synth], axis=1)  # rows = subjects, cols = raters
    k = 2

    grand_mean = X.mean()
    row_means = X.mean(axis=1)
    col_means = X.mean(axis=0)

    # Sums of squares
    SSR = k * np.sum((row_means - grand_mean) ** 2)
    SSC = n * np.sum((col_means - grand_mean) ** 2)
    SST = np.sum((X - grand_mean) ** 2)
    SSE = SST - SSR - SSC

    df_R = n - 1
    df_C = k - 1
    df_E = (n - 1) * (k - 1)

    if df_R <= 0 or df_E <= 0:
        return float("nan")

    MSR = SSR / df_R
    MSC = SSC / df_C
    MSE = SSE / df_E

    denom = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    if denom <= 0:
        return float("nan")
    return float((MSR - MSE) / denom)


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def agreement(real: np.ndarray, synth: np.ndarray) -> AgreementStats:
    """All headline agreement metrics for paired (real, synth) volumes."""
    real = np.asarray(real, dtype=float)
    synth = np.asarray(synth, dtype=float)
    n = real.size

    diff = synth - real
    bias = float(np.mean(diff)) if n else float("nan")
    sd_diff = float(np.std(diff, ddof=1)) if n > 1 else float("nan")

    # Bland-Altman 95% limits of agreement (assume normality of differences).
    loa_lower = bias - 1.96 * sd_diff if not math.isnan(sd_diff) else float("nan")
    loa_upper = bias + 1.96 * sd_diff if not math.isnan(sd_diff) else float("nan")

    # Percentage errors against the real (ground-truth) value.
    safe_real = np.where(real == 0, np.nan, real)
    pct_err = (synth - real) / safe_real * 100.0
    mae_pct = float(np.nanmean(np.abs(pct_err))) if n else float("nan")
    rmse_pct = float(np.sqrt(np.nanmean(pct_err ** 2))) if n else float("nan")

    return AgreementStats(
        n=n,
        icc_2_1=icc_2_1(real, synth),
        bias=bias,
        sd_diff=sd_diff,
        loa_lower=loa_lower,
        loa_upper=loa_upper,
        mae_pct=mae_pct,
        rmse_pct=rmse_pct,
        pearson_r=pearson_r(real, synth),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bland_altman(real: np.ndarray, synth: np.ndarray, stats: AgreementStats,
                      title: str, out_path: Path) -> None:
    """Standard Bland-Altman: x = mean of pair, y = synth − real, with bias + LoA."""
    import matplotlib
    matplotlib.use("Agg")  # safe headless on CERN/Kaggle
    import matplotlib.pyplot as plt

    means = (np.asarray(real) + np.asarray(synth)) / 2.0
    diffs = np.asarray(synth) - np.asarray(real)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(means, diffs, s=40, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axhline(stats.bias, color="C1", linestyle="-", lw=1.2,
               label=f"bias = {stats.bias:.1f}")
    ax.axhline(stats.loa_upper, color="C3", linestyle="--", lw=1.0,
               label=f"+1.96·SD = {stats.loa_upper:.1f}")
    ax.axhline(stats.loa_lower, color="C3", linestyle="--", lw=1.0,
               label=f"−1.96·SD = {stats.loa_lower:.1f}")
    ax.axhline(0, color="grey", linestyle=":", lw=0.7)
    ax.set_xlabel("Mean of (real, synth)  [mm³]")
    ax.set_ylabel("synth − real  [mm³]")
    ax.set_title(f"{title}\nICC(2,1)={stats.icc_2_1:.3f}, "
                 f"MAE={stats.mae_pct:.1f}%, n={stats.n}")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _process_pair(subject: str, real_path: Path, synth_path: Path,
                  regions: Tuple[str, ...]) -> List[Dict]:
    """Per-subject rows for each (region, side) pair."""
    real_seg, real_vox = load_aseg(real_path)
    synth_seg, synth_vox = load_aseg(synth_path)
    if real_seg.shape != synth_seg.shape:
        raise ValueError(
            f"[{subject}] shape mismatch: real {real_seg.shape} vs synth {synth_seg.shape}. "
            "Both aparc+aseg files must live on the same voxel grid; resample upstream."
        )

    rows = []
    for region in regions:
        labels = REGION_LABELS[region]
        for side, label_id in labels.items():
            v_real = region_volume(real_seg, label_id, real_vox)
            v_synth = region_volume(synth_seg, label_id, synth_vox)
            rows.append({
                "subject":    subject,
                "region":     region,
                "side":       side,
                "label":      label_id,
                "real_mm3":   round(v_real, 2),
                "synth_mm3":  round(v_synth, 2),
                "diff_mm3":   round(v_synth - v_real, 2),
                "diff_pct":   round((v_synth - v_real) / v_real * 100.0, 2)
                              if v_real > 0 else float("nan"),
            })
    return rows


def aggregate_agreement(per_subject: List[Dict],
                        regions: Tuple[str, ...]) -> Dict:
    """Group per-subject rows by region+side and compute agreement stats."""
    out: Dict = {}
    for region in regions:
        out[region] = {}
        for side in ("L", "R"):
            sub = [r for r in per_subject if r["region"] == region and r["side"] == side]
            if not sub:
                continue
            real = np.array([r["real_mm3"] for r in sub], dtype=float)
            synth = np.array([r["synth_mm3"] for r in sub], dtype=float)
            stats = agreement(real, synth)
            out[region][side] = {
                "n":        stats.n,
                "icc_2_1":  round(stats.icc_2_1, 4),
                "bias":     round(stats.bias, 2),
                "sd_diff":  round(stats.sd_diff, 2),
                "loa_lower": round(stats.loa_lower, 2),
                "loa_upper": round(stats.loa_upper, 2),
                "mae_pct":  round(stats.mae_pct, 2),
                "rmse_pct": round(stats.rmse_pct, 2),
                "pearson_r": round(stats.pearson_r, 4),
            }
    return out


def print_aggregate_report(agg: Dict) -> None:
    print(f"\n{'=' * 86}")
    print(f"  Volumetry agreement — real (FreeSurfer on real 7T) vs synthetic (FastSurfer on synth 7T)")
    print(f"{'=' * 86}")
    header = (f"  {'Region':<13} {'Side':<5} {'n':>3} {'ICC(2,1)':>9} "
              f"{'bias':>9} {'LoA lower':>10} {'LoA upper':>10} "
              f"{'MAE%':>7} {'r':>7}")
    print(header)
    print(f"  {'-' * 80}")
    for region, sides in agg.items():
        for side, s in sides.items():
            print(f"  {region:<13} {side:<5} {s['n']:>3} {s['icc_2_1']:>9.3f} "
                  f"{s['bias']:>9.1f} {s['loa_lower']:>10.1f} {s['loa_upper']:>10.1f} "
                  f"{s['mae_pct']:>7.1f} {s['pearson_r']:>7.3f}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    """Verify ICC, Bland-Altman, MAE%, Pearson r against known values.

    Synthesises paired (real, synth) volumes with controlled noise and
    bias and checks the implementation produces the expected stats.
    """
    print("Running hippocampal_volumetry self-test...")
    rng = np.random.default_rng(42)
    n = 60
    real = rng.normal(loc=4000, scale=400, size=n)             # ~4000 mm^3 hippo
    synth = real + 50 + rng.normal(loc=0, scale=80, size=n)    # +50 bias, sd=80 noise

    s = agreement(real, synth)
    print(f"  n = {s.n} (expected 60)")
    print(f"  ICC(2,1) = {s.icc_2_1:.3f} (expected ~0.97 — strong with bias and small noise)")
    print(f"  bias = {s.bias:+.1f} mm^3 (expected ~+50)")
    print(f"  sd_diff = {s.sd_diff:.1f} (expected ~80)")
    print(f"  LoA = [{s.loa_lower:.1f}, {s.loa_upper:.1f}]  (~ [-107, +207])")
    print(f"  MAE% = {s.mae_pct:.2f}%  (expected ~1.6%)")
    print(f"  Pearson r = {s.pearson_r:.3f}  (expected ~0.98)")

    # Bounds reflect typical sampling variability around the analytical
    # expectations (true bias=50, true sd_diff=80, ~3-sigma tolerance for n=60).
    assert s.n == 60
    assert 0.95 <= s.icc_2_1 <= 0.99, f"ICC unexpected: {s.icc_2_1}"
    assert 15 <= s.bias <= 85,        f"bias unexpected: {s.bias}"
    assert 50 <= s.sd_diff <= 110,    f"sd_diff unexpected: {s.sd_diff}"
    assert 0.95 <= s.pearson_r <= 0.99, f"r unexpected: {s.pearson_r}"

    # Edge: identical inputs -> ICC = 1.0, bias = 0
    s2 = agreement(real, real.copy())
    print(f"\n  identical inputs: ICC={s2.icc_2_1:.4f} (expected 1.0), bias={s2.bias:.4f}")
    assert math.isclose(s2.icc_2_1, 1.0, abs_tol=1e-6) or math.isnan(s2.icc_2_1)
    assert math.isclose(s2.bias, 0.0, abs_tol=1e-6)

    # Edge: completely random — low ICC.
    rand = rng.normal(loc=4000, scale=400, size=n)
    s3 = agreement(real, rand)
    print(f"  uncorrelated:    ICC={s3.icc_2_1:.3f}  (expected ~0)")
    assert -0.3 < s3.icc_2_1 < 0.3, f"random ICC unexpected: {s3.icc_2_1}"

    print("✓ All hippocampal_volumetry self-tests passed.")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pairs-csv", type=Path, default=None,
                        help="CSV with cols: subject, real_aparc_aseg, synth_aparc_aseg")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write CSV/JSON/PNG outputs")
    parser.add_argument("--regions", nargs="+", default=list(DEFAULT_REGIONS),
                        choices=sorted(REGION_LABELS),
                        help="Regions to analyse (default: hippocampus only)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip Bland-Altman PNG generation")
    parser.add_argument("--self-test", action="store_true",
                        help="Validate ICC / Bland-Altman / Pearson implementation")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    if args.self_test:
        return _self_test()

    if args.pairs_csv is None or args.output_dir is None:
        logging.error("--pairs-csv and --output-dir are required (or use --self-test).")
        return 2

    pairs_csv = args.pairs_csv.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pairs_csv.exists():
        logging.error("pairs CSV not found: %s", pairs_csv)
        return 2

    regions: Tuple[str, ...] = tuple(args.regions)
    logging.info("Regions: %s", ", ".join(regions))

    per_subject_rows: List[Dict] = []
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        required = {"subject", "real_aparc_aseg", "synth_aparc_aseg"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            logging.error("pairs CSV missing required cols: %s", sorted(missing))
            return 2

        for row in reader:
            subj = row["subject"]
            real_path = Path(row["real_aparc_aseg"])
            synth_path = Path(row["synth_aparc_aseg"])
            try:
                rows = _process_pair(subj, real_path, synth_path, regions)
                per_subject_rows.extend(rows)
                logging.info("[%s] ok (%d region-side pairs)", subj, len(rows))
            except Exception as e:
                logging.error("[%s] failed: %s", subj, e)

    if not per_subject_rows:
        logging.error("No subjects processed successfully.")
        return 1

    # Per-subject CSV
    per_subj_csv = out_dir / "per_subject_volumes.csv"
    fieldnames = ["subject", "region", "side", "label",
                  "real_mm3", "synth_mm3", "diff_mm3", "diff_pct"]
    with open(per_subj_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_subject_rows:
            writer.writerow(r)
    logging.info("Wrote %s", per_subj_csv)

    # Aggregate stats
    agg = aggregate_agreement(per_subject_rows, regions)
    agg_json = out_dir / "agreement.json"
    with open(agg_json, "w") as f:
        json.dump(agg, f, indent=2, default=lambda x: None if (
            isinstance(x, float) and math.isnan(x)) else x)
    logging.info("Wrote %s", agg_json)
    print_aggregate_report(agg)

    # Bland-Altman plots
    if not args.no_plots:
        for region in regions:
            for side in ("L", "R"):
                sub = [r for r in per_subject_rows
                       if r["region"] == region and r["side"] == side]
                if len(sub) < 2:
                    continue
                real = np.array([r["real_mm3"] for r in sub])
                synth = np.array([r["synth_mm3"] for r in sub])
                stats = agreement(real, synth)
                out = out_dir / f"bland_altman_{region}_{side}.png"
                plot_bland_altman(real, synth, stats,
                                  title=f"{region} ({side})", out_path=out)
                logging.info("Wrote %s", out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
