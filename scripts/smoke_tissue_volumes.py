"""
Quick AD-vs-CN tissue-volume comparison from the smoke-test synthesis outputs.

Provides ONE empirical AD-detection signal for the thesis:
    - Per-subject GM/WM/CSF volumes (mm^3) computed from predicted_seg.nii.gz
    - Group means by AD/CN, percent difference, atrophy proxies
    - CSF/GM ratio (a coarse "atrophy index" — higher in AD)

Limitations baked into the docstring honestly: 5+5 cohort, 4-class
segmentation (not parcellated subcortex), so this is a directionality
check against AD literature, not a statistical claim.

Reads:
    adni_smoke_results/results_adni_smoke/<ptid>/predicted_seg.nii.gz
    adni_preprocessed/adni_preprocessed/{AD,CN}/<ptid>_T1w_preprocessed.nii.gz
        (used only for group label discovery)

Writes:
    <out_dir>/tissue_volumes.csv         per-subject mm^3 columns
    <out_dir>/tissue_volumes_summary.txt thesis-ready table

Usage:
    python scripts/smoke_tissue_volumes.py \\
        --results-dir adni_smoke_results/results_adni_smoke \\
        --preprocessed-dir adni_preprocessed/adni_preprocessed \\
        --out-dir adni_smoke_analysis
"""
import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


TISSUE_LABELS = {1: "csf", 2: "gm", 3: "wm"}


def voxel_volume_mm3(affine):
    return float(abs(np.linalg.det(affine[:3, :3])))


def tissue_volumes(seg, vox_mm3):
    out = {}
    for label, name in TISSUE_LABELS.items():
        out[f"{name}_mm3"] = float(int(np.count_nonzero(seg == label)) * vox_mm3)
    out["brain_mm3"] = out["csf_mm3"] + out["gm_mm3"] + out["wm_mm3"]
    out["gm_frac"] = out["gm_mm3"] / out["brain_mm3"] if out["brain_mm3"] > 0 else 0.0
    out["csf_gm_ratio"] = out["csf_mm3"] / out["gm_mm3"] if out["gm_mm3"] > 0 else 0.0
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--preprocessed-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for subj_dir in sorted(args.results_dir.iterdir()):
        seg_path = subj_dir / "predicted_seg.nii.gz"
        if not seg_path.exists():
            continue
        ptid = subj_dir.name
        img = nib.load(str(seg_path))
        seg = np.round(img.get_fdata()).astype(np.int32)
        vox = voxel_volume_mm3(img.affine)
        vols = tissue_volumes(seg, vox)

        group = "?"
        if args.preprocessed_dir:
            for g in ("AD", "CN"):
                if (args.preprocessed_dir / g / f"{ptid}_T1w_preprocessed.nii.gz").exists():
                    group = g
                    break
        # Use 4 decimals for all values so small ratios (csf_gm_ratio ~0.03) survive.
        rows.append({"ptid": ptid, "group": group, "voxel_mm3": round(vox, 4),
                     **{k: round(v, 4) for k, v in vols.items()}})

    if not rows:
        print("No subjects found.", file=sys.stderr)
        return 1

    csv_path = args.out_dir / "tissue_volumes.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- group summary ----
    def _grp(g, key):
        return np.array([r[key] for r in rows if r["group"] == g], dtype=float)

    metrics = ("brain_mm3", "gm_mm3", "wm_mm3", "csf_mm3",
               "gm_frac", "csf_gm_ratio")
    ad_means = {k: float(_grp("AD", k).mean()) for k in metrics}
    cn_means = {k: float(_grp("CN", k).mean()) for k in metrics}
    ad_sd    = {k: float(_grp("AD", k).std(ddof=1)) for k in metrics}
    cn_sd    = {k: float(_grp("CN", k).std(ddof=1)) for k in metrics}

    # AD-literature direction key:
    #   atrophy   -> lower GM, lower brain, lower gm_frac
    #   ventricles-> higher CSF, higher CSF/GM ratio
    expected_direction = {
        "brain_mm3":    "AD < CN  (atrophy)",
        "gm_mm3":       "AD < CN  (cortical+subcortical loss)",
        "wm_mm3":       "AD ≤ CN  (mild WM loss)",
        "csf_mm3":      "AD > CN  (ventriculomegaly)",
        "gm_frac":      "AD < CN  (relative GM loss)",
        "csf_gm_ratio": "AD > CN  (atrophy index)",
    }

    table_lines = []
    table_lines.append("ADNI smoke-test tissue volumes — AD vs CN  (5 + 5)")
    table_lines.append("=" * 76)
    table_lines.append(f"{'metric':<14} {'AD mean ± SD':>18} {'CN mean ± SD':>18} "
                       f"{'Δ%':>6} {'matches AD lit?':>20}")
    table_lines.append("-" * 76)
    for k in metrics:
        a, c = ad_means[k], cn_means[k]
        ad_sd_v, cn_sd_v = ad_sd[k], cn_sd[k]
        delta_pct = ((a - c) / c) * 100 if c != 0 else 0.0

        # Decide if AD is supposed to be < or > CN
        if "csf" in k and "gm" in k:        # csf_gm_ratio
            expected_higher = "AD"
        elif k.startswith("csf"):
            expected_higher = "AD"
        else:                                # brain, gm, wm, gm_frac
            expected_higher = "CN"
        observed_higher = "AD" if a > c else "CN"
        matches = "✓ yes" if expected_higher == observed_higher else "✗ no"

        # mm^3 vs unitless formatting
        if k in ("gm_frac", "csf_gm_ratio"):
            ad_str = f"{a:.4f} ± {ad_sd_v:.4f}"
            cn_str = f"{c:.4f} ± {cn_sd_v:.4f}"
        else:
            ad_str = f"{a/1000:.1f} ± {ad_sd_v/1000:.1f} cm³"
            cn_str = f"{c/1000:.1f} ± {cn_sd_v/1000:.1f} cm³"

        table_lines.append(f"{k:<14} {ad_str:>18} {cn_str:>18} "
                           f"{delta_pct:>+5.1f}% {matches:>20}")
    table_lines.append("-" * 76)
    table_lines.append("Direction interpretation (per AD literature, e.g. Frisoni 2010):")
    for k, v in expected_direction.items():
        table_lines.append(f"  {k:<14}  expect  {v}")
    table_lines.append("")
    table_lines.append("Caveats:")
    table_lines.append("  - n=5 per group; statistical significance not claimed")
    table_lines.append("  - 4-class segmentation only (no parcellated subcortex)")
    table_lines.append("  - Subjects matched by demographics (age 70.8-82.8, ~50% male)")
    table_lines.append("  - Volumes from model's predicted_seg, not FreeSurfer")
    table_lines.append("  - Useful as DIRECTION-of-effect check, not as primary result")

    table = "\n".join(table_lines)
    print(table)
    (args.out_dir / "tissue_volumes_summary.txt").write_text(table, encoding="utf-8")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {args.out_dir / 'tissue_volumes_summary.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
