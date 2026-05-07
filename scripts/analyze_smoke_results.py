"""
Quick local analysis of the ADNI smoke-test outputs (no FastSurfer required).

Produces thesis-ready material in <out_dir>:
  - smoke_topology.csv      one row per subject: GM CC count, largest-CC fraction,
                            volumes per tissue class, total brain voxels
  - smoke_summary.json      cohort-level aggregate (mean/SD across 10 subjects)
  - figures/<ptid>_compare.png   3-plane side-by-side: input 3T vs synthetic 7T
  - figures/cohort_overview.png  9-panel grid of all 10 synthetic mid-coronal slices

Designed for "I have an hour, deadline tomorrow" scenarios — runs in <1 min on CPU.

Usage:
    python scripts/analyze_smoke_results.py \\
        --results-dir adni_smoke_results/results_adni_smoke \\
        --preprocessed-dir adni_preprocessed/adni_preprocessed \\
        --out-dir adni_smoke_analysis
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


TISSUE_LABELS = {1: "CSF", 2: "GM", 3: "WM"}


def topology_metrics(seg: np.ndarray, label_id: int):
    """Connected-component count and largest-CC fraction for one tissue label."""
    binary = (seg == label_id).astype(np.uint8)
    n_voxels = int(binary.sum())
    if n_voxels == 0:
        return {"n_voxels": 0, "n_cc": 0, "largest_frac": 0.0}
    labeled, n_cc = ndimage.label(binary)
    if n_cc == 0:
        return {"n_voxels": n_voxels, "n_cc": 0, "largest_frac": 0.0}
    sizes = ndimage.sum(binary, labeled, range(1, n_cc + 1))
    largest = float(sizes.max())
    return {"n_voxels": n_voxels, "n_cc": int(n_cc),
            "largest_frac": round(largest / n_voxels, 4)}


def show_three_planes(ax_row, vol, title_prefix):
    """Plot mid-axial / coronal / sagittal slices on a 3-axis row."""
    d, h, w = vol.shape
    planes = [
        ("axial",    np.rot90(vol[:, :, w // 2])),
        ("coronal",  np.rot90(vol[:, h // 2, :])),
        ("sagittal", np.rot90(vol[d // 2, :, :])),
    ]
    for ax, (name, slc) in zip(ax_row, planes):
        ax.imshow(slc, cmap="gray", aspect="equal")
        ax.set_title(f"{title_prefix} — {name}", fontsize=9)
        ax.axis("off")


def make_compare_figure(input_vol, synth_vol, ptid, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    show_three_planes(axes[0], input_vol, "Input 3T (preproc)")
    show_three_planes(axes[1], synth_vol, "Synthetic 7T")
    fig.suptitle(f"{ptid}", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_cohort_grid(synth_paths, out_path, ptids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(synth_paths)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
    axes = axes.flatten() if n > 1 else [axes]
    for ax, p, ptid in zip(axes, synth_paths, ptids):
        vol = nib.load(str(p)).get_fdata()
        h = vol.shape[1]
        ax.imshow(np.rot90(vol[:, h // 2, :]), cmap="gray", aspect="equal")
        ax.set_title(ptid, fontsize=8)
        ax.axis("off")
    for ax in axes[len(synth_paths):]:
        ax.axis("off")
    fig.suptitle("ADNI smoke test — synthetic 7T (mid-coronal slice)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Dir containing <ptid>/predicted_7T.nii.gz, predicted_seg.nii.gz")
    parser.add_argument("--preprocessed-dir", type=Path, default=None,
                        help="Optional: dir containing AD/CN/<ptid>_T1w_preprocessed.nii.gz "
                             "for input-vs-synth side-by-side figures")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"

    # Discover subjects
    subjects = sorted([p for p in args.results_dir.iterdir()
                       if p.is_dir() and (p / "predicted_seg.nii.gz").exists()])
    if not subjects:
        print(f"No subjects with predicted_seg.nii.gz under {args.results_dir}", file=sys.stderr)
        return 1

    rows = []
    synth_paths = []
    ptids = []
    print(f"Analyzing {len(subjects)} subjects...\n")
    print(f"{'PTID':<14} {'Group':<5} {'GM_CC':>6} {'GM_largest':>11} {'WM_CC':>6} "
          f"{'WM_largest':>11} {'CSF_vox':>8} {'GM_vox':>10} {'WM_vox':>10}")
    print("-" * 92)

    for subj_dir in subjects:
        ptid = subj_dir.name
        seg_path = subj_dir / "predicted_seg.nii.gz"
        synth_path = subj_dir / "predicted_7T.nii.gz"
        synth_paths.append(synth_path)
        ptids.append(ptid)

        seg = np.round(nib.load(str(seg_path)).get_fdata()).astype(int)
        results = {f"{TISSUE_LABELS[k]}": topology_metrics(seg, k)
                   for k in TISSUE_LABELS}

        # Resolve group from preprocessed dir layout if available
        group = "?"
        if args.preprocessed_dir:
            for g in ("AD", "CN"):
                if (args.preprocessed_dir / g / f"{ptid}_T1w_preprocessed.nii.gz").exists():
                    group = g
                    break

        gm_cc = results["GM"]["n_cc"]
        gm_lg = results["GM"]["largest_frac"]
        wm_cc = results["WM"]["n_cc"]
        wm_lg = results["WM"]["largest_frac"]
        print(f"{ptid:<14} {group:<5} {gm_cc:>6d} {gm_lg:>11.3f} "
              f"{wm_cc:>6d} {wm_lg:>11.3f} "
              f"{results['CSF']['n_voxels']:>8d} {results['GM']['n_voxels']:>10d} "
              f"{results['WM']['n_voxels']:>10d}")

        rows.append({
            "ptid": ptid, "group": group,
            "gm_cc": gm_cc, "gm_largest_frac": gm_lg, "gm_voxels": results["GM"]["n_voxels"],
            "wm_cc": wm_cc, "wm_largest_frac": wm_lg, "wm_voxels": results["WM"]["n_voxels"],
            "csf_cc": results["CSF"]["n_cc"], "csf_largest_frac": results["CSF"]["largest_frac"],
            "csf_voxels": results["CSF"]["n_voxels"],
        })

        # Per-subject input-vs-synth figure
        if args.preprocessed_dir and group in ("AD", "CN"):
            inp_path = args.preprocessed_dir / group / f"{ptid}_T1w_preprocessed.nii.gz"
            if inp_path.exists():
                inp_vol = nib.load(str(inp_path)).get_fdata()
                synth_vol = nib.load(str(synth_path)).get_fdata()
                make_compare_figure(inp_vol, synth_vol, f"{ptid} ({group})",
                                    fig_dir / f"{ptid}_compare.png")

    # Cohort overview grid
    make_cohort_grid(synth_paths, fig_dir / "cohort_overview.png", ptids)
    print(f"\nWrote {fig_dir}/cohort_overview.png  + {len([r for r in rows])} per-subject figures")

    # Save CSV
    csv_path = args.out_dir / "smoke_topology.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    # Aggregate summary
    arr = lambda key: np.array([r[key] for r in rows], dtype=float)
    summary = {
        "n_subjects": len(rows),
        "ad_count":   sum(1 for r in rows if r["group"] == "AD"),
        "cn_count":   sum(1 for r in rows if r["group"] == "CN"),
        "gm_largest_frac_mean": float(arr("gm_largest_frac").mean()),
        "gm_largest_frac_std":  float(arr("gm_largest_frac").std()),
        "wm_largest_frac_mean": float(arr("wm_largest_frac").mean()),
        "wm_largest_frac_std":  float(arr("wm_largest_frac").std()),
        "gm_cc_median":         float(np.median(arr("gm_cc"))),
        "wm_cc_median":         float(np.median(arr("wm_cc"))),
    }
    summary = {k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()}
    with open(args.out_dir / "smoke_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCohort summary:")
    for k, v in summary.items():
        print(f"  {k:<26} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
