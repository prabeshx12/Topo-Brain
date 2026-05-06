"""
Compute Betti numbers (β₀, β₁, β₂) per tissue class in a 3D segmentation.

Replaces scripts/verify_topology.py's `scipy.ndimage.label` (which only
counts β₀ = connected components) with `gudhi.CubicalComplex` so the
thesis can report real Betti numbers — components, handles, and voids.

Topological interpretation for tissue masks:
    β₀ : connected components       (e.g. left + right hemisphere = 2)
    β₁ : independent loops/handles  (e.g. cortical genus)
    β₂ : voids / enclosed cavities  (e.g. ventricles inside white matter)

Why this matters: the "topology loss" in src/topology_loss.py is currently
edge-aware CE+Dice (Sobel), not persistent homology — the thesis must
acknowledge this and back the "topology preservation" claim with
post-hoc Betti measurement of the predicted vs. ground-truth seg.

Usage:
    # Single subject
    python scripts/compute_betti.py \\
        --seg results/sub-06/predicted_seg.nii.gz \\
        --output results/sub-06/betti.json

    # With ground-truth comparison (delta-betti per tissue)
    python scripts/compute_betti.py \\
        --seg     results/sub-06/predicted_seg.nii.gz \\
        --gt-seg  /path/to/sub-06_ses-2_seg.nii \\
        --output  results/sub-06/betti.json

    # Batch over a pairs CSV (cols: subject, predicted_seg, ground_truth_seg?)
    python scripts/compute_betti.py \\
        --pairs-csv results/loocv/seg_pairs.csv \\
        --output-csv results/loocv/betti.csv

Install:
    pip install gudhi    # ships prebuilt wheels for linux/macos/windows
"""
import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


LABEL_NAMES = {0: "Background", 1: "CSF", 2: "GM", 3: "WM"}
TISSUE_LABELS = (1, 2, 3)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def betti_3d(binary: np.ndarray) -> Tuple[int, int, int]:
    """Compute (β₀, β₁, β₂) of the foreground of a 3D binary array.

    Uses gudhi's cubical complex with a sublevel-set filtration:
      foreground voxels -> filtration value 0
      background voxels -> filtration value 1

    `persistent_betti_numbers(from_value=0, to_value=0.5)` then counts
    features alive in the foreground only — i.e. before background voxels
    enter the complex.

    Returns zeros if the foreground is empty.
    """
    import gudhi as gd

    if not binary.any():
        return (0, 0, 0)

    # Pass the numpy array directly. Do NOT use flatten+dimensions — that
    # path misinterprets axis order in gudhi 3.12 and produces wrong Betti
    # numbers (caught by --self-test: two-cube test gave β₀=8 instead of 2).
    filt = np.where(binary, 0.0, 1.0).astype(np.float32)
    cc = gd.CubicalComplex(top_dimensional_cells=filt)
    cc.compute_persistence(homology_coeff_field=2, min_persistence=0.0)

    # Persistent Betti numbers in [from_value, to_value] count features
    # born at ≤ from_value and still alive after to_value. With our
    # filtration (foreground=0, background=1), at t=0.5 only foreground is
    # in the complex — so this returns the foreground's Betti numbers.
    raw = cc.persistent_betti_numbers(from_value=0.0, to_value=0.5)
    padded = (list(raw) + [0, 0, 0])[:3]
    return tuple(int(b) for b in padded)


def analyze_seg(seg: np.ndarray) -> Dict[str, Dict]:
    """Per-tissue Betti report for an integer-labelled 3D segmentation."""
    out: Dict[str, Dict] = {}
    for label_id in TISSUE_LABELS:
        name = LABEL_NAMES[label_id]
        binary = seg == label_id
        n_voxels = int(binary.sum())
        if n_voxels == 0:
            out[name] = {"voxels": 0, "betti_0": 0, "betti_1": 0, "betti_2": 0,
                         "elapsed_s": 0.0}
            continue
        t0 = time.time()
        b0, b1, b2 = betti_3d(binary)
        out[name] = {
            "voxels":   n_voxels,
            "betti_0":  b0,   # connected components
            "betti_1":  b1,   # handles / loops
            "betti_2":  b2,   # voids / cavities
            "elapsed_s": round(time.time() - t0, 2),
        }
        logging.info("  %-3s β₀=%d  β₁=%d  β₂=%d  voxels=%d  (%.1fs)",
                     name, b0, b1, b2, n_voxels, out[name]["elapsed_s"])
    return out


def diff_against_gt(pred: Dict, gt: Dict) -> Dict[str, Dict]:
    """For each tissue, compute (β_pred - β_gt) per dim. Lower |Δ| is better."""
    out: Dict[str, Dict] = {}
    for tissue in ("CSF", "GM", "WM"):
        if tissue not in pred or tissue not in gt:
            continue
        out[tissue] = {
            "delta_betti_0": pred[tissue]["betti_0"] - gt[tissue]["betti_0"],
            "delta_betti_1": pred[tissue]["betti_1"] - gt[tissue]["betti_1"],
            "delta_betti_2": pred[tissue]["betti_2"] - gt[tissue]["betti_2"],
        }
    return out


def load_seg(path: Path) -> np.ndarray:
    import nibabel as nib
    img = nib.load(str(path))
    return np.round(img.get_fdata()).astype(np.int32)


def print_report(results: Dict[str, Dict], title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")
    print(f"  {'Tissue':<8} {'Voxels':>10} {'β₀':>6} {'β₁':>6} {'β₂':>6}  "
          f"{'  meaning':<30}")
    print(f"  {'-' * 70}")
    interp = {
        "CSF": "ventricles + sulci",
        "GM":  "cortical ribbon + subcortex",
        "WM":  "white-matter tracts",
    }
    for tissue in ("CSF", "GM", "WM"):
        r = results.get(tissue)
        if r is None:
            continue
        print(f"  {tissue:<8} {r['voxels']:>10} {r['betti_0']:>6} "
              f"{r['betti_1']:>6} {r['betti_2']:>6}  {interp[tissue]:<30}")


def print_diff(diff: Dict[str, Dict]) -> None:
    print(f"\n  Δ Betti (predicted − ground truth)  — closer to 0 means better preservation")
    print(f"  {'Tissue':<8} {'Δβ₀':>8} {'Δβ₁':>8} {'Δβ₂':>8}")
    print(f"  {'-' * 35}")
    for tissue in ("CSF", "GM", "WM"):
        d = diff.get(tissue)
        if d is None:
            continue
        print(f"  {tissue:<8} {d['delta_betti_0']:>+8d} "
              f"{d['delta_betti_1']:>+8d} {d['delta_betti_2']:>+8d}")


def _self_test() -> int:
    """Validate gudhi integration against shapes of known Betti numbers.

    Catches version-incompatibility issues with gudhi's persistent_betti_numbers
    semantics, off-by-one errors in the filtration construction, etc.
    """
    print("Running compute_betti self-test (shapes with known Betti numbers)...")

    # 1) Empty volume -> all zeros.
    empty = np.zeros((10, 10, 10), dtype=bool)
    got = betti_3d(empty)
    print(f"  empty volume:        β = {got}   (expected (0, 0, 0))")
    assert got == (0, 0, 0), f"empty volume: got {got}"

    # 2) Solid cube — one connected component, no loops, no voids.
    cube = np.zeros((10, 10, 10), dtype=bool)
    cube[2:7, 2:7, 2:7] = True
    got = betti_3d(cube)
    print(f"  solid cube:          β = {got}   (expected (1, 0, 0))")
    assert got == (1, 0, 0), f"solid cube: got {got}"

    # 3) Two disconnected cubes — β₀ = 2.
    two = np.zeros((20, 10, 10), dtype=bool)
    two[2:7, 2:7, 2:7]    = True
    two[12:17, 2:7, 2:7]  = True
    got = betti_3d(two)
    print(f"  two cubes:           β = {got}   (expected (2, 0, 0))")
    assert got == (2, 0, 0), f"two cubes: got {got}"

    # 4) Hollow cube (shell with enclosed cavity) — β₂ = 1.
    shell = np.zeros((12, 12, 12), dtype=bool)
    shell[2:10, 2:10, 2:10] = True
    shell[4:8, 4:8, 4:8]    = False    # carve cavity that doesn't reach surface
    got = betti_3d(shell)
    print(f"  hollow cube:         β = {got}   (expected (1, 0, 1))")
    assert got == (1, 0, 1), f"hollow cube: got {got}"

    # 5) Block with a through-hole (homotopy-equivalent to a solid torus) — β₁ = 1.
    torus_like = np.zeros((20, 20, 10), dtype=bool)
    torus_like[2:18, 2:18, 2:8] = True
    torus_like[8:12, 8:12, :]   = False  # column carved through the full z-axis
    got = betti_3d(torus_like)
    print(f"  block w/ through:    β = {got}   (expected (1, 1, 0))")
    assert got == (1, 1, 0), f"block-with-hole: got {got}"

    print("✓ All Betti self-tests passed.")
    return 0


def _process_one(seg_path: Path, gt_path: Optional[Path]) -> Dict:
    logging.info("Computing Betti for %s", seg_path.name)
    seg = load_seg(seg_path)
    logging.info("  shape=%s, labels=%s",
                 seg.shape, dict(zip(*[a.tolist() for a in np.unique(seg, return_counts=True)])))
    pred_betti = analyze_seg(seg)

    out = {"predicted_seg": str(seg_path), "predicted": pred_betti}

    if gt_path is not None:
        if not gt_path.exists():
            logging.warning("Ground truth not found: %s — skipping diff", gt_path)
        elif gt_path == seg_path:
            logging.warning("--gt-seg same path as --seg; skipping diff")
        else:
            logging.info("Computing Betti for ground truth %s", gt_path.name)
            gt = load_seg(gt_path)
            if gt.shape != seg.shape:
                logging.error("Shape mismatch: pred %s vs gt %s — skipping diff",
                              seg.shape, gt.shape)
            else:
                gt_betti = analyze_seg(gt)
                out["ground_truth_seg"] = str(gt_path)
                out["ground_truth"] = gt_betti
                out["delta"] = diff_against_gt(pred_betti, gt_betti)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_mutually_exclusive_group(required=True)
    sub.add_argument("--seg", type=Path,
                     help="Single segmentation NIfTI")
    sub.add_argument("--pairs-csv", type=Path,
                     help="Batch mode CSV with cols: subject,predicted_seg,"
                          "ground_truth_seg (last optional)")
    sub.add_argument("--self-test", action="store_true",
                     help="Validate gudhi integration on shapes of known Betti")

    parser.add_argument("--gt-seg", type=Path, default=None,
                        help="Optional ground-truth NIfTI for delta-Betti")
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON output path (single mode)")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="CSV output path (batch mode)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    # Lazy import so --help works without gudhi installed.
    try:
        import gudhi  # noqa: F401
    except ImportError:
        logging.error("gudhi not installed. `pip install gudhi`.")
        return 2

    if args.self_test:
        return _self_test()

    if args.seg is not None:
        # Single mode
        result = _process_one(args.seg.resolve(),
                              args.gt_seg.resolve() if args.gt_seg else None)
        print_report(result["predicted"], "Predicted seg")
        if "ground_truth" in result:
            print_report(result["ground_truth"], "Ground truth seg")
            print_diff(result["delta"])

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            logging.info("Wrote %s", args.output)
        return 0

    # Batch mode
    pairs_csv = args.pairs_csv.resolve()
    if not pairs_csv.exists():
        logging.error("Pairs CSV not found: %s", pairs_csv)
        return 2
    if args.output_csv is None:
        logging.error("--output-csv is required in batch mode")
        return 2

    rows = []
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for sub in reader:
            seg_path = Path(sub["predicted_seg"])
            gt_path  = Path(sub["ground_truth_seg"]) if sub.get("ground_truth_seg") else None
            try:
                result = _process_one(seg_path, gt_path)
            except Exception as e:
                logging.error("[%s] failed: %s", sub.get("subject", seg_path.name), e)
                continue

            row = {"subject": sub.get("subject", seg_path.stem)}
            for tissue in ("CSF", "GM", "WM"):
                if tissue in result["predicted"]:
                    p = result["predicted"][tissue]
                    row[f"{tissue}_betti_0"] = p["betti_0"]
                    row[f"{tissue}_betti_1"] = p["betti_1"]
                    row[f"{tissue}_betti_2"] = p["betti_2"]
                if "delta" in result and tissue in result["delta"]:
                    d = result["delta"][tissue]
                    row[f"{tissue}_delta_betti_0"] = d["delta_betti_0"]
                    row[f"{tissue}_delta_betti_1"] = d["delta_betti_1"]
                    row[f"{tissue}_delta_betti_2"] = d["delta_betti_2"]
            rows.append(row)

    if not rows:
        logging.error("No subjects processed successfully")
        return 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r})
    cols = ["subject"] + [c for c in cols if c != "subject"]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logging.info("Wrote %s (%d subjects)", args.output_csv, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
