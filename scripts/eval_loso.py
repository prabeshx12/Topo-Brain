"""
Aggregate a completed LOSO sweep: evaluate each fold's FINAL checkpoint on its held-out subject,
then report mean +/- std across the 10 subjects. This is the number the paper reports.

The fold -> subject mapping reproduces train_cascaded's split EXACTLY (seed-42 shuffle), so the
subject evaluated for fold i is precisely the one that was held out of fold i's training.

Usage:
  python scripts/eval_loso.py --runs-dir /eos/user/p/ppokhrel/topobrain/runs/loso_topo0 \
      --pairs-csv /eos/user/p/ppokhrel/topobrain/data/norm/pairs_cern.csv
"""
import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fold_subject_map(pairs_csv, seed=42, n_folds=10):
    """Reproduce SubjectSplitter.create_folds: fresh default_rng(seed).shuffle over sorted subjects."""
    import csv
    subs = []
    with open(pairs_csv) as f:
        for row in csv.DictReader(f):
            subs.append(row["subject"])
    subs = sorted(set(subs))
    order = list(subs)
    np.random.default_rng(seed).shuffle(order)          # SAME shuffle as create_folds
    return {i: order[i] for i in range(min(n_folds, len(order)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, help="runs/loso_topo0 (holds fold_0 ... fold_9)")
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--config", default="configs/train_diffusion.yaml")
    ap.add_argument("--data-root", default=None, help="dir with the raw per-subject nii (for eval)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    fmap = fold_subject_map(a.pairs_csv)
    print("fold -> held-out subject:")
    for i, s in fmap.items():
        print(f"  fold {i}: {s}")

    # evaluate each fold's final checkpoint on its held-out subject via eval_cascaded (imported)
    rows = []
    for i, subj in fmap.items():
        fold_dir = Path(a.runs_dir) / f"fold_{i}"
        cks = sorted(glob.glob(str(fold_dir / "cascaded_*.pt")),
                     key=lambda p: int(Path(p).stem.split("_")[1]))
        if not cks:
            print(f"  [skip] fold {i}: no checkpoint in {fold_dir}")
            continue
        ck = cks[-1]                                    # FINAL checkpoint -- no selection
        print(f"\n=== fold {i}: eval {subj} with {Path(ck).name} ===")
        # eval_cern reads the NORMALISED pairs (complete segs), same data training used
        import subprocess
        cmd = [sys.executable, str(ROOT / "scripts" / "eval_cern.py"),
               "--ckpt", ck, "--pairs-csv", a.pairs_csv, "--subject", subj, "--out", str(fold_dir)]
        subprocess.run(cmd, check=False)
        res_path = fold_dir / "cascaded_eval.json"
        if res_path.exists():
            r = json.loads(res_path.read_text())
            r["_fold"] = i
            r["_subject"] = subj
            rows.append(r)

    if not rows:
        print("no fold results found -- have the training folds finished?")
        return

    # aggregate the headline metrics across subjects
    def grab(r, *keys):
        for k in keys:
            r = r.get(k, {}) if isinstance(r, dict) else {}
        return r if isinstance(r, (int, float)) else float("nan")

    metrics = {
        "brain PSNR": lambda r: r.get("image", {}).get("psnr_brain", float("nan")),
        "brain SSIM": lambda r: r.get("image", {}).get("ssim_brain", float("nan")),
        "whole PSNR": lambda r: r.get("whole_volume_no_paste", {}).get("psnr", float("nan")),
        "GM Dice": lambda r: r.get("tissue", {}).get("GM", {}).get("dice", float("nan")),
        "WM Dice": lambda r: r.get("tissue", {}).get("WM", {}).get("dice", float("nan")),
        "GM beta0": lambda r: (r.get("tissue", {}).get("GM", {}).get("betti_pred", [float("nan")]) or [float("nan")])[0],
    }
    print("\n" + "=" * 60)
    print(f"LOSO RESULT ({len(rows)} folds)   mean +/- std across held-out subjects")
    print("=" * 60)
    summary = {}
    for name, fn in metrics.items():
        vals = np.array([fn(r) for r in rows], float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            summary[name] = [float(vals.mean()), float(vals.std())]
            print(f"  {name:12} {vals.mean():8.3f} +/- {vals.std():.3f}   (n={len(vals)})")

    out = Path(a.out) if a.out else Path(a.runs_dir) / "loso_summary.json"
    out.write_text(json.dumps({"folds": rows, "summary": summary}, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
