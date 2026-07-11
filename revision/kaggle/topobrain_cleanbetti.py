"""
Fair topology comparison (CPU, Kaggle): raw vs small-island-cleaned predicted
segmentation vs FreeSurfer ground truth, for sub-06.

Tests whether the alarming raw Betti numbers (GM b0=5213 vs GT b0=1) are just
single-voxel speckle. Standard practice: drop connected components below a voxel
threshold, then recompute topology. No inference, no GPU.
"""
import glob
import importlib.util
import json
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gudhi"], check=False)

import numpy as np
import nibabel as nib
from scipy import ndimage


def log(*a):
    print(*a, flush=True)


hits = glob.glob("/kaggle/input/**/compute_betti.py", recursive=True)
spec = importlib.util.spec_from_file_location("cb", hits[0])
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)
betti_3d = cb.betti_3d

log("=== /kaggle/input ===")
for p in sorted(glob.glob("/kaggle/input/**/*.nii*", recursive=True))[:20]:
    log("  ", p)

# Kaggle decompresses uploaded .gz, so match .nii and .nii.gz
pred_hits = sorted(glob.glob("/kaggle/input/**/sub-06_predseg_ddim.nii*", recursive=True))
gt_hits = sorted(glob.glob("/kaggle/input/**/sub-06_ses-2_desc-preproc_T1w_seg.nii*", recursive=True))
assert pred_hits, "predicted seg not found"
assert gt_hits, "GT seg not found"
pred_p, gt_p = pred_hits[0], gt_hits[0]
log("pred:", pred_p)
log("gt  :", gt_p)
pred = np.rint(nib.load(pred_p).get_fdata()).astype(np.int32)
gt = np.rint(nib.load(gt_p).get_fdata()).astype(np.int32)
log("pred", pred.shape, np.unique(pred), "| gt", gt.shape, np.unique(gt))

TISSUE = {1: "CSF", 2: "GM", 3: "WM"}


def clean_small(binary, min_vox):
    lab, n = ndimage.label(binary)
    if n == 0:
        return binary
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    keep = np.where(counts >= min_vox)[0]
    return np.isin(lab, keep)


def cc_lcc(binary):
    if binary.sum() == 0:
        return 0, 0.0
    lab, n = ndimage.label(binary)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return int((counts > 0).sum()), float(counts.max() / binary.sum())


log(f"\n{'tissue':5} {'variant':>10} {'ncc':>7} {'largestCC':>10} {'b0':>7} {'b1':>7} {'b2':>7} {'euler':>7}")
log("-" * 72)
rows = []
for lb, name in TISSUE.items():
    gt_b = gt == lb
    ncc, lcc = cc_lcc(gt_b)
    b0, b1, b2 = betti_3d(gt_b)
    log(f"{name:5} {'GT':>10} {ncc:>7} {lcc:>10.3f} {b0:>7} {b1:>7} {b2:>7} {b0-b1+b2:>7}")
    rows.append(dict(tissue=name, variant="GT", ncc=ncc, largest_cc=round(lcc, 4),
                     b0=int(b0), b1=int(b1), b2=int(b2)))
    for k in [0, 20, 50, 100]:
        pb = pred == lb
        if k > 0:
            pb = clean_small(pb, k)
        ncc, lcc = cc_lcc(pb)
        b0, b1, b2 = betti_3d(pb)
        tag = "pred(raw)" if k == 0 else f"pred>={k}v"
        log(f"{name:5} {tag:>10} {ncc:>7} {lcc:>10.3f} {b0:>7} {b1:>7} {b2:>7} {b0-b1+b2:>7}")
        rows.append(dict(tissue=name, variant=tag, ncc=ncc, largest_cc=round(lcc, 4),
                         b0=int(b0), b1=int(b1), b2=int(b2)))
    log("-" * 72)

with open("/kaggle/working/cleaned_betti.json", "w") as f:
    json.dump(rows, f, indent=2)
log("wrote cleaned_betti.json")
