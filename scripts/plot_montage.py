"""
Qualitative montage: the 'show, don't tell' figure. One representative subject, one axial slice:

  real 7T | synthetic 7T | co-trained head's GM components | independent segmenter's GM components

The last two panels colour each connected component of grey matter differently, so the number of
colours IS beta0. The co-trained head shows a few large components (looks preserved); the
independent segmenter shows many specks (the over-smoothing it reveals). This makes the abstract
ratio visceral. GPU-light (one subject).

Usage:
  python scripts/plot_montage.py \
      --joint-ckpt /eos/user/p/ppokhrel/topobrain/runs/phase3_clean/cascaded_40000.pt \
      --pairs-csv  /eos/user/p/ppokhrel/topobrain/data/norm/pairs_cern.csv \
      --subject sub-06 \
      --pred7t /eos/user/p/ppokhrel/topobrain/runs/phase3_clean/eval_sub-06/sub-06_pred7T.nii.gz \
      --out /eos/user/p/ppokhrel/topobrain/runs/figures
"""
import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_s = importlib.util.spec_from_file_location("est", ROOT / "scripts" / "external_seg_topology.py")
est = importlib.util.module_from_spec(_s)
_s.loader.exec_module(est)


def gm_component_labels(seg, conn=26):
    """3D-label the GM (class 2) connected components; return (labelled array, count=beta0)."""
    st = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[conn])
    lab, n = ndimage.label(seg == 2, structure=st)
    return lab, int(n)


def best_axial_slice(seg):
    """axial = axis 2 (RAS); pick the slice with the most GM voxels."""
    gm = (seg == 2)
    areas = gm.sum(axis=(0, 1))
    return int(np.argmax(areas))


def render_montage(real, synth, seg_coupled, seg_indep, z, out, subject=""):
    """The 4-panel figure. Kept a pure function of arrays so it is testable without a model."""
    lab_c, n_c = gm_component_labels(seg_coupled)
    lab_i, n_i = gm_component_labels(seg_indep)

    def comp_rgb(lab2d):
        # colour each component; background -> black. Random permutation so neighbours differ.
        rng = np.random.default_rng(0)
        m = lab2d.max()
        lut = np.zeros((m + 1, 3))
        if m > 0:
            cols = plt.cm.nipy_spectral(np.linspace(0.05, 0.95, m))[:, :3]
            lut[1:] = cols[rng.permutation(m)]
        return lut[lab2d]

    def rot(a):
        return np.rot90(a)                       # display convention: superior up

    fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.8))
    ax[0].imshow(rot(real[:, :, z]), cmap="gray"); ax[0].set_title("real 7T")
    ax[1].imshow(rot(synth[:, :, z]), cmap="gray"); ax[1].set_title("synthetic 7T")
    ax[2].imshow(rot(comp_rgb(lab_c[:, :, z]))); ax[2].set_title(f"co-trained head\nGM $\\beta_0$ = {n_c}")
    ax[3].imshow(rot(comp_rgb(lab_i[:, :, z]))); ax[3].set_title(f"independent segmenter\nGM $\\beta_0$ = {n_i}")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{subject}: same synthetic image, two segmenters" if subject else
                 "same synthetic image, two segmenters", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(Path(out) / f"fig_montage_{subject or 'subj'}.png", dpi=300, bbox_inches="tight")
    fig.savefig(Path(out) / f"fig_montage_{subject or 'subj'}.pdf", bbox_inches="tight")
    plt.close(fig)
    return n_c, n_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-ckpt", required=True)
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--subject", default="sub-06")
    ap.add_argument("--pred7t", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    with open(a.pairs_csv) as f:
        row = next(r for r in csv.DictReader(f) if r["subject"] == a.subject)
    x3 = nib.load(row["input_3t"]).get_fdata().astype(np.float32)
    x7 = nib.load(row["target_7t"]).get_fdata().astype(np.float32)
    gt = np.rint(nib.load(row["seg"]).get_fdata()).astype(np.int32)
    brain = gt > 0

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stt = torch.load(a.joint_ckpt, map_location=dev, weights_only=False)
    cfg = stt["config"]["model"]
    model = est.CascadedSynthesisNet(in_channels=1, out_channels=1, num_classes=int(cfg["num_classes"]),
                                     features=tuple(cfg["features"]), use_attention=cfg["use_attention"]).to(dev)
    model.load_state_dict(stt["ema" if "ema" in stt else "model"]); model.eval()

    if a.pred7t and Path(a.pred7t).exists():
        synth = nib.load(a.pred7t).get_fdata().astype(np.float32)
        print(f"reused synth {a.pred7t}", flush=True)
    else:
        synth, _ = est.tiled_full(model, x3, dev)

    seg_coupled = est.tiled_seg_head(model, synth, x3, dev)     # co-trained head on synth
    seg_indep = est.classical_tissue_seg(synth, brain, seed=0)  # independent GMM on synth
    z = best_axial_slice(gt)
    n_c, n_i = render_montage(x7, synth, seg_coupled, seg_indep, z, out, a.subject)
    print(f"slice z={z} | co-trained GM beta0={n_c} | independent GM beta0={n_i}", flush=True)
    print(f"wrote {out}/fig_montage_{a.subject}.png", flush=True)


if __name__ == "__main__":
    main()
