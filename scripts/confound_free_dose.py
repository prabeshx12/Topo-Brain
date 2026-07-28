"""
Confound-free dose-response: apply THREE segmenters to the SAME images.

WHY. The joint->detached->independent dose-response (0.98->0.41->0.10) used a DIFFERENT generator
for the joint and detached arms, so a skeptic could attribute the trend to generator differences
rather than to segmenter-generator COUPLING. This script removes that confound entirely: it takes
ONE set of images (the joint model's synthetic 7T, and the real 7T) and segments those SAME images
with all three judges --

    1. joint co-trained head      (gradient-coupled to the generator that made the synth)
    2. detached head              (output-coupled: trained on outputs, no gradient)
    3. independent GMM probe      (uncoupled; image-only; no topology prior)

Because the images are identical across judges, any difference in the synth/real beta0 ratio is
attributable to the SEGMENTER'S coupling alone. If the ratio still falls monotonically
(coupled -> uncoupled), the dose-response is airtight.

beta0 = number of 26-connected components (field n_cc); --b0-only skips gudhi (default on -- fast).
Reuses the tested machinery in external_seg_topology.py. Deterministic (GMM seed=0).

Usage:
  python scripts/confound_free_dose.py \
      --joint-ckpt    /eos/user/p/ppokhrel/topobrain/runs/phase3_clean/cascaded_40000.pt \
      --detached-ckpt /eos/user/p/ppokhrel/topobrain/runs/phase3_detached/cascaded_40000.pt \
      --pairs-csv     /eos/user/p/ppokhrel/topobrain/data/norm/pairs_cern.csv \
      --subjects sub-06 sub-07 sub-01 sub-02 \
      --pred7t-dir /eos/user/p/ppokhrel/topobrain/runs/phase3_clean \
      --out /eos/user/p/ppokhrel/topobrain/runs/confound_free
"""
import argparse
import csv
import importlib.util
import json
import statistics as st
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# reuse external_seg_topology's machinery (tiled_seg_head, classical_tissue_seg, topo_of, etc.)
_s = importlib.util.spec_from_file_location("est", ROOT / "scripts" / "external_seg_topology.py")
est = importlib.util.module_from_spec(_s)
_s.loader.exec_module(est)

mh = est.mh
CascadedSynthesisNet = est.CascadedSynthesisNet
tiled_seg_head = est.tiled_seg_head
tiled_full = est.tiled_full
classical_tissue_seg = est.classical_tissue_seg
topo_of = est.topo_of
TISSUES = est.TISSUES


def log(m):
    print(m, flush=True)


def load_model(ckpt, dev):
    st = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = st["config"]["model"]
    m = CascadedSynthesisNet(
        in_channels=1, out_channels=1, num_classes=int(cfg["num_classes"]),
        features=tuple(cfg["features"]), use_attention=cfg["use_attention"]).to(dev)
    m.load_state_dict(st["ema" if "ema" in st else "model"])
    m.eval()
    return m


def row_for(pairs_csv, subject):
    with open(pairs_csv) as f:
        for row in csv.DictReader(f):
            if row["subject"] == subject:
                return row
    raise SystemExit(f"subject {subject} not in {pairs_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-ckpt", required=True)
    ap.add_argument("--detached-ckpt", required=True)
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--pred7t-dir", default=None,
                    help="if set, reuse <dir>/eval_<s>/<s>_pred7T.nii.gz as the synth (else generate)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--full-betti", action="store_true", help="use gudhi (slow); default is b0-only")
    a = ap.parse_args()
    b0_only = not a.full_betti

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"loading joint + detached models on {dev}")
    joint = load_model(a.joint_ckpt, dev)
    detached = load_model(a.detached_ckpt, dev)

    all_rows = []
    for s in a.subjects:
        log(f"\n===== {s} =====")
        row = row_for(a.pairs_csv, s)
        x3 = nib.load(row["input_3t"]).get_fdata().astype(np.float32)
        x7 = nib.load(row["target_7t"]).get_fdata().astype(np.float32)
        gt = np.rint(nib.load(row["seg"]).get_fdata()).astype(np.int32)
        brain = gt > 0

        # the ONE synthetic image every judge will score (from the JOINT generator)
        if a.pred7t_dir:
            p = Path(a.pred7t_dir) / f"eval_{s}" / f"{s}_pred7T.nii.gz"
            if p.exists():
                synth = nib.load(str(p)).get_fdata().astype(np.float32)
                log(f"  reused synth from {p}")
            else:
                log(f"  {p} missing -> generating"); synth, _ = tiled_full(joint, x3, dev)
        else:
            synth, _ = tiled_full(joint, x3, dev)

        # THREE judges, SAME two images (synth, real). Only the segmenter's coupling differs.
        judges = [
            ("joint_head",    lambda img: tiled_seg_head(joint, img, x3, dev)),
            ("detached_head", lambda img: tiled_seg_head(detached, img, x3, dev)),
            ("gmm_probe",     lambda img: classical_tissue_seg(img, brain, seed=0)),
        ]
        subj = {"_subject": s}
        for name, seg_fn in judges:
            log(f"  judge = {name}: segmenting SAME synth and SAME real ...")
            ts = topo_of(seg_fn(synth), f"{name}/synth", b0_only)
            tr = topo_of(seg_fn(x7),    f"{name}/real",  b0_only)
            subj[name] = {"synth": ts, "real": tr}
        all_rows.append(subj)

    # ---- report: synth/real beta0 ratio per judge, per tissue, on IDENTICAL images ----
    def ratio(subj, judge, t):
        s_ = subj[judge]["synth"][t]["n_cc"]
        r_ = subj[judge]["real"][t]["n_cc"]
        return (s_ / r_) if r_ else None

    order = ["joint_head", "detached_head", "gmm_probe"]
    label = {"joint_head": "joint head (coupled)", "detached_head": "detached head (output-coupled)",
             "gmm_probe": "GMM probe (uncoupled)"}
    log("\n" + "=" * 78)
    log("CONFOUND-FREE DOSE-RESPONSE: 3 judges on the SAME images  (GM synth/real beta0 ratio)")
    log("=" * 78)
    log(f"  {'subject':9} " + " ".join(f"{label[j]:>28}" for j in order))
    for subj in all_rows:
        log(f"  {subj['_subject']:9} " +
            " ".join(f"{(('%.2f' % ratio(subj, j, 'GM')) if ratio(subj,j,'GM') is not None else 'n/a'):>28}"
                     for j in order))

    log("\n  MEDIAN GM ratio (identical images -> isolates segmenter coupling):")
    meds = {}
    for j in order:
        vals = [ratio(subj, j, "GM") for subj in all_rows if ratio(subj, j, "GM") is not None]
        meds[j] = st.median(vals) if vals else float("nan")
        log(f"    {label[j]:32} {meds[j]:.2f}")

    mono = meds["joint_head"] >= meds["detached_head"] >= meds["gmm_probe"]
    log("\n  VERDICT: " + (
        f"MONOTONIC ({meds['joint_head']:.2f} -> {meds['detached_head']:.2f} -> {meds['gmm_probe']:.2f}) "
        "on IDENTICAL images -> the dose-response is the SEGMENTER's coupling, not the generator. "
        "Confound closed." if mono else
        f"NOT monotonic ({meds['joint_head']:.2f}, {meds['detached_head']:.2f}, {meds['gmm_probe']:.2f}) "
        "-- report honestly; the earlier trend was partly generator-driven."))

    outp = Path(a.out); outp.mkdir(parents=True, exist_ok=True)
    (outp / "confound_free_dose.json").write_text(json.dumps(all_rows, indent=2, default=float))
    log(f"\nwrote {outp / 'confound_free_dose.json'}")


if __name__ == "__main__":
    main()
