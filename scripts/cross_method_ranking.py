"""
Cross-method test: does a coupled segmenter distort the RANKING of synthesis methods?

The circular-evaluation finding so far concerns one generator. The cross-method claim is stronger:
if you compare DIFFERENT synthesis methods using a segmenter co-trained with ONE of them, that
segmenter is biased toward the method it was trained with, and can rank methods differently from an
independent judge. We test this with two methods on the same split:

    (i)  our cascaded model  -- the one our segmentation head was co-trained with
    (ii) FR-U-Net            -- an independent architecture (Acs & Zhuang), same split/seed

For each method's synthetic 7T we compute the grey-matter synthetic/real beta0 ratio under TWO
judges: our co-trained head, and the independent GMM probe. We then ask whether the two judges
RANK the methods the same way.

CRITICAL: FR-U-Net ends in a sigmoid ([0,1]); our head and metrics live in [-1,1]. We reuse
eval_frunet's tested inverse map (pred = pred01*2-1) with its assert, so the [0,1]-vs-[-1,1] trap
that would fake a result cannot recur. beta0 via connected components (fast, no gudhi).
Deterministic (GMM seed 0).

Usage:
  python scripts/cross_method_ranking.py \
      --joint-ckpt  /eos/user/p/ppokhrel/topobrain/runs/phase3_clean/cascaded_40000.pt \
      --frunet-ckpt /eos/user/p/ppokhrel/topobrain/runs/frunet_baseline/frunet_39999.pt \
      --pairs-csv   /eos/user/p/ppokhrel/topobrain/data/norm/pairs_cern.csv \
      --subjects sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10 \
      --out /eos/user/p/ppokhrel/topobrain/runs/cross_method
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


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


est = _load("est", "scripts/external_seg_topology.py")   # our judges + machinery
ef = _load("ef", "scripts/eval_frunet.py")               # FRUNet + tested range map

mh = est.mh
CascadedSynthesisNet = est.CascadedSynthesisNet
tiled_seg_head = est.tiled_seg_head
tiled_full = est.tiled_full
classical_tissue_seg = est.classical_tissue_seg
topo_of = est.topo_of


def log(m):
    print(m, flush=True)


def row_for(pairs_csv, subject):
    with open(pairs_csv) as f:
        for row in csv.DictReader(f):
            if row["subject"] == subject:
                return row
    raise SystemExit(f"subject {subject} not in {pairs_csv}")


def load_cascaded(ckpt, dev):
    stt = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = stt["config"]["model"]
    m = CascadedSynthesisNet(in_channels=1, out_channels=1, num_classes=int(cfg["num_classes"]),
                             features=tuple(cfg["features"]), use_attention=cfg["use_attention"]).to(dev)
    m.load_state_dict(stt["ema" if "ema" in stt else "model"])
    m.eval()
    return m


def gm_ratio(seg_synth, seg_real):
    a = topo_of(seg_synth, "synth", b0_only=True)["GM"]["n_cc"]
    b = topo_of(seg_real,  "real",  b0_only=True)["GM"]["n_cc"]
    return (a / b) if b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-ckpt", required=True)
    ap.add_argument("--frunet-ckpt", required=True)
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"loading our cascaded model + FR-U-Net on {dev}")
    ours = load_cascaded(a.joint_ckpt, dev)
    frunet = ef.FRUNet().to(dev)
    frunet.load_state_dict(torch.load(a.frunet_ckpt, map_location=dev)["model"])
    frunet.eval()

    rows = []
    for s in a.subjects:
        log(f"\n===== {s} =====")
        row = row_for(a.pairs_csv, s)
        x3 = nib.load(row["input_3t"]).get_fdata().astype(np.float32)
        x7 = nib.load(row["target_7t"]).get_fdata().astype(np.float32)
        gt = np.rint(nib.load(row["seg"]).get_fdata()).astype(np.int32)
        brain = gt > 0

        # --- generate each method's synthetic 7T on the SAME normalised input ---
        our_synth, _ = tiled_full(ours, x3, dev)                       # [-1,1] by construction
        fr01 = ef.tiled(frunet, x3, dev)                               # FR-U-Net sigmoid [0,1]
        assert 0.0 <= fr01.min() and fr01.max() <= 1.0, f"FR-U-Net not in [0,1]: {fr01.min()},{fr01.max()}"
        fr_synth = ef.to_model_range(fr01)                             # -> [-1,1], the tested map
        assert -1.001 <= fr_synth.min() and fr_synth.max() <= 1.001

        # --- real 7T segmentations (once per judge) ---
        head_real = tiled_seg_head(ours, x7, x3, dev)
        gmm_real = classical_tissue_seg(x7, brain, seed=0)

        # --- judge each method's synth with BOTH judges ---
        r = {"_subject": s}
        for method, synth in (("ours", our_synth), ("frunet", fr_synth)):
            head_ratio = gm_ratio(tiled_seg_head(ours, synth, x3, dev), head_real)
            gmm_ratio = gm_ratio(classical_tissue_seg(synth, brain, seed=0), gmm_real)
            r[method] = {"cotrained_head": head_ratio, "independent_gmm": gmm_ratio}
            log(f"  {method:6}: co-trained head ratio {head_ratio:.2f} | independent GMM {gmm_ratio:.2f}")
        rows.append(r)

    # ---- ranking under each judge ----
    def med(method, judge):
        vals = [x[method][judge] for x in rows if x[method][judge] is not None]
        return st.median(vals) if vals else float("nan")

    log("\n" + "=" * 78)
    log("CROSS-METHOD RANKING  (GM synth/real beta0 ratio; higher = judged 'more preserved')")
    log("=" * 78)
    log(f"  {'judge':22} {'ours':>10} {'FR-U-Net':>10}   ranking")
    for judge, name in (("cotrained_head", "our co-trained head"), ("independent_gmm", "independent GMM")):
        o, f = med("ours", judge), med("frunet", judge)
        rank = "ours > FR-U-Net" if o > f else ("FR-U-Net > ours" if f > o else "tie")
        log(f"  {name:22} {o:>10.2f} {f:>10.2f}   {rank}")

    o_h, f_h = med("ours", "cotrained_head"), med("frunet", "cotrained_head")
    o_g, f_g = med("ours", "independent_gmm"), med("frunet", "independent_gmm")
    coupled_pref_ours = o_h > f_h
    indep_pref_ours = o_g > f_g
    log("\n  READ-OUT:")
    if coupled_pref_ours and not indep_pref_ours:
        log("  -> RANK FLIP: the co-trained head ranks OUR method above FR-U-Net, but the independent")
        log("     judge ranks them the other way. A coupled segmenter favours the method it was")
        log("     trained with -- using it to COMPARE methods is biased. Strongest cross-method result.")
    elif coupled_pref_ours and indep_pref_ours and (o_h / max(f_h, 1e-6)) > 1.5 * (o_g / max(f_g, 1e-6)):
        log("  -> NO hard flip, but the co-trained head INFLATES our method's lead far beyond what the")
        log("     independent judge shows -- the coupled segmenter still distorts the comparison in our")
        log("     favour. Report the magnitude of the distortion.")
    else:
        log("  -> No ranking distortion detected between these two methods on GM. Report honestly;")
        log("     the single-method circularity + dose-response remain the core results.")
    log("  (Note: our co-trained head was trained on OUR generator's outputs; FR-U-Net's are")
    log("   out-of-distribution for it -- which is exactly the bias under test.)")

    outp = Path(a.out); outp.mkdir(parents=True, exist_ok=True)
    (outp / "cross_method_ranking.json").write_text(json.dumps(rows, indent=2, default=float))
    log(f"\nwrote {outp / 'cross_method_ranking.json'}")


if __name__ == "__main__":
    main()
