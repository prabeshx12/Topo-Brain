"""
Break the CIRCULARITY in the topology claim. This is the gate the whole paper rests on.

THE PROBLEM. eval_cern reports GM beta0 = 99 on the synthetic 7T against 2 on the real 7T's
FreeSurfer labels, while Dice (0.81) and HD95 (1.84 mm) look publishable. That contrast IS the
paper. But the 99 is produced by our CO-TRAINED segmentation head -- the very network whose
loss we are about to optimise with the topology term. A reviewer asks the obvious question:

    is the IMAGE topologically broken, or is our seg head simply bad on a held-out subject?

Answering that with the same head is circular. This script answers it with controls.

FOUR ARMS, one metric (src/metrics_honest.betti_numbers, connectivity 26), one LUT
(scripts/remap_aseg_complete.remap), so nothing differs except what we intend to vary:

    A   our head on the SYNTHETIC 7T    -- reproduces the headline
    B   our head on the REAL 7T        -- SAME head, SAME 3T channel, ONLY the image swapped
    C   an EXTERNAL segmenter on the SYNTHETIC 7T
    D   an EXTERNAL segmenter on the REAL 7T   -- the independent reference

WHY ARM B IS INFORMATIVE, AND FREE. The head's input is concat(image, 3T), because the model is
built with seg_sees_input=True (src/model_cascaded.py:189-191). So substituting the real 7T for
the generated image changes EXACTLY ONE CHANNEL and holds the weights and the 3T conditioning
fixed. If B is clean while A is shattered, the head is demonstrably CAPABLE of producing clean
GM, and the fragmentation is caused by the image content -- not by the instrument.
    Honest caveat: during training the head only ever saw GENERATED images in that channel, so
    a real 7T is mildly out-of-distribution for it. Therefore a CLEAN B is conclusive (an OOD
    input cannot make a bad head accidentally clean), while a BROKEN B is AMBIGUOUS and must be
    escalated to arms C/D. Do not over-read a broken B.

WHY C/D ARE DEFINITIVE. An external segmenter never saw our training data and never sees the 3T
-- it reads the image alone. If C >> D, the synthetic image really is topologically broken,
independently of anything we trained. If C ~= D, then the 99 was OUR head, and the benchmark
claim does not survive; we report that honestly and pivot.

HOW TO READ THE RESULT:
    B clean while A shattered      -> the image causes it; the head is capable
    C >> D                         -> DEFINITIVE: image is broken, independent of our head
    C ~= D, both near the GT       -> the 99 was our instrument -> the claim collapses. Say so.

RESOLUTION WARNING. Betti numbers are resolution- and segmenter-dependent. SynthSeg works at
1 mm while our data is 0.65 mm, so C/D are NOT comparable in absolute terms to A/B/GT. Compare
ONLY WITHIN a segmenter: A vs B, and C vs D. Never quote a C-vs-A delta.

Usage
-----
  # arms A + B (free -- no external tool needed) and export volumes for the segmenter:
  python scripts/external_seg_topology.py \
      --ckpt  /eos/user/p/ppokhrel/topobrain/runs/phase3_clean/cascaded_40000.pt \
      --pairs-csv /eos/user/p/ppokhrel/topobrain/data/norm/pairs_cern.csv \
      --subject sub-07 --out <dir> --export-for-synthseg

  # then segment the two exported volumes with ANY external tool, e.g.
  mri_synthseg --i <dir>/ext_in_synth.nii.gz --o <dir>/ext_out_synth.nii.gz --robust
  mri_synthseg --i <dir>/ext_in_real.nii.gz  --o <dir>/ext_out_real.nii.gz  --robust

  # finally arms C + D:
  python scripts/external_seg_topology.py ... --subject sub-07 --out <dir> \
      --ext-seg-synth <dir>/ext_out_synth.nii.gz --ext-seg-real <dir>/ext_out_real.nii.gz
"""
import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# same guarded module loading as eval_cern.py: keeps src/__init__'s heavy deps out of the way
pkg = importlib.util.module_from_spec(importlib.util.spec_from_loader("src", loader=None))
pkg.__path__ = [str(ROOT / "src")]
sys.modules.setdefault("src", pkg)
for _n in ("model", "model_cascaded", "metrics_honest"):
    _s = importlib.util.spec_from_file_location(f"src.{_n}", ROOT / "src" / f"{_n}.py")
    _m = importlib.util.module_from_spec(_s)
    sys.modules[f"src.{_n}"] = _m
    _s.loader.exec_module(_m)
mh = sys.modules["src.metrics_honest"]
CascadedSynthesisNet = sys.modules["src.model_cascaded"].CascadedSynthesisNet

_rs = importlib.util.spec_from_file_location("_remap", ROOT / "scripts" / "remap_aseg_complete.py")
_rm = importlib.util.module_from_spec(_rs)
_rs.loader.exec_module(_rm)
remap_aseg = _rm.remap                     # THE SAME LUT the ground truth was built with

# Reuse eval_cascaded's OWN tiling + window. This matters: eval_cascaded defines a CUSTOM
# tukey() (scripts/eval_cascaded.py:51) that is NOT scipy's. Importing scipy's instead would
# blend arm B with a different window than arm A, making the two arms subtly incomparable --
# which is precisely the confound this script exists to eliminate.
_ec = importlib.util.spec_from_file_location("_ec", ROOT / "scripts" / "eval_cascaded.py")
_ecm = importlib.util.module_from_spec(_ec)
_ec.loader.exec_module(_ecm)
tukey = _ecm.tukey
tiled_full = _ecm.tiled

TISSUES = {1: "CSF", 2: "GM", 3: "WM"}


def log(m):
    print(m, flush=True)


def row_for(pairs_csv, subject):
    with open(pairs_csv) as f:
        for row in csv.DictReader(f):
            if row["subject"] == subject:
                return row
    raise SystemExit(f"subject {subject} not in {pairs_csv}")


@torch.no_grad()
def tiled_seg_head(model, img_vol, x3_vol, dev, patch=64, overlap=32):
    """Run ONLY the seg head over a volume, feeding it concat(img_vol, x3_vol).

    This is the arm-B machinery: identical tiling/blending to eval_cascaded.tiled, but the
    image channel is supplied by the caller instead of being produced by the generator. That
    is what makes 'our head on the REAL 7T' a controlled swap of exactly one input channel.
    """
    D, H, W = img_vol.shape
    stride = patch - overlap
    sacc = np.zeros((4, D, H, W), np.float64)
    wacc = np.zeros((D, H, W), np.float64)
    w1 = tukey(patch, overlap / patch)          # eval_cascaded's own window -- see note above
    win = (w1[None, None, :] * w1[None, :, None] * w1[:, None, None])

    def starts(L):
        s = list(range(0, L - patch + 1, stride))
        if s and s[-1] + patch < L:
            s.append(L - patch)
        return s or [0]

    ds, hs, ws = starts(D), starts(H), starts(W)
    tot, n = len(ds) * len(hs) * len(ws), 0
    t0 = time.perf_counter()
    for d in ds:
        for h in hs:
            for w in ws:
                pi = img_vol[d:d + patch, h:h + patch, w:w + patch]
                px = x3_vol[d:d + patch, h:h + patch, w:w + patch]
                sh = pi.shape
                if sh != (patch,) * 3:                       # pad with -1.0 == background in [-1,1]
                    qi = np.full((patch,) * 3, -1.0, np.float32); qi[:sh[0], :sh[1], :sh[2]] = pi
                    qx = np.full((patch,) * 3, -1.0, np.float32); qx[:sh[0], :sh[1], :sh[2]] = px
                    pi, px = qi, qx
                ti = torch.from_numpy(pi).float()[None, None].to(dev)
                tx = torch.from_numpy(px).float()[None, None].to(dev)
                seg_in = torch.cat([ti, tx], dim=1) if model.seg_sees_input else ti
                sg = model.seg_head(seg_in).cpu().numpy().squeeze()[:, :sh[0], :sh[1], :sh[2]]
                ww = win[:sh[0], :sh[1], :sh[2]]
                for c in range(4):
                    sacc[c, d:d + sh[0], h:h + sh[1], w:w + sh[2]] += sg[c] * ww
                wacc[d:d + sh[0], h:h + sh[1], w:w + sh[2]] += ww
                n += 1
                if n % 100 == 0:
                    log(f"    seg-head patch {n}/{tot}  [{time.perf_counter()-t0:.0f}s]")
    m = wacc > 0
    for c in range(4):
        sacc[c][m] /= wacc[m]
    return np.argmax(sacc, 0).astype(np.uint8)


def topo_of(seg, name):
    """Betti + component count per tissue, with the honest metrics (connectivity 26)."""
    out = {}
    for lb, tname in TISSUES.items():
        m = seg == lb
        t0 = time.perf_counter()
        b0, b1, b2 = mh.betti_numbers(m, 26)
        ncc, lcc = mh.connected_components(m, 26)
        out[tname] = {"betti": [b0, b1, b2], "n_cc": ncc, "largest_cc_frac": round(lcc, 4),
                      "euler": b0 - b1 + b2, "voxels": int(m.sum())}
        log(f"    {name:26} {tname:3}: b0={b0:5d} b1={b1:4d} b2={b2:3d}  "
            f"cc={ncc:5d}  [{time.perf_counter()-t0:.0f}s]")
    return out


def classical_tissue_seg(vol, brain_mask, seed=0, max_fit=200_000):
    """Unsupervised 3-class Gaussian-mixture tissue segmentation from intensities alone.

    WHY THIS IS A LEGITIMATE INDEPENDENT PROBE (arms C/D without any external install):
      * it never saw our training data -- it is fit per-image, unsupervised;
      * it reads ONLY the image (no 3T channel, no labels, no network of ours);
      * it does NOT enforce topology. This is the crucial property. The FreeSurfer GT's
        GM beta0 ~= 2 is an ALGORITHMIC GUARANTEE of its topology-corrected surface
        reconstruction, not a measurement of the 7T image -- so comparing any voxel-wise
        segmenter to it is rigged. This probe has no such guarantee, so a difference
        between arms is attributable to the IMAGE;
      * identical procedure, identical brain mask, on both arms.

    This is essentially FSL FAST minus the MRF spatial prior. The MRF is omitted DELIBERATELY:
    it would smooth away exactly the speckle we are trying to measure, which is the thing under
    test. So this probe is, if anything, MORE sensitive to fragmentation than FAST would be --
    it cannot hide a difference, only reveal one.

    Cluster -> tissue by mean intensity (T1w: CSF < GM < WM). Fit on a random subsample for
    speed, then predict on every brain voxel.
    """
    v = vol[brain_mask].astype(np.float64).reshape(-1, 1)
    rng = np.random.default_rng(seed)
    fit_v = v if v.shape[0] <= max_fit else v[rng.choice(v.shape[0], max_fit, replace=False)]

    try:
        from sklearn.mixture import GaussianMixture
        g = GaussianMixture(n_components=3, covariance_type="full", random_state=seed,
                            max_iter=200, n_init=2).fit(fit_v)
        lab = g.predict(v)
        means = g.means_.ravel()
        how = "sklearn GaussianMixture"
    except Exception as e:                       # no sklearn in the LCG view -> 1-D k-means
        log(f"    (sklearn unavailable: {e}; falling back to 1-D k-means)")
        c = np.percentile(fit_v, [15, 50, 85]).astype(np.float64)
        for _ in range(100):
            d = np.abs(fit_v - c[None, :])
            a = np.argmin(d, 1)
            nc = np.array([fit_v[a == k].mean() if (a == k).any() else c[k] for k in range(3)])
            if np.allclose(nc, c, atol=1e-7):
                break
            c = nc
        lab = np.argmin(np.abs(v - c[None, :]), 1)
        means = c
        how = "1-D k-means (3 clusters)"

    order = np.argsort(means)                    # increasing intensity: CSF, GM, WM
    lut = np.zeros(3, np.uint8)
    lut[order[0]], lut[order[1]], lut[order[2]] = 1, 2, 3
    out = np.zeros(vol.shape, np.uint8)
    out[brain_mask] = lut[lab]
    log(f"    probe: {how}; cluster means {np.sort(means).round(3).tolist()} -> CSF/GM/WM")
    return out


def export_for_segmenter(vol, aff, path):
    """[-1,1] -> [0,255] float. IDENTICAL transform for real and synthetic, so the external
    segmenter cannot be advantaged on one arm by intensity scaling alone."""
    v = ((np.clip(vol, -1.0, 1.0) + 1.0) * 0.5 * 255.0).astype(np.float32)
    nib.save(nib.Nifti1Image(v, aff), str(path))
    log(f"  wrote {path}  (range {v.min():.1f}..{v.max():.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--subject", default="sub-07")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred7t", default=None,
                    help="reuse the pred7T.nii.gz eval_cern already wrote (skips inference)")
    ap.add_argument("--use-ema", type=int, default=1)
    ap.add_argument("--export-for-synthseg", action="store_true",
                    help="write [0,255] copies of the synthetic and real 7T for an external tool")
    ap.add_argument("--ext-seg-synth", default=None, help="aseg-style labels on the SYNTHETIC 7T")
    ap.add_argument("--ext-seg-real", default=None, help="aseg-style labels on the REAL 7T")
    ap.add_argument("--classical-probe", action="store_true",
                    help="arms C/D via the built-in unsupervised GMM tissue probe -- no external "
                         "tool needed, and (unlike FreeSurfer) it enforces NO topology prior")
    a = ap.parse_args()

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    row = row_for(a.pairs_csv, a.subject)
    x3_img = nib.load(row["input_3t"])
    x3 = x3_img.get_fdata().astype(np.float32)
    x7 = nib.load(row["target_7t"]).get_fdata().astype(np.float32)
    gt = np.rint(nib.load(row["seg"]).get_fdata()).astype(np.int32)
    aff = x3_img.affine
    log(f"{a.subject}: {x3.shape}  labels {sorted(np.unique(gt).tolist())}")

    res = {"_subject": a.subject, "_connectivity": 26}

    # ---- the reference the model was trained against -------------------------------------
    log("\nGT  (FreeSurfer aparc+aseg of the REAL 7T, the training reference)")
    res["GT_freesurfer_real7T"] = topo_of(gt, "GT/real7T")

    # ---- the synthetic volume -------------------------------------------------------------
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg = st["config"]["model"]
    model = CascadedSynthesisNet(
        in_channels=1, out_channels=1, num_classes=int(cfg["num_classes"]),
        features=tuple(cfg["features"]), use_attention=cfg["use_attention"]).to(dev)
    model.load_state_dict(st["ema" if (a.use_ema and "ema" in st) else "model"])
    model.eval()
    log(f"loaded step {st['step']} | dev {dev} | seg_sees_input={model.seg_sees_input}")

    if a.pred7t and Path(a.pred7t).exists():
        synth = nib.load(a.pred7t).get_fdata().astype(np.float32)
        log(f"reused synthetic volume from {a.pred7t} (no generator inference)")
        log("\nARM A: our head on the SYNTHETIC 7T (head re-run on the saved volume)")
        seg_A = tiled_seg_head(model, synth, x3, dev)
    else:
        log("\nARM A: full model on the 3T -> synthetic 7T + its segmentation")
        synth, seg_A = tiled_full(model, x3, dev)
    res["A_ourhead_on_synth"] = topo_of(seg_A, "A our head/SYNTH")

    # ---- ARM B: the controlled swap -- same head, same 3T, REAL image ---------------------
    log("\nARM B: our head on the REAL 7T (ONLY the image channel swapped; weights + 3T fixed)")
    seg_B = tiled_seg_head(model, x7, x3, dev)
    res["B_ourhead_on_real"] = topo_of(seg_B, "B our head/REAL")
    nib.save(nib.Nifti1Image(seg_B, aff), str(outdir / f"{a.subject}_segB_ourhead_on_real.nii.gz"))

    # ---- ARMS C/D via the built-in probe: independent, image-only, NO topology prior --------
    if a.classical_probe:
        brain = gt > 0
        log("\nARM C (probe): unsupervised GMM tissue seg on the SYNTHETIC 7T (image only)")
        seg_C = classical_tissue_seg(synth, brain, seed=0)
        res["C_probe_on_synth"] = topo_of(seg_C, "C probe/SYNTH")
        nib.save(nib.Nifti1Image(seg_C, aff), str(outdir / f"{a.subject}_segC_probe_synth.nii.gz"))

        log("\nARM D (probe): unsupervised GMM tissue seg on the REAL 7T (identical procedure)")
        seg_D = classical_tissue_seg(x7, brain, seed=0)
        res["D_probe_on_real"] = topo_of(seg_D, "D probe/REAL")
        nib.save(nib.Nifti1Image(seg_D, aff), str(outdir / f"{a.subject}_segD_probe_real.nii.gz"))

    # ---- ARMS C/D: an external, independent segmenter --------------------------------------
    if a.export_for_synthseg:
        log("\nexporting [0,255] volumes for an external segmenter (identical transform both arms)")
        export_for_segmenter(synth, aff, outdir / "ext_in_synth.nii.gz")
        export_for_segmenter(x7, aff, outdir / "ext_in_real.nii.gz")

    for tag, path, key in (("C ext/SYNTH", a.ext_seg_synth, "C_external_on_synth"),
                           ("D ext/REAL", a.ext_seg_real, "D_external_on_real")):
        if not path:
            continue
        log(f"\nARM {tag}: {path}")
        ext = np.rint(nib.load(path).get_fdata()).astype(int)
        seg_e = remap_aseg(ext)                       # SAME LUT as the ground truth
        res[key] = topo_of(seg_e, f"{tag}")

    # ---- verdict ---------------------------------------------------------------------------
    log("\n" + "=" * 78)
    log(f"TOPOLOGY vs SEGMENTER  ({a.subject}, connectivity 26)   beta0 per tissue")
    log("=" * 78)
    log(f"  {'arm':34} {'CSF':>6} {'GM':>6} {'WM':>6}")
    order = [("GT   FreeSurfer / REAL 7T  (*)", "GT_freesurfer_real7T"),
             ("A    our head  / SYNTHETIC", "A_ourhead_on_synth"),
             ("B    our head  / REAL 7T", "B_ourhead_on_real"),
             ("C    GMM probe / SYNTHETIC", "C_probe_on_synth"),
             ("D    GMM probe / REAL 7T", "D_probe_on_real"),
             ("C    external  / SYNTHETIC", "C_external_on_synth"),
             ("D    external  / REAL 7T", "D_external_on_real")]
    for label, key in order:
        if key not in res:
            continue
        r = res[key]
        log(f"  {label:34} {r['CSF']['betti'][0]:>6} {r['GM']['betti'][0]:>6} "
            f"{r['WM']['betti'][0]:>6}")
    log("  (*) NOTE: the FreeSurfer GT's low beta0 is an ALGORITHMIC GUARANTEE of its")
    log("      topology-corrected surface reconstruction, NOT a measurement of the 7T image.")
    log("      It is the right TRAINING TARGET but the WRONG topology reference: comparing any")
    log("      voxel-wise segmenter against it is rigged. Only same-segmenter comparisons")
    log("      (A vs B, C vs D) carry information about the IMAGE.")

    gm_a = res["A_ourhead_on_synth"]["GM"]["betti"][0]
    gm_b = res["B_ourhead_on_real"]["GM"]["betti"][0]
    gm_gt = res["GT_freesurfer_real7T"]["GM"]["betti"][0]
    log("\nREAD-OUT (arms A vs B -- the head is the constant, the image is the variable):")
    if gm_b <= max(3 * max(gm_gt, 1), gm_gt + 5) and gm_a > 3 * max(gm_b, 1):
        log(f"  GM: A={gm_a} shattered vs B={gm_b} clean (GT {gm_gt}).")
        log("  -> The SAME head reads the REAL 7T cleanly and the SYNTHETIC one as broken.")
        log("     The head is CAPABLE; the fragmentation is caused by the IMAGE. (World A)")
        log("     This is the result the benchmark claim needs. Confirm with arms C/D.")
    elif gm_b > 3 * max(gm_gt, 1):
        log(f"  GM: A={gm_a}, B={gm_b}, GT={gm_gt}. The head fragments the REAL 7T too.")
        log("  -> AMBIGUOUS: either the head is weak, or the real 7T is out-of-distribution")
        log("     for it (it only ever saw GENERATED images). Arms C/D are now REQUIRED.")
    else:
        log(f"  GM: A={gm_a}, B={gm_b}, GT={gm_gt} -- inconclusive; escalate to arms C/D.")

    pair = None
    if "C_external_on_synth" in res and "D_external_on_real" in res:
        pair = ("an INDEPENDENT external segmenter", "C_external_on_synth", "D_external_on_real")
    elif "C_probe_on_synth" in res and "D_probe_on_real" in res:
        pair = ("the unsupervised GMM probe (image-only, NO topology prior)",
                "C_probe_on_synth", "D_probe_on_real")

    if pair:
        what, ck, dk = pair
        log(f"\nREAD-OUT (arms C vs D -- {what}).")
        log("  This is THE test: same segmenter, same brain mask, only the IMAGE differs.")
        any_break = False
        for t in ("CSF", "GM", "WM"):
            c, d = res[ck][t]["betti"][0], res[dk][t]["betti"][0]
            ratio = c / max(d, 1)
            verdict = "SYNTH MORE BROKEN" if c > 2 * max(d, 1) else (
                      "synth SMOOTHER" if d > 2 * max(c, 1) else "comparable")
            any_break |= (c > 2 * max(d, 1))
            log(f"    {t:3}: synth b0={c:5d}  real b0={d:5d}   ratio {ratio:5.2f}x   {verdict}")
        if any_break:
            log("  -> The synthetic image IS measurably more fragmented than the real 7T under a")
            log("     segmenter that never saw our data and enforces no topology. The benchmark")
            log("     claim STANDS: standard metrics (Dice/HD95/PSNR) miss this, topology sees it.")
        else:
            log("  -> The independent segmenter does NOT find the synthetic image more broken")
            log("     than the real one. The GM=99 was OUR co-trained head, not the image.")
            log("     The benchmark claim does NOT survive. Report this honestly and pivot to")
            log("     what still stands (the fidelity wall + the protocol-sensitivity result).")
            log("     Do NOT publish the co-trained-head number as evidence about the image.")
        log("  (Compare only WITHIN a segmenter. Absolute values across segmenters differ by")
        log("   construction -- and across resolutions too, if the external tool resamples.)")
    else:
        log("\narms C/D not supplied. Re-run with --classical-probe (instant, no install), or")
        log("with --ext-seg-synth/--ext-seg-real from an external tool. Until one of those is")
        log("done, the benchmark claim is NOT defensible: arms A/B alone cannot separate")
        log("'the image is broken' from 'our head is weak'.")

    (outdir / "external_seg_topology.json").write_text(json.dumps(res, indent=2, default=float))
    log(f"\nwrote {outdir / 'external_seg_topology.json'}")


if __name__ == "__main__":
    main()
