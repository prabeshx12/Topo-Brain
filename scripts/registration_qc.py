"""
Registration QC for the paired 3T/7T volumes. This is a GATE, not a footnote.

WHY THIS MUST RUN BEFORE ANY CLAIM. Every headline number is referenced to the paired real 7T:
brain-masked SSIM/PSNR, tissue Dice, HD95, and -- critically -- the Betti "error vs the real
7T" that the whole topology thesis rests on. All of them silently assume the 3T and 7T are
voxel-aligned. If they are not:

  * the ~0.45 brain-only SSIM floor is partly MISREGISTRATION, not the perception-distortion
    wall, and the "more training does not help" conclusion has a mundane alternative cause;
  * HD95/Betti "errors" partly measure misalignment rather than synthesis quality;
  * worst, a topology loss that merely SMOOTHS could look good simply by being less sensitive
    to misalignment -- a fake win we would have no way to detect downstream.

THE TEST THAT ACTUALLY DECIDES IT. A similarity score on its own proves nothing (you cannot
say whether MI=0.42 is good). The decisive question is:

    does SHIFTING the 7T make it fit the 3T BETTER?

If the pair is correctly registered, ANY translation must make the match WORSE, so the
similarity profile peaks at zero shift. If the profile peaks at +2 voxels along an axis, the
pair is misaligned by 2 voxels (= 1.3 mm at 0.65 mm) along that axis, and that is measured, not
guessed. This is self-calibrating: it needs no threshold on the absolute score.

WHY MUTUAL INFORMATION. 3T and 7T have DIFFERENT contrast by construction (that is the whole
point of the task), so correlation-based scores are inappropriate -- a perfect registration can
have low correlation simply because the tissue intensities map differently. MI is the standard
cross-modality criterion: it measures statistical dependence, not intensity agreement.

Axis-wise 1-D profiles (not a full 3-D search) are used deliberately: 3 x (2R+1) evaluations
instead of (2R+1)^3, which is what makes this cheap enough to run on all 10 subjects. It
detects any gross translational misalignment, which is the failure mode that matters here.
It does NOT detect rotation or scaling -- stated honestly; if the profiles are clean but
something still looks wrong, escalate to a real registration tool.

Usage:
  python scripts/registration_qc.py \
      --pairs-csv /eos/user/p/ppokhrel/topobrain/data/norm/pairs_cern.csv \
      --out /eos/user/p/ppokhrel/topobrain/runs/regqc            # all subjects
  python scripts/registration_qc.py ... --subjects sub-06 sub-07 # or a subset
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AXES = {0: "sagittal (L-R)", 1: "coronal (A-P)", 2: "axial (S-I)"}   # RAS, matches metrics_protocols


def log(m):
    print(m, flush=True)


def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 64,
                       rng: tuple = (-1.0, 1.0)) -> float:
    """MI in nats between two intensity vectors. Cross-contrast safe (see module docstring)."""
    hist, _, _ = np.histogram2d(a, b, bins=bins, range=[list(rng), list(rng)])
    pxy = hist / hist.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])))


def qc_subject(row, max_shift=4, bins=64):
    x3_img = nib.load(row["input_3t"])
    x3 = x3_img.get_fdata().astype(np.float32)
    x7 = nib.load(row["target_7t"]).get_fdata().astype(np.float32)
    seg_img = nib.load(row["seg"])
    gt = np.rint(seg_img.get_fdata()).astype(np.int32)
    spacing = tuple(float(z) for z in x3_img.header.get_zooms()[:3])

    out = {"subject": row["subject"], "spacing_mm": spacing, "shape": list(x3.shape)}

    # ---- 1. geometry: are they even on the same grid? ------------------------------------
    same_shape = (x3.shape == x7.shape == gt.shape)
    same_affine = bool(np.allclose(x3_img.affine, nib.load(row["target_7t"]).affine, atol=1e-4)
                       and np.allclose(x3_img.affine, seg_img.affine, atol=1e-4))
    out["same_shape"] = bool(same_shape)
    out["same_affine"] = same_affine
    log(f"  grid: shape match {same_shape} | affine match {same_affine} | spacing {spacing}")
    if not same_shape:
        out["verdict"] = "FAIL: shapes differ -- the pair is not on a common grid"
        return out

    # ---- 2. crop to the brain, and build an eval mask immune to roll wrap-around ----------
    brain = gt > 0
    c = np.argwhere(brain)
    lo = np.maximum(c.min(0) - (max_shift + 4), 0)
    hi = np.minimum(c.max(0) + (max_shift + 4) + 1, np.array(x3.shape))
    sl = tuple(slice(int(a_), int(b_)) for a_, b_ in zip(lo, hi))
    a3, a7, ab = x3[sl], x7[sl], brain[sl]
    # np.roll wraps; erode the eval mask by more than max_shift so wrapped edge content can
    # never enter the score. This keeps the shift search cheap AND correct.
    eval_mask = ndimage.binary_erosion(
        ab, ndimage.generate_binary_structure(3, 3), iterations=max_shift + 1)
    if not eval_mask.any():
        eval_mask = ab
    out["eval_voxels"] = int(eval_mask.sum())

    mi0 = mutual_information(a3[eval_mask], a7[eval_mask], bins)
    out["mi_zero_shift"] = round(mi0, 5)
    log(f"  MI at zero shift: {mi0:.4f} nats  ({int(eval_mask.sum()):,} voxels)")

    # ---- 3. THE TEST: does shifting the 7T improve the match? -----------------------------
    profiles, peaks = {}, {}
    for ax, name in AXES.items():
        prof = {}
        for s in range(-max_shift, max_shift + 1):
            a7s = a7 if s == 0 else np.roll(a7, s, axis=ax)
            prof[s] = round(mutual_information(a3[eval_mask], a7s[eval_mask], bins), 5)
        best = max(prof, key=prof.get)
        profiles[name] = prof
        peaks[name] = best
        bar = "  ".join(f"{s:+d}:{prof[s]:.3f}" + ("*" if s == best else "")
                        for s in range(-max_shift, max_shift + 1))
        flag = "OK" if best == 0 else f"<-- PEAK OFF-CENTRE ({best:+d} vox = {best*spacing[ax]:+.2f} mm)"
        log(f"  axis {ax} {name:16}: {bar}   {flag}")
    out["mi_profiles"] = profiles
    out["mi_peak_shift_vox"] = peaks

    # ---- 4. verdict ------------------------------------------------------------------------
    off = {k: v for k, v in peaks.items() if v != 0}
    if not off:
        out["verdict"] = "PASS: MI peaks at zero shift on all three axes"
        out["max_misalign_mm"] = 0.0
        log("  VERDICT: PASS -- every axis peaks at zero shift; no gross translational error")
    else:
        mm = {k: round(abs(v) * spacing[[n for n, s in AXES.items() if s == k][0]], 2)
              for k, v in off.items()}
        out["verdict"] = f"FLAG: MI improves when shifted -- residual misalignment {mm}"
        out["max_misalign_mm"] = max(mm.values())
        log(f"  VERDICT: FLAG -- shifting IMPROVES the match: {mm}")
        log("           Real-7T-referenced metrics (SSIM/HD95/Betti) are CONFOUNDED for this "
            "subject.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--subjects", nargs="*", default=None, help="default: every subject in the csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-shift", type=int, default=4, help="voxels searched per axis (+/-)")
    a = ap.parse_args()

    with open(a.pairs_csv) as f:
        rows = list(csv.DictReader(f))
    if a.subjects:
        rows = [r for r in rows if r["subject"] in a.subjects]
    if not rows:
        raise SystemExit("no matching subjects")

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    results, t0 = [], time.perf_counter()
    for i, row in enumerate(rows, 1):
        log(f"\n[{i}/{len(rows)}] {row['subject']}")
        results.append(qc_subject(row, a.max_shift))

    log("\n" + "=" * 78)
    log("REGISTRATION QC SUMMARY")
    log("=" * 78)
    log(f"  {'subject':10} {'MI@0':>8}  {'peak shift (sag, cor, ax)':>28}  verdict")
    n_flag = 0
    for r in results:
        if "mi_peak_shift_vox" not in r:
            log(f"  {r['subject']:10} {'--':>8}  {'--':>28}  {r['verdict']}")
            n_flag += 1
            continue
        pk = r["mi_peak_shift_vox"]
        trip = f"({pk['sagittal (L-R)']:+d}, {pk['coronal (A-P)']:+d}, {pk['axial (S-I)']:+d})"
        ok = r["verdict"].startswith("PASS")
        n_flag += (not ok)
        log(f"  {r['subject']:10} {r['mi_zero_shift']:>8.4f}  {trip:>28}  "
            f"{'PASS' if ok else 'FLAG ' + str(r.get('max_misalign_mm')) + ' mm'}")

    log("")
    if n_flag == 0:
        log("  ALL SUBJECTS PASS. The 3T/7T pairs are translationally aligned, so the ~0.45")
        log("  brain-only SSIM floor is NOT a misregistration artifact, and the real-7T Betti")
        log("  reference is sound. The perception-distortion reading of the wall is supported.")
        log("  (Caveat, state it: this tests TRANSLATION only -- not rotation or scaling.)")
    else:
        log(f"  {n_flag}/{len(results)} SUBJECT(S) FLAGGED. Shifting the 7T improves the match,")
        log("  so those pairs carry residual misalignment. Every real-7T-referenced metric on")
        log("  them is confounded -- including the topology target. Do NOT build the causal-win")
        log("  claim on flagged subjects until they are re-registered.")

    (outdir / "registration_qc.json").write_text(json.dumps(results, indent=2, default=float))
    log(f"\nwrote {outdir / 'registration_qc.json'}   [{time.perf_counter()-t0:.0f}s]")


if __name__ == "__main__":
    main()
