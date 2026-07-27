"""
Is the fidelity wall real, or just residual misregistration? (robustness check)

CONTEXT. The independent-segmenter gate (external_seg_topology.py) refuted the topology-
brokenness claim, so the honest ~0.45 brain-only SSIM wall became a MAIN pillar of the paper.
registration_qc.py then flagged ~1-voxel (0.65 mm) residual 3T<->7T offsets on ~half the
subjects. A reviewer will therefore ask the obvious question: is your fidelity floor the
perception-distortion wall, or is it partly the pairs being misaligned?

THE CHECK. registration_qc already MEASURED, per subject and per axis, the integer shift that
best aligns the real 7T to the 3T (= the prediction's space; the model is a voxel-wise map on
the 3T grid). This script applies that shift to the real 7T (and its brain mask, together), then
recomputes the honest masked SSIM/PSNR. Interpretation:

    correction barely moves SSIM   -> the wall is real; misalignment is a negligible confound
                                       -> disclose the residual offset and keep the wall claim
    correction lifts SSIM a lot     -> the floor was partly misregistration
                                       -> re-register properly before quoting any fidelity number

BUILT-IN CONTROL. Subjects that PASSED registration (peak shift 0 on every axis, e.g. sub-02,
sub-06) must show ~zero change: their correction is a no-op. If a PASS subject's SSIM moves, the
check itself is wrong -- treat that as a red flag on this script, not on the data.

DIRECTION. registration_qc reports the shift s (per axis) at which MI(3T, roll(7T, s)) peaks, so
rolling the real 7T by exactly s aligns it to the 3T/prediction space. We roll the real 7T AND
the brain mask by s so the masked region stays consistent. np.roll wraps at the volume edge, but
the wrapped content lands in far background, never inside the interior brain mask.

No model inference -- reuses the saved *_pred7T.nii.gz. Runs in seconds per subject.

Usage:
  python scripts/fidelity_registration_robustness.py \
      --pairs-csv /eos/user/p/ppokhrel/topobrain/data/norm/pairs_cern.csv \
      --regqc-json /eos/user/p/ppokhrel/topobrain/runs/regqc/registration_qc.json \
      --pred-dir /eos/user/p/ppokhrel/topobrain/runs/phase3_clean \
      --out /eos/user/p/ppokhrel/topobrain/runs/regqc
  # pred volume for subject s is expected at <pred-dir>/eval_<s>/<s>_pred7T.nii.gz
"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_s = importlib.util.spec_from_file_location("mh", ROOT / "src" / "metrics_honest.py")
mh = importlib.util.module_from_spec(_s)
_s.loader.exec_module(mh)

# same RAS axis convention as registration_qc / metrics_protocols
AXIS_BY_NAME = {"sagittal (L-R)": 0, "coronal (A-P)": 1, "axial (S-I)": 2}


def log(m):
    print(m, flush=True)


def row_for(pairs_csv, subject):
    with open(pairs_csv) as f:
        for row in csv.DictReader(f):
            if row["subject"] == subject:
                return row
    return None


def shift_of(regqc_entry):
    """(dz, dy, dx) integer voxel shift that best aligns real 7T to the 3T space."""
    pk = regqc_entry.get("mi_peak_shift_vox", {})
    s = [0, 0, 0]
    for name, ax in AXIS_BY_NAME.items():
        s[ax] = int(pk.get(name, 0))
    return tuple(s)


def apply_shift(vol, shift):
    for ax, sft in enumerate(shift):
        if sft:
            vol = np.roll(vol, sft, axis=ax)
    return vol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--regqc-json", required=True)
    ap.add_argument("--pred-dir", required=True,
                    help="dir containing eval_<s>/<s>_pred7T.nii.gz")
    ap.add_argument("--pred-template", default="eval_{s}/{s}_pred7T.nii.gz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", nargs="*", default=None)
    a = ap.parse_args()

    regqc = {r["subject"]: r for r in json.load(open(a.regqc_json))}
    subjects = a.subjects or list(regqc.keys())
    pred_dir = Path(a.pred_dir)

    results, skipped = [], []
    for s in subjects:
        pred_path = pred_dir / a.pred_template.format(s=s)
        row = row_for(a.pairs_csv, s)
        if s not in regqc or row is None or not pred_path.exists():
            skipped.append((s, f"pred_exists={pred_path.exists()} in_regqc={s in regqc} "
                               f"in_pairs={row is not None}"))
            continue

        pred = nib.load(str(pred_path)).get_fdata().astype(np.float32)
        real = nib.load(row["target_7t"]).get_fdata().astype(np.float32)
        gt = np.rint(nib.load(row["seg"]).get_fdata()).astype(np.int32)
        brain = gt > 0

        if pred.shape != real.shape or pred.shape != brain.shape:
            skipped.append((s, f"shape mismatch pred{pred.shape} real{real.shape} seg{brain.shape}"))
            continue

        shift = shift_of(regqc[s])
        ssim_b = mh.masked_ssim(pred, real, brain)
        psnr_b = mh.masked_psnr(pred, real, brain)
        # align the real 7T (and its mask) to the prediction's space, then re-score
        real_c = apply_shift(real, shift)
        brain_c = apply_shift(brain, shift)
        ssim_a = mh.masked_ssim(pred, real_c, brain_c)
        psnr_a = mh.masked_psnr(pred, real_c, brain_c)

        passed = regqc[s].get("verdict", "").startswith("PASS")
        results.append({"subject": s, "regqc_pass": passed, "shift_vox": list(shift),
                        "ssim_before": round(ssim_b, 4), "ssim_after": round(ssim_a, 4),
                        "ssim_delta": round(ssim_a - ssim_b, 4),
                        "psnr_before": round(psnr_b, 3), "psnr_after": round(psnr_a, 3),
                        "psnr_delta": round(psnr_a - psnr_b, 3)})
        tag = "PASS(control)" if passed else "FLAG"
        log(f"  {s:8} {tag:14} shift={list(shift)}  "
            f"SSIM {ssim_b:.4f} -> {ssim_a:.4f} (d {ssim_a-ssim_b:+.4f})  "
            f"PSNR {psnr_b:.2f} -> {psnr_a:.2f} (d {psnr_a-psnr_b:+.2f})")

    log("\n" + "=" * 74)
    log("FIDELITY-WALL ROBUSTNESS TO REGISTRATION CORRECTION")
    log("=" * 74)
    if not results:
        log("  no subjects scored -- need the saved pred7T volumes. skipped:")
        for s, why in skipped:
            log(f"    {s}: {why}")
        return

    flagged = [r for r in results if not r["regqc_pass"]]
    controls = [r for r in results if r["regqc_pass"]]

    if controls:
        cmax = max(abs(r["ssim_delta"]) for r in controls)
        log(f"  CONTROL (registration-PASS subjects): max |SSIM change| = {cmax:.4f}")
        if cmax > 0.01:
            log("    ** WARNING: a PASS subject moved -- the check itself is suspect, not the data.")
        else:
            log("    OK: correcting an already-aligned subject is a no-op, as it must be.")

    if flagged:
        fmean = float(np.mean([r["ssim_delta"] for r in flagged]))
        fmax = max(r["ssim_delta"] for r in flagged)
        log(f"  FLAGGED subjects: mean SSIM change {fmean:+.4f}, max {fmax:+.4f} "
            f"(n={len(flagged)})")
        if fmax < 0.02:
            log("  VERDICT: the wall is ROBUST. Correcting the ~1-voxel offset barely moves SSIM,")
            log("  so the ~0.45 floor is the perception-distortion wall, not a registration")
            log("  artifact. Disclose the residual offset as a measured limitation and keep the")
            log("  claim. No re-registration needed for the fidelity conclusion.")
        elif fmax < 0.05:
            log("  VERDICT: MINOR sensitivity. The corrected numbers are a little higher; report")
            log("  the CORRECTED SSIM as primary and state the offset. The wall stands, slightly")
            log("  higher than the raw number.")
        else:
            log("  VERDICT: the floor is MATERIALLY affected by misalignment. Re-register the")
            log("  pairs properly (rigid) before quoting any fidelity number -- the raw wall is")
            log("  partly a registration artifact and must not be reported as-is.")
    else:
        log("  (no flagged subjects among those scored -- add --subjects sub-07 etc.)")

    if skipped:
        log("\n  skipped (missing pred/pairs/regqc):")
        for s, why in skipped:
            log(f"    {s}: {why}")

    outp = Path(a.out); outp.mkdir(parents=True, exist_ok=True)
    (outp / "fidelity_registration_robustness.json").write_text(
        json.dumps(results, indent=2, default=float))
    log(f"\nwrote {outp / 'fidelity_registration_robustness.json'}")


if __name__ == "__main__":
    main()
