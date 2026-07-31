"""
FastSurfer as an independent NEURAL judge -> GM beta0 ratio.

Closes the 'it's just a GMM' objection: if a standard neural segmenter (FastSurfer, trained on real
data, never on our generator) ALSO shows the synthetic 7T is topologically simpler than the real 7T
(ratio << 1), the reversal is not an artefact of using a classical GMM as the 'independent' judge.

Takes FastSurfer's aparc+aseg output for the SYNTHETIC and the REAL 7T (run FastSurfer --seg_only on
the [0,255] exports, e.g. runs/circularity_sub07/ext_in_synth.nii.gz / ext_in_real.nii.gz), remaps
each with the SAME 4-class LUT used everywhere else, and reports the GM synthetic/real beta0 ratio.
Compare this ratio to the GMM's (~0.10) and to the co-trained head's (~1.38): a FastSurfer ratio
well below 1 corroborates the finding with an independent neural segmenter.

nibabel reads FastSurfer's .mgz directly, so pass the .mgz path as-is. beta0 = 26-connected
components of grey matter (class 2). FastSurfer conforms to 1 mm, so its ABSOLUTE beta0 is not
comparable to our 0.65 mm numbers -- compare ONLY the synth-vs-real ratio WITHIN FastSurfer.

Usage:
  python scripts/fastsurfer_ratio.py \
      --synth-aseg /eos/.../runs/fastsurfer/synth_sub07/mri/aparc.DKTatlas+aseg.deep.mgz \
      --real-aseg  /eos/.../runs/fastsurfer/real_sub07/mri/aparc.DKTatlas+aseg.deep.mgz \
      --subject sub-07
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_r = importlib.util.spec_from_file_location("_remap", ROOT / "scripts" / "remap_aseg_complete.py")
_rm = importlib.util.module_from_spec(_r); _r.loader.exec_module(_rm)
remap = _rm.remap
_m = importlib.util.spec_from_file_location("mh", ROOT / "src" / "metrics_honest.py")
mh = importlib.util.module_from_spec(_m); _m.loader.exec_module(mh)


def gm_beta0(aseg_path):
    aseg = np.rint(nib.load(str(aseg_path)).get_fdata()).astype(int)
    seg4 = remap(aseg)                       # SAME LUT as the ground truth / all other judges
    ncc, _ = mh.connected_components(seg4 == 2, 26)   # GM = class 2; ncc == beta0
    return int(ncc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-aseg", required=True)
    ap.add_argument("--real-aseg", required=True)
    ap.add_argument("--subject", default="")
    a = ap.parse_args()

    bs = gm_beta0(a.synth_aseg)
    br = gm_beta0(a.real_aseg)
    ratio = bs / br if br else float("nan")
    tag = f"{a.subject} " if a.subject else ""
    print(f"{tag}FastSurfer (independent NEURAL judge):")
    print(f"  GM beta0  synth = {bs}   real = {br}")
    print(f"  synth/real ratio = {ratio:.3f}")
    if ratio < 0.5:
        print("  -> synth is topologically SIMPLER than real under an independent NEURAL segmenter:")
        print("     corroborates the over-smoothing finding; the 'it's just a GMM' objection does not hold.")
    else:
        print("  -> FastSurfer does NOT see the synth as simpler. Report honestly -- the neural")
        print("     independent judge disagrees with the GMM, which weakens the cliff and must be disclosed.")


if __name__ == "__main__":
    main()
