"""
Re-derive the 4-class tissue segmentation from the ORIGINAL FreeSurfer aparc+aseg, with a
COMPLETE label table -- recovering the cerebellum, corpus callosum and ventral DC that the
original preprocess_masks.py LUT silently dropped (see revision/SEGMENTATION_DECISION.md).

This replaces the incomplete `FS_MAPPING` in scripts/preprocess_masks.py. Every label is assigned
explicitly and the choices are documented, so the segmentation instrument is fully reproducible.

Classes: 0 background, 1 CSF, 2 GM, 3 WM.

Usage (per subject):
  python scripts/remap_aseg_complete.py --aseg <aparc+aseg.nii> --out <sub_seg_complete.nii.gz>
  # optional --ref <7T.nii> to resample onto the target grid (NN) if the aseg is off-grid
"""
import argparse
import numpy as np
import nibabel as nib

# --- COMPLETE FreeSurfer -> 4-class map, built against the labels actually present -----------
CSF = [4, 5, 14, 15, 24, 43, 44, 31, 63, 72]                         # ventricles + CSF + choroid
GM = [3, 42,                                                          # cerebral cortex (aseg)
      8, 47,                                                          # CEREBELLUM cortex (was dropped)
      10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 26, 58]         # subcortical GM nuclei
WM = [2, 41,                                                          # cerebral white matter
      7, 46,                                                          # CEREBELLUM WM (was dropped)
      251, 252, 253, 254, 255,                                        # CORPUS CALLOSUM (was dropped)
      28, 60,                                                         # ventral DC (was dropped)
      16,                                                             # brainstem -> WM (see note)
      77, 85, 5001, 5002]                                             # WM-hypo, optic chiasm, unseg WM
# NOTE on judgement calls (documented for the paper):
#  - Brainstem (16) -> WM: it is predominantly myelinated tracts. The old LUT put it in GM; this
#    is the one label whose class we deliberately change. Ventral DC (28/60) -> WM for the same
#    reason. Both are stated in the manuscript.
#  - Cortical parcellation 1000-2999 -> GM; WM parcellation 3000-4999 -> WM (range rules below).

FS_COMPLETE = {}
for l in CSF:
    FS_COMPLETE[l] = 1
for l in GM:
    FS_COMPLETE[l] = 2
for l in WM:
    FS_COMPLETE[l] = 3


def remap(aseg: np.ndarray) -> np.ndarray:
    out = np.zeros_like(aseg, dtype=np.uint8)
    for src, dst in FS_COMPLETE.items():
        out[aseg == src] = dst
    out[(aseg >= 1000) & (aseg < 3000)] = 2       # aparc cortical parcellation -> GM
    out[(aseg >= 3000) & (aseg < 5000)] = 3       # aparc WM parcellation -> WM (robustness)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aseg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None, help="optional 7T reference to resample onto (NN)")
    a = ap.parse_args()

    img = nib.load(a.aseg)
    aseg = np.rint(img.get_fdata()).astype(int)

    if a.ref:
        from nibabel.processing import resample_from_to
        ref = nib.load(a.ref)
        if img.shape != ref.shape or not np.allclose(img.affine, ref.affine):
            img = resample_from_to(img, ref, order=0)          # nearest-neighbour for labels
            aseg = np.rint(img.get_fdata()).astype(int)

    seg = remap(aseg)
    nib.save(nib.Nifti1Image(seg, img.affine), a.out)

    vals, counts = np.unique(seg, return_counts=True)
    names = {0: "BG", 1: "CSF", 2: "GM", 3: "WM"}
    print(f"wrote {a.out}")
    for v, c in zip(vals.tolist(), counts.tolist()):
        print(f"  {names.get(v, v):3}: {c:>10,} voxels ({100*c/seg.size:.1f}%)")
    unmapped = int(((aseg > 0) & (seg == 0)).sum())
    print(f"  aseg voxels left unmapped (dropped to BG): {unmapped:,}")


if __name__ == "__main__":
    main()
