"""
Prepare the CERN training data: normalise raw 3T/7T to [-1,1] and write pairs_cern.csv.

WHY THIS EXISTS. The raw files on Kaggle (preprocessed-mri-aligned) are z-scored, NOT [-1,1].
train_cascaded loads via _robust_normalize, which RAISES on anything outside [-1.1,1.1]. The
Kaggle training kernel normalised on the fly (revision/kaggle/topobrain_cascaded.py:112-117);
this script does exactly the same thing, once, on disk. It is byte-for-byte the same `dnorm` that
produced every existing result, so the numbers do not move.

Seg is rounded to uint8 and NOT normalised (it is a label map).

Usage (on lxplus, after both Kaggle datasets are unzipped into --data-root):
  python scripts/prep_cern_data.py \
      --data-root /eos/user/p/ppokhrel/topobrain/data \
      --out-dir   /eos/user/p/ppokhrel/topobrain/data/norm
"""
import argparse
import glob
import os
from pathlib import Path

import nibabel as nib
import numpy as np


def dnorm(vol, clip_lo=0.5, clip_hi=99.5, p_lo=1.0, p_hi=99.0):
    """The 'diffusion' normalisation -- identical to src/preprocessing.py and eval_cascaded.py."""
    v = vol.astype(np.float32).copy()
    roi = v[v > 0]
    if roi.size == 0:
        return v
    lo_c, hi_c = np.percentile(roi, clip_lo), np.percentile(roi, clip_hi)
    v = np.clip(v, lo_c, hi_c)
    roi = v[v > 0]
    lo, hi = np.percentile(roi, p_lo), np.percentile(roi, p_hi)
    if hi > lo:
        v = np.clip((v - lo) / (hi - lo), 0, 1) * 2.0 - 1.0
    v[vol <= 0] = -1.0
    return v


def find(root, s, rel):
    """Find a file for subject s, tolerating .nii / .nii.gz."""
    for ext in ("", ".gz"):
        p = Path(root) / s / rel
        p = p.with_name(p.name + ext) if ext else p
        if p.exists():
            return str(p)
    # last resort: recursive glob on the basename stem
    stem = Path(rel).name.split(".")[0]
    hits = sorted(glob.glob(str(Path(root) / s / "**" / f"{stem}.nii*"), recursive=True))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="dir holding sub-01 ... sub-10")
    ap.add_argument("--out-dir", required=True, help="where normalised files + csv go")
    ap.add_argument("--pairs-csv", default=None, help="output csv (default: <out-dir>/pairs_cern.csv)")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = Path(a.pairs_csv) if a.pairs_csv else out / "pairs_cern.csv"

    subs = sorted({os.path.basename(p) for p in glob.glob(os.path.join(a.data_root, "sub-*"))
                   if os.path.isdir(p)})
    print(f"found {len(subs)} subjects: {subs}")

    rows = ["subject,input_3t,target_7t,seg"]
    for s in subs:
        i3 = find(a.data_root, s, "ses-1/anat/" + s + "_ses-1_desc-preproc_T1w_registered.nii")
        t7 = find(a.data_root, s, "ses-2/anat/" + s + "_ses-2_desc-preproc_T1w.nii")
        sg = find(a.data_root, s, "ses-2/anat/" + s + "_ses-2_desc-preproc_T1w_seg.nii")
        if not (i3 and t7 and sg):
            print(f"  SKIP {s}: missing  3t={bool(i3)} 7t={bool(t7)} seg={bool(sg)}")
            continue

        oi, ot, osg = out / f"{s}_3t.nii.gz", out / f"{s}_7t.nii.gz", out / f"{s}_seg.nii.gz"
        if not oi.exists():
            a3 = nib.load(i3); nib.save(nib.Nifti1Image(dnorm(a3.get_fdata()), a3.affine), str(oi))
            b7 = nib.load(t7); nib.save(nib.Nifti1Image(dnorm(b7.get_fdata()), b7.affine), str(ot))
            cs = nib.load(sg)
            seg = np.rint(cs.get_fdata()).astype(np.uint8)
            nib.save(nib.Nifti1Image(seg, cs.affine), str(osg))
            print(f"  {s}: normalised  (7t range check: "
                  f"[{nib.load(str(ot)).get_fdata().min():.2f}, "
                  f"{nib.load(str(ot)).get_fdata().max():.2f}]  seg labels: "
                  f"{np.unique(seg).tolist()})")
        else:
            print(f"  {s}: already done")
        rows.append(f"{s},{oi},{ot},{osg}")

    csv_path.write_text("\n".join(rows) + "\n")
    print(f"\nwrote {csv_path} with {len(rows)-1} subjects")


if __name__ == "__main__":
    main()
