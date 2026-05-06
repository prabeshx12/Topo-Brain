"""
Preprocess ADNI 3T MPRAGE volumes for the topobrain diffusion model.

Two-stage per subject:
    1) HD-BET -> brain mask saved next to the input NIfTI
    2) Inline pipeline (nibabel + SimpleITK):
         - reorient to RAS via nibabel.as_closest_canonical
         - N4 bias correction (SimpleITK)
         - apply HD-BET brain mask
         - diffusion normalization to [-1, 1] (0.5/99.5 percentiles, bg = -1)

Matches configs/preprocess.yaml (the config the sub-06 model was trained
with): bias correction ON, no resampling, diffusion normalization. If you
change those, you'll break the model's input distribution.

We bypass the MRIPreprocessor MONAI pipeline used during training because
newer MONAI's Orientationd default `labels=None` requires meta-tensor
space info that doesn't survive the temp-file roundtrip. The math is
identical; only the orientation backend differs.

Input cohort CSV: rows from build_adni_cohort.py with `nifti_path` populated.
Output: preprocessed NIfTIs + an updated cohort CSV with `preprocessed_path`.

Usage:
    python scripts/preprocess_adni.py \\
        --cohort-csv adni_dataset/pairs_adni_smoke.csv \\
        --output-dir adni_preprocessed \\
        --output-csv adni_dataset/pairs_adni_smoke_preprocessed.csv

    # CPU fallback (slow but works on dev boxes without CUDA):
    python scripts/preprocess_adni.py ... --device cpu
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk

# Repo imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _n4_correct(arr: np.ndarray, spacing_xyz) -> np.ndarray:
    """N4 bias correction via SimpleITK on a numpy array (XYZ order)."""
    sitk_img = sitk.GetImageFromArray(arr.transpose(2, 1, 0))   # XYZ -> ZYX for sitk
    sitk_img.SetSpacing([float(s) for s in spacing_xyz[::-1]])
    sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)
    otsu = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
    corr = sitk.N4BiasFieldCorrectionImageFilter()
    corr.SetMaximumNumberOfIterations([50, 50, 50, 50])
    out = corr.Execute(sitk_img, otsu)
    return sitk.GetArrayFromImage(out).transpose(2, 1, 0)


def _diffusion_normalize(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Match configs/preprocess.yaml: clip at 0.5/99.5 percentiles within
    brain, scale to [-1, 1], set background to -1."""
    fg = arr[mask > 0]
    if fg.size == 0:
        raise ValueError("empty brain mask — diffusion normalization undefined")
    lo = float(np.percentile(fg, 0.5))
    hi = float(np.percentile(fg, 99.5))
    out = np.clip(arr, lo, hi)
    out = (out - lo) / max(hi - lo, 1e-8) * 2.0 - 1.0
    out[mask == 0] = -1.0
    return out.astype(np.float32)


def run_hdbet(image_path: Path, mask_path: Path, device: str) -> Path:
    """Run HD-BET on `image_path`, write a binary brain mask to `mask_path`.

    HD-BET writes its own outputs in a fixed pattern; we move the mask file
    into our chosen `mask_path` so MRIPreprocessor.find_mask can locate it.
    """
    import torch
    from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict
    from HD_BET.checkpoint_download import maybe_download_parameters

    maybe_download_parameters()

    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"

    predictor = get_hdbet_predictor(
        use_tta=True,            # "accurate" mode
        device=torch.device(device),
        verbose=False,
    )

    # HD-BET writes <out>.nii.gz (brain) and <out>_bet.nii.gz (mask).
    out_brain = mask_path.with_name(mask_path.stem.replace(".nii", "") + "_brain.nii.gz")
    out_brain.parent.mkdir(parents=True, exist_ok=True)

    hdbet_predict(
        str(image_path),
        str(out_brain),
        predictor=predictor,
        keep_brain_mask=True,
        compute_brain_extracted_image=True,
    )

    bet_mask = out_brain.parent / (out_brain.name.replace(".nii.gz", "_bet.nii.gz"))
    if not bet_mask.exists():
        raise RuntimeError(f"HD-BET did not produce mask: expected {bet_mask}")
    bet_mask.replace(mask_path)
    return mask_path


def preprocess_one(
    ptid: str,
    input_nifti: Path,
    output_nifti: Path,
    device: str,
    skip_hdbet_if_exists: bool,
    overwrite: bool,
) -> dict:
    """Run HD-BET + MRIPreprocessor for one subject. Returns a status dict."""
    t0 = time.time()
    if not input_nifti.exists():
        return {"ptid": ptid, "status": "input_missing", "message": str(input_nifti),
                "preprocessed_path": "", "elapsed_s": 0.0}

    if output_nifti.exists() and not overwrite:
        return {"ptid": ptid, "status": "skipped_exists",
                "message": str(output_nifti),
                "preprocessed_path": str(output_nifti),
                "elapsed_s": 0.0}

    # 1) HD-BET -> brain mask alongside the input.
    # MRIPreprocessor's SkullStripping looks for `*brain_mask.nii.gz` next to the image,
    # so we name the file accordingly.
    mask_path = input_nifti.parent / (input_nifti.name.replace(".nii.gz", "") + "_brain_mask.nii.gz")
    if mask_path.exists() and skip_hdbet_if_exists:
        logging.info("[%s] HD-BET mask exists, reusing: %s", ptid, mask_path.name)
    else:
        try:
            run_hdbet(input_nifti, mask_path, device=device)
            logging.info("[%s] HD-BET ok -> %s", ptid, mask_path.name)
        except Exception as e:
            return {"ptid": ptid, "status": "hdbet_failed", "message": str(e)[:300],
                    "preprocessed_path": "", "elapsed_s": time.time() - t0}

    # Sanity check the mask is non-trivial.
    try:
        mask_arr = nib.load(str(mask_path)).get_fdata()
        mask_voxels = int(np.count_nonzero(mask_arr))
        if mask_voxels < 100_000:
            logging.warning("[%s] brain mask is suspiciously small (%d voxels)",
                            ptid, mask_voxels)
    except Exception as e:
        return {"ptid": ptid, "status": "mask_unreadable", "message": str(e)[:300],
                "preprocessed_path": "", "elapsed_s": time.time() - t0}

    # 2) Inline preprocessing: reorient (nibabel) -> N4 (SimpleITK) ->
    # apply HD-BET mask -> diffusion-normalize -> save.
    # We bypass the MRIPreprocessor MONAI pipeline because newer MONAI's
    # Orientationd default `labels=None` requires meta-tensor space info
    # that doesn't survive the temp-file roundtrip MRIPreprocessor uses.
    output_nifti.parent.mkdir(parents=True, exist_ok=True)
    try:
        img_nib = nib.as_closest_canonical(nib.load(str(input_nifti)))
        mask_nib = nib.as_closest_canonical(nib.load(str(mask_path)))
        img = img_nib.get_fdata().astype(np.float32)
        mask = (mask_nib.get_fdata() > 0).astype(np.uint8)
        if img.shape != mask.shape:
            raise ValueError(f"image shape {img.shape} != mask shape {mask.shape}")

        spacing = img_nib.header.get_zooms()[:3]
        img_corrected = _n4_correct(img, spacing)
        normalized = _diffusion_normalize(img_corrected * mask, mask)

        nib.save(nib.Nifti1Image(normalized, img_nib.affine, img_nib.header),
                 str(output_nifti))
        rmin, rmax = float(normalized.min()), float(normalized.max())
        msg = f"range=[{rmin:.2f},{rmax:.2f}] voxels={mask_voxels}"
    except Exception as e:
        return {"ptid": ptid, "status": "preprocess_failed", "message": str(e)[:300],
                "preprocessed_path": "", "elapsed_s": time.time() - t0}

    return {"ptid": ptid, "status": "ok", "message": msg,
            "preprocessed_path": str(output_nifti),
            "elapsed_s": round(time.time() - t0, 1)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cohort-csv", required=True,
                        help="Cohort CSV from build_adni_cohort.py")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for preprocessed NIfTIs")
    parser.add_argument("--output-csv", required=True,
                        help="Output cohort CSV with preprocessed_path column")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Device for HD-BET (auto-falls-back to CPU if CUDA missing)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N subjects (0 = no limit). For smoke tests.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process subjects whose preprocessed output already exists")
    parser.add_argument("--no-skip-hdbet", action="store_true",
                        help="Re-run HD-BET even if a brain mask is already on disk")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    cohort = pd.read_csv(args.cohort_csv)
    if "nifti_path" not in cohort.columns:
        logging.error("Cohort CSV missing 'nifti_path' column. "
                      "Did you run build_adni_cohort.py with --conversion-manifest?")
        return 2
    if cohort["nifti_path"].isna().all() or (cohort["nifti_path"].astype(str).str.len() == 0).all():
        logging.error("All nifti_path values are empty in %s", args.cohort_csv)
        return 2

    if args.limit > 0:
        cohort = cohort.head(args.limit).copy()
    logging.info("Preprocessing %d subjects, device=%s", len(cohort), args.device)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, sub in cohort.iterrows():
        ptid = str(sub["ptid"])
        nifti = str(sub["nifti_path"]).strip()
        if not nifti or pd.isna(sub["nifti_path"]):
            rows.append({"ptid": ptid, "status": "no_nifti_path",
                         "preprocessed_path": "", "elapsed_s": 0.0, "message": ""})
            continue
        input_nifti = Path(nifti)
        # Group subdir keeps AD/CN separate; matches convert_adni_dicom.py layout.
        out_sub = output_dir / str(sub.get("group", "")) / f"{ptid}_T1w_preprocessed.nii.gz"

        result = preprocess_one(
            ptid=ptid,
            input_nifti=input_nifti,
            output_nifti=out_sub,
            device=args.device,
            skip_hdbet_if_exists=not args.no_skip_hdbet,
            overwrite=args.overwrite,
        )
        rows.append(result)
        logging.info("[%s] %s (%.1fs) %s", ptid, result["status"],
                     result["elapsed_s"], result["message"])

    # Merge results into cohort and write.
    results_df = pd.DataFrame(rows)
    out_cohort = cohort.merge(results_df[["ptid", "preprocessed_path", "status"]]
                              .rename(columns={"status": "preprocess_status"}),
                              on="ptid", how="left")
    out_path = Path(args.output_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_cohort.to_csv(out_path, index=False)

    n_ok = sum(1 for r in rows if r["status"] in ("ok", "skipped_exists"))
    n_fail = len(rows) - n_ok
    logging.info("Wrote %s | ok=%d, failed=%d", out_path, n_ok, n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
