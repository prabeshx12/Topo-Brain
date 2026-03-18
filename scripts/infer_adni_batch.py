"""
Integrated preprocessing + full-volume inference pipeline for ADNI-like unpaired 3D MRI.

This script supports variable image dimensions and voxel spacings by using tiled
64^3 inference with overlap blending.

Workflow:
1) Optional preprocessing via scripts.preprocess_bids
2) Discover preprocessed T1w volumes
3) Run full-volume tiled diffusion inference
4) Save synthesized 7T and segmentation outputs + diagnostics

Example:
    python scripts/infer_adni_batch.py \
      --checkpoint /path/to/checkpoint_100000.pt \
      --train-config configs/train_diffusion.yaml \
      --preprocessed-root /path/to/derivatives/topobrain-preproc \
      --output-root /path/to/adni_inference_results

With preprocessing:
    python scripts/infer_adni_batch.py \
      --checkpoint /path/to/checkpoint_100000.pt \
      --train-config configs/train_diffusion.yaml \
      --run-preprocess \
      --preprocess-config configs/preprocess.yaml \
      --data-root /path/to/adni_bids \
      --preprocessed-root /path/to/adni_bids/derivatives/topobrain-preproc \
      --output-root /path/to/adni_inference_results
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Ensure project root imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.diffusion import GaussianDiffusion
from src.model import AnatomyGuidedUNet


def _tukey_window_1d(n: int, alpha: float = 0.5) -> np.ndarray:
    if alpha <= 0:
        return np.ones(n, dtype=np.float64)
    if alpha >= 1:
        return np.hanning(n).astype(np.float64)

    w = np.ones(n, dtype=np.float64)
    taper = int(alpha * n / 2)
    if taper <= 0:
        return w

    w[:taper] = 0.5 * (1 - np.cos(np.pi * np.arange(taper) / taper))
    w[-taper:] = 0.5 * (1 - np.cos(np.pi * np.arange(taper, 0, -1) / taper))
    return w


def _get_starts(length: int, patch_size: int, stride: int) -> List[int]:
    starts = list(range(0, max(1, length - patch_size + 1), stride))
    if not starts:
        starts = [max(0, (length - patch_size) // 2)]
    elif starts[-1] + patch_size < length:
        starts.append(length - patch_size)
    return starts


def tiled_inference(
    diffusion: GaussianDiffusion,
    input_vol: np.ndarray,
    device: torch.device,
    num_classes: int,
    patch_size: int = 64,
    overlap: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    d, h, w = input_vol.shape
    stride = patch_size - overlap

    out_acc = np.zeros((d, h, w), dtype=np.float64)
    seg_acc = np.zeros((num_classes, d, h, w), dtype=np.float64)
    weight_acc = np.zeros((d, h, w), dtype=np.float64)

    alpha = overlap / patch_size
    w1d = _tukey_window_1d(patch_size, alpha=alpha)
    window = (w1d[None, None, :] * w1d[None, :, None] * w1d[:, None, None]).astype(np.float64)

    d_starts = _get_starts(d, patch_size, stride)
    h_starts = _get_starts(h, patch_size, stride)
    w_starts = _get_starts(w, patch_size, stride)

    total = len(d_starts) * len(h_starts) * len(w_starts)
    pbar = tqdm(total=total, desc="Tiled inference", unit="patch", leave=False)

    for ds in d_starts:
        for hs in h_starts:
            for ws in w_starts:
                patch = input_vol[ds:ds + patch_size, hs:hs + patch_size, ws:ws + patch_size]
                actual_shape = patch.shape

                if actual_shape != (patch_size, patch_size, patch_size):
                    padded = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
                    padded[:actual_shape[0], :actual_shape[1], :actual_shape[2]] = patch
                    patch = padded

                inp = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
                out_shape = inp.shape

                with torch.no_grad():
                    pred, seg = diffusion.p_sample_loop(conditioning=inp, shape=out_shape, return_all=True)

                pred_np = pred.cpu().numpy().squeeze()
                pred_np = pred_np[:actual_shape[0], :actual_shape[1], :actual_shape[2]]
                win = window[:actual_shape[0], :actual_shape[1], :actual_shape[2]]

                out_acc[ds:ds + actual_shape[0], hs:hs + actual_shape[1], ws:ws + actual_shape[2]] += pred_np * win
                weight_acc[ds:ds + actual_shape[0], hs:hs + actual_shape[1], ws:ws + actual_shape[2]] += win

                if seg is not None:
                    seg_np = seg.cpu().numpy().squeeze()  # [C, D, H, W]
                    seg_np = seg_np[:, :actual_shape[0], :actual_shape[1], :actual_shape[2]]
                    for c in range(min(seg_np.shape[0], num_classes)):
                        seg_acc[c, ds:ds + actual_shape[0], hs:hs + actual_shape[1], ws:ws + actual_shape[2]] += seg_np[c] * win

                pbar.update(1)

    pbar.close()

    mask = weight_acc > 0
    out_vol = np.zeros_like(out_acc, dtype=np.float32)
    out_vol[mask] = (out_acc[mask] / weight_acc[mask]).astype(np.float32)

    for c in range(num_classes):
        seg_acc[c][mask] /= weight_acc[mask]
    seg_vol = np.argmax(seg_acc, axis=0).astype(np.uint8)

    return out_vol, seg_vol


def run_preprocess_if_requested(args: argparse.Namespace) -> None:
    if not args.run_preprocess:
        return

    if not args.data_root:
        raise ValueError("--data-root is required when --run-preprocess is set")

    cmd = [
        sys.executable,
        "-m",
        "scripts.preprocess_bids",
        "--config",
        args.preprocess_config,
        "--data-root",
        args.data_root,
        "--output-root",
        args.preprocessed_root,
    ]

    if args.target_spacing:
        cmd.extend(["--target-spacing", str(args.target_spacing[0]), str(args.target_spacing[1]), str(args.target_spacing[2])])
    if args.overwrite_preprocess:
        cmd.append("--overwrite")

    print("Running preprocessing:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError("Preprocessing failed. See logs from scripts.preprocess_bids")


def discover_inputs(preprocessed_root: Path, manifest_path: Path | None = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    if manifest_path and manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") != "ok":
                    continue
                if row.get("modality") != "T1w":
                    continue
                output_path = row.get("output_path")
                if output_path and Path(output_path).exists():
                    rows.append(
                        {
                            "subject": row.get("subject", "unknown"),
                            "session": row.get("session", "unknown"),
                            "input_path": output_path,
                        }
                    )

    if rows:
        return rows

    # Fallback scan
    for p in sorted(preprocessed_root.rglob("*desc-preproc*T1w.nii*")):
        subject = "unknown"
        session = "unknown"
        for part in p.parts:
            if part.startswith("sub-"):
                subject = part
            elif part.startswith("ses-"):
                session = part
        rows.append({"subject": subject, "session": session, "input_path": str(p)})

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrated ADNI preprocessing + full-volume inference pipeline")

    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--train-config", default="configs/train_diffusion.yaml", help="Training YAML config")

    parser.add_argument("--run-preprocess", action="store_true", help="Run scripts.preprocess_bids before inference")
    parser.add_argument("--preprocess-config", default="configs/preprocess.yaml", help="Preprocessing YAML config")
    parser.add_argument("--data-root", default=None, help="Raw BIDS root for preprocessing")
    parser.add_argument("--preprocessed-root", required=True, help="Preprocessed root (input to inference)")
    parser.add_argument("--target-spacing", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"), help="Optional spacing override for preprocess step")
    parser.add_argument("--overwrite-preprocess", action="store_true", help="Force preprocessing overwrite")

    parser.add_argument("--manifest", default=None, help="Optional manifest.csv from preprocessing")
    parser.add_argument("--output-root", required=True, help="Output root for synthesized volumes")

    parser.add_argument("--patch-size", type=int, default=64, help="Patch size for tiled inference")
    parser.add_argument("--overlap", type=int, default=32, help="Patch overlap for tiled inference")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of subjects")

    args = parser.parse_args()

    if args.overlap >= args.patch_size:
        raise ValueError("--overlap must be smaller than --patch-size")

    run_preprocess_if_requested(args)

    preprocessed_root = Path(args.preprocessed_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(args.train_config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = AnatomyGuidedUNet(
        in_channels=config["model"]["in_channels"],
        cond_channels=config["model"].get("cond_channels", 1),
        out_channels=config["model"]["out_channels"],
        num_classes=config["model"].get("num_classes", 4),
        features=tuple(config["model"]["features"]),
        use_attention=config["model"].get("use_attention", False),
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        timesteps=config["diffusion"]["timesteps"],
        beta_schedule=config["diffusion"].get("beta_schedule", "cosine"),
        loss_type=config["diffusion"].get("loss_type", "l1"),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        print("Loaded EMA weights")
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print("Loaded model weights")
    else:
        model.load_state_dict(ckpt)
        print("Loaded raw state dict")
    model.eval()

    manifest_path = Path(args.manifest) if args.manifest else None
    inputs = discover_inputs(preprocessed_root=preprocessed_root, manifest_path=manifest_path)
    if args.limit is not None:
        inputs = inputs[: args.limit]

    if not inputs:
        raise RuntimeError("No preprocessed T1w inputs found. Check --preprocessed-root or --manifest")

    print(f"Found {len(inputs)} input volume(s)")

    summary = []
    for item in tqdm(inputs, desc="Subjects", unit="subj"):
        subject = item["subject"]
        session = item["session"]
        input_path = Path(item["input_path"])

        subj_out = output_root / subject / session / "anat"
        subj_out.mkdir(parents=True, exist_ok=True)

        try:
            nii = nib.load(str(input_path))
            vol = nii.get_fdata().astype(np.float32)
            affine = nii.affine

            # Diagnostics: expected range is roughly [-1, 1]
            vmin, vmax = float(vol.min()), float(vol.max())
            in_expected_range = (vmin >= -1.2) and (vmax <= 1.2)

            pred_vol, seg_vol = tiled_inference(
                diffusion=diffusion,
                input_vol=vol,
                device=device,
                num_classes=config["model"].get("num_classes", 4),
                patch_size=args.patch_size,
                overlap=args.overlap,
            )

            # Background cleanup to reduce out-of-brain noise
            brain_mask = vol > -0.95
            pred_masked = pred_vol.copy()
            pred_masked[~brain_mask] = vol[~brain_mask]

            out_pred = subj_out / f"{subject}_{session}_desc-synth7T_T1w.nii.gz"
            out_seg = subj_out / f"{subject}_{session}_desc-synth7Tseg_dseg.nii.gz"
            nib.save(nib.Nifti1Image(pred_masked, affine), str(out_pred))
            nib.save(nib.Nifti1Image(seg_vol, affine), str(out_seg))

            diag = {
                "subject": subject,
                "session": session,
                "input_path": str(input_path),
                "output_pred": str(out_pred),
                "output_seg": str(out_seg),
                "shape": list(vol.shape),
                "voxel_size": list(nii.header.get_zooms()[:3]),
                "input_range": [vmin, vmax],
                "expected_range": "[-1, 1]",
                "range_ok": in_expected_range,
                "status": "ok",
            }
            with open(subj_out / f"{subject}_{session}_inference_diagnostics.json", "w", encoding="utf-8") as f:
                json.dump(diag, f, indent=2)

            summary.append(diag)
        except Exception as exc:
            summary.append(
                {
                    "subject": subject,
                    "session": session,
                    "input_path": str(input_path),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    summary_path = output_root / "inference_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    n_ok = sum(1 for s in summary if s.get("status") == "ok")
    print(f"Done. Successful: {n_ok}/{len(summary)}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
