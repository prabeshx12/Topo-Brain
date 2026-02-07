"""
Full-volume evaluation of a single checkpoint with tiled patch inference.

- Tiles the full volume into overlapping 64³ patches
- Runs diffusion inference on each patch
- Stitches results back into a full volume
- Computes metrics on the FULL volume (consistent & reproducible)
- Saves multi-slice visualizations (axial, coronal, sagittal)
- Saves output as NIfTI for external inspection

Usage:
    python scripts/evaluate_full_volume.py \
        --checkpoint "/path/to/checkpoint_141000.pt" \
        --input "/path/to/sub-01_3T_preprocessed.nii.gz" \
        --target "/path/to/sub-01_7T_preprocessed.nii.gz" \
        --output_dir "results/sub-01_eval"
"""
import os
import sys
import torch
import yaml
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.diffusion import GaussianDiffusion
from src.model import AnatomyGuidedUNet


# ---------------------------------------------------------------------------
# Metrics (identical to sample_diffusion.py for consistency)
# ---------------------------------------------------------------------------

def compute_metrics(pred_np, target_np):
    """
    Compute SSIM and PSNR.  Both arrays are expected in [-1, 1].
    We rescale to [0, 1] and use data_range=1.0  (same as sample_diffusion.py).
    """
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr

    pred_01 = np.clip((pred_np + 1.0) / 2.0, 0.0, 1.0)
    tgt_01 = np.clip((target_np + 1.0) / 2.0, 0.0, 1.0)

    ssim_val = ssim(tgt_01, pred_01, data_range=1.0)
    psnr_val = psnr(tgt_01, pred_01, data_range=1.0)
    return ssim_val, psnr_val


# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------

def _tukey_window_1d(n, alpha=0.5):
    """1-D Tukey (tapered-cosine) window.  alpha=fraction that is cosine-tapered."""
    if alpha <= 0:
        return np.ones(n)
    if alpha >= 1:
        return np.hanning(n)
    w = np.ones(n)
    taper = int(alpha * n / 2)
    # left taper
    w[:taper] = 0.5 * (1 - np.cos(np.pi * np.arange(taper) / taper))
    # right taper
    w[-taper:] = 0.5 * (1 - np.cos(np.pi * np.arange(taper, 0, -1) / taper))
    return w


def tiled_inference(diffusion, input_vol, device, patch_size=64, overlap=32):
    """
    Run diffusion inference on overlapping 64³ tiles and stitch the result.

    Args:
        diffusion:  GaussianDiffusion model (already on device, eval mode)
        input_vol:  numpy array [D, H, W] in [-1, 1]
        device:     torch device
        patch_size: side length of cubic patch  (default 64)
        overlap:    overlap between adjacent patches (default 32 = 50%)

    Returns:
        output_vol: numpy array [D, H, W]  predicted 7T
        seg_vol:    numpy array [D, H, W]  predicted segmentation (argmax)
    """
    D, H, W = input_vol.shape
    stride = patch_size - overlap

    # Accumulators
    output_acc = np.zeros((D, H, W), dtype=np.float64)
    seg_acc = np.zeros((4, D, H, W), dtype=np.float64)  # num_classes=4
    weight_acc = np.zeros((D, H, W), dtype=np.float64)

    # Tukey window: flat center (weight=1) with cosine taper in the overlap zone.
    # alpha = overlap/patch_size  →  only the overlap fraction gets tapered.
    alpha = overlap / patch_size   # 0.5 when overlap=32, patch=64
    w1d = _tukey_window_1d(patch_size, alpha=alpha)
    window = w1d[None, None, :] * w1d[None, :, None] * w1d[:, None, None]
    window = window.astype(np.float64)

    # Generate tile coordinates
    def get_starts(length, ps, st):
        starts = list(range(0, length - ps + 1, st))
        if starts and starts[-1] + ps < length:
            starts.append(length - ps)
        if not starts:
            starts = [max(0, (length - ps) // 2)]
        return starts

    d_starts = get_starts(D, patch_size, stride)
    h_starts = get_starts(H, patch_size, stride)
    w_starts = get_starts(W, patch_size, stride)

    total = len(d_starts) * len(h_starts) * len(w_starts)
    pbar = tqdm(total=total, desc="Tiled inference", unit="patch")

    for ds in d_starts:
        for hs in h_starts:
            for ws in w_starts:
                # Extract patch
                patch = input_vol[ds:ds+patch_size, hs:hs+patch_size, ws:ws+patch_size]

                # Pad if patch is smaller than expected (edge case)
                actual_shape = patch.shape
                if actual_shape != (patch_size, patch_size, patch_size):
                    padded = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
                    padded[:actual_shape[0], :actual_shape[1], :actual_shape[2]] = patch
                    patch = padded

                # To tensor [1, 1, D, H, W]
                inp = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
                out_shape = inp.shape

                with torch.no_grad():
                    pred, seg = diffusion.p_sample_loop(
                        conditioning=inp,
                        shape=out_shape,
                        return_all=True,
                    )

                pred_np = pred.cpu().numpy().squeeze()
                # Trim to actual shape
                pred_np = pred_np[:actual_shape[0], :actual_shape[1], :actual_shape[2]]
                win = window[:actual_shape[0], :actual_shape[1], :actual_shape[2]]

                output_acc[ds:ds+actual_shape[0], hs:hs+actual_shape[1], ws:ws+actual_shape[2]] += pred_np * win
                weight_acc[ds:ds+actual_shape[0], hs:hs+actual_shape[1], ws:ws+actual_shape[2]] += win

                # Segmentation accumulation
                if seg is not None:
                    seg_np = seg.cpu().numpy().squeeze()  # [C, D, H, W]
                    seg_np = seg_np[:, :actual_shape[0], :actual_shape[1], :actual_shape[2]]
                    for c in range(seg_np.shape[0]):
                        seg_acc[c, ds:ds+actual_shape[0], hs:hs+actual_shape[1], ws:ws+actual_shape[2]] += seg_np[c] * win

                pbar.update(1)

    pbar.close()

    # Normalize by weights
    mask = weight_acc > 0
    output_vol = np.zeros_like(output_acc, dtype=np.float32)
    output_vol[mask] = (output_acc[mask] / weight_acc[mask]).astype(np.float32)

    # Segmentation argmax
    for c in range(seg_acc.shape[0]):
        seg_acc[c][mask] /= weight_acc[mask]
    seg_vol = np.argmax(seg_acc, axis=0).astype(np.uint8)

    return output_vol, seg_vol


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_multi_view(input_vol, pred_vol, target_vol, output_dir, tag="full"):
    """Save axial, coronal, sagittal views at multiple positions."""
    D, H, W = input_vol.shape

    slices_info = {
        'axial': {
            'func': lambda v, s: v[s, :, :],
            'positions': [D // 4, D // 2, 3 * D // 4],
        },
        'coronal': {
            'func': lambda v, s: v[:, s, :],
            'positions': [H // 4, H // 2, 3 * H // 4],
        },
        'sagittal': {
            'func': lambda v, s: v[:, :, s],
            'positions': [W // 4, W // 2, 3 * W // 4],
        },
    }

    for view_name, info in slices_info.items():
        for pos in info['positions']:
            inp_sl = np.rot90(info['func'](input_vol, pos))
            pred_sl = np.rot90(info['func'](pred_vol, pos))
            err_sl = np.abs(pred_sl - np.rot90(info['func'](target_vol, pos)))

            ncols = 4 if target_vol is not None else 2
            fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

            axes[0].imshow(inp_sl, cmap='gray')
            axes[0].set_title('Input 3T', fontsize=12)
            axes[0].axis('off')

            axes[1].imshow(pred_sl, cmap='gray')
            axes[1].set_title('Predicted 7T', fontsize=12)
            axes[1].axis('off')

            if target_vol is not None:
                tgt_sl = np.rot90(info['func'](target_vol, pos))
                axes[2].imshow(tgt_sl, cmap='gray')
                axes[2].set_title('Target 7T', fontsize=12)
                axes[2].axis('off')

                im = axes[3].imshow(err_sl, cmap='hot')
                axes[3].set_title('Abs Error', fontsize=12)
                axes[3].axis('off')
                plt.colorbar(im, ax=axes[3], fraction=0.046)

            plt.suptitle(f'{view_name.capitalize()} – slice {pos}', fontsize=14)
            plt.tight_layout()
            fname = output_dir / f'{tag}_{view_name}_slice{pos:03d}.png'
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_paths_from_csv(pairs_csv, subject):
    """Look up input_3t / target_7t paths for a subject in the pairs CSV."""
    import csv
    with open(pairs_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('subject', '') == subject:
                return row.get('input_3t', row.get('input')), row.get('target_7t', row.get('target'))
    raise ValueError(f"Subject '{subject}' not found in {pairs_csv}")


def main():
    parser = argparse.ArgumentParser(description='Full-volume evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/train_diffusion.yaml')
    parser.add_argument('--overlap', type=int, default=32, help='Patch overlap for tiling (default 32 = 50%%)')
    parser.add_argument('--output_dir', type=str, default='results/full_volume_eval')

    # Option A: direct paths
    parser.add_argument('--input', type=str, default=None, help='3T preprocessed NIfTI')
    parser.add_argument('--target', type=str, default=None, help='7T ground truth NIfTI')

    # Option B: resolve from pairs CSV
    parser.add_argument('--subject', type=str, default=None,
                        help='Subject ID (e.g. sub-01).  Looks up paths from --pairs_csv.')
    parser.add_argument('--pairs_csv', type=str, default=None,
                        help='Pairs CSV.  Defaults to dataset.pairs_csv from config.')
    args = parser.parse_args()

    # ---- config ----
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ---- resolve input / target paths ----
    if args.subject:
        csv_path = args.pairs_csv or config['dataset']['pairs_csv']
        input_path, target_path = resolve_paths_from_csv(csv_path, args.subject)
        print(f"Resolved from CSV for {args.subject}:")
        print(f"  input  = {input_path}")
        print(f"  target = {target_path}")
    elif args.input:
        input_path = args.input
        target_path = args.target
    else:
        parser.error("Provide either --input or --subject")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- model ----
    model = AnatomyGuidedUNet(
        in_channels=config['model']['in_channels'],
        cond_channels=config['model'].get('cond_channels', 1),
        out_channels=config['model']['out_channels'],
        num_classes=config['model']['num_classes'],
        features=config['model']['features'],
        use_attention=config['model']['use_attention'],
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        timesteps=config['diffusion']['timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
    ).to(device)

    # ---- load checkpoint ----
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'ema' in ckpt:
        print("Using EMA weights")
        model.load_state_dict(ckpt['ema'])
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # ---- load volumes ----
    print(f"Loading input:  {input_path}")
    inp_nii = nib.load(input_path)
    input_vol = inp_nii.get_fdata().astype(np.float32)
    affine = inp_nii.affine

    target_vol = None
    if target_path:
        print(f"Loading target: {target_path}")
        target_vol = nib.load(target_path).get_fdata().astype(np.float32)

    print(f"Volume shape: {input_vol.shape}")

    # ---- full-volume tiled inference ----
    patch_size = config['dataset']['patch_size'][0]  # 64
    pred_vol, seg_vol = tiled_inference(
        diffusion, input_vol, device,
        patch_size=patch_size,
        overlap=args.overlap,
    )

    # ---- brain mask: remove background noise from prediction ----
    # The diffusion model generates noise outside the brain.  We mask it out
    # by copying the input's background values into the prediction.
    brain_mask = (input_vol > -0.95)   # True inside brain
    pred_masked = pred_vol.copy()
    pred_masked[~brain_mask] = input_vol[~brain_mask]  # keep original BG

    # ---- output directory ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- save NIfTI ----
    # Raw (unmasked) – for debugging
    nib.save(nib.Nifti1Image(pred_vol, affine), out_dir / 'predicted_7T_raw.nii.gz')
    # Masked (clean) – primary output
    nib.save(nib.Nifti1Image(pred_masked, affine), out_dir / 'predicted_7T.nii.gz')
    print(f"Saved: {out_dir / 'predicted_7T.nii.gz'}  (brain-masked)")
    print(f"Saved: {out_dir / 'predicted_7T_raw.nii.gz'}  (raw)")

    seg_nii = nib.Nifti1Image(seg_vol, affine)
    nib.save(seg_nii, out_dir / 'predicted_seg.nii.gz')
    print(f"Saved: {out_dir / 'predicted_seg.nii.gz'}")

    # ---- compute metrics ----
    if target_vol is not None:
        # 1) Full-volume masked (primary metric – no BG noise)
        ssim_masked, psnr_masked = compute_metrics(pred_masked, target_vol)

        # 2) Brain-only: crop to tight bounding box of brain for SSIM
        coords = np.argwhere(brain_mask)
        lo = coords.min(axis=0)
        hi = coords.max(axis=0) + 1
        pred_bb = pred_masked[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        tgt_bb = target_vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        ssim_brain, psnr_brain = compute_metrics(pred_bb, tgt_bb)

        # 3) Center 64³ crop (directly comparable to sample_diffusion.py)
        D, H, W = input_vol.shape
        cd, ch, cw = D // 2, H // 2, W // 2
        rad = 32
        crop_pred = pred_masked[cd-rad:cd+rad, ch-rad:ch+rad, cw-rad:cw+rad]
        crop_tgt = target_vol[cd-rad:cd+rad, ch-rad:ch+rad, cw-rad:cw+rad]
        ssim_crop, psnr_crop = compute_metrics(crop_pred, crop_tgt)

        print("\n" + "=" * 60)
        print("METRICS (all use [0,1] range, data_range=1.0)")
        print("=" * 60)
        print(f"  Full masked :  SSIM = {ssim_masked:.4f}   PSNR = {psnr_masked:.2f} dB")
        print(f"  Brain bbox  :  SSIM = {ssim_brain:.4f}   PSNR = {psnr_brain:.2f} dB")
        print(f"  Center 64³  :  SSIM = {ssim_crop:.4f}   PSNR = {psnr_crop:.2f} dB")
        print("=" * 60)

        # Save metrics to file
        with open(out_dir / 'metrics.txt', 'w') as f:
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Input:      {input_path}\n")
            f.write(f"Target:     {target_path}\n")
            f.write(f"Volume shape: {input_vol.shape}\n")
            f.write(f"Overlap: {args.overlap}  (stride={patch_size - args.overlap})\n\n")
            f.write(f"Full masked :  SSIM = {ssim_masked:.4f}   PSNR = {psnr_masked:.2f} dB\n")
            f.write(f"Brain bbox  :  SSIM = {ssim_brain:.4f}   PSNR = {psnr_brain:.2f} dB\n")
            f.write(f"Center 64^3 :  SSIM = {ssim_crop:.4f}   PSNR = {psnr_crop:.2f} dB\n")

    # ---- visualizations (use the MASKED prediction for clean images) ----
    print("\nSaving visualizations ...")
    save_multi_view(input_vol, pred_masked, target_vol, out_dir, tag="eval")
    print(f"All outputs saved to: {out_dir}/")


if __name__ == '__main__':
    main()
