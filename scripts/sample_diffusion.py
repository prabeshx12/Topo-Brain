import argparse
import torch
import yaml
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.model import AnatomyGuidedUNet
from src.diffusion import GaussianDiffusion

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_comparison_plot(input_vol, gen_vol, target_vol, save_path):
    """
    Saves a side-by-side comparison of the center slices.
    """
    # Center slice index
    sl = input_vol.shape[2] // 2 
    
    # Extract slices & rotate for viewing (usually .T or rot90 helps)
    img_in = np.rot90(input_vol[0, 0, :, :, sl])
    img_gen = np.rot90(gen_vol[0, 0, :, :, sl])
    
    items = [("Input (3T)", img_in), ("Generated (7T)", img_gen)]
    
    if target_vol is not None:
        img_gt = np.rot90(target_vol[0, 0, :, :, sl])
        items.append(("Ground Truth (7T)", img_gt))
        
    fig, axes = plt.subplots(1, len(items), figsize=(5 * len(items), 5))
    if len(items) == 1: axes = [axes]
    
    for ax, (title, img) in zip(axes, items):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Checkpoint First (to check for config)
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 2. Determine Config
    if 'config' in checkpoint:
        print("Loaded config from checkpoint.")
        config = checkpoint['config']
    else:
        print(f"Config not found in checkpoint. Loading from: {args.config}")
        config = load_config(args.config)
    
    # 3. Model Setup
    use_t2 = config["dataset"].get("use_t2", False)
    in_channels = 2 if use_t2 else 1
    use_attention = config["model"].get("use_attention", False)
    num_classes = config["model"].get("num_classes", 4) # Default to 4 (BG, CSF, GM, WM)

    model = AnatomyGuidedUNet(
        in_channels=in_channels,
        cond_channels=1,
        out_channels=1,
        num_classes=num_classes,
        features=tuple(config["model"].get("features", (32, 64, 128, 256))),
        use_attention=use_attention
    ).to(device)

    # IMPORTANT: Use same beta_schedule as training!
    beta_schedule = config["diffusion"].get("beta_schedule", "cosine")
    print(f"Using architecture: {'Attention-UNet' if use_attention else 'Standard-UNet'}")
    print(f"Using beta_schedule: {beta_schedule}")
    
    diffusion = GaussianDiffusion(
        model, 
        timesteps=config["diffusion"].get("timesteps", 1000),
        beta_schedule=beta_schedule
    ).to(device)
    
    if 'ema' in checkpoint:
        print("Using EMA weights.")
        model.load_state_dict(checkpoint['ema'])
    else:
        print("Using standard weights.")
        model.load_state_dict(checkpoint['model'])
    model.eval()

    # 4. Input & Target Loading
    # Ensure 5D: [B, C, D, H, W]
    # Note: Data should be pre-normalized to [-1, 1] during preprocessing
    folder = Path(args.checkpoint).parent.parent
    
    def load_nii(path):
        if not path: return None, None
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)

        # Strict mode: inference expects preprocessed, diffusion-normalized inputs in [-1, 1].
        data_min = float(np.min(data))
        data_max = float(np.max(data))
        if data_min < -1.1 or data_max > 1.1:
            raise ValueError(
                "Input volume is not normalized to [-1, 1]. "
                "Run preprocessing with normalization.method='diffusion' "
                "and use those outputs for inference."
            )
        data = np.clip(data, -1.0, 1.0)
        
        tensor = torch.from_numpy(data).float()
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor, img.affine

    print(f"Loading Input: {args.input}")
    input_tensor, affine = load_nii(args.input)
    
    target_tensor = None
    if args.target:
        print(f"Loading Target: {args.target}")
        target_tensor, _ = load_nii(args.target)

    # 5. Crop Center (64^3)
    # We do this to ensure it matches training distribution and memory constraints
    _, _, d, h, w = input_tensor.shape
    cd, ch, cw = d//2, h//2, w//2
    rad = 32
    
    # Safe bounds check
    d_start = max(0, cd-rad); d_end = min(d, cd+rad)
    h_start = max(0, ch-rad); h_end = min(h, ch+rad)
    w_start = max(0, cw-rad); w_end = min(w, cw+rad)
    
    crop_in = input_tensor[:, :, d_start:d_end, h_start:h_end, w_start:w_end].to(device)
    
    crop_gt = None
    if target_tensor is not None:
        crop_gt = target_tensor[:, :, d_start:d_end, h_start:h_end, w_start:w_end].to(device)

    print(f"Inference input shape: {crop_in.shape}")

    # 6. Generate (Multi-Task)
    print("Running Diffusion...")
    with torch.no_grad():
        # conditioning: [B, C, D, H, W]
        # shape is the same as crop_in but always 1 channel for target 7T
        out_shape = (crop_in.shape[0], 1, *crop_in.shape[2:])
        generated, predicted_seg = diffusion.p_sample_loop(
            conditioning=crop_in, 
            shape=out_shape,
            return_all=True
        )

    # 7. Save Results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save NIfTI (Generated Image)
    out_nii = nib.Nifti1Image(generated.cpu().numpy().squeeze(), affine)
    nib.save(out_nii, out_path)
    
    # Save NIfTI (Segmentation Map)
    if predicted_seg is not None:
        seg_idx = torch.argmax(predicted_seg, dim=1).cpu().numpy().squeeze().astype(np.uint8)
        seg_path = out_path.with_name(out_path.name.replace(".nii", "_seg.nii"))
        seg_nii = nib.Nifti1Image(seg_idx, affine)
        nib.save(seg_nii, seg_path)
        print(f"Saved Segmentation: {seg_path.resolve()}")
    print(f"Saved Volume: {out_path.resolve()}")
    
    # Save Visualization
    vis_path = out_path.with_suffix("").with_suffix(".png") # output.nii.gz -> output.png
    save_comparison_plot(crop_in.cpu().numpy(), generated.cpu().numpy(), 
                         crop_gt.cpu().numpy() if crop_gt is not None else None, 
                         vis_path)
    # 8. Compute Metrics (Blueprint Section 8)
    if crop_gt is not None:
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr

        gen_np = generated.cpu().numpy().squeeze()
        gt_np = crop_gt.cpu().numpy().squeeze()

        # Scale to [0, 1] for metrics if originally [-1, 1]
        # Clamp to [-1,1] before rescaling (diffusion output may exceed range)
        gen_np = np.clip(gen_np, -1.0, 1.0)
        gt_np = np.clip(gt_np, -1.0, 1.0)
        gen_norm = (gen_np + 1) / 2
        gt_norm = (gt_np + 1) / 2

        cur_ssim = ssim(gen_norm, gt_norm, data_range=1.0)
        cur_psnr = psnr(gt_norm, gen_norm, data_range=1.0)

        # Compute Dice coefficient (volumetric overlap)
        gen_binary = (gen_np > 0.0).astype(np.float32)
        gt_binary = (gt_np > 0.0).astype(np.float32)
        intersection = np.sum(gen_binary * gt_binary)
        denominator = np.sum(gen_binary) + np.sum(gt_binary)
        dice = (2.0 * intersection / denominator) if denominator > 0 else 1.0

        # Compute HD95 (95th percentile Hausdorff Distance)
        try:
            from scipy.ndimage import distance_transform_edt, binary_dilation

            gen_bin = (gen_np > 0.0).astype(bool)
            gt_bin = (gt_np > 0.0).astype(bool)

            if gen_bin.any() and gt_bin.any():
                # Extract surfaces
                gen_surface = binary_dilation(gen_bin) ^ gen_bin
                gt_surface = binary_dilation(gt_bin) ^ gt_bin

                if gen_surface.any() and gt_surface.any():
                    # Distance transforms (assuming 1mm isotropic voxels for 64^3 crop)
                    dist_gen = distance_transform_edt(~gt_bin, sampling=(1.0, 1.0, 1.0))
                    dist_gt = distance_transform_edt(~gen_bin, sampling=(1.0, 1.0, 1.0))

                    distances_gen = dist_gen[gen_surface]
                    distances_gt = dist_gt[gt_surface]
                    all_distances = np.concatenate([distances_gen, distances_gt])

                    hd95 = np.percentile(all_distances, 95)
                else:
                    hd95 = float('nan')
            else:
                hd95 = float('nan')
        except ImportError:
            hd95 = float('nan')

        print("-" * 50)
        print(f"RESULTS (Quantitative - Patch 64³):")
        print(f"  SSIM:      {cur_ssim:.4f}")
        print(f"  PSNR:      {cur_psnr:.2f} dB")
        print(f"  Dice:      {dice:.4f}")
        if not np.isnan(hd95):
            print(f"  HD95:      {hd95:.2f} mm")
        else:
            print(f"  HD95:      N/A (scipy required)")
        
        # Clinical Volume Analysis (Section 8.3)
        if predicted_seg is not None:
            # seg classes: probably 1:CSF, 2:GM, 3:WM
            # Assuming seg_idx from earlier
            targets = [1, 2, 3]
            labels = {1: "CSF", 2: "Gray Matter", 3: "White Matter"}
            
            print(f"\nRESULTS (Clinical - Volume Counts):")
            for t in targets:
                gen_vol = np.sum(seg_idx == t)
                # If we have target masks, we could compare, but usually we compare volumes
                # of the synthetic output.
                print(f"  {labels[t]}: {gen_vol} voxels")
        print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Path to 3T input")
    parser.add_argument("--target", default=None, help="Optional: Path to 7T ground truth")
    parser.add_argument("--output", default="output.nii.gz")
    parser.add_argument("--config", default="configs/train_diffusion.yaml")
    args = parser.parse_args()
    run_inference(args)
