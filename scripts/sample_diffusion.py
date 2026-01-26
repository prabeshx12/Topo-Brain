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
    model = AnatomyGuidedUNet(
        in_channels=1, cond_channels=1, out_channels=1,
        num_classes=config["model"].get("num_classes", 3),
        features=tuple(config["model"].get("features", (32, 64, 128, 256)))
    ).to(device)

    diffusion = GaussianDiffusion(model, timesteps=config["diffusion"].get("timesteps", 1000)).to(device)
    
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
    def load_nii(path):
        if not path: return None, None
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        # Data is already normalized to [-1, 1] from preprocessing
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

    # 6. Generate
    print("Running Diffusion...")
    with torch.no_grad():
        generated = diffusion.p_sample_loop(conditioning=crop_in, shape=crop_in.shape)

    # 7. Save Results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save NIfTI
    out_nii = nib.Nifti1Image(generated.cpu().numpy().squeeze(), affine)
    nib.save(out_nii, out_path)
    print(f"Saved Volume: {out_path.resolve()}")
    
    # Save Visualization
    vis_path = out_path.with_suffix("").with_suffix(".png") # output.nii.gz -> output.png
    save_comparison_plot(crop_in.cpu().numpy(), generated.cpu().numpy(), 
                         crop_gt.cpu().numpy() if crop_gt is not None else None, 
                         vis_path)
    print(f"Saved Visualization: {vis_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Path to 3T input")
    parser.add_argument("--target", default=None, help="Optional: Path to 7T ground truth")
    parser.add_argument("--output", default="output.nii.gz")
    parser.add_argument("--config", default="configs/train_diffusion.yaml")
    args = parser.parse_args()
    run_inference(args)
