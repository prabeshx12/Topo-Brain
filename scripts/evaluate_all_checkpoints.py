"""
Evaluate all saved checkpoints to find the best one based on validation metrics.
Also generates visual comparison images for qualitative assessment.

Usage:
    python scripts/evaluate_all_checkpoints.py \
        --checkpoint_dir "/path/to/checkpoints/"
"""
import os
import sys
import torch
import yaml
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)  # Ensure relative paths work

from src.diffusion import GaussianDiffusion
from src.model import AnatomyGuidedUNet
from src.synthesis_dataset import (
    PatchConfig,
    load_pairs_manifest,
    create_synthesis_dataloaders,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_metrics(pred, target):
    """Calculate SSIM and PSNR between two 3D tensors.
    
    Matches sample_diffusion.py: rescale [-1, 1] -> [0, 1], data_range=1.0
    """
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr

    pred_np = pred.cpu().numpy().squeeze()
    target_np = target.cpu().numpy().squeeze()

    # Rescale from [-1, 1] to [0, 1] (same as sample_diffusion.py)
    pred_np = (pred_np + 1.0) / 2.0
    target_np = (target_np + 1.0) / 2.0

    # Clip to valid range
    pred_np = np.clip(pred_np, 0.0, 1.0)
    target_np = np.clip(target_np, 0.0, 1.0)

    ssim_val = ssim(target_np, pred_np, data_range=1.0)
    psnr_val = psnr(target_np, pred_np, data_range=1.0)

    return ssim_val, psnr_val


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_comparison_image(input_3t, pred_7t, target_7t, iteration, output_dir):
    """Save side-by-side comparison images (middle axial slice)."""
    mid = input_3t.shape[2] // 2

    inp = input_3t[0, 0, mid].cpu().numpy()
    pred = pred_7t[0, 0, mid].cpu().numpy()
    tgt = target_7t[0, 0, mid].cpu().numpy()
    err = np.abs(pred - tgt)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ['Input 3T', f'Predicted 7T\n(Iter {iteration})', 'Target 7T', 'Abs Error']
    images = [inp, pred, tgt, err]
    cmaps = ['gray', 'gray', 'gray', 'hot']

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        im = ax.imshow(img, cmap=cmap, aspect='auto')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    out_path = output_dir / f'checkpoint_{iteration:06d}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Single checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_path,
    model,
    diffusion,
    dataloader,
    device,
    save_images=False,
    image_output_dir=None,
    iteration=None,
):
    """Load one checkpoint, run the validation set, return metrics dict."""

    # ---- load weights ----
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Prefer EMA weights (usually better), fall back to model weights
    if 'ema' in ckpt:
        state = ckpt['ema']
    elif 'model' in ckpt:
        state = ckpt['model']
    elif 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt  # bare state-dict file

    model.load_state_dict(state)
    model.eval()

    # ---- evaluate ----
    all_ssim, all_psnr = [], []
    first_saved = False

    with torch.no_grad():
        for batch in dataloader:
            # Keys from PairedPatchDataset.__getitem__: 'input', 'target', 'seg'
            input_3t = batch['input'].to(device)
            target_7t = batch['target'].to(device)

            # Diffusion reverse-sampling
            # p_sample_loop(conditioning, shape, return_all=True) -> (img, seg)
            shape = target_7t.shape
            pred_7t, _ = diffusion.p_sample_loop(
                conditioning=input_3t,
                shape=shape,
                return_all=True,
            )

            # Save one image per checkpoint
            if save_images and not first_saved and image_output_dir is not None:
                save_comparison_image(input_3t, pred_7t, target_7t, iteration, image_output_dir)
                first_saved = True

            # Per-sample metrics
            for i in range(pred_7t.shape[0]):
                s, p = calculate_metrics(pred_7t[i:i+1], target_7t[i:i+1])
                all_ssim.append(s)
                all_psnr.append(p)

    return {
        'ssim_mean': float(np.mean(all_ssim)),
        'ssim_std': float(np.std(all_ssim)),
        'psnr_mean': float(np.mean(all_psnr)),
        'psnr_std': float(np.std(all_psnr)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate all checkpoints')
    parser.add_argument(
        '--checkpoint_dir', type=str, required=True,
        help='Directory containing checkpoint_*.pt files',
    )
    parser.add_argument(
        '--config', type=str, default='configs/train_diffusion.yaml',
        help='Path to training config YAML',
    )
    args = parser.parse_args()

    # ---- config ----
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---- model (matches train_diffusion.py) ----
    model = AnatomyGuidedUNet(
        in_channels=config['model']['in_channels'],
        cond_channels=config['model'].get('cond_channels', 1),
        out_channels=config['model']['out_channels'],
        num_classes=config['model']['num_classes'],
        features=config['model']['features'],
        use_attention=config['model']['use_attention'],
    ).to(device)

    # ---- diffusion wrapper ----
    diffusion = GaussianDiffusion(
        model=model,
        timesteps=config['diffusion']['timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
    ).to(device)

    # ---- validation data (uses same helpers as train_diffusion.py) ----
    print("Loading validation dataset ...")
    pairs = load_pairs_manifest(Path(config['dataset']['pairs_csv']))

    _, val_loader, _ = create_synthesis_dataloaders(
        pairs,
        config=PatchConfig(
            patch_size=tuple(config['dataset']['patch_size']),
            patches_per_volume=1,
        ),
        batch_size=1,
        num_workers=0,
        val_fold=0,
    )
    print("Validation dataset ready")

    # ---- find checkpoints ----
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        print(f"ERROR: checkpoint directory not found: {ckpt_dir}")
        return

    ckpt_files = sorted(ckpt_dir.glob('checkpoint_*.pt'))
    if not ckpt_files:
        print(f"No checkpoint_*.pt files in {ckpt_dir}")
        return

    print(f"Found {len(ckpt_files)} checkpoints")

    # ---- output dirs ----
    img_dir = Path('checkpoint_comparisons')
    img_dir.mkdir(exist_ok=True)

    # ---- evaluate each checkpoint ----
    results = []
    for path in tqdm(ckpt_files, desc='Evaluating'):
        # Extract iteration number from filename (skip non-numeric like checkpoint_latest.pt)
        parts = path.stem.split('_')
        if len(parts) < 2 or not parts[1].isdigit():
            tqdm.write(f"  Skipping {path.name} (not a numbered checkpoint)")
            continue
        iteration = int(parts[1])
        try:
            metrics = evaluate_checkpoint(
                path, model, diffusion, val_loader, device,
                save_images=True,
                image_output_dir=img_dir,
                iteration=iteration,
            )
            metrics['iteration'] = iteration
            metrics['checkpoint'] = path.name
            results.append(metrics)
            tqdm.write(
                f"  Iter {iteration:>6d}:  "
                f"SSIM={metrics['ssim_mean']:.4f}  "
                f"PSNR={metrics['psnr_mean']:.2f} dB"
            )
        except Exception as e:
            tqdm.write(f"  ERROR {path.name}: {e}")

    if not results:
        print("No checkpoints could be evaluated successfully.")
        return

    # ---- save CSV ----
    df = pd.DataFrame(results).sort_values('iteration')
    csv_path = 'checkpoint_evaluation_results.csv'
    df.to_csv(csv_path, index=False)

    # ---- report ----
    sep = '=' * 70
    print(f"\n{sep}")
    print(f"Results saved to: {csv_path}")
    print(sep)

    best_ssim = df.loc[df['ssim_mean'].idxmax()]
    print(f"\nBest SSIM : {best_ssim['ssim_mean']:.4f}  "
          f"(iter {int(best_ssim['iteration'])})  "
          f"PSNR={best_ssim['psnr_mean']:.2f} dB")

    best_psnr = df.loc[df['psnr_mean'].idxmax()]
    print(f"Best PSNR : {best_psnr['psnr_mean']:.2f} dB  "
          f"(iter {int(best_psnr['iteration'])})  "
          f"SSIM={best_psnr['ssim_mean']:.4f}")

    print(f"\n{sep}")
    print("TOP 5 BY SSIM:")
    print(sep)
    for _, row in df.nlargest(5, 'ssim_mean').iterrows():
        print(f"  Iter {int(row['iteration']):>6d}:  "
              f"SSIM={row['ssim_mean']:.4f}  "
              f"PSNR={row['psnr_mean']:.2f} dB  "
              f"({row['checkpoint']})")

    print(f"\nVisual comparisons saved to: {img_dir}/")
    print(sep)


if __name__ == '__main__':
    main()
