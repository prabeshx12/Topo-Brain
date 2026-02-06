"""
Evaluate all saved checkpoints to find the best one based on validation metrics.
Also generates visual comparison images for qualitative assessment.
"""
import os
import sys
import torch
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.diffusion import GaussianDiffusion
from src.model import AnatomyGuidedUNet
from src.synthesis_dataset import (
    PairedPatchDataset,
    PatchConfig,
    load_pairs_manifest,
    create_synthesis_dataloaders
)


def calculate_metrics(pred, target, seg_pred, seg_target):
    """Calculate SSIM, PSNR, and Dice scores."""
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    
    # Convert to numpy
    pred_np = pred.cpu().numpy().squeeze()
    target_np = target.cpu().numpy().squeeze()
    
    # Calculate SSIM and PSNR
    ssim_val = ssim(target_np, pred_np, data_range=target_np.max() - target_np.min())
    psnr_val = psnr(target_np, pred_np, data_range=target_np.max() - target_np.min())
    
    # Calculate Dice scores for each tissue type
    dice_scores = {}
    if seg_pred is not None and seg_target is not None:
        seg_pred_np = seg_pred.cpu().numpy()
        seg_target_np = seg_target.cpu().numpy()
        
        for tissue_idx, tissue_name in enumerate(['CSF', 'GM', 'WM']):
            pred_mask = (seg_pred_np == tissue_idx)
            target_mask = (seg_target_np == tissue_idx)
            
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = pred_mask.sum() + target_mask.sum()
            
            if union > 0:
                dice = 2.0 * intersection / union
                dice_scores[f'dice_{tissue_name}'] = dice
            else:
                dice_scores[f'dice_{tissue_name}'] = 0.0
    
    return ssim_val, psnr_val, dice_scores


def save_comparison_image(input_3t, pred_7t, target_7t, iteration, output_dir):
    """Save side-by-side comparison images."""
    # Take middle slice
    mid_slice = input_3t.shape[2] // 2
    
    input_slice = input_3t[0, 0, mid_slice].cpu().numpy()
    pred_slice = pred_7t[0, 0, mid_slice].cpu().numpy()
    target_slice = target_7t[0, 0, mid_slice].cpu().numpy()
    
    # Calculate error map
    error_map = np.abs(pred_slice - target_slice)
    
    # Create figure
    fig = plt.figure(figsize=(16, 4))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.3)
    
    # Input 3T
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(input_slice, cmap='gray', aspect='auto')
    ax1.set_title('Input 3T', fontsize=10)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # Predicted 7T
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(pred_slice, cmap='gray', aspect='auto')
    ax2.set_title(f'Predicted 7T\n(Iter {iteration})', fontsize=10)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Target 7T
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(target_slice, cmap='gray', aspect='auto')
    ax3.set_title('Target 7T', fontsize=10)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # Error map
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(error_map, cmap='hot', aspect='auto')
    ax4.set_title('Absolute Error', fontsize=10)
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    plt.tight_layout()
    output_path = output_dir / f'checkpoint_{iteration:06d}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_checkpoint(checkpoint_path, model, diffusion, dataloader, device, 
                       save_images=False, image_output_dir=None, iteration=None):
    """Evaluate a single checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_ssim = []
    all_psnr = []
    all_dice_csf = []
    all_dice_gm = []
    all_dice_wm = []
    
    first_batch_saved = False
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_3t = batch['input_3t'].to(device)
            target_7t = batch['target_7t'].to(device)
            
            if 'seg_7t' in batch:
                seg_target = batch['seg_7t'].to(device)
            else:
                seg_target = None
            
            # Generate sample
            pred_7t, seg_pred = diffusion.sample(
                input_3t, 
                return_all_timesteps=False,
                return_segmentation=True
            )
            
            # Save first batch images
            if save_images and not first_batch_saved and image_output_dir is not None:
                save_comparison_image(input_3t, pred_7t, target_7t, iteration, image_output_dir)
                first_batch_saved = True
            
            # Calculate metrics for each sample in batch
            for i in range(pred_7t.shape[0]):
                ssim_val, psnr_val, dice_scores = calculate_metrics(
                    pred_7t[i:i+1], 
                    target_7t[i:i+1],
                    seg_pred[i] if seg_pred is not None else None,
                    seg_target[i] if seg_target is not None else None
                )
                
                all_ssim.append(ssim_val)
                all_psnr.append(psnr_val)
                
                if dice_scores:
                    all_dice_csf.append(dice_scores.get('dice_CSF', 0.0))
                    all_dice_gm.append(dice_scores.get('dice_GM', 0.0))
                    all_dice_wm.append(dice_scores.get('dice_WM', 0.0))
    
    # Aggregate metrics
    results = {
        'ssim_mean': np.mean(all_ssim),
        'ssim_std': np.std(all_ssim),
        'psnr_mean': np.mean(all_psnr),
        'psnr_std': np.std(all_psnr),
    }
    
    if all_dice_csf:
        results.update({
            'dice_csf_mean': np.mean(all_dice_csf),
            'dice_gm_mean': np.mean(all_dice_gm),
            'dice_wm_mean': np.mean(all_dice_wm),
            'dice_avg': np.mean(all_dice_csf + all_dice_gm + all_dice_wm)
        })
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate all checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory containing checkpoint files')
    args = parser.parse_args()
    
    # Load config
    config_path = 'configs/train_diffusion.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = AnatomyGuidedUNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        features=config['model']['features'],
        use_attention=config['model']['use_attention']
    ).to(device)
    
    # Initialize diffusion
    diffusion = GaussianDiffusion(
        model=model,
        timesteps=config['diffusion']['timesteps'],
        beta_schedule=config['diffusion']['beta_schedule']
    ).to(device)
    
    # Load validation dataset
    print("Loading validation dataset...")
    pairs_path = Path(config['dataset']['pairs_csv'])
    pairs = load_pairs_manifest(pairs_path)
    
    # Create dataloaders
    _, val_loader, _ = create_synthesis_dataloaders(
        pairs,
        config=PatchConfig(
            patch_size=tuple(config['dataset']['patch_size']),
            patches_per_volume=1,  # Just 1 patch per volume for faster evaluation
        ),
        batch_size=1,
        num_workers=0,
        val_fold=0
    )
    
    print(f"Validation dataset ready")
    
    # Find all checkpoint files
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        print(f"Please provide the correct path using --checkpoint_dir argument")
        return
    
    checkpoint_files = sorted(checkpoint_dir.glob('checkpoint_*.pt'))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in '{checkpoint_dir}/' directory")
        print(f"Looking for files matching pattern: checkpoint_*.pt")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoints to evaluate")
    
    # Create output directory for images
    image_output_dir = Path('checkpoint_comparisons')
    image_output_dir.mkdir(exist_ok=True)
    print(f"Visual comparisons will be saved to: {image_output_dir}/")
    
    # Evaluate each checkpoint
    results = []
    for ckpt_path in tqdm(checkpoint_files, desc="Evaluating checkpoints"):
        # Extract iteration number from filename
        iteration = int(ckpt_path.stem.split('_')[1])
        
        try:
            metrics = evaluate_checkpoint(
                ckpt_path, model, diffusion, val_loader, device,
                save_images=True,
                image_output_dir=image_output_dir,
                iteration=iteration
            )
            metrics['iteration'] = iteration
            metrics['checkpoint'] = ckpt_path.name
            results.append(metrics)
            
            print(f"\nIteration {iteration:6d}: SSIM={metrics['ssim_mean']:.4f}, PSNR={metrics['psnr_mean']:.2f} dB")
            
        except Exception as e:
            print(f"\nError evaluating {ckpt_path.name}: {e}")
            continue
    
    # Create DataFrame and save results
    df = pd.DataFrame(results)
    df = df.sort_values('iteration')
    
    output_path = 'checkpoint_evaluation_results.csv'
    df.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    
    # Find best checkpoints
    print(f"\n{'='*80}")
    print("BEST CHECKPOINTS:")
    print(f"{'='*80}")
    
    best_ssim = df.loc[df['ssim_mean'].idxmax()]
    print(f"\nBest SSIM: {best_ssim['ssim_mean']:.4f} at iteration {int(best_ssim['iteration'])}")
    print(f"  Checkpoint: {best_ssim['checkpoint']}")
    print(f"  SSIM: {best_psnr['ssim_mean']:.4f}")
    
    if 'dice_avg' in df.columns:
        best_dice = df.loc[df['dice_avg'].idxmax()]
        print(f"\nBest Dice: {best_dice['dice_avg']:.4f} at iteration {int(best_dice['iteration'])}")
    
    print(f"\n{'='*80}")
    print(f"Visual comparisons saved to: checkpoint_comparisons/")
    print(f"Review the images to assess visual quality alongside metrics!")
    print(f"{'='*80}")
    print(f"  Checkpoint: {best_dice['checkpoint']}")
    print(f"  SSIM: {best_dice['ssim_mean']:.4f}, PSNR: {best_dice['psnr_mean']:.2f} dB")
    
    # Show top 5 overall (by SSIM)
    print(f"\n{'='*80}")
    print("TOP 5 CHECKPOINTS BY SSIM:")
    print(f"{'='*80}")
    top5 = df.nlargest(5, 'ssim_mean')
    for idx, row in top5.iterrows():
        print(f"Iter {int(row['iteration']):6d}: SSIM={row['ssim_mean']:.4f}, PSNR={row['psnr_mean']:.2f} dB - {row['checkpoint']}")


if __name__ == '__main__':
    main()
    
    print(f"\n{'='*80}")
    print(f"Visual comparisons saved to: checkpoint_comparisons/")
    print(f"Review the images to assess visual quality alongside metrics!")
    print(f"{'='*80}")
