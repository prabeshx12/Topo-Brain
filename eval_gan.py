"""
Evaluation and validation pipeline for trained 3T → 7T GAN.

Performs:
1. Quantitative metrics (PSNR, SSIM)
2. Visual inspection (multi-view slice comparisons)
3. Overfitting checks (loss curves)
4. Anatomical consistency checks (histogram, outliers)
5. Full-volume inference

Usage:
    python eval_gan.py --checkpoint checkpoints/best_model.pth --split val
"""
import argparse
import logging
from pathlib import Path
import sys
import csv
import warnings

import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import stats
import pandas as pd

from models.generator_unet3d import UNet3DGenerator
from models.paired_dataset import create_paired_data_list
from utils import discover_dataset, create_patient_level_split, setup_logging, set_random_seeds
from config import get_default_config

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


class GANEvaluator:
    """
    Comprehensive evaluator for trained 3T→7T GAN.
    
    Performs quantitative metrics, visual inspection, overfitting checks,
    and anatomical consistency validation.
    """
    
    def __init__(
        self,
        checkpoint_path: Path,
        output_dir: Path,
        device: torch.device,
        patch_size: tuple = (64, 64, 64),
        overlap: float = 0.5,
    ):
        """
        Initialize evaluator.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            output_dir: Directory for evaluation outputs
            device: Device to run inference on
            patch_size: Patch size for inference
            overlap: Overlap ratio for patch stitching (0.0-1.0)
        """
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.device = device
        self.patch_size = patch_size
        self.overlap = overlap
        
        # Create output directories
        self.metrics_dir = output_dir / "metrics"
        self.visual_dir = output_dir / "visualizations"
        self.plots_dir = output_dir / "plots"
        self.volumes_dir = output_dir / "generated_volumes"
        
        for dir_path in [self.metrics_dir, self.visual_dir, self.plots_dir, self.volumes_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load model
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        self.generator = self._load_generator()
        
        # Storage for results
        self.results = []
    
    def _load_generator(self) -> UNet3DGenerator:
        """Load trained generator from checkpoint."""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Extract config from checkpoint
        config_dict = checkpoint.get('config', {})
        
        # Create generator
        generator = UNet3DGenerator(
            in_channels=1,
            out_channels=1,
            base_features=config_dict.get('generator_base_features', 32),
            num_levels=config_dict.get('generator_num_levels', 4),
            norm_type=config_dict.get('norm_type', 'instance'),
        ).to(self.device)
        
        # Load weights
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()
        
        logger.info(f"Generator loaded (epoch {checkpoint['epoch']})")
        logger.info(f"Parameters: {generator.get_num_parameters():,}")
        
        return generator
    
    @torch.no_grad()
    def infer_full_volume(
        self,
        input_3t_path: Path,
    ) -> np.ndarray:
        """
        Run inference on full 3D volume using sliding window with overlap.
        
        Args:
            input_3t_path: Path to 3T input volume
        
        Returns:
            Generated 7T volume (numpy array)
        """
        # Load 3T volume
        nib_img = nib.load(str(input_3t_path))
        volume_3t = nib_img.get_fdata().astype(np.float32)
        
        if volume_3t.ndim == 4:
            volume_3t = volume_3t[..., 0]
        
        # Get volume shape
        D, H, W = volume_3t.shape
        pd, ph, pw = self.patch_size
        
        # Calculate stride based on overlap
        stride_d = int(pd * (1 - self.overlap))
        stride_h = int(ph * (1 - self.overlap))
        stride_w = int(pw * (1 - self.overlap))
        
        # Initialize output volume and count map
        output_volume = np.zeros_like(volume_3t)
        count_map = np.zeros_like(volume_3t)
        
        # Sliding window inference
        logger.info(f"Running sliding window inference on {input_3t_path.name}")
        logger.info(f"  Volume shape: {volume_3t.shape}")
        logger.info(f"  Patch size: {self.patch_size}")
        logger.info(f"  Stride: ({stride_d}, {stride_h}, {stride_w})")
        
        num_patches = 0
        
        for d_start in range(0, D - pd + 1, stride_d):
            for h_start in range(0, H - ph + 1, stride_h):
                for w_start in range(0, W - pw + 1, stride_w):
                    # Extract patch
                    patch_3t = volume_3t[
                        d_start:d_start + pd,
                        h_start:h_start + ph,
                        w_start:w_start + pw,
                    ]
                    
                    # Add batch and channel dimensions
                    patch_3t_tensor = torch.from_numpy(patch_3t[np.newaxis, np.newaxis, ...]).to(self.device)
                    
                    # Generate
                    patch_7t_tensor = self.generator(patch_3t_tensor)
                    
                    # Remove batch and channel dimensions
                    patch_7t = patch_7t_tensor[0, 0].cpu().numpy()
                    
                    # Add to output volume
                    output_volume[
                        d_start:d_start + pd,
                        h_start:h_start + ph,
                        w_start:w_start + pw,
                    ] += patch_7t
                    
                    count_map[
                        d_start:d_start + pd,
                        h_start:h_start + ph,
                        w_start:w_start + pw,
                    ] += 1
                    
                    num_patches += 1
        
        # Handle edge cases (regions not covered by patches)
        # Simply copy from input or use nearest patch
        uncovered_mask = count_map == 0
        if uncovered_mask.any():
            logger.warning(f"  {uncovered_mask.sum()} voxels not covered by patches")
            # For uncovered regions, use the input (no enhancement)
            output_volume[uncovered_mask] = volume_3t[uncovered_mask]
            count_map[uncovered_mask] = 1
        
        # Average overlapping regions
        output_volume = output_volume / count_map
        
        logger.info(f"  Processed {num_patches} patches")
        
        return output_volume
    
    def compute_metrics(
        self,
        input_3t: np.ndarray,
        generated_7t: np.ndarray,
        real_7t: np.ndarray,
    ) -> dict:
        """
        Compute quantitative metrics.
        
        Args:
            input_3t: 3T input volume
            generated_7t: Generated 7T volume
            real_7t: Real 7T ground truth volume
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # PSNR and SSIM: 3T vs real 7T (baseline)
        data_range = real_7t.max() - real_7t.min()
        
        try:
            metrics['psnr_3t_vs_7t'] = psnr(real_7t, input_3t, data_range=data_range)
        except Exception as e:
            logger.warning(f"Failed to compute PSNR (3T vs 7T): {e}")
            metrics['psnr_3t_vs_7t'] = np.nan
        
        try:
            metrics['ssim_3t_vs_7t'] = ssim(real_7t, input_3t, data_range=data_range)
        except Exception as e:
            logger.warning(f"Failed to compute SSIM (3T vs 7T): {e}")
            metrics['ssim_3t_vs_7t'] = np.nan
        
        # PSNR and SSIM: Generated vs real 7T (GAN performance)
        try:
            metrics['psnr_gen_vs_7t'] = psnr(real_7t, generated_7t, data_range=data_range)
        except Exception as e:
            logger.warning(f"Failed to compute PSNR (Gen vs 7T): {e}")
            metrics['psnr_gen_vs_7t'] = np.nan
        
        try:
            metrics['ssim_gen_vs_7t'] = ssim(real_7t, generated_7t, data_range=data_range)
        except Exception as e:
            logger.warning(f"Failed to compute SSIM (Gen vs 7T): {e}")
            metrics['ssim_gen_vs_7t'] = np.nan
        
        # L1 error
        metrics['l1_3t_vs_7t'] = np.mean(np.abs(input_3t - real_7t))
        metrics['l1_gen_vs_7t'] = np.mean(np.abs(generated_7t - real_7t))
        
        # Improvement metrics
        if not np.isnan(metrics['psnr_gen_vs_7t']) and not np.isnan(metrics['psnr_3t_vs_7t']):
            metrics['psnr_improvement'] = metrics['psnr_gen_vs_7t'] - metrics['psnr_3t_vs_7t']
        else:
            metrics['psnr_improvement'] = np.nan
        
        if not np.isnan(metrics['ssim_gen_vs_7t']) and not np.isnan(metrics['ssim_3t_vs_7t']):
            metrics['ssim_improvement'] = metrics['ssim_gen_vs_7t'] - metrics['ssim_3t_vs_7t']
        else:
            metrics['ssim_improvement'] = np.nan
        
        return metrics
    
    def check_anatomical_consistency(
        self,
        input_3t: np.ndarray,
        generated_7t: np.ndarray,
        real_7t: np.ndarray,
        subject_id: str,
    ) -> dict:
        """
        Check anatomical consistency of generated volume.
        
        Args:
            input_3t: 3T input volume
            generated_7t: Generated 7T volume
            real_7t: Real 7T volume
            subject_id: Subject identifier
        
        Returns:
            Dictionary of consistency checks
        """
        checks = {}
        
        # 1. Shape consistency
        checks['shape_match'] = generated_7t.shape == real_7t.shape
        
        # 2. Intensity range checks
        checks['gen_min'] = float(generated_7t.min())
        checks['gen_max'] = float(generated_7t.max())
        checks['gen_mean'] = float(generated_7t.mean())
        checks['gen_std'] = float(generated_7t.std())
        
        checks['real_min'] = float(real_7t.min())
        checks['real_max'] = float(real_7t.max())
        checks['real_mean'] = float(real_7t.mean())
        checks['real_std'] = float(real_7t.std())
        
        # 3. Check for extreme outliers (> 3 std from mean)
        gen_outliers = np.abs(generated_7t - generated_7t.mean()) > 3 * generated_7t.std()
        real_outliers = np.abs(real_7t - real_7t.mean()) > 3 * real_7t.std()
        
        checks['gen_outlier_ratio'] = float(gen_outliers.sum() / generated_7t.size)
        checks['real_outlier_ratio'] = float(real_outliers.sum() / real_7t.size)
        
        # 4. Check for saturation (values at min or max)
        gen_saturated_low = (generated_7t == generated_7t.min()).sum()
        gen_saturated_high = (generated_7t == generated_7t.max()).sum()
        checks['gen_saturation_ratio'] = float((gen_saturated_low + gen_saturated_high) / generated_7t.size)
        
        # 5. Histogram similarity (Kolmogorov-Smirnov test)
        # Flatten and sample for efficiency
        sample_size = min(10000, generated_7t.size)
        gen_sample = np.random.choice(generated_7t.flatten(), sample_size, replace=False)
        real_sample = np.random.choice(real_7t.flatten(), sample_size, replace=False)
        
        ks_statistic, ks_pvalue = stats.ks_2samp(gen_sample, real_sample)
        checks['ks_statistic'] = float(ks_statistic)
        checks['ks_pvalue'] = float(ks_pvalue)
        
        # Log warnings if issues detected
        if not checks['shape_match']:
            logger.warning(f"{subject_id}: Shape mismatch!")
        
        if checks['gen_outlier_ratio'] > 0.01:  # More than 1% outliers
            logger.warning(f"{subject_id}: High outlier ratio: {checks['gen_outlier_ratio']:.3f}")
        
        if checks['gen_saturation_ratio'] > 0.1:  # More than 10% saturated
            logger.warning(f"{subject_id}: High saturation: {checks['gen_saturation_ratio']:.3f}")
        
        if ks_pvalue < 0.01:  # Significant difference in distributions
            logger.warning(f"{subject_id}: Histogram distributions differ (KS p={ks_pvalue:.4f})")
        
        return checks
    
    def visualize_slices(
        self,
        input_3t: np.ndarray,
        generated_7t: np.ndarray,
        real_7t: np.ndarray,
        subject_id: str,
        slice_indices: dict = None,
    ):
        """
        Create multi-view slice comparisons.
        
        Args:
            input_3t: 3T input volume
            generated_7t: Generated 7T volume
            real_7t: Real 7T volume
            subject_id: Subject identifier
            slice_indices: Dict with 'axial', 'coronal', 'sagittal' indices
        """
        D, H, W = input_3t.shape
        
        # Default to middle slices if not specified
        if slice_indices is None:
            slice_indices = {
                'axial': D // 2,
                'coronal': H // 2,
                'sagittal': W // 2,
            }
        
        # Normalize to same range for visualization
        vmin = min(input_3t.min(), generated_7t.min(), real_7t.min())
        vmax = max(input_3t.max(), generated_7t.max(), real_7t.max())
        
        # Create figure with 3 rows (views) x 3 columns (3T, Gen, Real)
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        views = ['axial', 'coronal', 'sagittal']
        titles = ['3T Input', 'Generated 7T', 'Real 7T']
        
        for row, view in enumerate(views):
            idx = slice_indices[view]
            
            # Extract slices
            if view == 'axial':
                slice_3t = input_3t[idx, :, :]
                slice_gen = generated_7t[idx, :, :]
                slice_real = real_7t[idx, :, :]
            elif view == 'coronal':
                slice_3t = input_3t[:, idx, :]
                slice_gen = generated_7t[:, idx, :]
                slice_real = real_7t[:, idx, :]
            else:  # sagittal
                slice_3t = input_3t[:, :, idx]
                slice_gen = generated_7t[:, :, idx]
                slice_real = real_7t[:, :, idx]
            
            slices = [slice_3t, slice_gen, slice_real]
            
            for col, (slice_data, title) in enumerate(zip(slices, titles)):
                ax = axes[row, col]
                im = ax.imshow(slice_data, cmap='gray', vmin=vmin, vmax=vmax)
                
                if row == 0:
                    ax.set_title(title, fontsize=12, fontweight='bold')
                
                if col == 0:
                    ax.set_ylabel(f'{view.capitalize()}\n(slice {idx})', fontsize=11)
                
                ax.axis('off')
        
        plt.suptitle(f'Subject: {subject_id}', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save
        save_path = self.visual_dir / f"{subject_id}_slices.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization: {save_path}")
    
    def plot_histograms(
        self,
        input_3t: np.ndarray,
        generated_7t: np.ndarray,
        real_7t: np.ndarray,
        subject_id: str,
    ):
        """
        Plot intensity histograms for comparison.
        
        Args:
            input_3t: 3T input volume
            generated_7t: Generated 7T volume
            real_7t: Real 7T volume
            subject_id: Subject identifier
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        volumes = [input_3t, generated_7t, real_7t]
        labels = ['3T Input', 'Generated 7T', 'Real 7T']
        colors = ['blue', 'orange', 'green']
        
        for ax, volume, label, color in zip(axes, volumes, labels, colors):
            ax.hist(volume.flatten(), bins=100, alpha=0.7, color=color, edgecolor='black')
            ax.set_xlabel('Intensity')
            ax.set_ylabel('Frequency')
            ax.set_title(label)
            ax.grid(alpha=0.3)
            
            # Add stats
            stats_text = f'Mean: {volume.mean():.3f}\nStd: {volume.std():.3f}'
            ax.text(0.98, 0.98, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=9)
        
        plt.suptitle(f'Intensity Distributions - {subject_id}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        save_path = self.visual_dir / f"{subject_id}_histograms.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved histograms: {save_path}")
    
    def evaluate_subject(
        self,
        input_3t_path: Path,
        target_7t_path: Path,
        subject_id: str,
        save_volume: bool = True,
    ) -> dict:
        """
        Evaluate single subject.
        
        Args:
            input_3t_path: Path to 3T input
            target_7t_path: Path to 7T target
            subject_id: Subject identifier
            save_volume: Whether to save generated volume
        
        Returns:
            Dictionary of results
        """
        logger.info("="*80)
        logger.info(f"Evaluating: {subject_id}")
        logger.info("="*80)
        
        # Load real 7T
        logger.info(f"Loading real 7T: {target_7t_path.name}")
        nib_7t = nib.load(str(target_7t_path))
        real_7t = nib_7t.get_fdata().astype(np.float32)
        if real_7t.ndim == 4:
            real_7t = real_7t[..., 0]
        
        # Load 3T
        logger.info(f"Loading 3T input: {input_3t_path.name}")
        nib_3t = nib.load(str(input_3t_path))
        input_3t = nib_3t.get_fdata().astype(np.float32)
        if input_3t.ndim == 4:
            input_3t = input_3t[..., 0]
        
        # Generate 7T
        logger.info("Generating 7T...")
        generated_7t = self.infer_full_volume(input_3t_path)
        
        # Compute metrics
        logger.info("Computing metrics...")
        metrics = self.compute_metrics(input_3t, generated_7t, real_7t)
        
        # Anatomical consistency checks
        logger.info("Checking anatomical consistency...")
        consistency = self.check_anatomical_consistency(
            input_3t, generated_7t, real_7t, subject_id
        )
        
        # Visualizations
        logger.info("Creating visualizations...")
        self.visualize_slices(input_3t, generated_7t, real_7t, subject_id)
        self.plot_histograms(input_3t, generated_7t, real_7t, subject_id)
        
        # Save generated volume
        if save_volume:
            output_path = self.volumes_dir / f"{subject_id}_generated_7T.nii.gz"
            nib_out = nib.Nifti1Image(generated_7t, nib_7t.affine, nib_7t.header)
            nib.save(nib_out, str(output_path))
            logger.info(f"Saved generated volume: {output_path}")
        
        # Combine results
        result = {
            'subject': subject_id,
            **metrics,
            **consistency,
        }
        
        # Log summary
        logger.info(f"\nMetrics Summary for {subject_id}:")
        logger.info(f"  PSNR (3T vs 7T): {metrics['psnr_3t_vs_7t']:.2f} dB")
        logger.info(f"  PSNR (Gen vs 7T): {metrics['psnr_gen_vs_7t']:.2f} dB")
        logger.info(f"  PSNR Improvement: {metrics.get('psnr_improvement', 0):.2f} dB")
        logger.info(f"  SSIM (3T vs 7T): {metrics['ssim_3t_vs_7t']:.4f}")
        logger.info(f"  SSIM (Gen vs 7T): {metrics['ssim_gen_vs_7t']:.4f}")
        logger.info(f"  SSIM Improvement: {metrics.get('ssim_improvement', 0):.4f}")
        
        return result
    
    def plot_training_curves(self, checkpoint_path: Path):
        """
        Plot training and validation loss curves from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        train_history = checkpoint.get('train_history', {})
        val_history = checkpoint.get('val_history', {})
        
        if not train_history or not val_history:
            logger.warning("No training history found in checkpoint")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Generator loss
        ax = axes[0, 0]
        if 'g_loss' in train_history:
            ax.plot(train_history['g_loss'], label='Generator Loss', color='blue')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Generator Total Loss')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Plot 2: Discriminator loss
        ax = axes[0, 1]
        if 'd_loss' in train_history:
            ax.plot(train_history['d_loss'], label='Discriminator Loss', color='red')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Discriminator Loss')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Plot 3: L1 loss (train vs val)
        ax = axes[1, 0]
        if 'g_l1_loss' in train_history and 'l1_loss' in val_history:
            ax.plot(train_history['g_l1_loss'], label='Train L1', color='blue')
            ax.plot(val_history['l1_loss'], label='Val L1', color='orange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('L1 Loss')
            ax.set_title('L1 Loss (Train vs Val)')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Check for overfitting
            if len(val_history['l1_loss']) > 10:
                val_trend = np.polyfit(range(len(val_history['l1_loss'])), val_history['l1_loss'], 1)[0]
                if val_trend > 0:
                    ax.text(0.5, 0.95, 'WARNING: Validation loss increasing',
                           transform=ax.transAxes, ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
                           fontsize=10, fontweight='bold')
        
        # Plot 4: PSNR
        ax = axes[1, 1]
        if 'psnr' in val_history:
            ax.plot(val_history['psnr'], label='Val PSNR', color='green')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('PSNR (dB)')
            ax.set_title('Validation PSNR')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.suptitle('Training Curves', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        save_path = self.plots_dir / "training_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training curves: {save_path}")
    
    def save_metrics_summary(self, results: list):
        """
        Save metrics summary to CSV.
        
        Args:
            results: List of result dictionaries
        """
        if not results:
            logger.warning("No results to save")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Reorder columns
        priority_cols = ['subject', 'psnr_3t_vs_7t', 'psnr_gen_vs_7t', 'psnr_improvement',
                        'ssim_3t_vs_7t', 'ssim_gen_vs_7t', 'ssim_improvement']
        other_cols = [c for c in df.columns if c not in priority_cols]
        df = df[priority_cols + other_cols]
        
        # Save to CSV
        csv_path = self.metrics_dir / "metrics_summary.csv"
        df.to_csv(csv_path, index=False, float_format='%.6f')
        logger.info(f"Saved metrics summary: {csv_path}")
        
        # Compute and save aggregate statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats_df = df[numeric_cols].agg(['mean', 'std', 'min', 'max'])
        
        stats_path = self.metrics_dir / "aggregate_statistics.csv"
        stats_df.to_csv(stats_path, float_format='%.6f')
        logger.info(f"Saved aggregate statistics: {stats_path}")
        
        # Log summary
        logger.info("\n" + "="*80)
        logger.info("AGGREGATE METRICS")
        logger.info("="*80)
        logger.info(f"Number of subjects: {len(df)}")
        logger.info(f"\nPSNR (3T vs 7T):     {df['psnr_3t_vs_7t'].mean():.2f} ± {df['psnr_3t_vs_7t'].std():.2f} dB")
        logger.info(f"PSNR (Gen vs 7T):    {df['psnr_gen_vs_7t'].mean():.2f} ± {df['psnr_gen_vs_7t'].std():.2f} dB")
        logger.info(f"PSNR Improvement:    {df['psnr_improvement'].mean():.2f} ± {df['psnr_improvement'].std():.2f} dB")
        logger.info(f"\nSSIM (3T vs 7T):     {df['ssim_3t_vs_7t'].mean():.4f} ± {df['ssim_3t_vs_7t'].std():.4f}")
        logger.info(f"SSIM (Gen vs 7T):    {df['ssim_gen_vs_7t'].mean():.4f} ± {df['ssim_gen_vs_7t'].std():.4f}")
        logger.info(f"SSIM Improvement:    {df['ssim_improvement'].mean():.4f} ± {df['ssim_improvement'].std():.4f}")
        logger.info("="*80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained 3T→7T GAN')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                       help='Which split to evaluate')
    parser.add_argument('--output_dir', type=str, default='evaluation',
                       help='Output directory for results')
    parser.add_argument('--modality', type=str, default='T1w', choices=['T1w', 'T2w'])
    parser.add_argument('--patch_size', type=int, default=64, help='Patch size for inference')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap ratio (0.0-1.0)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_volumes', action='store_true', help='Save generated volumes')
    
    args = parser.parse_args()
    
    # Setup
    set_random_seeds(args.seed)
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_config = get_default_config().logging
    log_config.log_dir = output_dir / "logs"
    setup_logging(log_config)
    
    logger.info("="*80)
    logger.info("3T → 7T GAN Evaluation")
    logger.info("="*80)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Output: {output_dir}")
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load data
    config = get_default_config()
    logger.info("Discovering dataset...")
    data_list = discover_dataset(config.data.preprocessed_root, config.data)
    
    if len(data_list) == 0:
        logger.error("No preprocessed data found!")
        return
    
    # Create split
    logger.info("Creating patient-level split...")
    train_data, val_data, test_data = create_patient_level_split(
        data_list,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_seed=args.seed,
    )
    
    # Select split
    if args.split == 'train':
        eval_data = train_data
    elif args.split == 'val':
        eval_data = val_data
    else:
        eval_data = test_data
    
    # Create paired data
    from models.paired_dataset import create_paired_data_list
    paired_data = create_paired_data_list(eval_data, modality=args.modality)
    
    if len(paired_data) == 0:
        logger.error(f"No paired {args.split} data found!")
        return
    
    logger.info(f"Evaluating {len(paired_data)} subjects from {args.split} split")
    
    # Create evaluator
    evaluator = GANEvaluator(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        device=device,
        patch_size=(args.patch_size, args.patch_size, args.patch_size),
        overlap=args.overlap,
    )
    
    # Plot training curves
    logger.info("\nPlotting training curves...")
    evaluator.plot_training_curves(checkpoint_path)
    
    # Evaluate each subject
    results = []
    for i, pair in enumerate(paired_data, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Subject {i}/{len(paired_data)}")
        logger.info(f"{'='*80}")
        
        result = evaluator.evaluate_subject(
            input_3t_path=pair['input_3t'],
            target_7t_path=pair['target_7t'],
            subject_id=pair['subject'],
            save_volume=args.save_volumes,
        )
        
        results.append(result)
    
    # Save summary
    logger.info("\nSaving metrics summary...")
    evaluator.save_metrics_summary(results)
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  Metrics: {output_dir / 'metrics'}")
    logger.info(f"  Visualizations: {output_dir / 'visualizations'}")
    logger.info(f"  Plots: {output_dir / 'plots'}")
    if args.save_volumes:
        logger.info(f"  Generated volumes: {output_dir / 'generated_volumes'}")


if __name__ == "__main__":
    main()
