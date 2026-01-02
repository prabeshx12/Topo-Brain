"""
Enhanced training script for 3T → 7T MRI super-resolution GAN with:
- Topology-aware losses (Persistent Homology)
- Perceptual losses
- Spectral normalization
- Label smoothing
- Balanced G/D training
- Comprehensive metrics

This replaces train_gan.py with all critical fixes implemented.
"""
import argparse
import logging
from pathlib import Path
import time
import sys
import json
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.generator_unet3d import UNet3DGenerator
from models.discriminator_patchgan3d import PatchGANDiscriminator3D
from models.topology_loss import CombinedTopologyLoss
from models.perceptual_loss import MedicalPerceptualLoss
from models.paired_dataset import (
    Paired3T7TDataset,
    create_paired_data_list,
    build_gan_augmentation_transforms,
)
from src.config import get_default_config
from src.utils import discover_dataset, create_patient_level_split, setup_logging, set_random_seeds

logger = logging.getLogger(__name__)


class EnhancedGANConfig:
    """Configuration for enhanced GAN training with topology and perceptual losses."""
    def __init__(self):
        # Model architecture
        self.generator_base_features = 32
        self.generator_num_levels = 4
        self.discriminator_base_features = 64
        self.discriminator_num_layers = 3
        self.norm_type = "instance"
        self.use_spectral_norm = True  # CRITICAL FIX
        
        # Training
        self.num_epochs = 100
        self.batch_size = 2
        self.learning_rate_g = 2e-4  # Generator LR
        self.learning_rate_d = 5e-5  # Discriminator LR (4x SLOWER) - CRITICAL FIX
        self.beta1 = 0.5
        self.beta2 = 0.999
        
        # G/D training balance - CRITICAL FIX
        self.g_updates_per_d_update = 2  # Train G twice per D once
        
        # Loss weights
        self.lambda_l1 = 100.0
        self.lambda_adv = 1.0
        self.lambda_perceptual = 1.0  # NEW
        self.lambda_topology = 0.1     # NEW
        
        # Loss configuration
        self.adversarial_loss_type = "lsgan"
        self.use_label_smoothing = True   # CRITICAL FIX
        self.real_label_value = 0.9       # Instead of 1.0
        self.fake_label_value = 0.0
        
        # Topology loss settings
        self.use_topology_loss = True
        self.topology_ph_weight = 0.1
        self.topology_reg_weight = 1.0
        self.topology_frequency = 10  # Compute PH every N batches
        
        # Perceptual loss settings
        self.use_perceptual_loss = True
        self.perceptual_num_slices = 5
        
        # Patch sampling
        self.patch_size = (64, 64, 64)
        self.num_patches_per_volume = 10
        
        # Data augmentation
        self.use_augmentation = True
        self.augmentation_prob = 0.5
        
        # Optimization
        self.use_amp = False
        self.gradient_clip_value = 1.0
        
        # Logging and saving
        self.save_interval = 5
        self.log_interval = 10
        self.vis_interval = 1
        
        # Paths (relative)
        self.checkpoint_dir = Path("checkpoints_enhanced")
        self.visualization_dir = Path("visualizations_enhanced")
        self.log_dir = Path("logs_enhanced")
        
        # Data
        self.modality = "T1w"
        self.num_workers = 4
        self.pin_memory = True
        
        # Use registered data
        self.use_registered_data = True


class EnhancedGANTrainer:
    """Trainer class for enhanced 3T→7T GAN with topology and perceptual losses."""
    
    def __init__(
        self,
        config: EnhancedGANConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Create output directories
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Build models
        logger.info("Building models...")
        self.generator = UNet3DGenerator(
            in_channels=1,
            out_channels=1,
            base_features=config.generator_base_features,
            num_levels=config.generator_num_levels,
            norm_type=config.norm_type,
        ).to(device)
        
        self.discriminator = PatchGANDiscriminator3D(
            in_channels=1,
            base_features=config.discriminator_base_features,
            num_layers=config.discriminator_num_layers,
            norm_type=config.norm_type,
            use_spectral_norm=config.use_spectral_norm,  # CRITICAL FIX
        ).to(device)
        
        logger.info(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        logger.info(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        
        # Optimizers with DIFFERENT learning rates - CRITICAL FIX
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate_g,
            betas=(config.beta1, config.beta2),
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate_d,  # SLOWER than G
            betas=(config.beta1, config.beta2),
        )
        
        # Loss functions
        self.criterion_l1 = nn.L1Loss()
        
        if config.adversarial_loss_type == "lsgan":
            self.criterion_adv = nn.MSELoss()
        else:
            self.criterion_adv = nn.BCEWithLogitsLoss()
        
        # Topology loss - NEW
        if config.use_topology_loss:
            try:
                self.criterion_topology = CombinedTopologyLoss(
                    use_ph_loss=True,
                    ph_weight=config.topology_ph_weight,
                    reg_weight=config.topology_reg_weight,
                    ph_frequency=config.topology_frequency,
                    dimensions=[0, 1],  # Connected components and holes
                    use_3d=False,  # Use 2D slices for efficiency
                ).to(device)
                logger.info("✓ Topology loss enabled")
            except ImportError as e:
                logger.warning(f"Topology loss disabled: {e}")
                self.criterion_topology = None
        else:
            self.criterion_topology = None
        
        # Perceptual loss - NEW
        if config.use_perceptual_loss:
            try:
                self.criterion_perceptual = MedicalPerceptualLoss(
                    num_slices=config.perceptual_num_slices,
                    vgg_weight=1.0,
                    gradient_weight=1.0,
                    frequency_weight=0.5,
                ).to(device)
                logger.info("✓ Perceptual loss enabled")
            except Exception as e:
                logger.warning(f"Perceptual loss disabled: {e}")
                self.criterion_perceptual = None
        else:
            self.criterion_perceptual = None
        
        # Mixed precision
        self.scaler_g = GradScaler(enabled=config.use_amp)
        self.scaler_d = GradScaler(enabled=config.use_amp)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_psnr = 0.0
        
        # Metrics storage
        self.train_history = {
            'g_loss': [], 'g_l1_loss': [], 'g_adv_loss': [],
            'g_perceptual_loss': [], 'g_topology_loss': [],
            'd_loss': [], 'd_real_loss': [], 'd_fake_loss': [],
        }
        self.val_history = {'l1_loss': [], 'psnr': []}
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with enhanced losses."""
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = {
            'g_loss': 0.0, 'g_l1_loss': 0.0, 'g_adv_loss': 0.0,
            'g_perceptual_loss': 0.0, 'g_topology_loss': 0.0,
            'd_loss': 0.0, 'd_real_loss': 0.0, 'd_fake_loss': 0.0,
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for i, batch in enumerate(pbar):
            input_3t = batch['input_3t'].to(self.device)
            target_7t = batch['target_7t'].to(self.device)
            
            # ===== Train Discriminator =====
            self.optimizer_d.zero_grad()
            
            with autocast(enabled=self.config.use_amp):
                with torch.no_grad():
                    fake_7t = self.generator(input_3t)
                
                pred_real = self.discriminator(target_7t)
                pred_fake = self.discriminator(fake_7t.detach())
                
                # Label smoothing - CRITICAL FIX
                if self.config.use_label_smoothing:
                    label_real = torch.ones_like(pred_real) * self.config.real_label_value
                    label_fake = torch.ones_like(pred_fake) * self.config.fake_label_value
                else:
                    label_real = torch.ones_like(pred_real)
                    label_fake = torch.zeros_like(pred_fake)
                
                loss_d_real = self.criterion_adv(pred_real, label_real)
                loss_d_fake = self.criterion_adv(pred_fake, label_fake)
                loss_d = (loss_d_real + loss_d_fake) * 0.5
            
            self.scaler_d.scale(loss_d).backward()
            
            if self.config.gradient_clip_value > 0:
                self.scaler_d.unscale_(self.optimizer_d)
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(),
                    self.config.gradient_clip_value
                )
            
            self.scaler_d.step(self.optimizer_d)
            self.scaler_d.update()
            
            # ===== Train Generator (multiple times per D update) - CRITICAL FIX =====
            for _ in range(self.config.g_updates_per_d_update):
                self.optimizer_g.zero_grad()
                
                with autocast(enabled=self.config.use_amp):
                    fake_7t = self.generator(input_3t)
                    
                    # L1 reconstruction loss
                    loss_g_l1 = self.criterion_l1(fake_7t, target_7t)
                    
                    # Adversarial loss
                    pred_fake = self.discriminator(fake_7t)
                    if self.config.use_label_smoothing:
                        label_real = torch.ones_like(pred_fake) * self.config.real_label_value
                    else:
                        label_real = torch.ones_like(pred_fake)
                    loss_g_adv = self.criterion_adv(pred_fake, label_real)
                    
                    # Perceptual loss - NEW
                    loss_g_perceptual = 0.0
                    if self.criterion_perceptual is not None:
                        loss_perc, _ = self.criterion_perceptual(fake_7t, target_7t)
                        loss_g_perceptual = loss_perc
                    
                    # Topology loss - NEW
                    loss_g_topology = 0.0
                    if self.criterion_topology is not None:
                        loss_topo, _ = self.criterion_topology(fake_7t, target_7t)
                        loss_g_topology = loss_topo
                    
                    # Combined generator loss
                    loss_g = (
                        self.config.lambda_l1 * loss_g_l1 +
                        self.config.lambda_adv * loss_g_adv +
                        self.config.lambda_perceptual * loss_g_perceptual +
                        self.config.lambda_topology * loss_g_topology
                    )
                
                self.scaler_g.scale(loss_g).backward()
                
                if self.config.gradient_clip_value > 0:
                    self.scaler_g.unscale_(self.optimizer_g)
                    torch.nn.utils.clip_grad_norm_(
                        self.generator.parameters(),
                        self.config.gradient_clip_value
                    )
                
                self.scaler_g.step(self.optimizer_g)
                self.scaler_g.update()
            
            # Update metrics
            epoch_metrics['g_loss'] += loss_g.item()
            epoch_metrics['g_l1_loss'] += loss_g_l1.item()
            epoch_metrics['g_adv_loss'] += loss_g_adv.item()
            if isinstance(loss_g_perceptual, torch.Tensor):
                epoch_metrics['g_perceptual_loss'] += loss_g_perceptual.item()
            if isinstance(loss_g_topology, torch.Tensor):
                epoch_metrics['g_topology_loss'] += loss_g_topology.item()
            epoch_metrics['d_loss'] += loss_d.item()
            epoch_metrics['d_real_loss'] += loss_d_real.item()
            epoch_metrics['d_fake_loss'] += loss_d_fake.item()
            
            # Update progress bar
            pbar.set_postfix({
                'G': f"{loss_g.item():.3f}",
                'D': f"{loss_d.item():.3f}",
                'L1': f"{loss_g_l1.item():.3f}",
            })
            
            self.global_step += 1
        
        # Average metrics
        num_batches = len(self.train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.generator.eval()
        
        total_l1 = 0.0
        total_psnr = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            input_3t = batch['input_3t'].to(self.device)
            target_7t = batch['target_7t'].to(self.device)
            
            fake_7t = self.generator(input_3t)
            
            # L1 loss
            l1_loss = self.criterion_l1(fake_7t, target_7t)
            total_l1 += l1_loss.item()
            
            # PSNR
            mse = torch.mean((fake_7t - target_7t) ** 2)
            if mse > 0:
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                total_psnr += psnr.item()
            
            num_batches += 1
        
        metrics = {
            'l1_loss': total_l1 / num_batches,
            'psnr': total_psnr / num_batches,
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'best_val_psnr': self.best_val_psnr,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': vars(self.config),
        }
        
        # Save latest
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best
        if is_best:
            best_path = self.config.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"✓ New best model saved: PSNR = {self.best_val_psnr:.2f} dB")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting enhanced GAN training...")
        logger.info(f"  Epochs: {self.config.num_epochs}")
        logger.info(f"  Generator LR: {self.config.learning_rate_g}")
        logger.info(f"  Discriminator LR: {self.config.learning_rate_d}")
        logger.info(f"  G updates per D: {self.config.g_updates_per_d_update}")
        logger.info(f"  Label smoothing: {self.config.use_label_smoothing}")
        logger.info(f"  Spectral norm: {self.config.use_spectral_norm}")
        logger.info(f"  Topology loss: {self.config.use_topology_loss}")
        logger.info(f"  Perceptual loss: {self.config.use_perceptual_loss}")
        
        for epoch in range(1, self.config.num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Log training metrics
            logger.info(f"\nEpoch {epoch}/{self.config.num_epochs}")
            logger.info(f"  Train - G: {train_metrics['g_loss']:.4f}, D: {train_metrics['d_loss']:.4f}")
            logger.info(f"  L1: {train_metrics['g_l1_loss']:.4f}, Adv: {train_metrics['g_adv_loss']:.4f}")
            if train_metrics['g_perceptual_loss'] > 0:
                logger.info(f"  Perceptual: {train_metrics['g_perceptual_loss']:.4f}")
            if train_metrics['g_topology_loss'] > 0:
                logger.info(f"  Topology: {train_metrics['g_topology_loss']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            logger.info(f"  Val - L1: {val_metrics['l1_loss']:.4f}, PSNR: {val_metrics['psnr']:.2f} dB")
            
            # Update history
            for key, value in train_metrics.items():
                self.train_history[key].append(value)
            for key, value in val_metrics.items():
                self.val_history[key].append(value)
            
            # Check if best model
            is_best = val_metrics['psnr'] > self.best_val_psnr
            if is_best:
                self.best_val_psnr = val_metrics['psnr']
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
        
        logger.info(f"\n✓ Training complete!")
        logger.info(f"  Best validation PSNR: {self.best_val_psnr:.2f} dB")


def main():
    parser = argparse.ArgumentParser(description="Enhanced GAN Training")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr-g', type=float, default=2e-4)
    parser.add_argument('--lr-d', type=float, default=5e-5)
    parser.add_argument('--data-root', type=str, default='preprocessed_registered')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_enhanced')
    parser.add_argument('--no-topology', action='store_true')
    parser.add_argument('--no-perceptual', action='store_true')
    args = parser.parse_args()
    
    setup_logging()
    set_random_seeds(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Configuration
    config = EnhancedGANConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate_g = args.lr_g
    config.learning_rate_d = args.lr_d
    config.checkpoint_dir = Path(args.checkpoint_dir)
    config.use_topology_loss = not args.no_topology
    config.use_perceptual_loss = not args.no_perceptual
    
    # Load data
    data_root = Path(args.data_root)
    logger.info(f"Loading data from: {data_root}")
    
    # Create datasets (simplified - adapt to your data structure)
    from models.paired_dataset import Paired3T7TDataset
    
    # You need to implement your data loading here
    # This is a placeholder
    logger.warning("TODO: Implement actual data loading")
    
    # For now, create dummy loaders
    logger.error("Data loading not fully implemented. Please adapt to your dataset structure.")
    return
    
    # # Create trainer
    # trainer = EnhancedGANTrainer(config, train_loader, val_loader, device)
    # 
    # # Train
    # trainer.train()


if __name__ == "__main__":
    main()
