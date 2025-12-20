"""
Training script for 3T → 7T MRI super-resolution GAN.

Implements:
- Generator (3D U-Net)
- Discriminator (3D PatchGAN)
- L1 reconstruction loss
- Adversarial loss (LSGAN or BCE)
- Patch-based training
- Checkpoint saving
- Visualization
"""
import argparse
import logging
from pathlib import Path
import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.generator_unet3d import UNet3DGenerator
from models.discriminator_patchgan3d import PatchGANDiscriminator3D
from models.paired_dataset import (
    Paired3T7TDataset,
    create_paired_data_list,
    build_gan_augmentation_transforms,
)
from src.config import get_default_config
from src.utils import discover_dataset, create_patient_level_split, setup_logging, set_random_seeds


logger = logging.getLogger(__name__)


class GANConfig:
    """Configuration for GAN training."""
    def __init__(self):
        # Model architecture
        self.generator_base_features = 32
        self.generator_num_levels = 4
        self.discriminator_base_features = 64
        self.discriminator_num_layers = 3
        self.norm_type = "instance"  # "instance" or "group"
        
        # Training
        self.num_epochs = 100
        self.batch_size = 2  # Small for 3D volumes
        self.learning_rate_g = 2e-4
        self.learning_rate_d = 2e-4
        self.beta1 = 0.5  # Adam beta1
        self.beta2 = 0.999  # Adam beta2
        
        # Loss weights
        self.lambda_l1 = 100.0  # Weight for L1 loss
        self.lambda_adv = 1.0   # Weight for adversarial loss
        
        # Loss type
        self.adversarial_loss_type = "lsgan"  # "lsgan" or "bce"
        
        # Patch sampling
        self.patch_size = (64, 64, 64)
        self.num_patches_per_volume = 10
        
        # Data augmentation
        self.use_augmentation = True
        self.augmentation_prob = 0.5
        
        # Optimization
        self.use_amp = False  # Mixed precision
        self.gradient_clip_value = 1.0  # Gradient clipping for stability
        
        # Logging and saving
        self.save_interval = 5  # Save checkpoint every N epochs
        self.log_interval = 10  # Log every N iterations
        self.vis_interval = 1   # Save visualizations every N epochs
        
        # Paths
        self.checkpoint_dir = Path("checkpoints")
        self.visualization_dir = Path("visualizations")
        self.log_dir = Path("logs")
        
        # Data
        self.modality = "T1w"  # Use T1w or T2w
        self.num_workers = 4
        self.pin_memory = True


class GANTrainer:
    """Trainer class for 3T→7T GAN."""
    
    def __init__(
        self,
        config: GANConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Create directories
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.visualization_dir.mkdir(parents=True, exist_ok=True)
        config.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
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
        ).to(device)
        
        logger.info(f"Generator parameters: {self.generator.get_num_parameters():,}")
        logger.info(f"Discriminator parameters: {self.discriminator.get_num_parameters():,}")
        
        # Initialize optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate_g,
            betas=(config.beta1, config.beta2),
        )
        
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate_d,
            betas=(config.beta1, config.beta2),
        )
        
        # Loss functions
        self.criterion_l1 = nn.L1Loss()
        
        if config.adversarial_loss_type == "lsgan":
            # Least Squares GAN (more stable)
            self.criterion_adv = nn.MSELoss()
        elif config.adversarial_loss_type == "bce":
            # Binary Cross Entropy
            self.criterion_adv = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown adversarial loss: {config.adversarial_loss_type}")
        
        # Mixed precision scaler
        self.scaler_g = GradScaler(enabled=config.use_amp)
        self.scaler_d = GradScaler(enabled=config.use_amp)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        # Metrics storage
        self.train_history = {
            'g_loss': [],
            'g_l1_loss': [],
            'g_adv_loss': [],
            'd_loss': [],
            'd_real_loss': [],
            'd_fake_loss': [],
        }
        
        self.val_history = {
            'l1_loss': [],
            'psnr': [],
        }
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = {
            'g_loss': 0.0,
            'g_l1_loss': 0.0,
            'g_adv_loss': 0.0,
            'd_loss': 0.0,
            'd_real_loss': 0.0,
            'd_fake_loss': 0.0,
        }
        
        for i, batch in enumerate(self.train_loader):
            # Move to device
            input_3t = batch['input_3t'].to(self.device)  # (B, 1, D, H, W)
            target_7t = batch['target_7t'].to(self.device)  # (B, 1, D, H, W)
            
            batch_size = input_3t.size(0)
            
            # ===== Train Discriminator =====
            self.optimizer_d.zero_grad()
            
            with autocast(enabled=self.config.use_amp):
                # Generate fake 7T
                with torch.no_grad():
                    fake_7t = self.generator(input_3t)
                
                # Real predictions
                pred_real = self.discriminator(target_7t)
                # Fake predictions
                pred_fake = self.discriminator(fake_7t.detach())
                
                # Real/fake labels
                if self.config.adversarial_loss_type == "lsgan":
                    # LSGAN: real=1, fake=0
                    label_real = torch.ones_like(pred_real)
                    label_fake = torch.zeros_like(pred_fake)
                else:
                    # BCE: real=1, fake=0
                    label_real = torch.ones_like(pred_real)
                    label_fake = torch.zeros_like(pred_fake)
                
                # Discriminator losses
                loss_d_real = self.criterion_adv(pred_real, label_real)
                loss_d_fake = self.criterion_adv(pred_fake, label_fake)
                loss_d = (loss_d_real + loss_d_fake) * 0.5
            
            # Backward
            self.scaler_d.scale(loss_d).backward()
            
            # Gradient clipping
            if self.config.gradient_clip_value > 0:
                self.scaler_d.unscale_(self.optimizer_d)
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(),
                    self.config.gradient_clip_value
                )
            
            self.scaler_d.step(self.optimizer_d)
            self.scaler_d.update()
            
            # ===== Train Generator =====
            self.optimizer_g.zero_grad()
            
            with autocast(enabled=self.config.use_amp):
                # Generate fake 7T
                fake_7t = self.generator(input_3t)
                
                # L1 reconstruction loss
                loss_g_l1 = self.criterion_l1(fake_7t, target_7t)
                
                # Adversarial loss (fool discriminator)
                pred_fake = self.discriminator(fake_7t)
                
                if self.config.adversarial_loss_type == "lsgan":
                    label_real = torch.ones_like(pred_fake)
                else:
                    label_real = torch.ones_like(pred_fake)
                
                loss_g_adv = self.criterion_adv(pred_fake, label_real)
                
                # Total generator loss
                loss_g = (
                    self.config.lambda_l1 * loss_g_l1 +
                    self.config.lambda_adv * loss_g_adv
                )
            
            # Backward
            self.scaler_g.scale(loss_g).backward()
            
            # Gradient clipping
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
            epoch_metrics['d_loss'] += loss_d.item()
            epoch_metrics['d_real_loss'] += loss_d_real.item()
            epoch_metrics['d_fake_loss'] += loss_d_fake.item()
            
            # Logging
            if (i + 1) % self.config.log_interval == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{self.config.num_epochs}] "
                    f"Iter [{i+1}/{len(self.train_loader)}] "
                    f"G_loss: {loss_g.item():.4f} (L1: {loss_g_l1.item():.4f}, Adv: {loss_g_adv.item():.4f}) "
                    f"D_loss: {loss_d.item():.4f} (Real: {loss_d_real.item():.4f}, Fake: {loss_d_fake.item():.4f})"
                )
            
            self.global_step += 1
        
        # Average metrics
        num_batches = len(self.train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            self.train_history[key].append(epoch_metrics[key])
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate the model."""
        self.generator.eval()
        
        val_l1_loss = 0.0
        val_psnr = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            input_3t = batch['input_3t'].to(self.device)
            target_7t = batch['target_7t'].to(self.device)
            
            # Generate
            fake_7t = self.generator(input_3t)
            
            # L1 loss
            l1_loss = self.criterion_l1(fake_7t, target_7t)
            val_l1_loss += l1_loss.item()
            
            # PSNR
            mse = torch.mean((fake_7t - target_7t) ** 2)
            if mse > 0:
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                val_psnr += psnr.item()
            
            num_batches += 1
        
        # Average
        val_l1_loss /= num_batches
        val_psnr /= num_batches
        
        self.val_history['l1_loss'].append(val_l1_loss)
        self.val_history['psnr'].append(val_psnr)
        
        logger.info(f"Validation - L1: {val_l1_loss:.4f}, PSNR: {val_psnr:.2f} dB")
        
        return val_l1_loss, val_psnr
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config.__dict__,
        }
        
        # Save regular checkpoint
        path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
        
        # Save best model
        if is_best:
            best_path = self.config.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    @torch.no_grad()
    def save_visualizations(self, epoch: int):
        """Save example visualizations."""
        self.generator.eval()
        
        # Get one batch from validation
        batch = next(iter(self.val_loader))
        input_3t = batch['input_3t'].to(self.device)[:1]  # First sample
        target_7t = batch['target_7t'].to(self.device)[:1]
        
        # Generate
        fake_7t = self.generator(input_3t)
        
        # Move to CPU and numpy
        input_3t = input_3t[0, 0].cpu().numpy()  # (D, H, W)
        target_7t = target_7t[0, 0].cpu().numpy()
        fake_7t = fake_7t[0, 0].cpu().numpy()
        
        # Take middle slices
        d, h, w = input_3t.shape
        slice_idx = d // 2
        
        slice_3t = input_3t[slice_idx]
        slice_7t_real = target_7t[slice_idx]
        slice_7t_fake = fake_7t[slice_idx]
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(slice_3t, cmap='gray')
        axes[0].set_title('Input 3T')
        axes[0].axis('off')
        
        axes[1].imshow(slice_7t_fake, cmap='gray')
        axes[1].set_title('Generated 7T')
        axes[1].axis('off')
        
        axes[2].imshow(slice_7t_real, cmap='gray')
        axes[2].set_title('Real 7T (Ground Truth)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save
        save_path = self.config.visualization_dir / f"epoch_{epoch:03d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization: {save_path}")
    
    def train(self):
        """Main training loop."""
        logger.info("="*80)
        logger.info("Starting GAN Training")
        logger.info("="*80)
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Patch size: {self.config.patch_size}")
        logger.info(f"Device: {self.device}")
        logger.info("="*80)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_l1, val_psnr = self.validate(epoch)
            
            # Check if best
            is_best = val_l1 < best_val_loss
            if is_best:
                best_val_loss = val_l1
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch + 1, is_best=is_best)
            
            # Save visualizations
            if (epoch + 1) % self.config.vis_interval == 0:
                self.save_visualizations(epoch + 1)
            
            epoch_time = time.time() - start_time
            
            logger.info(
                f"Epoch [{epoch+1}/{self.config.num_epochs}] completed in {epoch_time:.2f}s - "
                f"G_loss: {train_metrics['g_loss']:.4f}, D_loss: {train_metrics['d_loss']:.4f}, "
                f"Val_L1: {val_l1:.4f}, Val_PSNR: {val_psnr:.2f} dB"
            )
            logger.info("-"*80)
        
        logger.info("="*80)
        logger.info("Training completed!")
        logger.info(f"Best validation L1 loss: {best_val_loss:.4f}")
        logger.info("="*80)


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train 3T→7T GAN')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr_g', type=float, default=2e-4, help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=2e-4, help='Discriminator learning rate')
    parser.add_argument('--lambda_l1', type=float, default=100.0, help='L1 loss weight')
    parser.add_argument('--patch_size', type=int, default=64, help='Patch size')
    parser.add_argument('--num_patches', type=int, default=10, help='Patches per volume')
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--modality', type=str, default='T1w', choices=['T1w', 'T2w'])
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    set_random_seeds(args.seed)
    
    # Create GAN config
    gan_config = GANConfig()
    gan_config.num_epochs = args.epochs
    gan_config.batch_size = args.batch_size
    gan_config.learning_rate_g = args.lr_g
    gan_config.learning_rate_d = args.lr_d
    gan_config.lambda_l1 = args.lambda_l1
    gan_config.patch_size = (args.patch_size, args.patch_size, args.patch_size)
    gan_config.num_patches_per_volume = args.num_patches
    gan_config.use_amp = args.use_amp
    gan_config.modality = args.modality
    
    # Setup logging
    log_config = get_default_config().logging
    log_config.log_dir = gan_config.log_dir
    setup_logging(log_config)
    
    logger.info("="*80)
    logger.info("3T → 7T MRI Super-Resolution GAN")
    logger.info("="*80)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data configuration
    data_config = get_default_config()
    
    # Discover dataset
    logger.info("Discovering dataset...")
    data_list = discover_dataset(data_config.data.preprocessed_root, data_config.data)
    
    if len(data_list) == 0:
        logger.error("No preprocessed data found!")
        logger.info("Please run preprocessing first: python example_pipeline.py --preprocess")
        return
    
    # Create patient-level split
    logger.info("Creating patient-level split...")
    train_data, val_data, test_data = create_patient_level_split(
        data_list,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_seed=args.seed,
    )
    
    # Create paired datasets
    logger.info("Creating paired 3T→7T datasets...")
    train_pairs = create_paired_data_list(train_data, modality=args.modality)
    val_pairs = create_paired_data_list(val_data, modality=args.modality)
    
    if len(train_pairs) == 0 or len(val_pairs) == 0:
        logger.error("No paired 3T→7T data found!")
        logger.info("Make sure you have both ses-1 (3T) and ses-2 (7T) preprocessed data.")
        return
    
    # Build augmentation transforms
    transform = None
    if gan_config.use_augmentation:
        transform = build_gan_augmentation_transforms(
            augmentation_prob=gan_config.augmentation_prob
        )
    
    # Create datasets
    train_dataset = Paired3T7TDataset(
        data_pairs=train_pairs,
        patch_size=gan_config.patch_size,
        num_patches_per_volume=gan_config.num_patches_per_volume,
        transform=transform,
        deterministic=True,
        random_seed=args.seed,
    )
    
    val_dataset = Paired3T7TDataset(
        data_pairs=val_pairs,
        patch_size=gan_config.patch_size,
        num_patches_per_volume=5,  # Fewer patches for validation
        transform=None,  # No augmentation for validation
        deterministic=True,
        random_seed=args.seed + 1,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=gan_config.batch_size,
        shuffle=True,
        num_workers=gan_config.num_workers,
        pin_memory=gan_config.pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=gan_config.batch_size,
        shuffle=False,
        num_workers=gan_config.num_workers,
        pin_memory=gan_config.pin_memory,
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Create trainer
    trainer = GANTrainer(
        config=gan_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
    
    # Train
    trainer.train()
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
