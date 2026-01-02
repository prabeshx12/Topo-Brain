"""
Complete 3-Stage Pipeline: GAN → Persistent Homology → Latent Diffusion

This script orchestrates the full pipeline for 3T to 7T MRI enhancement:

Stage 1: GAN (with topology-aware and perceptual losses)
    - Generates initial 7T estimate from 3T input
    - Trained with L1, adversarial, perceptual, and topology losses

Stage 2: Persistent Homology Refinement
    - Analyzes and corrects topological features
    - Ensures anatomical structure preservation

Stage 3: Latent Diffusion Model
    - Final high-quality refinement in latent space
    - Conditioned on 3T input and GAN+PH outputs

Usage:
    # Train all stages
    python scripts/run_full_pipeline.py --mode train --data-root preprocessed_registered
    
    # Inference only
    python scripts/run_full_pipeline.py --mode inference --checkpoint checkpoints/full_pipeline.pth
"""
import argparse
import logging
from pathlib import Path
import sys
import json
from typing import Dict, Any, Optional
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from models.generator_unet3d import UNet3DGenerator
from models.discriminator_patchgan3d import PatchGANDiscriminator3D
from models.topology_loss import CombinedTopologyLoss
from models.perceptual_loss import MedicalPerceptualLoss
from models.ph_refiner import PHRefiner
from models.latent_diffusion import (
    LatentDiffusionModel,
    VAEEncoder3D,
    VAEDecoder3D,
    DenoisingUNet3D,
)
from src.utils import setup_logging, set_random_seeds

logger = logging.getLogger(__name__)


class ThreeStagePipeline:
    """
    Complete three-stage MRI enhancement pipeline.
    """
    def __init__(
        self,
        device: torch.device,
        checkpoint_dir: Path = Path("checkpoints_pipeline"),
    ):
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Stage 1: GAN
        self.generator = None
        self.discriminator = None
        
        # Stage 2: PH Refiner
        self.ph_refiner = None
        
        # Stage 3: Latent Diffusion
        self.ldm = None
        
        # Metrics storage
        self.metrics_history = {
            'stage1': [],
            'stage2': [],
            'stage3': [],
        }
    
    def build_models(self):
        """Initialize all models for the pipeline."""
        logger.info("Building three-stage pipeline models...")
        
        # Stage 1: GAN
        logger.info("  Stage 1: GAN...")
        self.generator = UNet3DGenerator(
            in_channels=1,
            out_channels=1,
            base_features=32,
            num_levels=4,
        ).to(self.device)
        
        self.discriminator = PatchGANDiscriminator3D(
            in_channels=1,
            base_features=64,
            num_layers=3,
            use_spectral_norm=True,
        ).to(self.device)
        
        logger.info(f"    Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")
        logger.info(f"    Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        
        # Stage 2: PH Refiner
        logger.info("  Stage 2: Persistent Homology Refiner...")
        self.ph_refiner = PHRefiner(device=self.device)
        
        # Stage 3: Latent Diffusion
        logger.info("  Stage 3: Latent Diffusion Model...")
        vae_encoder = VAEEncoder3D(in_channels=1, latent_channels=4)
        vae_decoder = VAEDecoder3D(latent_channels=4, out_channels=1)
        denoising_unet = DenoisingUNet3D(
            in_channels=4,
            model_channels=128,
            out_channels=4,
            condition_channels=8,
        )
        
        self.ldm = LatentDiffusionModel(
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
            denoising_unet=denoising_unet,
            num_timesteps=1000,
            device=self.device,
        )
        
        logger.info("✓ All models built successfully")
    
    def train_stage1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        lr_g: float = 2e-4,
        lr_d: float = 5e-5,
    ) -> Dict[str, Any]:
        """
        Train Stage 1: GAN with topology and perceptual losses.
        """
        logger.info("\n" + "="*60)
        logger.info("STAGE 1: Training GAN")
        logger.info("="*60)
        
        # Import the enhanced trainer
        from scripts.train_gan_enhanced import EnhancedGANConfig, EnhancedGANTrainer
        
        config = EnhancedGANConfig()
        config.num_epochs = num_epochs
        config.learning_rate_g = lr_g
        config.learning_rate_d = lr_d
        config.checkpoint_dir = self.checkpoint_dir / "stage1"
        
        trainer = EnhancedGANTrainer(config, train_loader, val_loader, self.device)
        trainer.generator = self.generator
        trainer.discriminator = self.discriminator
        
        trainer.train()
        
        # Load best model
        best_checkpoint = config.checkpoint_dir / "best_model.pth"
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            logger.info(f"✓ Loaded best Stage 1 model: PSNR = {checkpoint['best_val_psnr']:.2f} dB")
            
            return {'best_psnr': checkpoint['best_val_psnr']}
        
        return {}
    
    def train_stage2(
        self,
        train_loader: DataLoader,
        num_epochs: int = 50,
        lr: float = 1e-4,
    ) -> Dict[str, Any]:
        """
        Train Stage 2: PH Refiner.
        
        Uses Stage 1 GAN outputs to train topology corrector.
        """
        logger.info("\n" + "="*60)
        logger.info("STAGE 2: Training Persistent Homology Refiner")
        logger.info("="*60)
        
        self.generator.eval()
        
        # Create training data: GAN outputs + targets
        gan_outputs_loader = self._create_gan_outputs_dataset(train_loader)
        
        # Train PH corrector
        save_path = self.checkpoint_dir / "stage2" / "ph_corrector.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.ph_refiner.train_corrector(
            gan_outputs_loader,
            num_epochs=num_epochs,
            learning_rate=lr,
            save_path=str(save_path),
        )
        
        logger.info(f"✓ Stage 2 training complete")
        
        return {'checkpoint': str(save_path)}
    
    def train_stage3(
        self,
        train_loader: DataLoader,
        num_epochs: int = 100,
        lr: float = 1e-4,
    ) -> Dict[str, Any]:
        """
        Train Stage 3: Latent Diffusion Model.
        
        First train VAE, then train diffusion model conditioned on
        3T input and GAN+PH outputs.
        """
        logger.info("\n" + "="*60)
        logger.info("STAGE 3: Training Latent Diffusion Model")
        logger.info("="*60)
        
        # Freeze Stage 1 and 2
        self.generator.eval()
        self.ph_refiner.corrector.eval()
        
        logger.info("Step 1: Training VAE...")
        self._train_vae(train_loader, num_epochs=50, lr=lr)
        
        logger.info("Step 2: Training Diffusion Model...")
        self._train_diffusion(train_loader, num_epochs=num_epochs, lr=lr)
        
        logger.info(f"✓ Stage 3 training complete")
        
        return {}
    
    def _train_vae(self, train_loader: DataLoader, num_epochs: int, lr: float):
        """Train VAE for latent compression."""
        optimizer = torch.optim.Adam(
            list(self.ldm.vae_encoder.parameters()) + 
            list(self.ldm.vae_decoder.parameters()),
            lr=lr
        )
        
        criterion_recon = nn.L1Loss()
        
        self.ldm.vae_encoder.train()
        self.ldm.vae_decoder.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"VAE Epoch {epoch+1}/{num_epochs}"):
                target_7t = batch['target_7t'].to(self.device)
                
                # Encode
                mean, logvar = self.ldm.vae_encoder(target_7t)
                z = self.ldm.vae_encoder.reparameterize(mean, logvar)
                
                # Decode
                recon = self.ldm.vae_decoder(z)
                
                # Loss: reconstruction + KL divergence
                loss_recon = criterion_recon(recon, target_7t)
                loss_kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
                
                loss = loss_recon + 0.0001 * loss_kl
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(train_loader)
            logger.info(f"  Epoch {epoch+1}: Loss = {epoch_loss:.4f}")
        
        # Save VAE
        vae_path = self.checkpoint_dir / "stage3" / "vae.pth"
        vae_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'encoder': self.ldm.vae_encoder.state_dict(),
            'decoder': self.ldm.vae_decoder.state_dict(),
        }, vae_path)
        
        self.ldm.vae_encoder.eval()
        self.ldm.vae_decoder.eval()
    
    def _train_diffusion(self, train_loader: DataLoader, num_epochs: int, lr: float):
        """Train diffusion model in latent space."""
        optimizer = torch.optim.Adam(
            self.ldm.denoising_unet.parameters(),
            lr=lr
        )
        
        self.ldm.denoising_unet.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Diffusion Epoch {epoch+1}/{num_epochs}"):
                input_3t = batch['input_3t'].to(self.device)
                target_7t = batch['target_7t'].to(self.device)
                
                # Get GAN + PH outputs
                with torch.no_grad():
                    gan_output = self.generator(input_3t)
                    ph_output, _ = self.ph_refiner.refine(gan_output, target_7t)
                    
                    # Encode to latent
                    latent_3t = self.ldm.encode_to_latent(input_3t)
                    latent_ph = self.ldm.encode_to_latent(ph_output)
                    latent_target, _ = self.ldm.vae_encoder(target_7t)
                    
                    condition = torch.cat([latent_3t, latent_ph], dim=1)
                
                # Sample random timestep
                t = torch.randint(0, self.ldm.num_timesteps, (input_3t.shape[0],), device=self.device)
                
                # Add noise to target latent
                noise = torch.randn_like(latent_target)
                alpha_t = self.ldm.alphas_cumprod[t][:, None, None, None, None]
                noisy_latent = torch.sqrt(alpha_t) * latent_target + torch.sqrt(1 - alpha_t) * noise
                
                # Predict noise
                predicted_noise = self.ldm.denoising_unet(noisy_latent, t, condition)
                
                # Loss
                loss = F.mse_loss(predicted_noise, noise)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(train_loader)
            logger.info(f"  Epoch {epoch+1}: Loss = {epoch_loss:.4f}")
        
        # Save diffusion model
        diffusion_path = self.checkpoint_dir / "stage3" / "diffusion_unet.pth"
        torch.save(self.ldm.denoising_unet.state_dict(), diffusion_path)
        
        self.ldm.denoising_unet.eval()
    
    def _create_gan_outputs_dataset(self, loader: DataLoader) -> DataLoader:
        """Helper to create dataset of GAN outputs for Stage 2 training."""
        # This is a placeholder - implement according to your data structure
        return loader
    
    @torch.no_grad()
    def inference(
        self,
        input_3t: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Run complete three-stage inference.
        
        Args:
            input_3t: 3T MRI input (B, 1, D, H, W)
            return_intermediates: Whether to return outputs from each stage
            
        Returns:
            Dictionary with final output and optionally intermediate results
        """
        self.generator.eval()
        self.ph_refiner.corrector.eval()
        self.ldm.vae_encoder.eval()
        self.ldm.vae_decoder.eval()
        self.ldm.denoising_unet.eval()
        
        results = {}
        
        # Stage 1: GAN
        logger.info("Running Stage 1: GAN...")
        stage1_output = self.generator(input_3t)
        if return_intermediates:
            results['stage1_output'] = stage1_output
        
        # Stage 2: PH Refinement
        logger.info("Running Stage 2: PH Refinement...")
        stage2_output, ph_metrics = self.ph_refiner.refine(stage1_output)
        if return_intermediates:
            results['stage2_output'] = stage2_output
            results['ph_metrics'] = ph_metrics
        
        # Stage 3: Latent Diffusion
        logger.info("Running Stage 3: Latent Diffusion...")
        stage3_output = self.ldm.enhance(
            input_3t,
            stage2_output,
            num_inference_steps=50,
        )
        
        results['final_output'] = stage3_output
        
        return results
    
    def save_checkpoint(self, path: Path):
        """Save complete pipeline checkpoint."""
        checkpoint = {
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'ph_corrector': self.ph_refiner.corrector.state_dict(),
            'vae_encoder': self.ldm.vae_encoder.state_dict(),
            'vae_decoder': self.ldm.vae_decoder.state_dict(),
            'denoising_unet': self.ldm.denoising_unet.state_dict(),
            'metrics_history': self.metrics_history,
        }
        
        torch.save(checkpoint, path)
        logger.info(f"✓ Saved complete pipeline checkpoint: {path}")
    
    def load_checkpoint(self, path: Path):
        """Load complete pipeline checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.ph_refiner.corrector.load_state_dict(checkpoint['ph_corrector'])
        self.ldm.vae_encoder.load_state_dict(checkpoint['vae_encoder'])
        self.ldm.vae_decoder.load_state_dict(checkpoint['vae_decoder'])
        self.ldm.denoising_unet.load_state_dict(checkpoint['denoising_unet'])
        self.metrics_history = checkpoint.get('metrics_history', {})
        
        logger.info(f"✓ Loaded complete pipeline from: {path}")


def main():
    parser = argparse.ArgumentParser(description="Three-Stage MRI Enhancement Pipeline")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'])
    parser.add_argument('--data-root', type=str, default='preprocessed_registered')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_pipeline')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for inference')
    parser.add_argument('--epochs-stage1', type=int, default=100)
    parser.add_argument('--epochs-stage2', type=int, default=50)
    parser.add_argument('--epochs-stage3', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    
    setup_logging()
    set_random_seeds(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create pipeline
    pipeline = ThreeStagePipeline(
        device=device,
        checkpoint_dir=Path(args.checkpoint_dir),
    )
    
    pipeline.build_models()
    
    if args.mode == 'train':
        logger.info("\n" + "="*60)
        logger.info("COMPLETE THREE-STAGE TRAINING PIPELINE")
        logger.info("="*60)
        logger.info(f"Data root: {args.data_root}")
        logger.info(f"Checkpoint dir: {args.checkpoint_dir}")
        logger.info(f"Epochs: Stage1={args.epochs_stage1}, Stage2={args.epochs_stage2}, Stage3={args.epochs_stage3}")
        
        # Load data
        logger.warning("TODO: Implement data loading for your dataset structure")
        logger.error("Data loading not implemented. Please adapt to your dataset.")
        return
        
        # # Train all stages
        # pipeline.train_stage1(train_loader, val_loader, args.epochs_stage1)
        # pipeline.train_stage2(train_loader, args.epochs_stage2)
        # pipeline.train_stage3(train_loader, args.epochs_stage3)
        # 
        # # Save final checkpoint
        # final_checkpoint = Path(args.checkpoint_dir) / "complete_pipeline.pth"
        # pipeline.save_checkpoint(final_checkpoint)
        # 
        # logger.info(f"\n✓ Complete pipeline training finished!")
        # logger.info(f"  Checkpoint saved: {final_checkpoint}")
    
    elif args.mode == 'inference':
        if args.checkpoint is None:
            logger.error("--checkpoint required for inference mode")
            return
        
        # Load checkpoint
        pipeline.load_checkpoint(Path(args.checkpoint))
        
        logger.info("\nRunning inference...")
        logger.warning("TODO: Implement inference loop for your data")


if __name__ == "__main__":
    main()
