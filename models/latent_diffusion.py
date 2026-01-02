"""
Stage 3: Latent Diffusion Model (LDM) for Final 7T Enhancement

Implements a latent diffusion model that operates in compressed latent space
for efficient high-quality refinement of 7T MRI estimates.

Architecture:
1. VAE Encoder/Decoder for latent space compression
2. Denoising U-Net for diffusion in latent space
3. Conditioning on 3T input and GAN+PH outputs

References:
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022)
- Özbey et al., "Unsupervised Medical Image Translation with Adversarial Diffusion Models" (2023)
- Pinaya et al., "Brain Imaging Generation with Latent Diffusion Models" (MICCAI 2022)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import logging
import math

logger = logging.getLogger(__name__)


class VAEEncoder3D(nn.Module):
    """
    3D Variational Autoencoder Encoder for compressing MRI to latent space.
    
    Reduces spatial dimensions while preserving semantic information.
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        latent_channels: int = 4,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        
        # Downsampling path
        self.conv_in = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Down blocks (each reduces spatial dims by 2)
        self.down1 = self._make_down_block(base_channels, base_channels * 2, num_res_blocks)
        self.down2 = self._make_down_block(base_channels * 2, base_channels * 4, num_res_blocks)
        self.down3 = self._make_down_block(base_channels * 4, base_channels * 4, num_res_blocks)
        
        # Middle
        self.mid = ResidualBlock3D(base_channels * 4, base_channels * 4)
        
        # To latent space
        self.conv_out = nn.Conv3d(base_channels * 4, latent_channels * 2, kernel_size=3, padding=1)
    
    def _make_down_block(self, in_ch: int, out_ch: int, num_res_blocks: int) -> nn.Module:
        layers = []
        for i in range(num_res_blocks):
            layers.append(ResidualBlock3D(in_ch if i == 0 else out_ch, out_ch))
        layers.append(nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode to latent space with reparameterization.
        
        Args:
            x: Input image (B, C, D, H, W)
            
        Returns:
            Latent encoding and log variance
        """
        x = self.conv_in(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.mid(x)
        x = self.conv_out(x)
        
        # Split to mean and logvar
        mean, logvar = torch.chunk(x, 2, dim=1)
        
        return mean, logvar
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std


class VAEDecoder3D(nn.Module):
    """
    3D VAE Decoder for reconstructing from latent space.
    """
    def __init__(
        self,
        latent_channels: int = 4,
        base_channels: int = 64,
        out_channels: int = 1,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        
        self.conv_in = nn.Conv3d(latent_channels, base_channels * 4, kernel_size=3, padding=1)
        
        # Middle
        self.mid = ResidualBlock3D(base_channels * 4, base_channels * 4)
        
        # Up blocks
        self.up3 = self._make_up_block(base_channels * 4, base_channels * 4, num_res_blocks)
        self.up2 = self._make_up_block(base_channels * 4, base_channels * 2, num_res_blocks)
        self.up1 = self._make_up_block(base_channels * 2, base_channels, num_res_blocks)
        
        # Output
        self.conv_out = nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1)
    
    def _make_up_block(self, in_ch: int, out_ch: int, num_res_blocks: int) -> nn.Module:
        layers = [
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        ]
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock3D(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from latent space.
        
        Args:
            z: Latent encoding (B, latent_channels, D', H', W')
            
        Returns:
            Reconstructed image (B, 1, D, H, W)
        """
        x = self.conv_in(z)
        x = self.mid(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.conv_out(x)
        
        return x


class ResidualBlock3D(nn.Module):
    """3D Residual block with group normalization."""
    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or channels
        
        self.conv1 = nn.Conv3d(channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)
        
        if channels != out_channels:
            self.skip = nn.Conv3d(channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.gn2(h)
        
        return F.silu(h + self.skip(x))


class AttentionBlock3D(nn.Module):
    """3D Self-attention block for diffusion model."""
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.gn = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv3d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        
        h = self.gn(x)
        qkv = self.qkv(h)
        
        # Reshape for multi-head attention
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, D * H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, DHW, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        h = torch.matmul(attn, v)
        
        # Reshape back
        h = h.permute(0, 1, 3, 2).reshape(B, C, D, H, W)
        h = self.proj_out(h)
        
        return x + h


class DenoisingUNet3D(nn.Module):
    """
    3D U-Net for denoising in latent space with time conditioning.
    
    Used in the diffusion process to predict noise at each timestep.
    """
    def __init__(
        self,
        in_channels: int = 4,
        model_channels: int = 128,
        out_channels: int = 4,
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [2, 4],
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_heads: int = 8,
        condition_channels: int = 8,  # Conditioning from 3T + GAN/PH outputs
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Condition embedding (from 3T input)
        self.cond_conv = nn.Conv3d(condition_channels, model_channels, kernel_size=3, padding=1)
        
        # Input conv
        self.conv_in = nn.Conv3d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Downsampling
        self.down_blocks = nn.ModuleList([])
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResBlockWithTimeEmbed(ch, out_ch, time_embed_dim)
                )
                ch = out_ch
                
                if level in attention_resolutions:
                    self.down_blocks.append(AttentionBlock3D(ch, num_heads))
            
            if level < len(channel_mult) - 1:
                self.down_blocks.append(Downsample3D(ch))
        
        # Middle
        self.mid_block1 = ResBlockWithTimeEmbed(ch, ch, time_embed_dim)
        self.mid_attn = AttentionBlock3D(ch, num_heads)
        self.mid_block2 = ResBlockWithTimeEmbed(ch, ch, time_embed_dim)
        
        # Upsampling
        self.up_blocks = nn.ModuleList([])
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            for i in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResBlockWithTimeEmbed(ch + out_ch, out_ch, time_embed_dim)
                )
                ch = out_ch
                
                if level in attention_resolutions:
                    self.up_blocks.append(AttentionBlock3D(ch, num_heads))
            
            if level > 0:
                self.up_blocks.append(Upsample3D(ch))
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv3d(ch, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Denoise latent at given timesteps.
        
        Args:
            x: Noisy latent (B, C, D', H', W')
            timesteps: Diffusion timesteps (B,)
            condition: Conditioning information (B, cond_ch, D', H', W')
            
        Returns:
            Predicted noise
        """
        # Time embedding
        t_emb = self.get_timestep_embedding(timesteps, self.model_channels)
        t_emb = self.time_embed(t_emb)
        
        # Condition embedding
        cond_emb = self.cond_conv(condition)
        
        # Input + condition
        h = self.conv_in(x) + cond_emb
        
        # Downsampling with skip connections
        skip_connections = [h]
        for module in self.down_blocks:
            h = module(h, t_emb) if isinstance(module, ResBlockWithTimeEmbed) else module(h)
            skip_connections.append(h)
        
        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Upsampling with skip connections
        for module in self.up_blocks:
            if isinstance(module, ResBlockWithTimeEmbed):
                h = torch.cat([h, skip_connections.pop()], dim=1)
                h = module(h, t_emb)
            else:
                h = module(h)
        
        # Output
        h = self.conv_out(h)
        
        return h
    
    @staticmethod
    def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        """Sinusoidal timestep embedding."""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class ResBlockWithTimeEmbed(nn.Module):
    """Residual block with time embedding conditioning."""
    def __init__(self, channels: int, out_channels: int, time_embed_dim: int):
        super().__init__()
        
        self.conv1 = nn.Conv3d(channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)
        
        if channels != out_channels:
            self.skip = nn.Conv3d(channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.silu(h)
        
        # Add time embedding
        t_emb = self.time_proj(F.silu(t_emb))
        h = h + t_emb[:, :, None, None, None]
        
        h = self.conv2(h)
        h = self.gn2(h)
        
        return F.silu(h + self.skip(x))


class Downsample3D(nn.Module):
    """3D downsampling layer."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    """3D upsampling layer."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        return self.conv(x)


class LatentDiffusionModel:
    """
    Complete Latent Diffusion Model for 7T MRI enhancement.
    
    Combines VAE for compression and diffusion model for refinement.
    """
    def __init__(
        self,
        vae_encoder: VAEEncoder3D,
        vae_decoder: VAEDecoder3D,
        denoising_unet: DenoisingUNet3D,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        device: torch.device = torch.device('cpu'),
    ):
        self.vae_encoder = vae_encoder.to(device)
        self.vae_decoder = vae_decoder.to(device)
        self.denoising_unet = denoising_unet.to(device)
        self.device = device
        self.num_timesteps = num_timesteps
        
        # Set up diffusion schedule
        self.betas = self._get_beta_schedule(beta_schedule, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.vae_encoder.eval()
        self.vae_decoder.eval()
        self.denoising_unet.eval()
    
    def _get_beta_schedule(self, schedule: str, num_timesteps: int) -> torch.Tensor:
        """Get noise schedule."""
        if schedule == "linear":
            beta_start, beta_end = 0.0001, 0.02
            return torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
    
    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space."""
        mean, logvar = self.vae_encoder(x)
        z = self.vae_encoder.reparameterize(mean, logvar)
        return z
    
    @torch.no_grad()
    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.vae_decoder(z)
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        condition: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        """
        Sample from diffusion model (DDPM or DDIM).
        
        Args:
            shape: Shape of latent to generate (B, C, D', H', W')
            condition: Conditioning (3T + GAN/PH outputs encoded)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Generated latent
        """
        # Start from random noise
        x_t = torch.randn(shape, device=self.device)
        
        # Denoising loop
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long)
        
        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.denoising_unet(x_t, t_batch, condition)
            
            # DDPM update
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
            
            beta_t = self.betas[t]
            
            # Compute x_{t-1}
            pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            # Clip for stability
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Compute direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_t_prev) * predicted_noise
            
            # Compute x_{t-1}
            x_t = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
            
            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = x_t + torch.sqrt(beta_t) * noise
        
        return x_t
    
    @torch.no_grad()
    def enhance(
        self,
        input_3t: torch.Tensor,
        gan_ph_output: torch.Tensor,
        num_inference_steps: int = 50,
    ) -> torch.Tensor:
        """
        Complete enhancement pipeline.
        
        Args:
            input_3t: Original 3T input (B, 1, D, H, W)
            gan_ph_output: Output from GAN+PH stages (B, 1, D, H, W)
            num_inference_steps: Number of diffusion steps
            
        Returns:
            Final enhanced 7T output
        """
        # Encode inputs to latent space
        latent_3t = self.encode_to_latent(input_3t)
        latent_gan_ph = self.encode_to_latent(gan_ph_output)
        
        # Concatenate as condition
        condition = torch.cat([latent_3t, latent_gan_ph], dim=1)
        
        # Sample from diffusion
        latent_shape = latent_gan_ph.shape
        enhanced_latent = self.sample(latent_shape, condition, num_inference_steps)
        
        # Decode to image space
        enhanced_image = self.decode_from_latent(enhanced_latent)
        
        return enhanced_image


if __name__ == "__main__":
    # Test latent diffusion model
    print("Testing Latent Diffusion Model components...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test VAE
    print("\n1. Testing VAE...")
    encoder = VAEEncoder3D(in_channels=1, latent_channels=4).to(device)
    decoder = VAEDecoder3D(latent_channels=4, out_channels=1).to(device)
    
    x = torch.randn(1, 1, 32, 64, 64).to(device)
    mean, logvar = encoder(x)
    z = encoder.reparameterize(mean, logvar)
    recon = decoder(z)
    
    print(f"   Input: {x.shape}")
    print(f"   Latent: {z.shape}")
    print(f"   Reconstruction: {recon.shape}")
    
    # Test Denoising U-Net
    print("\n2. Testing Denoising U-Net...")
    unet = DenoisingUNet3D(in_channels=4, condition_channels=8).to(device)
    
    z = torch.randn(1, 4, 4, 8, 8).to(device)
    t = torch.tensor([500]).to(device)
    cond = torch.randn(1, 8, 4, 8, 8).to(device)
    
    noise_pred = unet(z, t, cond)
    print(f"   Noisy latent: {z.shape}")
    print(f"   Predicted noise: {noise_pred.shape}")
    
    print("\n✓ Latent Diffusion Model tests complete")
