"""
Model Architecture for Topo-Brain: Anatomy-Guided Conditional Diffusion Model.
Implements Blueprint Section 7.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Simple 3D Residual Block.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm3d(channels)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)

class AnatomyGuidedUNet(nn.Module):
    """
    3D U-Net backbone for Diffusion Model.
    Conditioned on 3T input via concatenation.
    """
    def __init__(
        self,
        in_channels: int = 1,     # Target channels (7T t1w)
        cond_channels: int = 1,   # Conditioning channels (3T t1w + optional t2w)
        out_channels: int = 1,    # Output channels (Noise or 7T prediction)
        num_classes: int = 3,     # Segmentation classes (Background, GM, WM)
        features: tuple = (32, 64, 128, 256),
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # We model p(x_7t | x_3t). 
        # The input to the UNet at each step is Concatenate(x_t, x_3t).
        # x_t: Noisy 7T guess (in_channels)
        # x_3t: Clean 3T conditional (cond_channels)
        
        total_in_channels = in_channels + cond_channels
        
        # Using MONAI's basic UNet structure but we might need 
        # a custom one for Time Embedding injection if we do full DDPM.
        # For now, let's implement a custom simplified U-Net to handle 
        # Time Embeddings explicitly, as MONAI's basic UNet doesn't easy expose it.
        
        self.inc = nn.Conv3d(total_in_channels, features[0], kernel_size=3, padding=1)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        # Downsample
        in_ch = features[0]
        for feat in features[1:]:
            self.downs.append(
                nn.Sequential(
                    nn.Conv3d(in_ch, feat, kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm3d(feat),
                    nn.LeakyReLU(0.2),
                    ResidualBlock(feat)
                )
            )
            in_ch = feat
            
        # Bottleneck
        self.bottleneck = ResidualBlock(features[-1])
        
        # Upsample (Denoising)
        for feat in reversed(features[:-1]):
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_ch, feat, kernel_size=2, stride=2),
                    nn.InstanceNorm3d(feat),
                    nn.LeakyReLU(0.2),
                    nn.Conv3d(feat, feat, kernel_size=3, padding=1), # Refinement
                    ResidualBlock(feat)
                )
            )
            in_ch = feat
            
        self.outc = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
        # Segmentation Decoder (Mirrors the Denoising Decoder structure)
        self.seg_ups = nn.ModuleList()
        in_ch_seg = features[-1]
        for feat in reversed(features[:-1]):
            self.seg_ups.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_ch_seg, feat, kernel_size=2, stride=2),
                    nn.InstanceNorm3d(feat),
                    nn.LeakyReLU(0.2),
                    nn.Conv3d(feat, feat, kernel_size=3, padding=1),
                    ResidualBlock(feat)
                )
            )
            in_ch_seg = feat
            
        # Segmentation output (Logits)
        self.seg_outc = nn.Conv3d(features[0], num_classes, kernel_size=1)
        
        # Time Embedding (Sinusoidal usually, simplified here for placeholder)
        self.time_embed = nn.Sequential(
            nn.Linear(1, features[0]),
            nn.SiLU(),
            nn.Linear(features[0], features[0])
        )

    def forward(self, x, t, conditioning):
        """
        x: Noisy 7T input [B, C_in, D, H, W]
        t: Time step [B, 1]
        conditioning: 3T input [B, C_cond, D, H, W]
        
        Returns:
            {'prediction': ..., 'segmentation': ...}
        """
        # 1. Conditioning via Concatenation
        x_in = torch.cat([x, conditioning], dim=1)
        
        # 2. Time Embedding
        # Ensure t is float and [B, 1]
        t = t.float().view(-1, 1)
        t_emb = self.time_embed(t).view(-1, self.inc.out_channels, 1, 1, 1)
        
        # 3. Encoder
        x1 = self.inc(x_in)
        x1 = x1 + t_emb # Add time embedding to first layer features
        
        skips = [x1]
        feat = x1
        for down in self.downs:
            feat = down(feat)
            skips.append(feat)
            
        # 4. Bottleneck (Shared Features)
        feat = self.bottleneck(feat)
        
        # 5a. Denoising Decoder
        denoise_feat = feat
        denoise_skips = reversed(skips[:-1])
        for up, skip in zip(self.ups, denoise_skips):
            denoise_feat = up(denoise_feat)
            denoise_feat = denoise_feat + skip 
            
        out_noise = self.outc(denoise_feat)
        
        # 5b. Segmentation Decoder (Multi-Task)
        # Shares the bottleneck features 'feat' and skip connections 'skips'
        # To avoid interference, we clone feat or use separate path if we wanted
        # discrete weights. Here we share 'feat' but have separate weights for decoder.
        seg_feat = feat
        seg_skips = reversed(skips[:-1])
        for up, skip in zip(self.seg_ups, seg_skips):
            seg_feat = up(seg_feat)
            seg_feat = seg_feat + skip
            
        out_seg = self.seg_outc(seg_feat)
        
        return {
            "prediction": out_noise, 
            "segmentation": out_seg
        }

if __name__ == "__main__":
    # Smoke Test
    model = AnatomyGuidedUNet(in_channels=1, cond_channels=1, out_channels=1, num_classes=5)
    x = torch.randn(2, 1, 64, 64, 64)
    cond = torch.randn(2, 1, 64, 64, 64) # 3T
    t = torch.randn(2, 1) # Timesteps
    
    out = model(x, t, cond)
    print(f"Noise Output: {out['prediction'].shape}")
    print(f"Seg Output: {out['segmentation'].shape}")
    
    assert out['prediction'].shape == x.shape
    assert out['segmentation'].shape == (2, 5, 64, 64, 64)
    print("Test Passed!")
