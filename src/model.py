"""
Model Architecture for Topo-Brain: Anatomy-Guided Conditional Diffusion Model.
Upgraded with Sinusoidal Time Embeddings and Concatenated Skip Connections.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        # Use GroupNorm instead of InstanceNorm for better stability at batch size 2
        num_groups = min(groups, out_ch)
        if out_ch % num_groups != 0: num_groups = 1
        
        self.proj = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        ) if time_emb_dim is not None else None

        self.block1 = Block(in_ch, out_ch, groups=groups)
        self.block2 = Block(out_ch, out_ch, groups=groups)
        self.res_conv = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            # Add time embedding to spatial features
            h = h + time_emb.view(time_emb.shape[0], time_emb.shape[1], 1, 1, 1)

        h = self.block2(h)
        return h + self.res_conv(x)

class SelfAttention3D(nn.Module):
    """
    3D Self-Attention module for global context.
    As specified in Blueprint Section 4: 'Global Context Module'.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        
        self.qkv = nn.Conv3d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv3d(channels, channels, 1)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        b, c, d, h, w = x.shape
        qkv = self.qkv(self.norm(x)).view(b, 3, self.num_heads, c // self.num_heads, -1)
        q, k, v = qkv.unbind(1) # [b, heads, head_dim, length]
        
        # Attention: (length, length) map
        attn = (q.transpose(-2, -1) @ k) * self.scale # [b, heads, length, length]
        attn = attn.softmax(dim=-1)
        
        out = (v @ attn.transpose(-2, -1)).view(b, c, d, h, w)
        return x + self.proj(out)

class AnatomyGuidedUNet(nn.Module):
    """
    3D U-Net backbone for Diffusion Model.
    Conditioned on 3T input via concatenation.
    Injects time embeddings into every block.
    Uses concatenation for skip connections.
    """
    def __init__(
        self,
        in_channels: int = 1,
        cond_channels: int = 1,
        out_channels: int = 1,
        num_classes: int = 3,
        features: tuple = (32, 64, 128, 256),
        use_attention: bool = False, # Safe default to avoid breaking old checkpoints
    ):
        super().__init__()
        self.use_attention = use_attention
        
        # Time embedding
        dim = features[0]
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        total_in_channels = in_channels + cond_channels
        
        # Initial convolution
        self.inc = nn.Conv3d(total_in_channels, features[0], kernel_size=3, padding=1)
        
        # Encoder (Downsampling)
        self.downs = nn.ModuleList()
        in_ch = features[0]
        for feat in features[1:]:
            self.downs.append(nn.ModuleList([
                ResnetBlock(in_ch, in_ch, time_emb_dim=dim),
                nn.Conv3d(in_ch, feat, 3, stride=2, padding=1) # Downsample
            ]))
            in_ch = feat
            
        # Bottleneck
        mid_dim = features[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = SelfAttention3D(mid_dim) if use_attention else nn.Identity()
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        
        # Decoder (Upsampling)
        self.ups = nn.ModuleList()
        rev_features = list(reversed(features))
        for i in range(len(rev_features) - 1):
            curr_feat = rev_features[i]
            next_feat = rev_features[i+1]
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose3d(curr_feat, next_feat, 2, stride=2), # Upsample
                # Skip connection uses concatenation: next_feat (from up) + next_feat (from skip)
                ResnetBlock(next_feat * 2, next_feat, time_emb_dim=dim)
            ]))
            
        # Final prediction head
        self.outc = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
        # Segmentation head (Multi-task)
        self.seg_ups = nn.ModuleList()
        for i in range(len(rev_features) - 1):
            curr_feat = rev_features[i]
            next_feat = rev_features[i+1]
            self.seg_ups.append(nn.ModuleList([
                nn.ConvTranspose3d(curr_feat, next_feat, 2, stride=2),
                ResnetBlock(next_feat * 2, next_feat, time_emb_dim=dim)
            ]))
        self.seg_outc = nn.Conv3d(features[0], num_classes, kernel_size=1)

    def forward(self, x, t, conditioning):
        # 1. Conditioning via Concatenation
        x = torch.cat([x, conditioning], dim=1)
        
        # 2. Time Embedding
        t = self.time_mlp(t.view(-1))
        
        # 3. Encoder
        x = self.inc(x)
        h = [x]
        
        for res, down in self.downs:
            x = res(x, t)
            x = down(x)
            h.append(x)
            
        # 4. Bottleneck
        x = self.mid_block1(x, t)
        if self.use_attention:
            x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        # 5a. Denoising Decoder
        denoise_x = x
        for i, (up, res) in enumerate(self.ups):
            denoise_x = up(denoise_x)
            # Concatenate skip connection
            denoise_x = torch.cat([denoise_x, h.pop()], dim=1)
            denoise_x = res(denoise_x, t)
            
        out_noise = self.outc(denoise_x)
        
        # 5b. Segmentation Decoder
        # Since we popped from h, we need to handle x/h for segmentation too.
        # However, h is consumed. Let's re-save or use a different approach.
        # Re-populating for multi-task:
        # Actually, let's just use a fresh list for the shared features if we want to be safe.
        
        # For simplicity in this version, if we need full segmentation, we should re-forward or store.
        # But wait, 'h' is just a list of tensors. We can just keep it or slice it.
        # Let's fix the encoder loop to not consume h.
        
        return {
            "prediction": out_noise,
            "segmentation": torch.zeros((x.shape[0], 3, *x.shape[2:]), device=x.device) # Placeholder for now
        }

    # Redefine forward to be more robust for multi-task if needed
    def forward_full(self, x, t, conditioning):
        x_in = torch.cat([x, conditioning], dim=1)
        t_emb = self.time_mlp(t.view(-1))
        
        feat = self.inc(x_in)
        skips = [feat]
        
        for res, down in self.downs:
            feat = res(feat, t_emb)
            feat = down(feat)
            skips.append(feat)
            
        feat = self.mid_block1(feat, t_emb)
        if self.use_attention:
            feat = self.mid_attn(feat)
        feat = self.mid_block2(feat, t_emb)
        
        # Denoising path
        d_feat = feat
        for i, (up, res) in enumerate(self.ups):
            d_feat = up(d_feat)
            d_feat = torch.cat([d_feat, skips[-(i+2)]], dim=1)
            d_feat = res(d_feat, t_emb)
        out_noise = self.outc(d_feat)
        
        # Segmentation path
        s_feat = feat
        for i, (up, res) in enumerate(self.seg_ups):
            s_feat = up(s_feat)
            s_feat = torch.cat([s_feat, skips[-(i+2)]], dim=1)
            s_feat = res(s_feat, t_emb)
        out_seg = self.seg_outc(s_feat)
        
        return {"prediction": out_noise, "segmentation": out_seg}
    
    # Overwrite forward with forward_full
    forward = forward_full

if __name__ == "__main__":
    model = AnatomyGuidedUNet(in_channels=1, cond_channels=1, out_channels=1, num_classes=5)
    x = torch.randn(2, 1, 64, 64, 64)
    cond = torch.randn(2, 1, 64, 64, 64)
    t = torch.randint(0, 1000, (2,))
    
    out = model(x, t, cond)
    print(f"Noise Output: {out['prediction'].shape}")
    print(f"Seg Output: {out['segmentation'].shape}")
    assert out['prediction'].shape == x.shape
    assert out['segmentation'].shape == (2, 5, 64, 64, 64)
    print("Test Passed!")
