"""
3D U-Net Generator for 3T → 7T MRI super-resolution.

Architecture:
- Encoder-decoder with skip connections
- 3D convolutions throughout
- InstanceNorm for stability (better than BatchNorm for small batches)
- LeakyReLU activations
- Output same spatial size as input
"""
import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """
    3D Convolutional block with normalization and activation.
    
    Conv3D -> InstanceNorm3D -> LeakyReLU
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_norm: bool = True,
        norm_type: str = "instance",  # "instance" or "group"
        num_groups: int = 8,
    ):
        super().__init__()
        
        layers = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_norm,  # No bias if using normalization
            )
        ]
        
        if use_norm:
            if norm_type == "instance":
                layers.append(nn.InstanceNorm3d(out_channels, affine=True))
            elif norm_type == "group":
                layers.append(nn.GroupNorm(num_groups, out_channels))
            else:
                raise ValueError(f"Unknown norm_type: {norm_type}")
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class DownBlock3D(nn.Module):
    """
    Downsampling block for encoder.
    
    Two conv blocks followed by downsampling (stride-2 conv).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "instance",
        num_groups: int = 8,
    ):
        super().__init__()
        
        self.conv1 = ConvBlock3D(
            in_channels, out_channels,
            norm_type=norm_type, num_groups=num_groups
        )
        self.conv2 = ConvBlock3D(
            out_channels, out_channels,
            norm_type=norm_type, num_groups=num_groups
        )
        self.downsample = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, stride=2, padding=1
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x  # Save for skip connection
        x = self.downsample(x)
        return x, skip


class UpBlock3D(nn.Module):
    """
    Upsampling block for decoder with skip connections.
    
    Upsampling -> Concatenate with skip -> Two conv blocks
    """
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm_type: str = "instance",
        num_groups: int = 8,
    ):
        super().__init__()
        
        # Transposed convolution for upsampling
        self.upsample = nn.ConvTranspose3d(
            in_channels, in_channels,
            kernel_size=3, stride=2, padding=1, output_padding=1
        )
        
        # After concatenation: in_channels + skip_channels
        self.conv1 = ConvBlock3D(
            in_channels + skip_channels, out_channels,
            norm_type=norm_type, num_groups=num_groups
        )
        self.conv2 = ConvBlock3D(
            out_channels, out_channels,
            norm_type=norm_type, num_groups=num_groups
        )
    
    def forward(self, x, skip):
        x = self.upsample(x)
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet3DGenerator(nn.Module):
    """
    3D U-Net Generator for 3T → 7T MRI enhancement.
    
    Architecture:
        Input: (B, 1, D, H, W) - 3T MRI patch
        Output: (B, 1, D, H, W) - Generated 7T MRI patch
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale MRI)
        out_channels: Number of output channels (default: 1 for grayscale MRI)
        base_features: Base number of features (doubled at each level)
        num_levels: Number of encoder/decoder levels
        norm_type: Normalization type ("instance" or "group")
        num_groups: Number of groups for GroupNorm
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 32,
        num_levels: int = 4,
        norm_type: str = "instance",
        num_groups: int = 8,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_features = base_features
        self.num_levels = num_levels
        
        # Initial convolution
        self.init_conv = ConvBlock3D(
            in_channels, base_features,
            norm_type=norm_type, num_groups=num_groups
        )
        
        # Build encoder feature sizes list
        encoder_channels = [base_features]
        for i in range(num_levels):
            encoder_channels.append(encoder_channels[-1] * 2)
        
        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        
        for i in range(num_levels):
            self.encoder_blocks.append(
                DownBlock3D(
                    encoder_channels[i], encoder_channels[i+1],
                    norm_type=norm_type, num_groups=num_groups
                )
            )
        
        # Bottleneck
        bottleneck_channels = encoder_channels[-1]
        self.bottleneck = nn.Sequential(
            ConvBlock3D(
                bottleneck_channels, bottleneck_channels,
                norm_type=norm_type, num_groups=num_groups
            ),
            ConvBlock3D(
                bottleneck_channels, bottleneck_channels,
                norm_type=norm_type, num_groups=num_groups
            ),
        )
        
        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        decoder_in_channels = bottleneck_channels
        
        for i in range(num_levels):
            # After upsampling, we concat with skip from same spatial level
            # Skip from encoder j has encoder_channels[j+1] features
            # Decoder 0 needs skip from encoder 3 → encoder_channels[4]
            # Decoder 1 needs skip from encoder 2 → encoder_channels[3]
            # Decoder i needs skip from encoder (num_levels-1-i) → encoder_channels[num_levels-i]
            skip_channels = encoder_channels[num_levels - i]
            decoder_out_channels = decoder_in_channels // 2
            
            self.decoder_blocks.append(
                UpBlock3D(
                    decoder_in_channels, skip_channels, decoder_out_channels,
                    norm_type=norm_type, num_groups=num_groups
                )
            )
            decoder_in_channels = decoder_out_channels
        
        # Final output convolution (no activation - we want full intensity range)
        self.output_conv = nn.Conv3d(
            decoder_in_channels, out_channels,
            kernel_size=1, stride=1, padding=0
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input 3T MRI patch (B, 1, D, H, W)
        
        Returns:
            Generated 7T MRI patch (B, 1, D, H, W)
        """
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder with skip connections
        skips = []
        for encoder_block in self.encoder_blocks:
            x, skip = encoder_block(x)
            skips.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections (in reverse order)
        for decoder_block, skip in zip(self.decoder_blocks, reversed(skips)):
            x = decoder_block(x, skip)
        
        # Output
        x = self.output_conv(x)
        
        return x
    
    def get_num_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_generator():
    """Test the generator architecture."""
    print("Testing UNet3DGenerator...")
    
    # Create model
    model = UNet3DGenerator(
        in_channels=1,
        out_channels=1,
        base_features=32,
        num_levels=4,
    )
    
    print(f"Total parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    batch_size = 2
    patch_size = 64
    
    x = torch.randn(batch_size, 1, patch_size, patch_size, patch_size)
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        y = model(x)
    
    print(f"Output shape: {y.shape}")
    assert y.shape == x.shape, "Output shape must match input shape!"
    
    print("✓ Generator test passed!")


if __name__ == "__main__":
    test_generator()
