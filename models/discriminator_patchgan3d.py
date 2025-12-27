"""
3D PatchGAN Discriminator for 3T → 7T MRI super-resolution.

Architecture:
- Fully convolutional network
- Operates on local 3D patches
- Outputs spatial map of real/fake predictions
- No pooling - uses strided convolutions for downsampling
"""
import torch
import torch.nn as nn


class DiscriminatorBlock3D(nn.Module):
    """
    Discriminator convolutional block.
    
    Conv3D (stride 2) -> InstanceNorm3D -> LeakyReLU
    
    Note: First block doesn't use normalization (common practice).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_norm: bool = True,
        norm_type: str = "instance",
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
                bias=not use_norm,
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


class PatchGANDiscriminator3D(nn.Module):
    """
    3D PatchGAN Discriminator for MRI.
    
    Classifies whether 3D patches are real (7T) or fake (generated).
    Output is a spatial map of predictions rather than a single scalar.
    
    Architecture:
        Input: (B, 1, D, H, W) - Real or fake 7T MRI patch
        Output: (B, 1, D', H', W') - Real/fake logits map
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale MRI)
        base_features: Base number of features (doubled at each layer)
        num_layers: Number of discriminator layers
        norm_type: Normalization type ("instance" or "group")
        num_groups: Number of groups for GroupNorm
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_features: int = 64,
        num_layers: int = 3,
        norm_type: str = "instance",
        num_groups: int = 8,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_features = base_features
        self.num_layers = num_layers
        
        # Build discriminator layers
        layers = []
        
        # First layer: no normalization
        layers.append(
            DiscriminatorBlock3D(
                in_channels, base_features,
                kernel_size=4, stride=2, padding=1,
                use_norm=False,  # No norm in first layer
            )
        )
        
        # Middle layers: with normalization
        current_features = base_features
        for i in range(1, num_layers):
            next_features = min(current_features * 2, 512)  # Cap at 512
            layers.append(
                DiscriminatorBlock3D(
                    current_features, next_features,
                    kernel_size=4, stride=2, padding=1,
                    use_norm=True,
                    norm_type=norm_type,
                    num_groups=num_groups,
                )
            )
            current_features = next_features
        
        # Penultimate layer: stride 1 to maintain some spatial resolution
        next_features = min(current_features * 2, 512)
        layers.append(
            DiscriminatorBlock3D(
                current_features, next_features,
                kernel_size=4, stride=1, padding=1,
                use_norm=True,
                norm_type=norm_type,
                num_groups=num_groups,
            )
        )
        current_features = next_features
        
        # Final layer: output logits (no normalization, no activation)
        layers.append(
            nn.Conv3d(
                current_features, 1,
                kernel_size=4, stride=1, padding=1,
                bias=True,
            )
        )
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input 7T MRI patch (real or fake) (B, 1, D, H, W)
        
        Returns:
            Logits map (B, 1, D', H', W')
            Each spatial location predicts real/fake for local patch
        """
        return self.model(x)
    
    def get_num_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_discriminator():
    """Test the discriminator architecture."""
    print("Testing PatchGANDiscriminator3D...")
    
    # Create model
    model = PatchGANDiscriminator3D(
        in_channels=1,
        base_features=64,
        num_layers=3,
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
    print(f"Output spatial size: {y.shape[2:]}")
    
    # Output should be smaller than input (due to strided convs)
    assert y.shape[0] == batch_size
    assert y.shape[1] == 1
    assert all(s < patch_size for s in y.shape[2:]), "Output should be downsampled"
    
    print("✓ Discriminator test passed!")


if __name__ == "__main__":
    test_discriminator()
