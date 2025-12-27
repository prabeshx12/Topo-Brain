
import unittest
import torch
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Import models
from models.generator_unet3d import UNet3DGenerator
from models.discriminator_patchgan3d import PatchGANDiscriminator3D

class TestGANModels(unittest.TestCase):
    """
    Unit tests for 3D GAN models.
    """
    
    def setUp(self):
        # Common parameters
        self.batch_size = 2
        self.channels = 1
        self.patch_size = 64
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_generator_structure(self):
        """Test generator instantiation."""
        model = UNet3DGenerator(
            in_channels=self.channels,
            out_channels=self.channels,
            base_features=16, # Use smaller features for test speed
            num_levels=3
        ).to(self.device)
        
        self.assertIsInstance(model, UNet3DGenerator)
        
    def test_generator_forward(self):
        """Test generator forward pass (shape preservation)."""
        model = UNet3DGenerator(
            in_channels=self.channels,
            out_channels=self.channels,
            base_features=16,
            num_levels=3
        ).to(self.device)
        model.eval()
        
        input_tensor = torch.randn(
            self.batch_size, self.channels, 
            self.patch_size, self.patch_size, self.patch_size
        ).to(self.device)
        
        with torch.no_grad():
            output_tensor = model(input_tensor)
            
        # Check output shape matches input shape (B, C, D, H, W)
        self.assertEqual(output_tensor.shape, input_tensor.shape)
        
    def test_discriminator_forward(self):
        """Test discriminator forward pass (downsampling)."""
        model = PatchGANDiscriminator3D(
            in_channels=self.channels,
            base_features=16,
            num_layers=3
        ).to(self.device)
        model.eval()
        
        input_tensor = torch.randn(
            self.batch_size, self.channels, 
            self.patch_size, self.patch_size, self.patch_size
        ).to(self.device)
        
        with torch.no_grad():
            output_tensor = model(input_tensor)
            
        # Discriminator should downsample spatially
        # Shape: (B, 1, D', H', W')
        self.assertEqual(output_tensor.shape[0], self.batch_size)
        self.assertEqual(output_tensor.shape[1], 1)
        
        # Verify spatial dimensions are reduced
        self.assertLess(output_tensor.shape[2], self.patch_size)
        self.assertLess(output_tensor.shape[3], self.patch_size)
        self.assertLess(output_tensor.shape[4], self.patch_size)

if __name__ == '__main__':
    print(f"Running tests on device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    unittest.main()
