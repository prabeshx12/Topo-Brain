
import torch
import unittest
import sys
import os

# Add src to path to import modules directly, bypassing src/__init__.py
src_path = os.path.join(os.getcwd(), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

import model as model_module
import diffusion as diffusion_module

AnatomyGuidedUNet = model_module.AnatomyGuidedUNet
GaussianDiffusion = diffusion_module.GaussianDiffusion

class TestDiffusion(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = AnatomyGuidedUNet(
            in_channels=1, 
            cond_channels=1, 
            out_channels=1,
            num_classes=3,
            features=(8, 16, 32, 64) # Small model for testing
        ).to(self.device)
        
        self.diffusion = GaussianDiffusion(
            model=self.model,
            timesteps=10, # Few steps for testing
            beta_schedule="linear",
            loss_type="l1"
        ).to(self.device)
        
        self.b, self.c, self.d, self.h, self.w = 2, 1, 32, 32, 32
        
    def test_loss_computation(self):
        """Test if forward pass computes all weighted losses correctly."""
        x_start = torch.randn(self.b, self.c, self.d, self.h, self.w).to(self.device)
        conditioning = torch.randn(self.b, self.c, self.d, self.h, self.w).to(self.device)
        seg_target = torch.randint(0, 3, (self.b, self.d, self.h, self.w)).to(self.device) # Class labels
        
        # Enable all losses
        losses = self.diffusion(
            x_start=x_start,
            conditioning=conditioning,
            seg_target=seg_target,
            lambda_pixel=1.0,
            lambda_percep=0.1,
            lambda_topo=0.1
        )
        
        print("Losses:", {k: v.item() for k, v in losses.items()})
        
        self.assertIn("loss", losses)
        self.assertIn("loss_diff", losses)
        self.assertIn("loss_pixel", losses)
        self.assertIn("loss_vgg", losses)
        self.assertIn("loss_topo", losses)
        
        # Check total loss formula approx
        # loss = diff + 1.0*pixel + 0.1*vgg + 0.1*topo
        expected = losses["loss_diff"] + 1.0*losses["loss_pixel"] + 0.1*losses["loss_vgg"] + 0.1*losses["loss_topo"]
        self.assertTrue(torch.isclose(losses["loss"], expected, atol=1e-5))
        
    def test_sampling_loop(self):
        """Test if sampling loop generates correct output shape."""
        conditioning = torch.randn(self.b, self.c, self.d, self.h, self.w).to(self.device)
        
        with torch.no_grad():
            sampled = self.diffusion.p_sample_loop(
                conditioning=conditioning,
                shape=(self.b, self.c, self.d, self.h, self.w)
            )
            
        print(f"Sampled shape: {sampled.shape}")
        self.assertEqual(sampled.shape, (self.b, self.c, self.d, self.h, self.w))

if __name__ == "__main__":
    unittest.main()
