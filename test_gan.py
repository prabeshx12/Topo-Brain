"""
Quick test script to verify GAN models work correctly.

Tests:
1. Generator architecture
2. Discriminator architecture
3. Forward pass
4. Loss computation
5. Backward pass
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add models to path
sys.path.append(str(Path(__file__).parent))

from models.generator_unet3d import UNet3DGenerator
from models.discriminator_patchgan3d import PatchGANDiscriminator3D


def test_models():
    """Test all GAN components."""
    print("="*80)
    print("Testing 3T→7T GAN Implementation")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Configuration
    batch_size = 2
    patch_size = 64
    
    print(f"\nTest Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Patch size: {patch_size}³")
    
    # ===== Test Generator =====
    print("\n" + "-"*80)
    print("1. Testing Generator (3D U-Net)")
    print("-"*80)
    
    generator = UNet3DGenerator(
        in_channels=1,
        out_channels=1,
        base_features=32,
        num_levels=4,
        norm_type="instance",
    ).to(device)
    
    print(f"✓ Generator created")
    print(f"  Parameters: {generator.get_num_parameters():,}")
    
    # Test forward pass
    input_3t = torch.randn(batch_size, 1, patch_size, patch_size, patch_size).to(device)
    print(f"✓ Input created: {input_3t.shape}")
    
    with torch.no_grad():
        output_7t = generator(input_3t)
    
    print(f"✓ Forward pass successful: {output_7t.shape}")
    assert output_7t.shape == input_3t.shape, "Output shape mismatch!"
    print(f"✓ Output shape matches input")
    
    # ===== Test Discriminator =====
    print("\n" + "-"*80)
    print("2. Testing Discriminator (PatchGAN)")
    print("-"*80)
    
    discriminator = PatchGANDiscriminator3D(
        in_channels=1,
        base_features=64,
        num_layers=3,
        norm_type="instance",
    ).to(device)
    
    print(f"✓ Discriminator created")
    print(f"  Parameters: {discriminator.get_num_parameters():,}")
    
    # Test forward pass
    target_7t = torch.randn(batch_size, 1, patch_size, patch_size, patch_size).to(device)
    
    with torch.no_grad():
        pred_real = discriminator(target_7t)
        pred_fake = discriminator(output_7t)
    
    print(f"✓ Forward pass successful")
    print(f"  Real predictions: {pred_real.shape}")
    print(f"  Fake predictions: {pred_fake.shape}")
    assert pred_real.shape == pred_fake.shape, "Prediction shapes mismatch!"
    print(f"✓ Prediction shapes match")
    
    # ===== Test Loss Functions =====
    print("\n" + "-"*80)
    print("3. Testing Loss Functions")
    print("-"*80)
    
    # L1 loss
    criterion_l1 = nn.L1Loss()
    l1_loss = criterion_l1(output_7t, target_7t)
    print(f"✓ L1 loss: {l1_loss.item():.6f}")
    
    # Adversarial loss (LSGAN)
    criterion_adv = nn.MSELoss()
    label_real = torch.ones_like(pred_real)
    label_fake = torch.zeros_like(pred_fake)
    
    adv_loss_real = criterion_adv(pred_real, label_real)
    adv_loss_fake = criterion_adv(pred_fake, label_fake)
    print(f"✓ Adversarial loss (real): {adv_loss_real.item():.6f}")
    print(f"✓ Adversarial loss (fake): {adv_loss_fake.item():.6f}")
    
    # ===== Test Backward Pass =====
    print("\n" + "-"*80)
    print("4. Testing Backward Pass")
    print("-"*80)
    
    # Generator backward
    generator.train()
    discriminator.eval()
    
    input_3t = torch.randn(batch_size, 1, patch_size, patch_size, patch_size).to(device)
    target_7t = torch.randn(batch_size, 1, patch_size, patch_size, patch_size).to(device)
    
    # Forward
    fake_7t = generator(input_3t)
    
    # Generator loss
    loss_g_l1 = criterion_l1(fake_7t, target_7t)
    pred_fake = discriminator(fake_7t)
    loss_g_adv = criterion_adv(pred_fake, torch.ones_like(pred_fake))
    loss_g = 100.0 * loss_g_l1 + 1.0 * loss_g_adv
    
    # Backward
    loss_g.backward()
    
    # Check gradients exist
    has_grads = any(p.grad is not None for p in generator.parameters())
    print(f"✓ Generator backward pass successful")
    print(f"  Gradients computed: {has_grads}")
    
    # Discriminator backward
    discriminator.train()
    
    with torch.no_grad():
        fake_7t = generator(input_3t)
    
    pred_real = discriminator(target_7t)
    pred_fake = discriminator(fake_7t)
    
    loss_d_real = criterion_adv(pred_real, torch.ones_like(pred_real))
    loss_d_fake = criterion_adv(pred_fake, torch.zeros_like(pred_fake))
    loss_d = (loss_d_real + loss_d_fake) * 0.5
    
    loss_d.backward()
    
    has_grads = any(p.grad is not None for p in discriminator.parameters())
    print(f"✓ Discriminator backward pass successful")
    print(f"  Gradients computed: {has_grads}")
    
    # ===== Test Optimizer Step =====
    print("\n" + "-"*80)
    print("5. Testing Optimizer")
    print("-"*80)
    
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    # Get initial parameter value
    initial_param_g = next(generator.parameters()).clone()
    initial_param_d = next(discriminator.parameters()).clone()
    
    # Training step
    optimizer_g.zero_grad()
    fake_7t = generator(input_3t)
    loss_g = criterion_l1(fake_7t, target_7t)
    loss_g.backward()
    optimizer_g.step()
    
    optimizer_d.zero_grad()
    pred_real = discriminator(target_7t)
    loss_d = criterion_adv(pred_real, torch.ones_like(pred_real))
    loss_d.backward()
    optimizer_d.step()
    
    # Check parameters changed
    param_changed_g = not torch.equal(initial_param_g, next(generator.parameters()))
    param_changed_d = not torch.equal(initial_param_d, next(discriminator.parameters()))
    
    print(f"✓ Optimizer step successful")
    print(f"  Generator parameters updated: {param_changed_g}")
    print(f"  Discriminator parameters updated: {param_changed_d}")
    
    # ===== Summary =====
    print("\n" + "="*80)
    print("All Tests Passed! ✓")
    print("="*80)
    print("\nModel Summary:")
    print(f"  Generator parameters: {generator.get_num_parameters():,}")
    print(f"  Discriminator parameters: {discriminator.get_num_parameters():,}")
    print(f"  Total parameters: {generator.get_num_parameters() + discriminator.get_num_parameters():,}")
    print("\nMemory Usage:")
    if torch.cuda.is_available():
        print(f"  GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"  GPU memory reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    else:
        print(f"  CPU mode (no GPU detected)")
    
    print("\n" + "="*80)
    print("Ready for training!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Ensure preprocessing is done: python example_pipeline.py --preprocess")
    print("  2. Start training: python train_gan.py")
    print("  3. Monitor in checkpoints/ and visualizations/")


if __name__ == "__main__":
    try:
        test_models()
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"Test Failed: {e}")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
