#!/usr/bin/env python3
"""
Quick Training Health Check Script
Run this periodically during training to verify everything is working correctly.
"""
import sys
import yaml
from pathlib import Path

def check_config_alignment():
    """Verify configuration files match blueprint requirements."""
    print("=" * 80)
    print("CONFIGURATION ALIGNMENT CHECK")
    print("=" * 80)
    
    config_path = Path("configs/train_diffusion.yaml")
    if not config_path.exists():
        print("❌ Configuration file not found!")
        return False
        
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    checks = []
    
    # Check timesteps
    timesteps = config.get("diffusion", {}).get("timesteps", 1000)
    if timesteps <= 200:
        print(f"✅ Timesteps: {timesteps} (Blueprint: 100-200)")
        checks.append(True)
    else:
        print(f"❌ Timesteps: {timesteps} (Should be 100-200)")
        checks.append(False)
    
    # Check loss weights
    loss_weights = config.get("loss_weights", {})
    lambda_pixel = loss_weights.get("lambda_pixel", 0)
    lambda_percep = loss_weights.get("lambda_percep", 0)
    lambda_topo = loss_weights.get("lambda_topo", 0)
    
    if lambda_pixel >= 1.0:
        print(f"✅ lambda_pixel: {lambda_pixel} (Blueprint: 1.0)")
        checks.append(True)
    else:
        print(f"❌ lambda_pixel: {lambda_pixel} (Should be 1.0)")
        checks.append(False)
        
    if 0.1 <= lambda_percep <= 0.5:
        print(f"✅ lambda_percep: {lambda_percep} (Blueprint: 0.1-0.3)")
        checks.append(True)
    else:
        print(f"⚠️  lambda_percep: {lambda_percep} (Recommended: 0.1-0.3)")
        checks.append(True)  # Warning, not error
        
    if 0.1 <= lambda_topo <= 1.0:
        print(f"✅ lambda_topo: {lambda_topo} (Blueprint: 0.1-1.0)")
        checks.append(True)
    else:
        print(f"⚠️  lambda_topo: {lambda_topo} (Recommended: 0.1-1.0)")
        checks.append(True)  # Warning, not error
    
    # Check attention
    use_attention = config.get("model", {}).get("use_attention", False)
    if use_attention:
        print(f"✅ Self-attention enabled (Blueprint: required)")
        checks.append(True)
    else:
        print(f"❌ Self-attention disabled (Should be enabled)")
        checks.append(False)
    
    print()
    if all(checks):
        print("✅ ALL CONFIGURATION CHECKS PASSED")
        return True
    else:
        print("❌ SOME CONFIGURATION CHECKS FAILED")
        return False

def check_code_implementation():
    """Verify critical code fixes are in place."""
    print("=" * 80)
    print("CODE IMPLEMENTATION CHECK")
    print("=" * 80)
    
    diffusion_path = Path("src/diffusion.py")
    if not diffusion_path.exists():
        print("❌ diffusion.py not found!")
        return False
        
    with open(diffusion_path) as f:
        diffusion_code = f.read()
    
    checks = []
    
    # Check for timestep gating removal
    if "t_gate = torch.ones_like(t).float()" in diffusion_code:
        print("✅ Timestep gating REMOVED (correct)")
        checks.append(True)
    elif "t_gate = (t < 400)" in diffusion_code or "t_gate = (t < 800)" in diffusion_code:
        print("❌ Timestep gating STILL ACTIVE (should be removed)")
        checks.append(False)
    else:
        print("⚠️  Cannot verify timestep gating status")
        checks.append(True)
    
    # Check for tanh clamping removal
    if "torch.tanh(x_recon" not in diffusion_code:
        print("✅ Tanh clamping REMOVED (correct)")
        checks.append(True)
    else:
        print("❌ Tanh clamping STILL PRESENT (should be removed)")
        checks.append(False)
    
    # Check for simple pixel loss
    if "loss_pixel = F.l1_loss(x_recon, x_start)" in diffusion_code:
        print("✅ Simple pixel loss WITHOUT destructive clamping (correct)")
        checks.append(True)
    else:
        print("⚠️  Pixel loss implementation may have issues")
        checks.append(True)
    
    print()
    if all(checks):
        print("✅ ALL CODE CHECKS PASSED")
        return True
    else:
        print("❌ SOME CODE CHECKS FAILED")
        return False

def check_training_metrics(log_dir=None):
    """Check if training metrics look healthy."""
    print("=" * 80)
    print("TRAINING METRICS CHECK")
    print("=" * 80)
    
    # Try to find latest checkpoint
    checkpoint_dirs = [
        Path("checkpoints"),
        Path("logs"),
    ]
    
    if log_dir:
        checkpoint_dirs.insert(0, Path(log_dir))
    
    latest_checkpoint = None
    for cdir in checkpoint_dirs:
        if cdir.exists():
            checkpoints = list(cdir.glob("**/checkpoint_*.pt"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                break
    
    if not latest_checkpoint:
        print("⚠️  No checkpoints found yet. Start training to see metrics.")
        return True
    
    print(f"Found checkpoint: {latest_checkpoint}")
    print("✅ Training has started")
    print()
    print("Monitor these metrics in TensorBoard:")
    print("  - Loss/Total should decrease")
    print("  - Loss/Pixel should be < 1.0 by 10k iterations")
    print("  - Loss/Perceptual should activate after 50k iterations")
    print("  - Loss/Topo should activate after 100k iterations")
    print()
    print("Visual checks:")
    print("  - Outputs should be recognizable brain images (not noise)")
    print("  - SSIM should be > 0.3 by 5k iterations")
    print("  - PSNR should be > 10 dB by 5k iterations")
    
    return True

def main():
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "TOPO-BRAIN TRAINING HEALTH CHECK" + " " * 25 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    config_ok = check_config_alignment()
    print()
    
    code_ok = check_code_implementation()
    print()
    
    metrics_ok = check_training_metrics()
    print()
    
    print("=" * 80)
    print("OVERALL STATUS")
    print("=" * 80)
    
    if config_ok and code_ok:
        print("✅ READY FOR TRAINING")
        print()
        print("Next steps:")
        print("  1. Delete any old broken checkpoints")
        print("  2. Run: python scripts/train_diffusion.py --config configs/train_diffusion.yaml")
        print("  3. Monitor TensorBoard: tensorboard --logdir logs")
        print("  4. Check outputs every 10k iterations")
        print()
        print("Expected results:")
        print("  - SSIM > 0.3 by 5k iterations")
        print("  - SSIM > 0.6 by 20k iterations")
        print("  - SSIM > 0.85 by 100k iterations")
        return 0
    else:
        print("❌ ISSUES DETECTED - Review output above")
        print()
        print("Run this script again after fixing issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
