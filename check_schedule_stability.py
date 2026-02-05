#!/usr/bin/env python3
"""
Check Diffusion Schedule Numerical Stability
Verify that the cosine schedule with 200 timesteps produces stable coefficients.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule from the config."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)

def check_schedule_stability(timesteps=200):
    """Check if the schedule produces numerically stable coefficients."""
    print("=" * 80)
    print(f"DIFFUSION SCHEDULE STABILITY CHECK (T={timesteps})")
    print("=" * 80)
    
    betas = cosine_beta_schedule(timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    
    sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)
    
    print(f"\nBeta schedule statistics:")
    print(f"  Min beta: {betas.min().item():.6f}")
    print(f"  Max beta: {betas.max().item():.6f}")
    print(f"  Mean beta: {betas.mean().item():.6f}")
    
    print(f"\nAlpha_cumprod statistics:")
    print(f"  Min: {alphas_cumprod.min().item():.6f}")
    print(f"  Max: {alphas_cumprod.max().item():.6f}")
    print(f"  At t=0: {alphas_cumprod[0].item():.6f}")
    print(f"  At t={timesteps-1}: {alphas_cumprod[-1].item():.6f}")
    
    print(f"\nsqrt_recip_alphas_cumprod (multiplier for x_t):")
    print(f"  Min: {sqrt_recip_alphas_cumprod.min().item():.4f}")
    print(f"  Max: {sqrt_recip_alphas_cumprod.max().item():.4f}")
    print(f"  At t=0: {sqrt_recip_alphas_cumprod[0].item():.4f}")
    print(f"  At t={timesteps-1}: {sqrt_recip_alphas_cumprod[-1].item():.4f}")
    
    print(f"\nsqrt_recipm1_alphas_cumprod (multiplier for noise):")
    print(f"  Min: {sqrt_recipm1_alphas_cumprod.min().item():.4f}")
    print(f"  Max: {sqrt_recipm1_alphas_cumprod.max().item():.4f}")
    print(f"  At t=0: {sqrt_recipm1_alphas_cumprod[0].item():.4f}")
    print(f"  At t={timesteps-1}: {sqrt_recipm1_alphas_cumprod[-1].item():.4f}")
    
    # Check for dangerous timesteps
    print(f"\nDangerous timestep detection:")
    dangerous_recip = sqrt_recip_alphas_cumprod > 10.0
    dangerous_recipm1 = sqrt_recipm1_alphas_cumprod > 10.0
    
    if dangerous_recip.any():
        dangerous_t = torch.where(dangerous_recip)[0]
        print(f"  ⚠️  sqrt_recip > 10 at timesteps: {dangerous_t.tolist()}")
    else:
        print(f"  ✅ No dangerous sqrt_recip values")
    
    if dangerous_recipm1.any():
        dangerous_t = torch.where(dangerous_recipm1)[0]
        print(f"  ⚠️  sqrt_recipm1 > 10 at timesteps: {dangerous_t.tolist()}")
    else:
        print(f"  ✅ No dangerous sqrt_recipm1 values")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    t = np.arange(timesteps)
    
    # Plot betas
    axes[0, 0].plot(t, betas.numpy())
    axes[0, 0].set_title('Beta Schedule')
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Beta')
    axes[0, 0].grid(True)
    
    # Plot alphas_cumprod
    axes[0, 1].plot(t, alphas_cumprod.numpy())
    axes[0, 1].set_title('Cumulative Alpha Product')
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('Alpha_cumprod')
    axes[0, 1].grid(True)
    
    # Plot sqrt_recip
    axes[1, 0].plot(t, sqrt_recip_alphas_cumprod.numpy())
    axes[1, 0].axhline(y=10.0, color='r', linestyle='--', label='Danger threshold')
    axes[1, 0].set_title('sqrt(1/alpha_cumprod) - x_t multiplier')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Coefficient')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot sqrt_recipm1
    axes[1, 1].plot(t, sqrt_recipm1_alphas_cumprod.numpy())
    axes[1, 1].axhline(y=10.0, color='r', linestyle='--', label='Danger threshold')
    axes[1, 1].set_title('sqrt(1/alpha_cumprod - 1) - noise multiplier')
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylabel('Coefficient')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    output_path = Path('diffusion_schedule_stability.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")
    
    # Provide recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    max_recip = sqrt_recip_alphas_cumprod.max().item()
    max_recipm1 = sqrt_recipm1_alphas_cumprod.max().item()
    
    if max_recip > 50 or max_recipm1 > 50:
        print("❌ CRITICAL: Coefficients are extremely large (>50)")
        print("   Recommendation: Increase timesteps to 500 or 1000")
        return False
    elif max_recip > 20 or max_recipm1 > 20:
        print("⚠️  WARNING: Coefficients are large (>20)")
        print("   Recommendation: Consider increasing timesteps to 500")
        print("   Current safeguards (clamping) should handle this")
        return True
    elif max_recip > 10 or max_recipm1 > 10:
        print("⚠️  CAUTION: Some coefficients exceed 10")
        print("   Current safeguards (clamping to 10) will activate")
        print("   Training should be stable but may clip some gradients")
        return True
    else:
        print("✅ EXCELLENT: All coefficients are in safe range (<10)")
        print("   No clamping needed, training should be very stable")
        return True

def main():
    print("\n╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "DIFFUSION SCHEDULE NUMERICAL STABILITY" + " " * 24 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Check 200 timesteps (current config)
    stable_200 = check_schedule_stability(200)
    print()
    
    # Also check 1000 for comparison
    print("\n" + "=" * 80)
    print("COMPARISON: 1000 timesteps (original)")
    print("=" * 80)
    stable_1000 = check_schedule_stability(1000)
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    if stable_200:
        print("✅ 200 timesteps schedule is STABLE enough for training")
        print("   Loss spikes are likely from other causes (data, model, etc.)")
    else:
        print("❌ 200 timesteps schedule has NUMERICAL ISSUES")
        print("   Consider switching back to 1000 timesteps")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
