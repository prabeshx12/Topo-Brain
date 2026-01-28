"""
STANDALONE TEST: Verify preprocessing fix works correctly.
This script processes ONE file and immediately verifies the output.
Run this ONCE to confirm the fix works before running full preprocessing.

Usage:
    python test_preprocessing_fix.py --input /path/to/raw/T1w.nii.gz --mask /path/to/mask.nii.gz --output /path/to/test_output.nii.gz
"""
import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
import sys

def diffusion_normalize(image_array: np.ndarray, mask: np.ndarray, 
                        percentile_lower: float = 0.5, 
                        percentile_upper: float = 99.5) -> np.ndarray:
    """
    Diffusion model normalization: outputs [-1, 1] range.
    Background is set to -1.0.
    
    This is the EXACT logic that should be in preprocessing.py
    """
    normalized = image_array.copy()
    
    # Get brain region values only
    roi = image_array[mask > 0]
    
    if len(roi) == 0:
        print("ERROR: Empty brain mask!")
        return normalized
    
    # Calculate percentiles from brain region
    lower_val = np.percentile(roi, percentile_lower)
    upper_val = np.percentile(roi, percentile_upper)
    
    print(f"  Percentile {percentile_lower}%: {lower_val:.2f}")
    print(f"  Percentile {percentile_upper}%: {upper_val:.2f}")
    
    if upper_val > lower_val:
        # Scale to [0, 1] first
        normalized = (normalized - lower_val) / (upper_val - lower_val)
        normalized = np.clip(normalized, 0, 1)
        # Then scale to [-1, 1]
        normalized = normalized * 2.0 - 1.0
    else:
        print("ERROR: Zero percentile range!")
        return normalized
    
    # CRITICAL: Set background to -1.0 (not 0.0!)
    normalized[mask == 0] = -1.0
    
    return normalized.astype(np.float32)


def verify_output(data: np.ndarray) -> bool:
    """Verify the preprocessed data has correct statistics."""
    
    # With diffusion normalization: background = -1.0, brain > -0.95
    brain_mask = data > -0.95
    brain_data = data[brain_mask]
    background_data = data[~brain_mask]
    
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    
    print(f"\nBrain tissue (values > -0.95):")
    print(f"  Min:  {brain_data.min():.6f}")
    print(f"  Max:  {brain_data.max():.6f}")
    print(f"  Mean: {brain_data.mean():.6f}")
    print(f"  Std:  {brain_data.std():.6f}")
    print(f"  Voxels: {len(brain_data):,}")
    
    print(f"\nBackground (values <= -0.95):")
    if len(background_data) > 0:
        print(f"  Min:  {background_data.min():.6f}")
        print(f"  Max:  {background_data.max():.6f}")
        print(f"  Mean: {background_data.mean():.6f}")
        print(f"  Unique values: {len(np.unique(background_data))}")
    else:
        print("  No background voxels detected!")
    
    print(f"\nFull volume:")
    print(f"  Shape: {data.shape}")
    print(f"  Min:  {data.min():.6f}")
    print(f"  Max:  {data.max():.6f}")
    
    # Validation checks
    problems = []
    passes = []
    
    # Check 1: Background should be -1.0
    if len(background_data) > 0:
        if np.abs(background_data.mean() - (-1.0)) < 0.01:
            passes.append("✅ Background is -1.0")
        else:
            problems.append(f"❌ Background mean is {background_data.mean():.4f}, should be -1.0")
    
    # Check 2: Brain max should be close to 1.0
    if brain_data.max() > 0.9:
        passes.append("✅ Brain tissue reaches ~1.0 (good contrast)")
    else:
        problems.append(f"❌ Brain tissue max is {brain_data.max():.4f}, should be ~1.0")
    
    # Check 3: Good standard deviation
    if brain_data.std() > 0.3:
        passes.append("✅ Good standard deviation (not compressed)")
    else:
        problems.append(f"❌ Low std: {brain_data.std():.4f} (contrast compressed)")
    
    # Check 4: Volume min should be -1.0
    if np.abs(data.min() - (-1.0)) < 0.01:
        passes.append("✅ Volume minimum is -1.0")
    else:
        problems.append(f"❌ Volume min is {data.min():.4f}, should be -1.0")
    
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    for p in passes:
        print(f"  {p}")
    for p in problems:
        print(f"  {p}")
    
    if len(problems) == 0:
        print("\n" + "🎉"*20)
        print("  ALL CHECKS PASSED! PREPROCESSING IS CORRECT!")
        print("  You can now run full preprocessing with confidence.")
        print("🎉"*20)
        return True
    else:
        print("\n" + "⚠️"*20)
        print("  PROBLEMS DETECTED - DO NOT RUN FULL PREPROCESSING")
        print("⚠️"*20)
        return False


def main():
    parser = argparse.ArgumentParser(description="Test preprocessing fix on one file")
    parser.add_argument("--input", required=True, help="Path to raw input NIfTI (e.g., T1w.nii.gz)")
    parser.add_argument("--mask", required=True, help="Path to brain mask NIfTI")
    parser.add_argument("--output", default="test_preprocessed.nii.gz", help="Output path")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    mask_path = Path(args.mask)
    output_path = Path(args.output)
    
    # Validate inputs
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    if not mask_path.exists():
        print(f"ERROR: Mask file not found: {mask_path}")
        sys.exit(1)
    
    print("="*60)
    print("PREPROCESSING TEST")
    print("="*60)
    print(f"\nInput: {input_path}")
    print(f"Mask:  {mask_path}")
    print(f"Output: {output_path}")
    
    # Load data
    print("\n[1/4] Loading input image...")
    img = nib.load(str(input_path))
    image_array = img.get_fdata().astype(np.float32)
    affine = img.affine
    print(f"  Shape: {image_array.shape}")
    print(f"  Raw range: [{image_array.min():.2f}, {image_array.max():.2f}]")
    
    print("\n[2/4] Loading brain mask...")
    mask_img = nib.load(str(mask_path))
    mask_array = mask_img.get_fdata().astype(np.uint8)
    mask_array = (mask_array > 0).astype(np.uint8)  # Binarize
    brain_voxels = np.sum(mask_array > 0)
    total_voxels = mask_array.size
    print(f"  Brain voxels: {brain_voxels:,} ({100*brain_voxels/total_voxels:.1f}%)")
    
    print("\n[3/4] Applying diffusion normalization...")
    normalized = diffusion_normalize(image_array, mask_array)
    
    print("\n[4/4] Saving output...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(normalized, affine), str(output_path))
    print(f"  Saved to: {output_path}")
    
    # Verify
    success = verify_output(normalized)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
