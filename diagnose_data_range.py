#!/usr/bin/env python3
"""
Emergency Data Range Diagnostic
Check what ranges the model is predicting vs what the data actually contains.
"""
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
import sys

def check_preprocessed_data_ranges():
    """Check a few preprocessed files to verify their intensity ranges."""
    print("=" * 80)
    print("PREPROCESSED DATA RANGE CHECK")
    print("=" * 80)
    
    # Try to find some preprocessed files
    pairs_csv = Path("/eos/home-i04/p/ppokhrel/Untitled Folder 1/tissue_masks/pairs_with_tissue_masks.csv")
    
    if not pairs_csv.exists():
        print(f"❌ Pairs file not found: {pairs_csv}")
        return False
    
    # Read first few pairs
    import pandas as pd
    df = pd.read_csv(pairs_csv)
    
    print(f"Found {len(df)} pairs in dataset")
    print()
    
    # Check first 3 files
    for idx in range(min(3, len(df))):
        row = df.iloc[idx]
        
        # Check 3T input
        input_path = Path(row['input_3t'])
        if input_path.exists():
            img = nib.load(str(input_path))
            data = img.get_fdata()
            
            print(f"File {idx+1}: {input_path.name}")
            print(f"  3T Input:")
            print(f"    Shape: {data.shape}")
            print(f"    Min: {data.min():.4f}")
            print(f"    Max: {data.max():.4f}")
            print(f"    Mean: {data.mean():.4f}")
            print(f"    Std: {data.std():.4f}")
            print(f"    Non-zero values: {(data != 0).sum()}")
            print(f"    Background value: {data[data == data.min()][0] if len(data[data == data.min()]) > 0 else 'N/A'}")
            
            # Check if in [-1, 1] range
            if data.min() >= -1.1 and data.max() <= 1.1:
                print(f"    ✅ Range looks correct for diffusion normalization")
            else:
                print(f"    ❌ Range is WRONG! Should be [-1, 1]")
        
        # Check 7T target  
        target_path = Path(row['target_7t'])
        if target_path.exists():
            img = nib.load(str(target_path))
            data = img.get_fdata()
            
            print(f"  7T Target:")
            print(f"    Shape: {data.shape}")
            print(f"    Min: {data.min():.4f}")
            print(f"    Max: {data.max():.4f}")
            print(f"    Mean: {data.mean():.4f}")
            print(f"    Std: {data.std():.4f}")
            
            # Check if in [-1, 1] range
            if data.min() >= -1.1 and data.max() <= 1.1:
                print(f"    ✅ Range looks correct for diffusion normalization")
            else:
                print(f"    ❌ Range is WRONG! Should be [-1, 1]")
        
        print()
    
    return True

def check_model_output_range(checkpoint_path=None):
    """Check what range the model is actually predicting."""
    print("=" * 80)
    print("MODEL OUTPUT RANGE CHECK")
    print("=" * 80)
    
    if checkpoint_path is None:
        # Try to find latest checkpoint
        checkpoint_paths = [
            Path("checkpoints/checkpoint_10000.pt"),
            Path("checkpoints/checkpoint_latest.pt"),
        ]
        
        checkpoint_path = None
        for cp in checkpoint_paths:
            if cp.exists():
                checkpoint_path = cp
                break
    
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        print("⚠️  No checkpoint found. Train first, then run this check.")
        return True
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"Checkpoint at step: {checkpoint.get('step', 'unknown')}")
    
    # Try to load config
    config = checkpoint.get('config', {})
    print(f"Config keys: {list(config.keys())}")
    
    return True

def main():
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 18 + "EMERGENCY DATA RANGE DIAGNOSTIC" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    data_ok = check_preprocessed_data_ranges()
    print()
    
    model_ok = check_model_output_range()
    print()
    
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print()
    print("If you see:")
    print("  - 3T/7T ranges are NOT [-1, 1]: Preprocessing failed")
    print("  - 3T/7T ranges ARE [-1, 1] but PSNR is negative: Model output range issue")
    print("  - Pixel loss > 10: Severe range mismatch between prediction and target")
    print()
    print("Solutions:")
    print("  1. If preprocessing is wrong: Re-run preprocessing with correct config")
    print("  2. If model output is wrong: Check for any output activation in model.py")
    print("  3. If ranges look correct: May need to add output clamping in diffusion.py")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
