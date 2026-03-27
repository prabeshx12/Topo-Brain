"""
Script to visualize patches from the PairedPatchDataset.
Demonstrates:
1. Paired extraction (3T and 7T alignment)
2. Advanced Augmentation effects (Elastic, Affine)
3. Data Formulation correctness (Shapes, Intensities)
"""
import argparse
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.synthesis_dataset import load_pairs_manifest, PairedPatchDataset, PatchConfig

def save_patch_slices(
    batch: dict,
    output_path: Path,
    idx: int,
    suffix: str = "",
):
    """Save middle slices of the 3D patches."""
    input_vol = batch["input"][0,0].numpy() # Take 1st batch, 1st channel (T1)
    target_vol = batch["target"][0,0].numpy()
    
    # Get middle slice indices
    d, h, w = input_vol.shape
    mid_d, mid_h, mid_w = d // 2, h // 2, w // 2
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"Patch {idx} {suffix} | Sub: {batch['subject']}", fontsize=14)
    
    # Input slices
    axes[0, 0].imshow(input_vol[mid_d, :, :], cmap="gray", origin="lower")
    axes[0, 0].set_title("Input 3T (Axial)")
    axes[0, 1].imshow(input_vol[:, mid_h, :], cmap="gray", origin="lower")
    axes[0, 1].set_title("Input 3T (Coronal)")
    axes[0, 2].imshow(input_vol[:, :, mid_w], cmap="gray", origin="lower")
    axes[0, 2].set_title("Input 3T (Sagittal)")
    
    # Target slices
    axes[1, 0].imshow(target_vol[mid_d, :, :], cmap="gray", origin="lower")
    axes[1, 0].set_title("Target 7T (Axial)")
    axes[1, 1].imshow(target_vol[:, mid_h, :], cmap="gray", origin="lower")
    axes[1, 1].set_title("Target 7T (Coronal)")
    axes[1, 2].imshow(target_vol[:, :, mid_w], cmap="gray", origin="lower")
    axes[1, 2].set_title("Target 7T (Sagittal)")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize synthesis patches")
    parser.add_argument("--pairs-file", type=str, default="derivatives/topobrain-preproc/pairs.csv")
    parser.add_argument("--data-root", type=str, help="Override root directory for image files (useful if data moved)")
    parser.add_argument("--output-dir", type=str, default="viz_patches")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Pairs
    pairs_path = Path(args.pairs_file)
    # 1. Load Pairs
    pairs_path = Path(args.pairs_file)
    if not pairs_path.exists():
        print(f"Error: Pairs manifest not found at: {pairs_path}")
        print("Please provide the correct path using: --pairs-file /path/to/pairs.csv")
        return

    # Pass data_root to load_pairs_manifest for automatic rebasing
    data_root = Path(args.data_root) if args.data_root else None
    pairs = load_pairs_manifest(pairs_path, data_root=data_root)
    
    if not pairs:
        print("Error: Pairs manifest is empty!")
        return

    # Filter invalid pairs (where files still don't exist)
    valid_pairs = [
        p for p in pairs 
        if Path(p["input_3t"]).exists() and Path(p["target_7t"]).exists()
    ]
    
    if not valid_pairs:
        print("Error: No valid pairs found!")
        first_pair = pairs[0] if pairs else {}
        sample_rel = first_pair.get('input_3t', 'unknown')
        print(f"Sample path from CSV: {sample_rel}")
        
        if data_root:
            print(f"Data Root Provided: {data_root.resolve()}")
            expected_full = data_root / sample_rel
            print(f"Expected Full Path: {expected_full}")
            print(f"Details: exists={expected_full.exists()}")
            
            # List root contents to help user
            if data_root.exists():
                print(f"Contents of {data_root}:")
                for x in list(data_root.iterdir())[:5]:
                    print(f" - {x.name}")
            else:
                print(f"Warning: Data root {data_root} does not exist!")
        else:
            print("Try providing --data-root /path/to/preprocessed_data")
        return
        
    print(f"Found {len(valid_pairs)} valid pairs.")
    pairs = valid_pairs
        
    # 2. Configure Dataset (Non-Augmented for Baseline)
    config = PatchConfig(patch_size=(64, 64, 64), patches_per_volume=4)
    ds_raw = PairedPatchDataset(pairs[:2], config=config, augment=False) # Use first 2 subjects
    
    print("Generating Non-Augmented Samples...")
    for i in range(args.num_samples):
        sample = ds_raw[i]
        # Fake batch dim
        batch = {
            "input": sample["input"].unsqueeze(0),
            "target": sample["target"].unsqueeze(0),
            "subject": sample["subject"]
        }
        save_patch_slices(batch, output_dir / f"sample_{i}_original.png", i, suffix="(Original)")
        
    # 3. Configure Dataset (Augmented)
    print("Generating Augmented Samples (Affine/Elastic)...")
    ds_aug = PairedPatchDataset(pairs[:2], config=config, augment=True)
    
    # We visualize the SAME indices to see random variations if we re-access, 
    # but here we just take fresh samples to show distortions.
    for i in range(args.num_samples):
        sample = ds_aug[i]
        batch = {
            "input": sample["input"].unsqueeze(0),
            "target": sample["target"].unsqueeze(0),
            "subject": sample["subject"]
        }
        save_patch_slices(batch, output_dir / f"sample_{i}_augmented.png", i, suffix="(Augmented)")

    print(f"Done! Check {output_dir} for visualizations.")

if __name__ == "__main__":
    main()
