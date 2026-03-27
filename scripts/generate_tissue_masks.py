"""
Generate synthetic tissue segmentations from 7T T1w images.
This creates approximate GM/WM/CSF masks using intensity thresholding.
"""
import nibabel as nib
import numpy as np
from pathlib import Path
from scipy import ndimage
import argparse

def generate_tissue_mask(t1w_path, brain_mask_path, output_path):
    """
    Generate 3-class tissue segmentation from T1w and brain mask.
    
    Classes:
    0 - Background
    1 - CSF (darkest in brain)
    2 - Gray Matter (medium intensity)
    3 - White Matter (brightest in brain)
    """
    # Load data
    t1w_img = nib.load(t1w_path)
    t1w_data = t1w_img.get_fdata()
    
    brain_mask_img = nib.load(brain_mask_path)
    brain_mask = brain_mask_img.get_fdata() > 0
    
    # Initialize output
    tissue_mask = np.zeros_like(t1w_data, dtype=np.uint8)
    
    # Get brain voxel intensities
    brain_intensities = t1w_data[brain_mask]
    
    # Compute percentile thresholds
    # T1w: WM is brightest, GM is medium, CSF is darkest
    p33 = np.percentile(brain_intensities, 33)
    p66 = np.percentile(brain_intensities, 66)
    
    # Assign tissue classes within brain
    tissue_mask[brain_mask & (t1w_data < p33)] = 1   # CSF (darkest)
    tissue_mask[brain_mask & (t1w_data >= p33) & (t1w_data < p66)] = 2  # GM
    tissue_mask[brain_mask & (t1w_data >= p66)] = 3  # WM (brightest)
    
    # Clean up with morphological operations
    for label in [1, 2, 3]:
        mask = tissue_mask == label
        # Remove small islands
        mask = ndimage.binary_opening(mask, structure=np.ones((3,3,3)))
        # Fill small holes
        mask = ndimage.binary_closing(mask, structure=np.ones((3,3,3)))
        tissue_mask[mask] = label
    
    # Save
    tissue_img = nib.Nifti1Image(tissue_mask, t1w_img.affine, t1w_img.header)
    nib.save(tissue_img, output_path)
    
    # Print statistics
    unique, counts = np.unique(tissue_mask, return_counts=True)
    print(f"Generated tissue mask: {output_path}")
    for label, count in zip(unique, counts):
        label_names = {0: "Background", 1: "CSF", 2: "GM", 3: "WM"}
        pct = 100 * count / tissue_mask.size
        print(f"  Class {label} ({label_names[label]}): {count} voxels ({pct:.1f}%)")
    
    return tissue_mask


def process_all_volumes(manifest_path, output_dir):
    """Process all 7T volumes in the manifest."""
    import pandas as pd
    
    manifest = pd.read_csv(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, row in manifest.iterrows():
        # Assuming manifest has columns: target_7t_path, mask_output_path
        t1w_path = row['target_7t_path']
        mask_path = row['mask_output_path']
        
        # Generate output path
        t1w_name = Path(t1w_path).stem.replace('_preprocessed', '')
        tissue_path = output_dir / f"{t1w_name}_tissue_mask.nii.gz"
        
        print(f"\nProcessing {idx+1}/{len(manifest)}: {t1w_name}")
        
        try:
            generate_tissue_mask(t1w_path, mask_path, tissue_path)
            
            # Update manifest with new tissue mask path
            manifest.at[idx, 'tissue_mask_path'] = str(tissue_path)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # Save updated manifest
    output_manifest = output_dir / "pairs_with_tissue_masks.csv"
    manifest.to_csv(output_manifest, index=False)
    print(f"\n✓ Updated manifest saved to: {output_manifest}")
    print(f"  Total volumes processed: {len(manifest)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic tissue masks")
    parser.add_argument("--manifest", required=True, help="Path to pairs.csv")
    parser.add_argument("--output_dir", default="tissue_masks", help="Output directory")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Synthetic Tissue Mask Generation")
    print("=" * 60)
    print()
    
    process_all_volumes(args.manifest, args.output_dir)
    
    print()
    print("=" * 60)
    print("✅ Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Update configs/train_diffusion.yaml:")
    print("   - Change num_classes: 2 → 4")
    print("   - Update dataset.pairs_csv to point to new manifest")
    print("2. Resume training from 100k checkpoint")
    print("3. The model will now learn GM/WM/CSF boundaries")
