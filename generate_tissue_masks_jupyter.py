"""
Generate Synthetic Tissue Masks - CERNbox Jupyter Version
Run this directly in a Jupyter notebook or as a Python script.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from scipy import ndimage
import pandas as pd

def generate_tissue_mask(t1w_path, brain_mask_path, output_path):
    """
    Generate 4-class tissue segmentation from T1w and brain mask.
    
    Classes:
    0 - Background
    1 - CSF (darkest in brain)
    2 - Gray Matter (medium intensity)
    3 - White Matter (brightest in brain)
    """
    print(f"  Processing: {Path(t1w_path).name}")
    
    # Load data
    t1w_img = nib.load(t1w_path)
    t1w_data = t1w_img.get_fdata()
    
    brain_mask_img = nib.load(brain_mask_path)
    brain_mask = brain_mask_img.get_fdata() > 0
    
    # Initialize output
    tissue_mask = np.zeros_like(t1w_data, dtype=np.uint8)
    
    # Get brain voxel intensities
    brain_intensities = t1w_data[brain_mask]
    
    if len(brain_intensities) == 0:
        print(f"    ⚠ Warning: Empty brain mask, skipping")
        return None
    
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
    label_names = {0: "Background", 1: "CSF", 2: "GM", 3: "WM"}
    for label, count in zip(unique, counts):
        pct = 100 * count / tissue_mask.size
        print(f"    Class {label} ({label_names[label]}): {pct:.1f}%")
    
    return str(output_path)


def main():
    """Main processing function."""
    
    # Configuration
    base_dir = Path("/eos/home-i04/p/ppokhrel/Untitled Folder 1")
    manifest_path = base_dir / "Topo-Brain" / "pairs.csv"
    output_dir = base_dir / "tissue_masks"
    
    print("=" * 60)
    print("Synthetic Tissue Mask Generation")
    print("=" * 60)
    print()
    print(f"Manifest: {manifest_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    manifest = pd.read_csv(manifest_path)
    print(f"Found {len(manifest)} volume pairs")
    print()
    
    # Process each volume
    tissue_mask_paths = []
    for idx, row in manifest.iterrows():
        print(f"[{idx+1}/{len(manifest)}]")
        
        t1w_path = row['target_7t_path']
        mask_path = row['mask_output_path']
        
        # Generate output path
        t1w_name = Path(t1w_path).stem.replace('_preprocessed', '')
        tissue_path = output_dir / f"{t1w_name}_tissue_mask.nii.gz"
        
        try:
            result = generate_tissue_mask(t1w_path, mask_path, tissue_path)
            tissue_mask_paths.append(result)
            print(f"  ✓ Saved: {tissue_path.name}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            tissue_mask_paths.append(None)
        
        print()
    
    # Update manifest
    manifest['tissue_mask_path'] = tissue_mask_paths
    output_manifest = output_dir / "pairs_with_tissue_masks.csv"
    manifest.to_csv(output_manifest, index=False)
    
    print("=" * 60)
    print("✅ Complete!")
    print("=" * 60)
    print()
    print(f"Updated manifest: {output_manifest}")
    print(f"Total volumes processed: {len([p for p in tissue_mask_paths if p is not None])}")
    print()
    print("Next steps:")
    print("1. Update configs/train_diffusion.yaml:")
    print(f'   dataset.pairs_csv: "{output_manifest.relative_to(base_dir / "Topo-Brain")}"')
    print("2. Resume training from 100k checkpoint")
    print("3. Watch loss_topo decrease!")


if __name__ == "__main__":
    main()
