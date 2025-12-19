"""
Quick visualization script to view brain masked images
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths for first subject
subject_dir = Path("Nifti/sub-01/ses-1/anat")
original_file = subject_dir / "sub-01_ses-1_T1w_defaced.nii.gz"
brain_file = subject_dir / "sub-01_ses-1_T1w_brain.nii.gz"
mask_file = subject_dir / "sub-01_ses-1_T1w_brain_mask.nii.gz"

# Load the images
original_img = nib.load(original_file)
brain_img = nib.load(brain_file)
mask_img = nib.load(mask_file)

original_data = original_img.get_fdata()
brain_data = brain_img.get_fdata()
mask_data = mask_img.get_fdata()

# Get middle slices for visualization
mid_sagittal = original_data.shape[0] // 2
mid_coronal = original_data.shape[1] // 2
mid_axial = original_data.shape[2] // 2

# Create figure with comparison
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('Subject 01 - Brain Extraction Results', fontsize=16, fontweight='bold')

views = [
    ('Sagittal', 0, mid_sagittal),
    ('Coronal', 1, mid_coronal),
    ('Axial', 2, mid_axial)
]

for idx, (view_name, axis, slice_idx) in enumerate(views):
    # Get slices
    if axis == 0:
        orig_slice = original_data[slice_idx, :, :]
        brain_slice = brain_data[slice_idx, :, :]
        mask_slice = mask_data[slice_idx, :, :]
    elif axis == 1:
        orig_slice = original_data[:, slice_idx, :]
        brain_slice = brain_data[:, slice_idx, :]
        mask_slice = mask_data[:, slice_idx, :]
    else:
        orig_slice = original_data[:, :, slice_idx]
        brain_slice = brain_data[:, :, slice_idx]
        mask_slice = mask_data[:, :, slice_idx]
    
    # Original
    axes[idx, 0].imshow(orig_slice.T, cmap='gray', origin='lower')
    axes[idx, 0].set_title(f'{view_name} - Original')
    axes[idx, 0].axis('off')
    
    # Brain mask
    axes[idx, 1].imshow(mask_slice.T, cmap='gray', origin='lower')
    axes[idx, 1].set_title(f'{view_name} - Brain Mask')
    axes[idx, 1].axis('off')
    
    # Brain extracted
    axes[idx, 2].imshow(brain_slice.T, cmap='gray', origin='lower')
    axes[idx, 2].set_title(f'{view_name} - Brain Extracted')
    axes[idx, 2].axis('off')

plt.tight_layout()
plt.savefig('brain_extraction_result_sub01.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved as 'brain_extraction_result_sub01.png'")
plt.show()

# Print statistics
print("\n" + "="*60)
print("BRAIN EXTRACTION STATISTICS - Subject 01")
print("="*60)
print(f"Original image shape: {original_data.shape}")
print(f"Brain mask coverage: {np.sum(mask_data > 0) / mask_data.size * 100:.2f}% of volume")
print(f"Brain voxels: {np.sum(mask_data > 0):,}")
print(f"Original intensity range: [{original_data.min():.1f}, {original_data.max():.1f}]")
print(f"Brain intensity range: [{brain_data[brain_data > 0].min():.1f}, {brain_data[brain_data > 0].max():.1f}]")
print("="*60)
