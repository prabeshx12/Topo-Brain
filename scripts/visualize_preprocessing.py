"""
Visualize preprocessing effects: original vs preprocessed images with sample patches.
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def normalize_for_display(img, percentile_clip=True):
    """Normalize image to 0-1 range for display with outlier clipping."""
    img = img.astype(float)
    if percentile_clip:
        # Clip outliers for better contrast
        p1, p99 = np.percentile(img, [1, 99])
        img = np.clip(img, p1, p99)
    if img.max() > img.min():
        return (img - img.min()) / (img.max() - img.min())
    return img

def extract_patch(volume, center, size=32):
    """Extract a 3D patch from volume."""
    z, y, x = center
    half = size // 2
    
    z_start, z_end = max(0, z - half), min(volume.shape[0], z + half)
    y_start, y_end = max(0, y - half), min(volume.shape[1], y + half)
    x_start, x_end = max(0, x - half), min(volume.shape[2], x + half)
    
    return volume[z_start:z_end, y_start:y_end, x_start:x_end]

# Load images
print("Loading images...")
subject = "sub-01"
modality = "T1w"

# Original images
orig_3t = nib.load(f'Nifti/{subject}/ses-1/anat/{subject}_ses-1_{modality}_defaced.nii.gz')
orig_7t = nib.load(f'Nifti/{subject}/ses-2/anat/{subject}_ses-2_{modality}_defaced.nii.gz')

# Preprocessed images
prep_3t = nib.load(f'preprocessed/{subject}/ses-1/anat/{subject}_ses-1_{modality}_defaced_preprocessed.nii.gz')
prep_7t = nib.load(f'preprocessed/{subject}/ses-2/anat/{subject}_ses-2_{modality}_defaced_preprocessed.nii.gz')

# Get data arrays
orig_3t_data = orig_3t.get_fdata()
orig_7t_data = orig_7t.get_fdata()
prep_3t_data = np.squeeze(prep_3t.get_fdata())
prep_7t_data = np.squeeze(prep_7t.get_fdata())

print(f"Original 3T shape: {orig_3t_data.shape}, spacing: {orig_3t.header.get_zooms()}")
print(f"Original 7T shape: {orig_7t_data.shape}, spacing: {orig_7t.header.get_zooms()}")
print(f"Preprocessed 3T shape: {prep_3t_data.shape}")
print(f"Preprocessed 7T shape: {prep_7t_data.shape}")

# Get middle slices
mid_orig_3t = orig_3t_data.shape[2] // 2
mid_orig_7t = orig_7t_data.shape[2] // 2
mid_prep_3t = prep_3t_data.shape[2] // 2
mid_prep_7t = prep_7t_data.shape[2] // 2

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# Row 1: Original vs Preprocessed - 3T
ax1 = fig.add_subplot(gs[0, 0])
slice_3t_orig = normalize_for_display(orig_3t_data[:, :, mid_orig_3t])
ax1.imshow(slice_3t_orig.T, cmap='gray', origin='lower', vmin=0, vmax=1)
ax1.set_title(f'Original 3T - Axial Slice\nShape: {orig_3t_data.shape}', fontsize=10)
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 1])
slice_3t_prep = normalize_for_display(prep_3t_data[:, :, mid_prep_3t])
ax2.imshow(slice_3t_prep.T, cmap='gray', origin='lower', vmin=0, vmax=1)
ax2.set_title(f'Preprocessed 3T - Axial Slice\nShape: {prep_3t_data.shape}\nIntensity: [{prep_3t_data.min():.2f}, {prep_3t_data.max():.2f}]', fontsize=9)
ax2.axis('off')

# Row 1: Original vs Preprocessed - 7T
ax3 = fig.add_subplot(gs[0, 2])
slice_7t_orig = normalize_for_display(orig_7t_data[:, :, mid_orig_7t])
ax3.imshow(slice_7t_orig.T, cmap='gray', origin='lower', vmin=0, vmax=1)
ax3.set_title(f'Original 7T - Axial Slice\nShape: {orig_7t_data.shape}', fontsize=10)
ax3.axis('off')

ax4 = fig.add_subplot(gs[0, 3])
slice_7t_prep = normalize_for_display(prep_7t_data[:, :, mid_prep_7t])
ax4.imshow(slice_7t_prep.T, cmap='gray', origin='lower', vmin=0, vmax=1)
ax4.set_title(f'Preprocessed 7T - Axial Slice\nShape: {prep_7t_data.shape}\nIntensity: [{prep_7t_data.min():.2f}, {prep_7t_data.max():.2f}]', fontsize=9)
ax4.axis('off')

# Row 2: Coronal slices
mid_coronal_orig_3t = orig_3t_data.shape[1] // 2
mid_coronal_orig_7t = orig_7t_data.shape[1] // 2
mid_coronal_prep_3t = prep_3t_data.shape[1] // 2
mid_coronal_prep_7t = prep_7t_data.shape[1] // 2

ax5 = fig.add_subplot(gs[1, 0])
slice_cor_3t_orig = normalize_for_display(orig_3t_data[:, mid_coronal_orig_3t, :])
ax5.imshow(slice_cor_3t_orig.T, cmap='gray', origin='lower', vmin=0, vmax=1)
ax5.set_title('Original 3T - Coronal', fontsize=10)
ax5.axis('off')

ax6 = fig.add_subplot(gs[1, 1])
slice_cor_3t_prep = normalize_for_display(prep_3t_data[:, mid_coronal_prep_3t, :])
ax6.imshow(slice_cor_3t_prep.T, cmap='gray', origin='lower', vmin=0, vmax=1)
ax6.set_title('Preprocessed 3T - Coronal', fontsize=10)
ax6.axis('off')

ax7 = fig.add_subplot(gs[1, 2])
slice_cor_7t_orig = normalize_for_display(orig_7t_data[:, mid_coronal_orig_7t, :])
ax7.imshow(slice_cor_7t_orig.T, cmap='gray', origin='lower', vmin=0, vmax=1)
ax7.set_title('Original 7T - Coronal', fontsize=10)
ax7.axis('off')

ax8 = fig.add_subplot(gs[1, 3])
slice_cor_7t_prep = normalize_for_display(prep_7t_data[:, mid_coronal_prep_7t, :])
ax8.imshow(slice_cor_7t_prep.T, cmap='gray', origin='lower', vmin=0, vmax=1)
ax8.set_title('Preprocessed 7T - Coronal', fontsize=10)
ax8.axis('off')



# Row 3: Sample patches (zoomed regions)
# Extract patches from center of brain
patch_size = 64
center_3t_orig = (orig_3t_data.shape[0]//2, orig_3t_data.shape[1]//2, mid_orig_3t)
center_7t_orig = (orig_7t_data.shape[0]//2, orig_7t_data.shape[1]//2, mid_orig_7t)
center_3t_prep = (prep_3t_data.shape[0]//2, prep_3t_data.shape[1]//2, mid_prep_3t)
center_7t_prep = (prep_7t_data.shape[0]//2, prep_7t_data.shape[1]//2, mid_prep_7t)

patch_3t_orig = extract_patch(orig_3t_data, center_3t_orig, patch_size)
patch_7t_orig = extract_patch(orig_7t_data, center_7t_orig, patch_size)
patch_3t_prep = extract_patch(prep_3t_data, center_3t_prep, patch_size)
patch_7t_prep = extract_patch(prep_7t_data, center_7t_prep, patch_size)

ax9 = fig.add_subplot(gs[2, 0])
patch_slice_3t_orig = normalize_for_display(patch_3t_orig[:, :, patch_3t_orig.shape[2]//2])
ax9.imshow(patch_slice_3t_orig.T, cmap='gray', origin='lower', vmin=0, vmax=1)
ax9.set_title(f'Original 3T Patch\n{patch_3t_orig.shape}', fontsize=10)
ax9.axis('off')

ax10 = fig.add_subplot(gs[2, 1])
patch_slice_3t_prep = normalize_for_display(patch_3t_prep[:, :, patch_3t_prep.shape[2]//2])
ax10.imshow(patch_slice_3t_prep.T, cmap='gray', origin='lower', vmin=0, vmax=1)
ax10.set_title(f'Preprocessed 3T Patch\n{patch_3t_prep.shape}', fontsize=10)
ax10.axis('off')

ax11 = fig.add_subplot(gs[2, 2])
patch_slice_7t_orig = normalize_for_display(patch_7t_orig[:, :, patch_7t_orig.shape[2]//2])
ax11.imshow(patch_slice_7t_orig.T, cmap='gray', origin='lower', vmin=0, vmax=1)
ax11.set_title(f'Original 7T Patch\n{patch_7t_orig.shape}', fontsize=10)
ax11.axis('off')

ax12 = fig.add_subplot(gs[2, 3])
patch_slice_7t_prep = normalize_for_display(patch_7t_prep[:, :, patch_7t_prep.shape[2]//2])
ax12.imshow(patch_slice_7t_prep.T, cmap='gray', origin='lower', vmin=0, vmax=1)
ax12.set_title(f'Preprocessed 7T Patch\n{patch_7t_prep.shape}', fontsize=10)
ax12.axis('off')



plt.suptitle(f'Preprocessing Comparison - {subject} {modality}\nSpatial Resampling + Intensity Normalization (Z-score)', 
             fontsize=14, fontweight='bold')

# Save figure
output_path = Path('logs/qc/preprocessing_comparison.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to: {output_path}")

# Create intensity distribution comparison
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original distributions
axes[0, 0].hist(orig_3t_data.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
axes[0, 0].set_title(f'Original 3T Intensity Distribution\nMean: {orig_3t_data.mean():.2f}, Std: {orig_3t_data.std():.2f}')
axes[0, 0].set_xlabel('Intensity')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].hist(orig_7t_data.flatten(), bins=100, alpha=0.7, color='red', edgecolor='black')
axes[0, 1].set_title(f'Original 7T Intensity Distribution\nMean: {orig_7t_data.mean():.2f}, Std: {orig_7t_data.std():.2f}')
axes[0, 1].set_xlabel('Intensity')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(alpha=0.3)

# Preprocessed distributions
axes[1, 0].hist(prep_3t_data.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
axes[1, 0].set_title(f'Preprocessed 3T Intensity Distribution\nMean: {prep_3t_data.mean():.4f}, Std: {prep_3t_data.std():.4f}')
axes[1, 0].set_xlabel('Intensity (Z-score normalized)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(alpha=0.3)

axes[1, 1].hist(prep_7t_data.flatten(), bins=100, alpha=0.7, color='red', edgecolor='black')
axes[1, 1].set_title(f'Preprocessed 7T Intensity Distribution\nMean: {prep_7t_data.mean():.4f}, Std: {prep_7t_data.std():.4f}')
axes[1, 1].set_xlabel('Intensity (Z-score normalized)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(alpha=0.3)

plt.suptitle('Intensity Distribution: Before vs After Preprocessing', fontsize=14, fontweight='bold')
plt.tight_layout()

output_path2 = Path('logs/qc/intensity_distributions.png')
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Saved intensity distributions to: {output_path2}")

plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Original 3T: {orig_3t_data.shape}, spacing: {orig_3t.header.get_zooms()[:3]}")
print(f"Original 7T: {orig_7t_data.shape}, spacing: {orig_7t.header.get_zooms()[:3]}")
print(f"\nPreprocessed 3T: {prep_3t_data.shape}, spacing: ~0.8mm isotropic")
print(f"Preprocessed 7T: {prep_7t_data.shape}, spacing: ~0.8mm isotropic")
print(f"\nIntensity ranges:")
print(f"  Original 3T: [{orig_3t_data.min():.1f}, {orig_3t_data.max():.1f}]")
print(f"  Original 7T: [{orig_7t_data.min():.1f}, {orig_7t_data.max():.1f}]")
print(f"  Preprocessed 3T: [{prep_3t_data.min():.4f}, {prep_3t_data.max():.4f}]")
print(f"  Preprocessed 7T: [{prep_7t_data.min():.4f}, {prep_7t_data.max():.4f}]")
print("="*70)
