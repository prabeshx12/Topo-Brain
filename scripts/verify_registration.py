"""
Quick script to manually verify registration quality.
"""
import nibabel as nib
import numpy as np
from pathlib import Path

print("="*70)
print("CHECKING REGISTRATION QUALITY FOR SUB-01 T1w")
print("="*70)

# Load preprocessed BEFORE registration
print("\n1. BEFORE Registration:")
orig_3t = nib.load('preprocessed/sub-01/ses-1/anat/sub-01_ses-1_T1w_defaced_preprocessed.nii.gz')
orig_7t = nib.load('preprocessed/sub-01/ses-2/anat/sub-01_ses-2_T1w_defaced_preprocessed.nii.gz')

data_3t_orig = np.squeeze(orig_3t.get_fdata())
data_7t_orig = np.squeeze(orig_7t.get_fdata())

corr_before = np.corrcoef(data_3t_orig.flatten(), data_7t_orig.flatten())[0, 1]
print(f"   Correlation: {corr_before:.4f}")

# Load registered AFTER registration
print("\n2. AFTER Registration:")
reg_3t = nib.load('preprocessed_registered/sub-01/ses-1/anat/sub-01_ses-1_T1w_registered.nii.gz')
ref_7t = nib.load('preprocessed_registered/sub-01/ses-2/anat/sub-01_ses-2_T1w_preprocessed.nii.gz')

data_3t_reg = reg_3t.get_fdata()
data_7t_ref = np.squeeze(ref_7t.get_fdata())

print(f"   Registered 3T shape: {data_3t_reg.shape}")
print(f"   7T reference shape: {data_7t_ref.shape}")

corr_after = np.corrcoef(data_3t_reg.flatten(), data_7t_ref.flatten())[0, 1]
print(f"   Correlation: {corr_after:.4f}")

# Summary
print("\n" + "="*70)
print(f"IMPROVEMENT: {corr_after - corr_before:+.4f}")
if corr_after > 0.7:
    print("✓ GOOD quality")
elif corr_after > 0.5:
    print("⚠ REVIEW - acceptable but could be better")
else:
    print("✗ POOR quality - registration may have failed")
print("="*70)
