
import os
import numpy as np
import nibabel as nib
from pathlib import Path
import json

def create_dummy_data():
    """Create dummy preprocessed data for testing train_gan.py."""
    
    # Setup paths
    root = Path("test_data_dummy")
    data_dir = root / "preprocessed"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 2 subjects, with paired sessions (3T and 7T)
    # sub-01_ses-1_T1w_preprocessed.nii.gz (3T)
    # sub-01_ses-2_T1w_preprocessed.nii.gz (7T)
    
    subjects = [f"sub-{i:02d}" for i in range(1, 6)]
    
    print(f"Creating dummy data in {data_dir}...")
    
    for sub in subjects:
        # Create 3T session (ses-1)
        # Shape must contain 64^3 patch, so make it 80x80x80
        shape = (80, 80, 80)
        
        # 3T
        data_3t = np.random.rand(*shape).astype(np.float32)
        affine = np.eye(4)
        img_3t = nib.Nifti1Image(data_3t, affine)
        
        # Folder structure: sub/ses/anat
        out_dir_3t = data_dir / sub / "ses-1" / "anat"
        out_dir_3t.mkdir(parents=True, exist_ok=True)
        
        fname_3t = f"{sub}_ses-1_T1w_preprocessed.nii.gz"
        nib.save(img_3t, out_dir_3t / fname_3t)
        
        # Metadata 3T
        meta_3t = {"spacing": [1.0, 1.0, 1.0]}
        with open(out_dir_3t / f"{sub}_ses-1_T1w_preprocessed_metadata.json", 'w') as f:
            json.dump(meta_3t, f)
            
        # 7T (ses-2)
        data_7t = np.random.rand(*shape).astype(np.float32)
        img_7t = nib.Nifti1Image(data_7t, affine)
        
        out_dir_7t = data_dir / sub / "ses-2" / "anat"
        out_dir_7t.mkdir(parents=True, exist_ok=True)
        
        fname_7t = f"{sub}_ses-2_T1w_preprocessed.nii.gz"
        nib.save(img_7t, out_dir_7t / fname_7t)
        
        # Metadata 7T
        with open(out_dir_7t / f"{sub}_ses-2_T1w_preprocessed_metadata.json", 'w') as f:
            json.dump(meta_3t, f)
            
    print("Done! Dummy data created.")

if __name__ == "__main__":
    create_dummy_data()
