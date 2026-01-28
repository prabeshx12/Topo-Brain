"""Quick script to check preprocessed data statistics."""
import nibabel as nib
import numpy as np
from pathlib import Path
import sys

# Update this to your actual data root path on CERNBox/EOS
data_root = input("Enter your data root path (e.g., /eos/home-i04/p/ppokhrel/Untitled Folder 1/Topo-brain/derivatives/topobrain-preproc): ").strip()

data_root = Path(data_root)

if not data_root.exists():
    print(f"ERROR: Path does not exist: {data_root}")
    sys.exit(1)

# Load first preprocessed file
test_file = data_root / "sub-01/ses-1/anat/sub-01_ses-1_desc-preproc_T1w_registered.nii.gz"

if not test_file.exists():
    print(f"ERROR: Test file not found: {test_file}")
    print(f"Looking for files in: {data_root / 'sub-01'}")
    if (data_root / "sub-01").exists():
        files = list((data_root / "sub-01").rglob("*.nii.gz"))
        print(f"Found {len(files)} .nii.gz files:")
        for f in files[:5]:
            print(f"  {f.relative_to(data_root)}")
    sys.exit(1)

print(f"Loading: {test_file}")
img = nib.load(str(test_file))
data = img.get_fdata()

# Calculate statistics
brain_mask = np.abs(data) > 0.01
brain_data = data[brain_mask]
background_data = data[~brain_mask]

print("\n" + "="*60)
print("PREPROCESSED DATA STATISTICS")
print("="*60)
print(f"\nBrain tissue (abs > 0.01):")
print(f"  Min:  {brain_data.min():.6f}")
print(f"  Max:  {brain_data.max():.6f}")
print(f"  Mean: {brain_data.mean():.6f}")
print(f"  Std:  {brain_data.std():.6f}")
print(f"  Voxels: {len(brain_data)}")

print(f"\nBackground (abs <= 0.01):")
print(f"  Min:  {background_data.min():.6f}")
print(f"  Max:  {background_data.max():.6f}")
print(f"  Mean: {background_data.mean():.6f}")
print(f"  Unique values: {len(np.unique(background_data))}")

print(f"\nFull volume:")
print(f"  Shape: {data.shape}")
print(f"  Min:  {data.min():.6f}")
print(f"  Max:  {data.max():.6f}")
print(f"  Mean: {data.mean():.6f}")

print("\n" + "="*60)
print("EXPECTED VALUES:")
print("="*60)
print("Brain tissue: Should span [-1, 1] with good distribution")
print("Background: Should ALL be -1.0 (not 0.0)")
print("Full volume: Min=-1.0, Max=~1.0")
print("="*60)

# Check for problems
problems = []
if background_data.mean() > -0.9:
    problems.append("❌ Background is NOT -1.0! (old preprocessing)")
if brain_data.max() < 0.5:
    problems.append("❌ Brain tissue max < 0.5 (poor contrast)")
if brain_data.std() < 0.2:
    problems.append("❌ Low standard deviation (compressed range)")

if problems:
    print("\n⚠️  PROBLEMS DETECTED:")
    for p in problems:
        print(f"  {p}")
    print("\n→ You need to re-run preprocessing with the fixed config!")
else:
    print("\n✅ Data looks correct! Ready for training.")
