"""Quick script to check preprocessed data statistics."""
import nibabel as nib
import numpy as np
from pathlib import Path
import sys
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Check preprocessed data statistics")
parser.add_argument("--data-root", type=str, default=None, 
                    help="Path to preprocessed data folder")
parser.add_argument("--file", type=str, default=None,
                    help="Direct path to a specific NIfTI file to check")
parser.add_argument("--all", action="store_true",
                    help="Check ALL NIfTI files in the folder (not just first)")
parser.add_argument("--max-files", type=int, default=None,
                    help="Maximum number of files to check (default: all)")
args = parser.parse_args()

def check_single_file(filepath):
    """Check a single NIfTI file and return (passed, stats_dict)"""
    img = nib.load(str(filepath))
    data = img.get_fdata()
    
    # With diffusion normalization: background = -1.0, brain > -0.95
    brain_mask = data > -0.95
    brain_data = data[brain_mask]
    background_data = data[~brain_mask]
    
    stats = {
        "file": filepath.name,
        "shape": data.shape,
        "min": data.min(),
        "max": data.max(),
        "brain_mean": brain_data.mean() if len(brain_data) > 0 else 0,
        "brain_std": brain_data.std() if len(brain_data) > 0 else 0,
        "brain_max": brain_data.max() if len(brain_data) > 0 else 0,
        "bg_mean": background_data.mean() if len(background_data) > 0 else -1.0,
    }
    
    # Validation
    problems = []
    if len(background_data) > 0 and stats["bg_mean"] > -0.9:
        problems.append("bg_not_minus1")
    if len(brain_data) > 0 and stats["brain_max"] < 0.5:
        problems.append("low_contrast")
    if len(brain_data) > 0 and stats["brain_std"] < 0.2:
        problems.append("compressed_range")
    
    stats["problems"] = problems
    return len(problems) == 0, stats

# Determine files to check
files_to_check = []

if args.file:
    test_file = Path(args.file)
    if not test_file.exists():
        print(f"ERROR: File not found: {test_file}")
        sys.exit(1)
    files_to_check = [test_file]
elif args.data_root:
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"ERROR: Path does not exist: {data_root}")
        sys.exit(1)
    # Find NIfTI files
    nii_files = list(data_root.rglob("*.nii.gz")) + list(data_root.rglob("*.nii"))
    if not nii_files:
        print(f"ERROR: No NIfTI files found in {data_root}")
        sys.exit(1)
    
    if args.all or args.max_files:
        files_to_check = nii_files[:args.max_files] if args.max_files else nii_files
    else:
        files_to_check = [nii_files[0]]
        print(f"Found {len(nii_files)} NIfTI files. Checking first one only.")
        print("Use --all to check all files, or --max-files N to check N files.\n")
else:
    # Default paths
    for default_path in [
        "/eos/home-i04/p/ppokhrel/Untitled Folder 1/preprocessed",
        "/kaggle/working/preprocessed",
        "/kaggle/working"
    ]:
        data_root = Path(default_path)
        if data_root.exists():
            break
    else:
        print("ERROR: No data path specified and defaults don't exist.")
        print("Usage: python check_data_stats.py --data-root /path/to/preprocessed")
        print("   or: python check_data_stats.py --file /path/to/specific/file.nii.gz")
        sys.exit(1)
    
    nii_files = list(data_root.rglob("*desc-preproc*.nii.gz"))
    if not nii_files:
        nii_files = list(data_root.rglob("*.nii.gz"))
    if not nii_files:
        print(f"ERROR: No NIfTI files found in {data_root}")
        sys.exit(1)
    
    if args.all or args.max_files:
        files_to_check = nii_files[:args.max_files] if args.max_files else nii_files
    else:
        files_to_check = [nii_files[0]]
        print(f"Found {len(nii_files)} NIfTI files. Checking first one only.")
        print("Use --all to check all files, or --max-files N to check N files.\n")

# Check files
print("="*60)
print(f"CHECKING {len(files_to_check)} FILE(S)")
print("="*60)

passed_count = 0
failed_count = 0
failed_files = []

for i, filepath in enumerate(files_to_check):
    print(f"\n[{i+1}/{len(files_to_check)}] {filepath.name}...", end=" ")
    try:
        passed, stats = check_single_file(filepath)
        if passed:
            print(f"✅ OK (bg={stats['bg_mean']:.3f}, max={stats['brain_max']:.3f}, std={stats['brain_std']:.3f})")
            passed_count += 1
        else:
            print(f"❌ FAILED: {', '.join(stats['problems'])}")
            failed_count += 1
            failed_files.append((filepath, stats))
    except Exception as e:
        print(f"❌ ERROR: {e}")
        failed_count += 1
        failed_files.append((filepath, {"error": str(e)}))

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total files checked: {len(files_to_check)}")
print(f"Passed: {passed_count} ✅")
print(f"Failed: {failed_count} ❌")

if failed_files:
    print("\nFailed files:")
    for fp, stats in failed_files:
        if "error" in stats:
            print(f"  - {fp.name}: {stats['error']}")
        else:
            print(f"  - {fp.name}: {', '.join(stats['problems'])} (bg={stats['bg_mean']:.3f})")

print("\n" + "="*60)
if failed_count == 0:
    print("🎉 ALL FILES PASSED! Data is ready for training.")
    print("="*60)
    sys.exit(0)
else:
    print("⚠️  SOME FILES FAILED! Re-run preprocessing with fixed config.")
    print("="*60)
    sys.exit(1)
