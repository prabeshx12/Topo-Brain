#!/usr/bin/env python
"""
Quick setup and validation script for the brain MRI preprocessing pipeline.
Run this first to verify your environment and dataset.
"""
import sys
from pathlib import Path

def check_imports():
    """Check if all required packages are installed."""
    print("Checking required packages...")
    required_packages = {
        'torch': 'PyTorch',
        'monai': 'MONAI',
        'nibabel': 'NiBabel',
        'SimpleITK': 'SimpleITK',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("✓ All required packages are installed\n")
    return True


def check_dataset(data_root):
    """Check if dataset exists and has expected structure."""
    print(f"Checking dataset at: {data_root}")
    
    if not data_root.exists():
        print(f"  ✗ Dataset not found at {data_root}")
        print("  Please update the data_root path in config.py")
        return False
    
    # Count subjects
    subjects = list(data_root.glob("sub-*"))
    if len(subjects) == 0:
        print(f"  ✗ No subjects found in {data_root}")
        return False
    
    print(f"  ✓ Found {len(subjects)} subjects")
    
    # Check for NIfTI files
    nifti_files = list(data_root.glob("**/*.nii.gz"))
    if len(nifti_files) == 0:
        print(f"  ✗ No NIfTI files found")
        return False
    
    print(f"  ✓ Found {len(nifti_files)} NIfTI files")
    
    # Check for JSON metadata
    json_files = list(data_root.glob("**/*.json"))
    print(f"  ✓ Found {len(json_files)} JSON metadata files")
    
    return True


def validate_configuration():
    """Validate configuration file."""
    print("Validating configuration...")
    
    try:
        from config import get_default_config
        config = get_default_config()
        print("  ✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"  ✗ Configuration error: {e}")
        return False


def run_quick_test():
    """Run a quick test of the pipeline."""
    print("Running quick pipeline test...")
    
    try:
        from config import get_default_config
        from utils import discover_dataset, create_patient_level_split
        
        config = get_default_config()
        
        # Discover dataset
        data_list = discover_dataset(config.data.data_root, config.data)
        if len(data_list) == 0:
            print("  ✗ No data discovered")
            return False
        
        print(f"  ✓ Discovered {len(data_list)} volumes")
        
        # Test data splitting
        train_data, val_data, test_data = create_patient_level_split(
            data_list,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_seed=42,
        )
        
        print(f"  ✓ Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        # Test DataLoader
        from dataset import create_data_loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            config, train_data, val_data, test_data
        )
        
        print(f"  ✓ DataLoaders created successfully")
        
        # Load one batch
        batch = next(iter(train_loader))
        print(f"  ✓ Successfully loaded batch: shape={batch['image'].shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_next_steps():
    """Print recommended next steps."""
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("\n1. Generate brain masks (IMPORTANT!):")
    print("   - Install HD-BET: pip install HD-BET")
    print("   - Run: python generate_masks.py")
    print("   - Or use SynthStrip/FSL BET")
    
    print("\n2. Run the full pipeline:")
    print("   python example_pipeline.py --preprocess --config default")
    
    print("\n3. Explore interactively:")
    print("   jupyter notebook notebooks/interactive_pipeline.ipynb")
    
    print("\n4. Customize configuration:")
    print("   - Edit config.py")
    print("   - Adjust preprocessing parameters")
    print("   - Set your data paths")
    
    print("\n5. Start model development:")
    print("   - Implement your architecture")
    print("   - Use the DataLoaders from dataset.py")
    print("   - Monitor with TensorBoard")
    
    print("\n" + "="*70)
    print("For detailed documentation, see README.md")
    print("For improvements and suggestions, see IMPROVEMENTS.md")
    print("="*70 + "\n")


def main():
    """Main setup and validation routine."""
    print("="*70)
    print("BRAIN MRI PREPROCESSING PIPELINE - SETUP VALIDATION")
    print("="*70 + "\n")
    
    # Step 1: Check imports
    if not check_imports():
        sys.exit(1)
    
    # Step 2: Check configuration
    if not validate_configuration():
        sys.exit(1)
    
    # Step 3: Check dataset
    from config import get_default_config
    config = get_default_config()
    
    if not check_dataset(config.data.data_root):
        print("\n❌ Dataset validation failed")
        print("Please check your data_root path in config.py")
        sys.exit(1)
    
    print()  # Blank line
    
    # Step 4: Run quick test
    if not run_quick_test():
        print("\n❌ Pipeline test failed")
        print("Please check the error messages above")
        sys.exit(1)
    
    # Success!
    print("\n" + "="*70)
    print("✅ ALL CHECKS PASSED!")
    print("="*70)
    
    print_next_steps()


if __name__ == "__main__":
    main()
