#!/usr/bin/env python
"""
Simple preprocessing script for MRI data.
Run this to preprocess all data before training.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import MRIPreprocessor
from src.config import get_default_config
from src.utils import discover_dataset
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run preprocessing on all discovered data."""
    
    print("=" * 70)
    print("STEP 3: PREPROCESSING - N4 Correction + Skull Stripping + Normalization")
    print("=" * 70)
    print("\n⏱️  Expected time: 1-2 hours (CPU)")
    print("📂 Output directory: preprocessed/\n")
    
    # Load configuration
    config = get_default_config()
    preprocessor = MRIPreprocessor(config.preprocessing)
    
    # Discover all files
    logger.info(f"Discovering dataset in: {config.data.data_root}")
    dataset_info = discover_dataset(config.data.data_root, config.data)
    
    print(f"\n✓ Found {len(dataset_info)} files to process")
    print(f"  - Input:  {config.data.data_root}")
    print(f"  - Output: {config.data.output_root}\n")
    
    if len(dataset_info) == 0:
        print("❌ No data found! Check your Nifti folder structure.")
        return
    
    # Show sample of what will be processed
    print("Sample files to process:")
    for item in dataset_info[:5]:
        print(f"  - {item['subject']} / {item['session']} / {item['modality']}")
    if len(dataset_info) > 5:
        print(f"  ... and {len(dataset_info) - 5} more\n")
    
    input("\n⏸️  Press Enter to start preprocessing (or Ctrl+C to cancel)...")
    
    # Process each file
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    print("\n" + "=" * 70)
    print("Processing files...")
    print("=" * 70 + "\n")
    
    for file_info in tqdm(dataset_info, desc="Preprocessing", unit="file"):
        try:
            input_path = Path(file_info['image'])  # Fixed: use 'image' key instead of 'file_path'
            
            # Create output path
            relative_path = input_path.relative_to(config.data.data_root)
            output_filename = input_path.name.replace('.nii.gz', '_preprocessed.nii.gz')
            output_path = config.data.output_root / relative_path.parent / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if already exists
            if output_path.exists():
                logger.debug(f"Skipping {output_path.name} (already exists)")
                skipped_count += 1
                continue
            
            # Preprocess
            logger.info(f"Processing: {input_path.name}")
            result = preprocessor.preprocess_single(input_path, output_path)
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {file_info['subject']}: {e}")
            error_count += 1
            continue
    
    # Summary
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"✓ Processed: {processed_count} files")
    print(f"⏭️  Skipped:   {skipped_count} files (already existed)")
    print(f"❌ Errors:    {error_count} files")
    print(f"📂 Output:    {config.data.output_root}")
    print("=" * 70)
    
    if processed_count > 0 or skipped_count > 0:
        print("\n✅ Ready for next step: REGISTRATION")
        print("   Run: python scripts/register_dataset.py --input preprocessed --output preprocessed_registered\n")
    else:
        print("\n❌ No files were processed. Check the errors above.\n")


if __name__ == "__main__":
    main()
