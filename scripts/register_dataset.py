"""
Register all 3T volumes to corresponding 7T volumes.
This script should be run BEFORE training the GAN to ensure spatial alignment.

Usage:
    python scripts/register_dataset.py --input preprocessed --output preprocessed_registered
"""
import argparse
import logging
from pathlib import Path
import sys
import json
from tqdm import tqdm
import shutil

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import ImageRegistration
from src.utils import setup_logging, discover_dataset

logger = logging.getLogger(__name__)


def register_dataset(
    preprocessed_dir: Path,
    output_dir: Path,
    registration_type: str = "rigid",
    validate: bool = True,
):
    """
    Register all 3T-7T pairs in dataset.
    
    Args:
        preprocessed_dir: Directory with preprocessed data
        output_dir: Where to save registered data
        registration_type: "rigid" or "affine"
        validate: Whether to compute validation metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize registration
    registrator = ImageRegistration(
        registration_type=registration_type,
        num_iterations=100,
        learning_rate=1.0,
    )
    
    # Find all subjects
    subjects = sorted([d for d in preprocessed_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    
    if len(subjects) == 0:
        logger.error(f"No subjects found in {preprocessed_dir}")
        return
    
    logger.info(f"Found {len(subjects)} subjects to register")
    
    registration_report = []
    
    for subject_dir in tqdm(subjects, desc="Registering subjects"):
        subject_id = subject_dir.name
        
        # Find 3T and 7T files (ses-1 = 3T, ses-2 = 7T)
        # Try both filename patterns: with and without '_defaced'
        t1w_3t = subject_dir / "ses-1" / "anat" / f"{subject_id}_ses-1_T1w_defaced_preprocessed.nii.gz"
        if not t1w_3t.exists():
            t1w_3t = subject_dir / "ses-1" / "anat" / f"{subject_id}_ses-1_T1w_preprocessed.nii.gz"
        
        t1w_7t = subject_dir / "ses-2" / "anat" / f"{subject_id}_ses-2_T1w_defaced_preprocessed.nii.gz"
        if not t1w_7t.exists():
            t1w_7t = subject_dir / "ses-2" / "anat" / f"{subject_id}_ses-2_T1w_preprocessed.nii.gz"
        
        t2w_3t = subject_dir / "ses-1" / "anat" / f"{subject_id}_ses-1_T2w_defaced_preprocessed.nii.gz"
        if not t2w_3t.exists():
            t2w_3t = subject_dir / "ses-1" / "anat" / f"{subject_id}_ses-1_T2w_preprocessed.nii.gz"
        
        t2w_7t = subject_dir / "ses-2" / "anat" / f"{subject_id}_ses-2_T2w_defaced_preprocessed.nii.gz"
        if not t2w_7t.exists():
            t2w_7t = subject_dir / "ses-2" / "anat" / f"{subject_id}_ses-2_T2w_preprocessed.nii.gz"
        
        subject_report = {
            'subject': subject_id,
            'status': 'success',
            'modalities': {}
        }
        
        # Register T1w
        if t1w_3t.exists() and t1w_7t.exists():
            try:
                output_3t = output_dir / subject_id / "ses-1" / "anat" / f"{subject_id}_ses-1_T1w_registered.nii.gz"
                output_7t = output_dir / subject_id / "ses-2" / "anat" / f"{subject_id}_ses-2_T1w_preprocessed.nii.gz"
                
                # Register 3T to 7T space
                registrator.register(t1w_3t, t1w_7t, output_3t)
                
                # Copy 7T (reference) to output
                output_7t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(t1w_7t, output_7t)
                
                # Validate registration
                if validate:
                    checkerboard_path = output_dir / subject_id / f"{subject_id}_T1w_checkerboard.nii.gz"
                    metrics = registrator.validate_registration(
                        output_3t, output_7t, checkerboard_path
                    )
                    subject_report['modalities']['T1w'] = metrics
                    logger.info(f"{subject_id} T1w: Correlation={metrics['correlation']:.3f}, NCC={metrics['normalized_cross_correlation']:.3f}")
                
            except Exception as e:
                logger.error(f"Registration failed for {subject_id} T1w: {e}")
                subject_report['status'] = 'partial_failure'
                subject_report['modalities']['T1w'] = {'error': str(e)}
        else:
            logger.warning(f"Missing T1w files for {subject_id}")
        
        # Register T2w
        if t2w_3t.exists() and t2w_7t.exists():
            try:
                output_3t = output_dir / subject_id / "ses-1" / "anat" / f"{subject_id}_ses-1_T2w_registered.nii.gz"
                output_7t = output_dir / subject_id / "ses-2" / "anat" / f"{subject_id}_ses-2_T2w_preprocessed.nii.gz"
                
                # Register 3T to 7T space
                registrator.register(t2w_3t, t2w_7t, output_3t)
                
                # Copy 7T (reference) to output
                output_7t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(t2w_7t, output_7t)
                
                # Validate registration
                if validate:
                    checkerboard_path = output_dir / subject_id / f"{subject_id}_T2w_checkerboard.nii.gz"
                    metrics = registrator.validate_registration(
                        output_3t, output_7t, checkerboard_path
                    )
                    subject_report['modalities']['T2w'] = metrics
                    logger.info(f"{subject_id} T2w: Correlation={metrics['correlation']:.3f}, NCC={metrics['normalized_cross_correlation']:.3f}")
                
            except Exception as e:
                logger.error(f"Registration failed for {subject_id} T2w: {e}")
                subject_report['status'] = 'partial_failure'
                subject_report['modalities']['T2w'] = {'error': str(e)}
        else:
            logger.warning(f"Missing T2w files for {subject_id}")
        
        registration_report.append(subject_report)
    
    # Save report
    report_path = output_dir / "registration_report.json"
    with open(report_path, 'w') as f:
        json.dump(registration_report, f, indent=2)
    
    logger.info(f"Registration complete. Report saved to {report_path}")
    
    # Summary statistics
    success_count = sum(1 for r in registration_report if r['status'] == 'success')
    logger.info(f"\nSummary:")
    logger.info(f"  Successfully registered: {success_count}/{len(subjects)} subjects")
    
    if validate:
        # Compute average metrics
        t1w_corrs = [r['modalities'].get('T1w', {}).get('correlation', 0) 
                     for r in registration_report if 'correlation' in r['modalities'].get('T1w', {})]
        t2w_corrs = [r['modalities'].get('T2w', {}).get('correlation', 0) 
                     for r in registration_report if 'correlation' in r['modalities'].get('T2w', {})]
        
        if t1w_corrs:
            logger.info(f"  Average T1w correlation: {sum(t1w_corrs)/len(t1w_corrs):.3f}")
        if t2w_corrs:
            logger.info(f"  Average T2w correlation: {sum(t2w_corrs)/len(t2w_corrs):.3f}")
        
        logger.info(f"\nQuality check:")
        logger.info(f"  Good registration: correlation > 0.7")
        logger.info(f"  Acceptable: 0.5 - 0.7")
        logger.info(f"  Poor: < 0.5 (should be re-examined)")


def main():
    parser = argparse.ArgumentParser(description="Register 3T to 7T MRI volumes")
    parser.add_argument(
        '--input',
        type=str,
        default='preprocessed',
        help='Input directory with preprocessed data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='preprocessed_registered',
        help='Output directory for registered data'
    )
    parser.add_argument(
        '--registration-type',
        type=str,
        default='rigid',
        choices=['rigid', 'affine'],
        help='Type of registration (rigid=6 DOF, affine=12 DOF)'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation metrics (faster)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging (simple version without config)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Convert paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    logger.info(f"Starting dataset registration...")
    logger.info(f"  Input: {input_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Registration type: {args.registration_type}")
    logger.info(f"  Validate: {not args.no_validate}")
    
    # Run registration
    register_dataset(
        preprocessed_dir=input_dir,
        output_dir=output_dir,
        registration_type=args.registration_type,
        validate=not args.no_validate,
    )
    
    logger.info("✓ Dataset registration complete!")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review registration_report.json in {output_dir}")
    logger.info(f"  2. Inspect checkerboard visualizations for quality")
    logger.info(f"  3. Update config.py to use registered data path")
    logger.info(f"  4. Run training: python scripts/train_gan.py")


if __name__ == "__main__":
    main()
