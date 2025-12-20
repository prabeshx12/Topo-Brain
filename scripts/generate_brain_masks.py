"""
Script to generate brain masks for all subjects in the dataset.

This is a CRITICAL preprocessing step. The current pipeline uses simple
thresholding as a fallback, which is unreliable for production use.

This script provides wrappers for multiple brain extraction tools:
1. HD-BET (recommended) - Deep learning-based, works well with 3T and 7T
2. SynthStrip (FreeSurfer) - Fast and accurate
3. FSL BET - Classic tool (requires FSL installation)

Usage:
    # Using HD-BET (recommended)
    python generate_brain_masks.py --method hd-bet --device cuda
    
    # Using SynthStrip
    python generate_brain_masks.py --method synthstrip
    
    # Using FSL BET
    python generate_brain_masks.py --method fsl-bet
"""

import argparse
import logging
from pathlib import Path
import subprocess
import sys

from tqdm import tqdm

from src.config import get_default_config


logger = logging.getLogger(__name__)


def generate_masks_hd_bet(input_files, device='cuda', mode='accurate'):
    """
    Generate brain masks using HD-BET.
    
    Installation:
        pip install HD-BET
        
    Reference:
        Isensee et al. (2019) - Automated brain extraction of multisequence MRI 
        using artificial neural networks
    
    Args:
        input_files: List of input NIfTI paths
        device: 'cuda' or 'cpu'
        mode: 'accurate' or 'fast'
    """
    try:
        import torch
        from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict
        from HD_BET.checkpoint_download import maybe_download_parameters
    except ImportError as e:
        logger.error(f"HD-BET not installed. Error: {e}")
        logger.error("Install with: pip install HD-BET")
        return False
    
    logger.info(f"Using HD-BET with device={device}, mode={mode}")
    
    # Download parameters if needed
    logger.info("Checking/downloading HD-BET parameters...")
    maybe_download_parameters()
    
    # Set device
    torch_device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    
    # Initialize predictor once (more efficient)
    use_tta = (mode == 'accurate')
    logger.info("Initializing HD-BET predictor...")
    predictor = get_hdbet_predictor(use_tta=use_tta, device=torch_device, verbose=False)
    
    for input_path in tqdm(input_files, desc="HD-BET Processing"):
        input_path = Path(input_path)
        
        # Create output paths
        output_brain = input_path.parent / input_path.name.replace("_defaced", "_brain")
        output_mask = input_path.parent / input_path.name.replace("_defaced", "_brain_mask")
        
        # Skip if already processed
        if output_mask.exists():
            logger.debug(f"Skipping {input_path.name} (mask already exists)")
            continue
        
        try:
            # Run HD-BET prediction
            hdbet_predict(
                str(input_path),
                str(output_brain),
                predictor=predictor,
                keep_brain_mask=True,
                compute_brain_extracted_image=True
            )
            
            # Rename the mask file from _bet.nii.gz to _brain_mask.nii.gz
            bet_mask = output_brain.parent / (output_brain.stem.replace('.nii', '') + '_bet.nii.gz')
            if bet_mask.exists():
                bet_mask.rename(output_mask)
            
            logger.info(f"✓ Processed: {input_path.name}")
            
        except Exception as e:
            logger.error(f"✗ Failed to process {input_path.name}: {e}")
            continue
    
    return True


def generate_masks_synthstrip(input_files):
    """
    Generate brain masks using SynthStrip (FreeSurfer).
    
    Installation:
        Requires FreeSurfer 7.3+
        https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
        
    Reference:
        Hoopes et al. (2022) - SynthStrip: skull-stripping for any brain image
    
    Args:
        input_files: List of input NIfTI paths
    """
    # Check if SynthStrip is available
    try:
        result = subprocess.run(
            ['mri_synthstrip', '--help'],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise FileNotFoundError
    except FileNotFoundError:
        logger.error("SynthStrip not found. Please install FreeSurfer 7.3+")
        return False
    
    logger.info("Using SynthStrip (FreeSurfer)")
    
    for input_path in tqdm(input_files, desc="SynthStrip Processing"):
        input_path = Path(input_path)
        
        # Create output paths
        output_brain = input_path.parent / input_path.name.replace("_defaced", "_brain")
        output_mask = input_path.parent / input_path.name.replace("_defaced", "_brain_mask")
        
        # Skip if already processed
        if output_mask.exists():
            logger.debug(f"Skipping {input_path.name} (mask already exists)")
            continue
        
        try:
            # Run SynthStrip
            cmd = [
                'mri_synthstrip',
                '-i', str(input_path),
                '-o', str(output_brain),
                '-m', str(output_mask),
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"✓ Processed: {input_path.name}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to process {input_path.name}: {e}")
            continue
    
    return True


def generate_masks_fsl_bet(input_files, fractional_intensity=0.5):
    """
    Generate brain masks using FSL BET.
    
    Installation:
        Requires FSL
        https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
        
    Reference:
        Smith (2002) - Fast robust automated brain extraction
    
    Args:
        input_files: List of input NIfTI paths
        fractional_intensity: BET fractional intensity threshold (0-1)
    """
    # Check if FSL BET is available
    try:
        result = subprocess.run(
            ['bet'],
            capture_output=True,
            text=True
        )
    except FileNotFoundError:
        logger.error("FSL BET not found. Please install FSL")
        return False
    
    logger.info(f"Using FSL BET with fractional_intensity={fractional_intensity}")
    
    for input_path in tqdm(input_files, desc="FSL BET Processing"):
        input_path = Path(input_path)
        
        # Create output paths
        output_brain = input_path.parent / input_path.name.replace("_defaced", "_brain")
        output_mask = input_path.parent / input_path.name.replace("_defaced", "_brain_mask")
        
        # Skip if already processed
        if output_mask.exists():
            logger.debug(f"Skipping {input_path.name} (mask already exists)")
            continue
        
        try:
            # Run BET
            cmd = [
                'bet',
                str(input_path),
                str(output_brain),
                '-f', str(fractional_intensity),
                '-m',  # Generate mask
                '-R',  # Robust brain centre estimation
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"✓ Processed: {input_path.name}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to process {input_path.name}: {e}")
            continue
    
    return True


def verify_masks(input_files):
    """
    Verify that masks were generated successfully.
    
    Args:
        input_files: List of input NIfTI paths
        
    Returns:
        Dictionary with verification statistics
    """
    stats = {
        'total': len(input_files),
        'with_mask': 0,
        'without_mask': [],
    }
    
    for input_path in input_files:
        input_path = Path(input_path)
        mask_path = input_path.parent / input_path.name.replace("_defaced", "_brain_mask")
        
        if mask_path.exists():
            stats['with_mask'] += 1
        else:
            stats['without_mask'].append(str(input_path))
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate brain masks for all subjects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using HD-BET (recommended)
  python generate_brain_masks.py --method hd-bet --device cuda
  
  # Using SynthStrip (FreeSurfer)
  python generate_brain_masks.py --method synthstrip
  
  # Using FSL BET
  python generate_brain_masks.py --method fsl-bet --bet-threshold 0.5
        """
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='hd-bet',
        choices=['hd-bet', 'synthstrip', 'fsl-bet'],
        help='Brain extraction method to use'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for HD-BET (only used with --method hd-bet)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='accurate',
        choices=['accurate', 'fast'],
        help='HD-BET mode (only used with --method hd-bet)'
    )
    
    parser.add_argument(
        '--bet-threshold',
        type=float,
        default=0.5,
        help='FSL BET fractional intensity threshold (only used with --method fsl-bet)'
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default=None,
        help='Data root directory (default: from config.py)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Get data root
    if args.data_root is not None:
        data_root = Path(args.data_root)
    else:
        config = get_default_config()
        data_root = config.data.data_root
    
    logger.info(f"Data root: {data_root}")
    
    # Find all input files
    input_files = sorted(data_root.glob("**/*_defaced.nii.gz"))
    
    if len(input_files) == 0:
        logger.error(f"No *_defaced.nii.gz files found in {data_root}")
        sys.exit(1)
    
    logger.info(f"Found {len(input_files)} files to process")
    
    # Generate masks based on selected method
    success = False
    
    if args.method == 'hd-bet':
        success = generate_masks_hd_bet(
            input_files,
            device=args.device,
            mode=args.mode
        )
    elif args.method == 'synthstrip':
        success = generate_masks_synthstrip(input_files)
    elif args.method == 'fsl-bet':
        success = generate_masks_fsl_bet(
            input_files,
            fractional_intensity=args.bet_threshold
        )
    
    if not success:
        logger.error("Brain mask generation failed")
        sys.exit(1)
    
    # Verify masks
    stats = verify_masks(input_files)
    
    logger.info("\n" + "="*70)
    logger.info("VERIFICATION RESULTS")
    logger.info("="*70)
    logger.info(f"Total files: {stats['total']}")
    logger.info(f"Files with masks: {stats['with_mask']}")
    logger.info(f"Files without masks: {len(stats['without_mask'])}")
    
    if stats['without_mask']:
        logger.warning("\nFiles without masks:")
        for path in stats['without_mask']:
            logger.warning(f"  - {path}")
    
    if stats['with_mask'] == stats['total']:
        logger.info("\n✓ All files have brain masks!")
    else:
        logger.warning(f"\n⚠ {len(stats['without_mask'])} files are missing masks")
    
    logger.info("="*70)
    
    # Next steps
    logger.info("\nNEXT STEPS:")
    logger.info("1. Visually inspect a few masks to verify quality")
    logger.info("2. Run the preprocessing pipeline: python example_pipeline.py --preprocess")
    logger.info("3. Start model development!")


if __name__ == "__main__":
    main()
