"""
Example script demonstrating Quality Control (QC) and Intensity Harmonization.

This script shows how to:
1. Run quality control on preprocessed data
2. Detect outliers and artifacts
3. Harmonize intensity across 3T and 7T scans
4. Generate QC reports

Usage:
    python example_qc_harmonization.py
"""
import logging
from pathlib import Path

from config import get_default_config
from utils import discover_dataset, setup_logging
from quality_control import PreprocessingQC, QCMetrics
from harmonization import IntensityHarmonizer

logger = logging.getLogger(__name__)


def run_quality_control_example(config):
    """
    Run quality control on preprocessed data.
    
    Args:
        config: MRIConfig instance
    """
    logger.info("="*80)
    logger.info("QUALITY CONTROL PIPELINE")
    logger.info("="*80)
    
    # Discover preprocessed data
    logger.info("Discovering preprocessed data...")
    data_list = discover_dataset(config.data.preprocessed_root, config.data)
    
    if len(data_list) == 0:
        logger.warning("No preprocessed data found. Please run preprocessing first.")
        logger.info("Run: python example_pipeline.py --preprocess")
        return
    
    logger.info(f"Found {len(data_list)} preprocessed scans")
    
    # Run QC metrics on first few samples
    logger.info("\n" + "="*80)
    logger.info("Computing QC metrics on sample scans...")
    logger.info("="*80)
    
    sample_data = data_list[:5]  # First 5 samples
    qc_results = []
    
    for i, data_dict in enumerate(sample_data, 1):
        image_path = data_dict["image"]
        logger.info(f"\n[{i}/{len(sample_data)}] Processing: {image_path.name}")
        
        # Compute QC metrics
        metrics = QCMetrics.compute_metrics(str(image_path))
        
        logger.info(f"  SNR: {metrics['snr']:.2f}")
        logger.info(f"  CNR: {metrics['cnr']:.2f}")
        logger.info(f"  Entropy: {metrics['entropy']:.2f}")
        logger.info(f"  Mean Intensity: {metrics['mean_intensity']:.2f}")
        logger.info(f"  Std Intensity: {metrics['std_intensity']:.2f}")
        logger.info(f"  Has Artifacts: {metrics['has_artifacts']}")
        
        qc_results.append({
            'file': image_path.name,
            'subject': data_dict['subject'],
            'session': data_dict['session'],
            'modality': data_dict['modality'],
            **metrics
        })
    
    # Run comprehensive QC with outlier detection
    logger.info("\n" + "="*80)
    logger.info("Running comprehensive QC with outlier detection...")
    logger.info("="*80)
    
    qc = PreprocessingQC(output_dir=config.logging.qc_dir)
    
    # Get all image paths
    image_paths = [str(d['image']) for d in data_list]
    
    # Run QC
    qc_df = qc.run_qc(image_paths)
    
    # Generate report
    report_path = config.logging.qc_dir / "qc_report.html"
    qc.generate_report(qc_df, str(report_path))
    
    logger.info(f"\n✓ QC completed!")
    logger.info(f"  Total scans: {len(qc_df)}")
    logger.info(f"  Outliers detected: {qc_df['is_outlier'].sum()}")
    logger.info(f"  Scans with artifacts: {qc_df['has_artifacts'].sum()}")
    logger.info(f"\n  Report saved to: {report_path}")
    
    return qc_df


def run_harmonization_example(config):
    """
    Run intensity harmonization across 3T and 7T scans.
    
    Args:
        config: MRIConfig instance
    """
    logger.info("\n" + "="*80)
    logger.info("INTENSITY HARMONIZATION PIPELINE")
    logger.info("="*80)
    
    # Discover preprocessed data
    logger.info("Discovering preprocessed data...")
    data_list = discover_dataset(config.data.preprocessed_root, config.data)
    
    if len(data_list) == 0:
        logger.warning("No preprocessed data found. Please run preprocessing first.")
        return
    
    # Separate 3T and 7T scans
    scans_3t = [str(d['image']) for d in data_list if d['session'] == 'ses-1']
    scans_7t = [str(d['image']) for d in data_list if d['session'] == 'ses-2']
    
    logger.info(f"\nFound {len(scans_3t)} 3T scans (ses-1)")
    logger.info(f"Found {len(scans_7t)} 7T scans (ses-2)")
    
    if len(scans_3t) == 0 or len(scans_7t) == 0:
        logger.warning("Need both 3T and 7T scans for harmonization.")
        return
    
    # Test different harmonization methods
    methods = ['histogram', 'zscore', 'quantile']
    
    for method in methods:
        logger.info(f"\n{'-'*80}")
        logger.info(f"Testing {method.upper()} harmonization method")
        logger.info(f"{'-'*80}")
        
        # Create harmonizer
        harmonizer = IntensityHarmonizer(method=method)
        
        # Fit on 3T scans (reference distribution)
        logger.info("Fitting harmonizer on 3T scans...")
        harmonizer.fit(scans_3t)
        
        # Save harmonizer
        harmonizer_path = config.data.cache_dir / f"harmonizer_{method}.pkl"
        harmonizer.save(str(harmonizer_path))
        logger.info(f"✓ Harmonizer saved to: {harmonizer_path}")
        
        # Transform a 7T scan
        sample_7t = scans_7t[0]
        logger.info(f"\nTransforming sample 7T scan: {Path(sample_7t).name}")
        
        harmonized_data = harmonizer.transform(sample_7t)
        
        # Save harmonized scan
        output_path = config.data.output_root / f"harmonized_{method}_sample.nii.gz"
        harmonizer.save_harmonized(harmonized_data, str(output_path))
        logger.info(f"✓ Harmonized scan saved to: {output_path}")
        
        # Compute before/after statistics
        import nibabel as nib
        import numpy as np
        
        original_data = nib.load(sample_7t).get_fdata()
        
        logger.info(f"\nIntensity statistics:")
        logger.info(f"  Original  - Mean: {np.mean(original_data):.2f}, Std: {np.std(original_data):.2f}")
        logger.info(f"  Harmonized - Mean: {np.mean(harmonized_data):.2f}, Std: {np.std(harmonized_data):.2f}")
    
    logger.info("\n" + "="*80)
    logger.info("✓ Harmonization examples completed!")
    logger.info("="*80)


def main():
    """Main function."""
    # Setup
    config = get_default_config()
    setup_logging(config.logging)
    
    logger.info("="*80)
    logger.info("QUALITY CONTROL AND HARMONIZATION EXAMPLE")
    logger.info("="*80)
    
    # Run QC
    try:
        qc_df = run_quality_control_example(config)
        
        if qc_df is not None:
            # Show summary
            logger.info("\nQC Summary Statistics:")
            logger.info(f"  SNR:     Mean={qc_df['snr'].mean():.2f}, Std={qc_df['snr'].std():.2f}")
            logger.info(f"  CNR:     Mean={qc_df['cnr'].mean():.2f}, Std={qc_df['cnr'].std():.2f}")
            logger.info(f"  Entropy: Mean={qc_df['entropy'].mean():.2f}, Std={qc_df['entropy'].std():.2f}")
    except Exception as e:
        logger.error(f"QC pipeline failed: {e}", exc_info=True)
    
    # Run Harmonization
    try:
        run_harmonization_example(config)
    except Exception as e:
        logger.error(f"Harmonization pipeline failed: {e}", exc_info=True)
    
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE COMPLETED")
    logger.info("="*80)
    logger.info(f"\nOutput locations:")
    logger.info(f"  QC reports: {config.logging.qc_dir}")
    logger.info(f"  Harmonized scans: {config.data.output_root}")
    logger.info(f"  Harmonizer models: {config.data.cache_dir}")


if __name__ == "__main__":
    main()
