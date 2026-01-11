"""
CLI script for running enhanced Quality Control on preprocessed data.
Generates alignment visualizations, mask quality reports, and integrates MRIQC metrics.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import pandas as pd

from src.quality_control import (
    AlignmentQC,
    MaskQualityValidator,
    MRIQCParser,
    PreprocessingQC,
)
from src.bids import discover_bids_files, create_3t_7t_pairs


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_alignment_qc(
    pairs_manifest: Path,
    output_dir: Path,
    max_samples: int = -1,
) -> List[Dict]:
    """
    Run alignment QC on 3T-7T pairs.
    
    Args:
        pairs_manifest: Path to pairs.csv or pairs.jsonl from preprocessing
        output_dir: Directory to save QC visualizations
        max_samples: Maximum number of pairs to process (-1 for all)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load pairs
    if pairs_manifest.suffix == ".csv":
        pairs_df = pd.read_csv(pairs_manifest)
        pairs = pairs_df.to_dict("records")
    else:
        pairs = []
        with open(pairs_manifest) as f:
            for line in f:
                pairs.append(json.loads(line.strip()))
    
    if max_samples > 0:
        pairs = pairs[:max_samples]
    
    results = []
    
    for pair in pairs:
        subject = pair.get("subject", "unknown")
        input_3t = Path(pair["input_3t"])
        target_7t = Path(pair["target_7t"])
        
        if not input_3t.exists() or not target_7t.exists():
            logger.warning(f"Missing files for {subject}: {input_3t} or {target_7t}")
            continue
        
        logger.info(f"Processing alignment QC for {subject}")
        
        try:
            # Load images
            img_3t = nib.load(str(input_3t))
            img_7t = nib.load(str(target_7t))
            
            arr_3t = img_3t.get_fdata().astype(np.float32)
            arr_7t = img_7t.get_fdata().astype(np.float32)
            
            # Compute alignment score
            alignment_score = AlignmentQC.compute_alignment_score(arr_3t, arr_7t)
            
            # Generate visualization
            viz_path = output_dir / f"{subject}_alignment_qc.png"
            AlignmentQC.save_alignment_visualization(
                arr_3t, arr_7t, viz_path,
                title=f"{subject} - 3T-7T Alignment QC"
            )
            
            result = {
                "subject": subject,
                "input_3t": str(input_3t),
                "target_7t": str(target_7t),
                "visualization": str(viz_path),
                **alignment_score,
            }
            results.append(result)
            
            logger.info(f"  NCC: {alignment_score['ncc']:.3f}, Edge MSE: {alignment_score['edge_mse']:.4f}, Quality: {alignment_score['alignment_quality']}")
            
        except Exception as e:
            logger.error(f"Failed to process {subject}: {e}")
            results.append({
                "subject": subject,
                "error": str(e),
            })
    
    # Save results
    results_path = output_dir / "alignment_qc_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Alignment QC results saved to {results_path}")
    return results


def run_mask_qc(
    manifest: Path,
    output_dir: Path,
) -> List[Dict]:
    """
    Run mask quality validation on preprocessed volumes.
    
    Args:
        manifest: Path to manifest.csv or manifest.jsonl from preprocessing
        output_dir: Directory to save QC results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    if manifest.suffix == ".csv":
        df = pd.read_csv(manifest)
        entries = df.to_dict("records")
    else:
        entries = []
        with open(manifest) as f:
            for line in f:
                entries.append(json.loads(line.strip()))
    
    results = []
    
    for entry in entries:
        if entry.get("status") != "ok":
            continue
        
        subject = entry.get("subject", "unknown")
        session = entry.get("session", "")
        modality = entry.get("modality", "")
        output_path = Path(entry["output_path"])
        mask_path = Path(entry.get("mask_output_path", ""))
        
        if not mask_path.exists():
            continue
        
        logger.info(f"Validating mask for {subject}/{session}/{modality}")
        
        try:
            # Load mask
            mask_img = nib.load(str(mask_path))
            mask_arr = mask_img.get_fdata().astype(np.uint8)
            voxel_spacing = tuple(mask_img.header.get_zooms()[:3])
            
            # Load image for coverage check
            img_arr = None
            if output_path.exists():
                img_arr = nib.load(str(output_path)).get_fdata().astype(np.float32)
            
            # Validate mask
            validation = MaskQualityValidator.validate_mask(mask_arr, voxel_spacing, img_arr)
            
            result = {
                "subject": subject,
                "session": session,
                "modality": modality,
                "mask_path": str(mask_path),
                **validation,
            }
            results.append(result)
            
            if validation["warnings"]:
                logger.warning(f"  Warnings: {validation['warnings']}")
            else:
                logger.info(f"  Mask OK: volume={validation['volume_mm3']:.0f}mm³")
            
        except Exception as e:
            logger.error(f"Failed to validate mask for {subject}: {e}")
            results.append({
                "subject": subject,
                "session": session,
                "modality": modality,
                "error": str(e),
            })
    
    # Save results
    results_path = output_dir / "mask_qc_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Mask QC results saved to {results_path}")
    return results


def run_mriqc_integration(
    manifest: Path,
    derivatives_root: Path,
    output_dir: Path,
) -> List[Dict]:
    """
    Integrate MRIQC metrics into QC report.
    
    Args:
        manifest: Path to manifest from preprocessing
        derivatives_root: BIDS derivatives root (containing mriqc folder)
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    if manifest.suffix == ".csv":
        df = pd.read_csv(manifest)
        entries = df.to_dict("records")
    else:
        entries = []
        with open(manifest) as f:
            for line in f:
                entries.append(json.loads(line.strip()))
    
    results = []
    found_mriqc = 0
    
    for entry in entries:
        subject = entry.get("subject", "unknown")
        input_path = Path(entry.get("input_path", ""))
        
        # Try to find MRIQC for original input
        mriqc_metrics = MRIQCParser.get_metrics_for_image(input_path, derivatives_root)
        
        if mriqc_metrics:
            found_mriqc += 1
            result = {
                "subject": subject,
                "session": entry.get("session", ""),
                "modality": entry.get("modality", ""),
                **mriqc_metrics,
            }
            results.append(result)
            
            quality = mriqc_metrics.get("mriqc_overall_quality", "unknown")
            logger.info(f"  {subject}: MRIQC found, quality={quality}")
    
    logger.info(f"Found MRIQC data for {found_mriqc}/{len(entries)} volumes")
    
    # Save results
    results_path = output_dir / "mriqc_integration.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"MRIQC integration results saved to {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run enhanced Quality Control on preprocessed MRI data"
    )
    parser.add_argument(
        "--preprocessed-root",
        type=str,
        required=True,
        help="Root directory of preprocessed data (e.g., derivatives/topobrain-preproc)",
    )
    parser.add_argument(
        "--derivatives-root",
        type=str,
        default=None,
        help="BIDS derivatives root for MRIQC (defaults to parent of preprocessed-root)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for QC results (defaults to preprocessed-root/qc)",
    )
    parser.add_argument(
        "--alignment-only",
        action="store_true",
        help="Run only alignment QC",
    )
    parser.add_argument(
        "--mask-only",
        action="store_true",
        help="Run only mask QC",
    )
    parser.add_argument(
        "--mriqc-only",
        action="store_true",
        help="Run only MRIQC integration",
    )
    parser.add_argument(
        "--max-alignment-samples",
        type=int,
        default=-1,
        help="Maximum number of pairs for alignment QC (-1 for all)",
    )
    
    args = parser.parse_args()
    
    preprocessed_root = Path(args.preprocessed_root)
    derivatives_root = Path(args.derivatives_root) if args.derivatives_root else preprocessed_root.parent
    output_dir = Path(args.output_dir) if args.output_dir else preprocessed_root / "qc"
    
    manifest_path = preprocessed_root / "manifest.csv"
    if not manifest_path.exists():
        manifest_path = preprocessed_root / "manifest.jsonl"
    
    pairs_path = preprocessed_root / "pairs.csv"
    if not pairs_path.exists():
        pairs_path = preprocessed_root / "pairs.jsonl"
    
    run_all = not (args.alignment_only or args.mask_only or args.mriqc_only)
    
    # Run alignment QC
    if run_all or args.alignment_only:
        if pairs_path.exists():
            logger.info("=" * 50)
            logger.info("Running Alignment QC...")
            logger.info("=" * 50)
            run_alignment_qc(pairs_path, output_dir / "alignment", args.max_alignment_samples)
        else:
            logger.warning(f"Pairs manifest not found: {pairs_path}")
    
    # Run mask QC
    if run_all or args.mask_only:
        if manifest_path.exists():
            logger.info("=" * 50)
            logger.info("Running Mask Quality Validation...")
            logger.info("=" * 50)
            run_mask_qc(manifest_path, output_dir / "masks")
        else:
            logger.warning(f"Manifest not found: {manifest_path}")
    
    # Run MRIQC integration
    if run_all or args.mriqc_only:
        if manifest_path.exists():
            logger.info("=" * 50)
            logger.info("Running MRIQC Integration...")
            logger.info("=" * 50)
            run_mriqc_integration(manifest_path, derivatives_root, output_dir / "mriqc")
        else:
            logger.warning(f"Manifest not found: {manifest_path}")
    
    logger.info("=" * 50)
    logger.info(f"QC complete. Results saved to {output_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
