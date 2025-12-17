"""
End-to-end example of the brain MRI preprocessing and data loading pipeline.

This script demonstrates:
1. Dataset discovery
2. Patient-level train/val/test split
3. Preprocessing pipeline execution
4. DataLoader creation
5. Statistics computation
6. Visualization

Usage:
    python example_pipeline.py [--preprocess] [--config {default,highres,fast}]
"""
import argparse
import logging
from pathlib import Path
import sys

import torch

from config import (
    get_default_config,
    get_highres_config,
    get_fast_config,
    MRIConfig,
)
from preprocessing import MRIPreprocessor
from dataset import (
    BrainMRIDataset,
    create_data_loaders,
    save_data_split,
    load_data_split,
    compute_dataset_statistics,
)
from utils import (
    setup_logging,
    discover_dataset,
    create_patient_level_split,
    visualize_sample,
    visualize_batch,
    compute_and_save_statistics,
    verify_preprocessing,
    set_random_seeds,
)


logger = logging.getLogger(__name__)


def run_preprocessing_pipeline(config: MRIConfig, force_reprocess: bool = False):
    """
    Run the complete preprocessing pipeline.
    
    Args:
        config: MRI configuration
        force_reprocess: Whether to reprocess existing files
    """
    logger.info("="*80)
    logger.info("STEP 1: PREPROCESSING PIPELINE")
    logger.info("="*80)
    
    # Discover all images in the dataset
    data_list = discover_dataset(config.data.data_root, config.data)
    
    if len(data_list) == 0:
        logger.error("No data found! Check your data_root path.")
        return []
    
    # Initialize preprocessor
    preprocessor = MRIPreprocessor(config.preprocessing)
    
    # Collect all image paths
    image_paths = [item["image"] for item in data_list]
    
    # Preprocess all images
    preprocessed_paths = preprocessor.preprocess_dataset(
        image_paths,
        config.data.output_root,
        save_intermediate=False,  # Set to True to save intermediate steps
    )
    
    logger.info(f"Preprocessed {len(preprocessed_paths)} volumes")
    
    # Update data_list with preprocessed paths
    preprocessed_data_list = []
    for original_item, preprocessed_path in zip(data_list, preprocessed_paths):
        updated_item = original_item.copy()
        updated_item["image"] = preprocessed_path
        updated_item["original_image"] = original_item["image"]
        preprocessed_data_list.append(updated_item)
    
    # Verify preprocessing on a sample
    if len(preprocessed_data_list) > 0 and config.logging.save_visualizations:
        logger.info("Creating preprocessing verification plots...")
        verify_dir = config.logging.log_dir / "preprocessing_verification"
        verify_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify first few samples
        for i in range(min(3, len(preprocessed_data_list))):
            verify_preprocessing(
                preprocessed_data_list[i]["original_image"],
                preprocessed_data_list[i]["image"],
                verify_dir,
            )
    
    return preprocessed_data_list


def run_data_split_pipeline(config: MRIConfig, data_list: list):
    """
    Create and save patient-level data splits.
    
    Args:
        config: MRI configuration
        data_list: List of data samples
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    logger.info("="*80)
    logger.info("STEP 2: DATA SPLITTING")
    logger.info("="*80)
    
    # Check if split already exists
    split_path = config.data.cache_dir / "data_split.json"
    
    if split_path.exists():
        logger.info(f"Loading existing data split from {split_path}")
        train_data, val_data, test_data = load_data_split(split_path)
    else:
        logger.info("Creating new patient-level split...")
        
        # Set random seeds for reproducibility
        set_random_seeds(config.split.random_seed)
        
        # Create split
        train_data, val_data, test_data = create_patient_level_split(
            data_list,
            config.split.train_ratio,
            config.split.val_ratio,
            config.split.test_ratio,
            random_seed=config.split.random_seed,
        )
        
        # Save split for reproducibility
        save_data_split(train_data, val_data, test_data, split_path)
    
    return train_data, val_data, test_data


def run_dataloader_pipeline(
    config: MRIConfig,
    train_data: list,
    val_data: list,
    test_data: list,
):
    """
    Create DataLoaders and compute statistics.
    
    Args:
        config: MRI configuration
        train_data: Training data list
        val_data: Validation data list
        test_data: Test data list
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info("="*80)
    logger.info("STEP 3: DATALOADER CREATION")
    logger.info("="*80)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config, train_data, val_data, test_data
    )
    
    # Compute and save statistics
    if config.logging.save_statistics:
        logger.info("Computing dataset statistics...")
        
        stats_dir = config.logging.log_dir / "statistics"
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute statistics for each split
        train_stats = compute_and_save_statistics(
            train_loader,
            stats_dir / "train_statistics.json",
            max_samples=50,  # Limit for speed
        )
        
        val_stats = compute_and_save_statistics(
            val_loader,
            stats_dir / "val_statistics.json",
            max_samples=None,  # Use all validation data
        )
        
        test_stats = compute_and_save_statistics(
            test_loader,
            stats_dir / "test_statistics.json",
            max_samples=None,  # Use all test data
        )
        
        logger.info("Statistics summary:")
        logger.info(f"  Train - Mean: {train_stats.get('mean', 0):.4f}, Std: {train_stats.get('std', 0):.4f}")
        logger.info(f"  Val   - Mean: {val_stats.get('mean', 0):.4f}, Std: {val_stats.get('std', 0):.4f}")
        logger.info(f"  Test  - Mean: {test_stats.get('mean', 0):.4f}, Std: {test_stats.get('std', 0):.4f}")
    
    return train_loader, val_loader, test_loader


def run_visualization_pipeline(
    config: MRIConfig,
    train_loader,
    val_loader,
    test_loader,
):
    """
    Create visualizations of data samples.
    
    Args:
        config: MRI configuration
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
    """
    logger.info("="*80)
    logger.info("STEP 4: VISUALIZATION")
    logger.info("="*80)
    
    if not config.logging.save_visualizations:
        logger.info("Visualization disabled in config")
        return
    
    viz_dir = config.logging.log_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize training samples
    logger.info("Visualizing training samples...")
    train_batch = next(iter(train_loader))
    visualize_batch(
        train_batch,
        viz_dir / "train",
        num_samples=min(config.logging.num_viz_samples, len(train_batch["image"])),
    )
    
    # Visualize validation samples
    logger.info("Visualizing validation samples...")
    val_batch = next(iter(val_loader))
    visualize_batch(
        val_batch,
        viz_dir / "val",
        num_samples=min(config.logging.num_viz_samples, len(val_batch["image"])),
    )
    
    # Visualize test samples
    if len(test_loader) > 0:
        logger.info("Visualizing test samples...")
        test_batch = next(iter(test_loader))
        visualize_batch(
            test_batch,
            viz_dir / "test",
            num_samples=min(config.logging.num_viz_samples, len(test_batch["image"])),
        )


def demo_training_iteration(train_loader, device: str = "cpu"):
    """
    Demonstrate a simple training iteration.
    
    Args:
        train_loader: Training DataLoader
        device: Device to use ('cpu' or 'cuda')
    """
    logger.info("="*80)
    logger.info("STEP 5: DEMO TRAINING ITERATION")
    logger.info("="*80)
    
    logger.info(f"Using device: {device}")
    
    # Iterate through a few batches
    num_batches = min(3, len(train_loader))
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
        
        # Get data
        images = batch["image"].to(device)
        subjects = batch["subject"]
        
        logger.info(f"Batch {batch_idx + 1}:")
        logger.info(f"  Image shape: {images.shape}")
        logger.info(f"  Image dtype: {images.dtype}")
        logger.info(f"  Image range: [{images.min():.4f}, {images.max():.4f}]")
        logger.info(f"  Subjects: {subjects}")
        logger.info(f"  Device: {images.device}")
        
        # Simulate forward pass (just a simple operation)
        # In practice, this would be your model's forward pass
        output = images.mean(dim=[2, 3, 4])  # Global average pooling
        logger.info(f"  Output shape: {output.shape}")
        
        # Check for NaN or Inf
        if torch.isnan(images).any():
            logger.warning("  WARNING: NaN values detected in images!")
        if torch.isinf(images).any():
            logger.warning("  WARNING: Inf values detected in images!")
    
    logger.info("Demo training iteration completed successfully ✓")


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="Brain MRI Preprocessing Pipeline")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run preprocessing (skip if already preprocessed)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "highres", "fast"],
        help="Configuration preset to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for demo training iteration",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config == "highres":
        config = get_highres_config()
    elif args.config == "fast":
        config = get_fast_config()
    else:
        config = get_default_config()
    
    # Setup logging
    setup_logging(config)
    
    logger.info("="*80)
    logger.info("BRAIN MRI PREPROCESSING AND DATA LOADING PIPELINE")
    logger.info("="*80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Data root: {config.data.data_root}")
    logger.info(f"Output root: {config.data.output_root}")
    logger.info(f"Log directory: {config.logging.log_dir}")
    
    try:
        # Step 1: Preprocessing (optional)
        if args.preprocess:
            preprocessed_data_list = run_preprocessing_pipeline(config)
            if len(preprocessed_data_list) == 0:
                logger.error("Preprocessing failed. Exiting.")
                sys.exit(1)
        else:
            logger.info("Skipping preprocessing. Using raw data.")
            logger.info("To preprocess data, run with --preprocess flag")
            preprocessed_data_list = discover_dataset(config.data.data_root, config.data)
        
        # Step 2: Data splitting
        train_data, val_data, test_data = run_data_split_pipeline(
            config, preprocessed_data_list
        )
        
        # Step 3: DataLoader creation and statistics
        train_loader, val_loader, test_loader = run_dataloader_pipeline(
            config, train_data, val_data, test_data
        )
        
        # Step 4: Visualization
        run_visualization_pipeline(config, train_loader, val_loader, test_loader)
        
        # Step 5: Demo training iteration
        device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
        demo_training_iteration(train_loader, device=device)
        
        logger.info("="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY ✓")
        logger.info("="*80)
        logger.info(f"Logs saved to: {config.logging.log_dir}")
        logger.info(f"Preprocessed data: {config.data.output_root}")
        logger.info(f"Data split: {config.data.cache_dir / 'data_split.json'}")
        
        # Print summary
        logger.info("\nSUMMARY:")
        logger.info(f"  Total samples: {len(preprocessed_data_list)}")
        logger.info(f"  Train samples: {len(train_data)} ({len(train_data)/len(preprocessed_data_list)*100:.1f}%)")
        logger.info(f"  Val samples: {len(val_data)} ({len(val_data)/len(preprocessed_data_list)*100:.1f}%)")
        logger.info(f"  Test samples: {len(test_data)} ({len(test_data)/len(preprocessed_data_list)*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
