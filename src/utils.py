"""
Utility functions for dataset discovery, splitting, logging, and visualization.
"""
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import random
import hashlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import nibabel as nib

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")

from .config import MRIConfig, DataConfig


logger = logging.getLogger(__name__)


def setup_logging(config: MRIConfig) -> None:
    """
    Setup logging configuration.
    
    Args:
        config: MRI configuration
    """
    log_config = config.logging
    
    # Create log directory
    log_config.log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_config.log_dir / 'pipeline.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging initialized at level {log_config.log_level}")


def discover_dataset(data_root: Path, data_config: DataConfig) -> List[Dict[str, any]]:
    """
    Discover all MRI volumes in the dataset following BIDS structure.
    
    Args:
        data_root: Root directory of BIDS dataset
        data_config: Data configuration
        
    Returns:
        List of dictionaries containing file paths and metadata
    """
    logger.info(f"Discovering dataset in: {data_root}")
    
    data_list = []
    
    # Iterate through subjects
    for subject_dir in sorted(data_root.glob("sub-*")):
        if not subject_dir.is_dir():
            continue
        
        subject_id = subject_dir.name
        
        # Iterate through sessions
        for session_dir in sorted(subject_dir.glob("ses-*")):
            if not session_dir.is_dir():
                continue
            
            session_id = session_dir.name
            
            # Determine field strength based on session
            if session_id == data_config.session_3t:
                field_strength = "3T"
            elif session_id == data_config.session_7t:
                field_strength = "7T"
            else:
                field_strength = "unknown"
            
            # Look in anat folder
            anat_dir = session_dir / "anat"
            if not anat_dir.exists():
                continue
            
            # Find NIfTI files
            for nifti_file in sorted(anat_dir.glob(data_config.file_pattern)):
                # Extract modality from filename
                filename = nifti_file.name
                
                # Determine modality
                modality = None
                for mod in data_config.modalities:
                    if mod in filename:
                        modality = mod
                        break
                
                if modality is None:
                    logger.debug(f"Skipping file (unknown modality): {nifti_file}")
                    continue
                
                # Check if corresponding JSON exists
                # Handle both _defaced and _preprocessed naming conventions
                json_file_name = nifti_file.name
                for suffix in ["_defaced.nii.gz", "_preprocessed.nii.gz", ".nii.gz"]:
                    if suffix in json_file_name:
                        json_file_name = json_file_name.replace(suffix, "_metadata.json")
                        break
                json_file = nifti_file.parent / json_file_name
                metadata = {}
                if json_file.exists():
                    try:
                        with open(json_file, 'r') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to load JSON metadata: {json_file}, {e}")
                
                # Create data entry
                data_entry = {
                    "image": nifti_file,
                    "subject": subject_id,
                    "session": session_id,
                    "modality": modality,
                    "field_strength": field_strength,
                    "json_metadata": metadata,
                }
                
                data_list.append(data_entry)
    
    logger.info(f"Discovered {len(data_list)} volumes")
    
    # Print summary
    subjects = set(item["subject"] for item in data_list)
    sessions = set(item["session"] for item in data_list)
    modalities = set(item["modality"] for item in data_list)
    field_strengths = set(item["field_strength"] for item in data_list)
    
    logger.info(f"Summary:")
    logger.info(f"  Subjects: {len(subjects)} - {sorted(subjects)}")
    logger.info(f"  Sessions: {len(sessions)} - {sorted(sessions)}")
    logger.info(f"  Modalities: {len(modalities)} - {sorted(modalities)}")
    logger.info(f"  Field strengths: {len(field_strengths)} - {sorted(field_strengths)}")
    
    return data_list


def create_patient_level_split(
    data_list: List[Dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create patient-level train/val/test split.
    Ensures no data leakage between sessions of the same patient.
    
    Args:
        data_list: List of all data samples
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Get unique subjects
    subjects = sorted(list(set(item["subject"] for item in data_list)))
    num_subjects = len(subjects)
    
    logger.info(f"Creating patient-level split for {num_subjects} subjects")
    logger.info(f"Ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    
    # Shuffle subjects deterministically
    random.shuffle(subjects)
    
    # Calculate split indices
    train_count = int(num_subjects * train_ratio)
    val_count = int(num_subjects * val_ratio)
    
    # Split subjects
    train_subjects = set(subjects[:train_count])
    val_subjects = set(subjects[train_count:train_count + val_count])
    test_subjects = set(subjects[train_count + val_count:])
    
    logger.info(f"Split subjects:")
    logger.info(f"  Train: {len(train_subjects)} subjects - {sorted(train_subjects)}")
    logger.info(f"  Val: {len(val_subjects)} subjects - {sorted(val_subjects)}")
    logger.info(f"  Test: {len(test_subjects)} subjects - {sorted(test_subjects)}")
    
    # Assign data to splits
    train_data = [item for item in data_list if item["subject"] in train_subjects]
    val_data = [item for item in data_list if item["subject"] in val_subjects]
    test_data = [item for item in data_list if item["subject"] in test_subjects]
    
    logger.info(f"Split samples:")
    logger.info(f"  Train: {len(train_data)} samples")
    logger.info(f"  Val: {len(val_data)} samples")
    logger.info(f"  Test: {len(test_data)} samples")
    
    # Verify no leakage
    train_subjects_check = set(item["subject"] for item in train_data)
    val_subjects_check = set(item["subject"] for item in val_data)
    test_subjects_check = set(item["subject"] for item in test_data)
    
    assert len(train_subjects_check & val_subjects_check) == 0, "Train-Val leakage detected!"
    assert len(train_subjects_check & test_subjects_check) == 0, "Train-Test leakage detected!"
    assert len(val_subjects_check & test_subjects_check) == 0, "Val-Test leakage detected!"
    
    logger.info("✓ No data leakage detected")
    
    return train_data, val_data, test_data


def create_kfold_splits(
    data_list: List[Dict],
    k_folds: int = 5,
    random_seed: int = 42,
) -> List[Tuple[List[Dict], List[Dict]]]:
    """
    Create k-fold cross-validation splits at patient level.
    
    Args:
        data_list: List of all data samples
        k_folds: Number of folds
        random_seed: Random seed for reproducibility
        
    Returns:
        List of (train_data, val_data) tuples for each fold
    """
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Get unique subjects
    subjects = sorted(list(set(item["subject"] for item in data_list)))
    num_subjects = len(subjects)
    
    # Shuffle subjects
    random.shuffle(subjects)
    
    # Create folds
    fold_size = num_subjects // k_folds
    folds = []
    
    for i in range(k_folds):
        # Determine validation subjects for this fold
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k_folds - 1 else num_subjects
        
        val_subjects = set(subjects[val_start:val_end])
        train_subjects = set(subjects) - val_subjects
        
        # Create data lists
        train_data = [item for item in data_list if item["subject"] in train_subjects]
        val_data = [item for item in data_list if item["subject"] in val_subjects]
        
        folds.append((train_data, val_data))
        
        logger.info(f"Fold {i+1}/{k_folds}:")
        logger.info(f"  Train: {len(train_subjects)} subjects, {len(train_data)} samples")
        logger.info(f"  Val: {len(val_subjects)} subjects, {len(val_data)} samples")
    
    return folds


def visualize_sample(
    image_path: Path,
    output_path: Optional[Path] = None,
    slice_indices: Optional[Tuple[int, int, int]] = None,
    title: Optional[str] = None,
) -> None:
    """
    Visualize a 3D MRI volume with orthogonal slices.
    
    Args:
        image_path: Path to NIfTI file
        output_path: Optional path to save figure
        slice_indices: Tuple of (sagittal, coronal, axial) slice indices (None for middle)
        title: Optional title for the plot
    """
    # Load image
    nib_img = nib.load(str(image_path))
    image_array = nib_img.get_fdata()
    
    # Get middle slices if not specified
    if slice_indices is None:
        slice_indices = (
            image_array.shape[0] // 2,
            image_array.shape[1] // 2,
            image_array.shape[2] // 2,
        )
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot sagittal slice
    axes[0].imshow(image_array[slice_indices[0], :, :].T, cmap='gray', origin='lower')
    axes[0].set_title(f'Sagittal (x={slice_indices[0]})')
    axes[0].axis('off')
    
    # Plot coronal slice
    axes[1].imshow(image_array[:, slice_indices[1], :].T, cmap='gray', origin='lower')
    axes[1].set_title(f'Coronal (y={slice_indices[1]})')
    axes[1].axis('off')
    
    # Plot axial slice
    axes[2].imshow(image_array[:, :, slice_indices[2]].T, cmap='gray', origin='lower')
    axes[2].set_title(f'Axial (z={slice_indices[2]})')
    axes[2].axis('off')
    
    # Set main title
    if title is None:
        title = image_path.name
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save or show
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_batch(
    batch: Dict,
    output_dir: Path,
    num_samples: int = 3,
) -> None:
    """
    Visualize samples from a batch.
    
    Args:
        batch: Batch dictionary from DataLoader
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images = batch["image"]
    subjects = batch["subject"]
    
    num_samples = min(num_samples, len(images))
    
    for i in range(num_samples):
        image = images[i, 0].cpu().numpy()  # Remove channel dim
        subject = subjects[i]
        
        # Get middle slices
        slice_indices = (
            image.shape[0] // 2,
            image.shape[1] // 2,
            image.shape[2] // 2,
        )
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Compute intensity range for consistent visualization
        vmin, vmax = np.percentile(image[image != 0], [1, 99])
        
        # Plot orthogonal slices
        axes[0].imshow(image[slice_indices[0], :, :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[0].set_title('Sagittal')
        axes[0].axis('off')
        
        axes[1].imshow(image[:, slice_indices[1], :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[1].set_title('Coronal')
        axes[1].axis('off')
        
        axes[2].imshow(image[:, :, slice_indices[2]].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[2].set_title('Axial')
        axes[2].axis('off')
        
        fig.suptitle(f'Subject: {subject}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = output_dir / f"sample_{i}_{subject}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization: {output_path}")


def compute_and_save_statistics(
    data_loader,
    output_path: Path,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute dataset statistics and save to JSON.
    
    Args:
        data_loader: DataLoader to compute statistics from
        output_path: Path to save JSON file
        max_samples: Maximum samples to use
        
    Returns:
        Dictionary of statistics
    """
    from dataset import compute_dataset_statistics
    
    stats = compute_dataset_statistics(data_loader, max_samples)
    
    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved statistics to {output_path}")
    
    return stats


def verify_preprocessing(
    original_path: Path,
    preprocessed_path: Path,
    output_dir: Path,
) -> None:
    """
    Create side-by-side comparison of original and preprocessed images.
    
    Args:
        original_path: Path to original NIfTI file
        preprocessed_path: Path to preprocessed NIfTI file
        output_dir: Directory to save comparison
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load images
    orig_img = nib.load(str(original_path)).get_fdata()
    prep_img = nib.load(str(preprocessed_path)).get_fdata()
    
    # Get middle slices
    orig_mid = orig_img.shape[2] // 2
    prep_mid = prep_img.shape[2] // 2
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(orig_img[:, :, orig_mid].T, cmap='gray', origin='lower')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Preprocessed image
    axes[0, 1].imshow(prep_img[:, :, prep_mid].T, cmap='gray', origin='lower')
    axes[0, 1].set_title('Preprocessed')
    axes[0, 1].axis('off')
    
    # Histograms
    orig_nonzero = orig_img[orig_img != 0]
    prep_nonzero = prep_img[prep_img != 0]
    
    axes[1, 0].hist(orig_nonzero.flatten(), bins=100, alpha=0.7, label='Original')
    axes[1, 0].set_title('Original Intensity Distribution')
    axes[1, 0].set_xlabel('Intensity')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    axes[1, 1].hist(prep_nonzero.flatten(), bins=100, alpha=0.7, label='Preprocessed', color='orange')
    axes[1, 1].set_title('Preprocessed Intensity Distribution')
    axes[1, 1].set_xlabel('Intensity')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    output_path = output_dir / f"comparison_{original_path.stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved preprocessing comparison to {output_path}")


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    logger.info(f"Set random seed to {seed}")


class TensorBoardLogger:
    """
    TensorBoard logging for training metrics and visualizations.
    """
    
    def __init__(self, log_dir: Path, enabled: bool = True):
        self.log_dir = Path(log_dir)
        self.enabled = enabled and TENSORBOARD_AVAILABLE
        
        if self.enabled:
            self.writer = SummaryWriter(str(self.log_dir))
            logger.info(f"TensorBoard logging enabled: {self.log_dir}")
        else:
            self.writer = None
            if not TENSORBOARD_AVAILABLE:
                logger.warning("TensorBoard not available")
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int) -> None:
        """Log multiple scalar values."""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        """Log a 2D image."""
        if self.enabled:
            # Normalize to [0, 1]
            if image.max() > image.min():
                image_norm = (image - image.min()) / (image.max() - image.min())
            else:
                image_norm = image
            self.writer.add_image(tag, image_norm, step, dataformats='HW')
    
    def log_volume_slices(self, tag: str, volume: np.ndarray, step: int) -> None:
        """Log orthogonal slices from a 3D volume."""
        if not self.enabled:
            return
        
        # Get middle slices
        mid_x = volume.shape[0] // 2
        mid_y = volume.shape[1] // 2
        mid_z = volume.shape[2] // 2
        
        self.log_image(f"{tag}/sagittal", volume[mid_x, :, :], step)
        self.log_image(f"{tag}/coronal", volume[:, mid_y, :], step)
        self.log_image(f"{tag}/axial", volume[:, :, mid_z], step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """Log a histogram."""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text."""
        if self.enabled:
            self.writer.add_text(tag, text, step)
    
    def log_batch_statistics(self, batch: dict, step: int, prefix: str = "batch") -> None:
        """Log statistics from a batch."""
        if not self.enabled:
            return
        
        images = batch['image'].cpu().numpy()
        
        # Log statistics
        self.log_scalar(f"{prefix}/mean", float(images.mean()), step)
        self.log_scalar(f"{prefix}/std", float(images.std()), step)
        self.log_scalar(f"{prefix}/min", float(images.min()), step)
        self.log_scalar(f"{prefix}/max", float(images.max()), step)
        
        # Log sample image
        if len(images) > 0:
            sample_volume = images[0, 0]  # First sample, first channel
            self.log_volume_slices(f"{prefix}/sample", sample_volume, step)
    
    def close(self) -> None:
        """Close the writer."""
        if self.enabled and self.writer is not None:
            self.writer.close()
            logger.info("Closed TensorBoard writer")


def run_kfold_cross_validation(
    data_list: List[Dict],
    k_folds: int,
    train_function: callable,
    random_seed: int = 42,
) -> List[Dict]:
    """
    Convenience wrapper for k-fold cross-validation.
    
    Args:
        data_list: Complete dataset
        k_folds: Number of folds
        train_function: Function that takes (train_data, val_data) and returns metrics dict
        random_seed: Random seed
        
    Returns:
        List of metrics dictionaries from each fold
    """
    folds = create_kfold_splits(data_list, k_folds, random_seed)
    
    all_results = []
    
    for fold_idx, (train_data, val_data) in enumerate(folds):
        logger.info(f"\n{'='*70}")
        logger.info(f"Training Fold {fold_idx + 1}/{k_folds}")
        logger.info(f"{'='*70}")
        
        # Train and get metrics
        metrics = train_function(train_data, val_data)
        metrics['fold'] = fold_idx + 1
        
        all_results.append(metrics)
        
        logger.info(f"Fold {fold_idx + 1} Results: {metrics}")
    
    # Aggregate results
    logger.info(f"\n{'='*70}")
    logger.info("Cross-Validation Results")
    logger.info(f"{'='*70}")
    
    # Compute mean and std across folds
    if all_results:
        metric_names = [k for k in all_results[0].keys() if k != 'fold' and isinstance(all_results[0][k], (int, float))]
        
        for metric_name in metric_names:
            values = [r[metric_name] for r in all_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")
    
    return all_results


if __name__ == "__main__":
    # Example usage
    from config import get_default_config
    
    logging.basicConfig(level=logging.INFO)
    
    config = get_default_config()
    setup_logging(config)
    
    # Discover dataset
    data_list = discover_dataset(config.data.data_root, config.data)
    
    # Create splits
    train_data, val_data, test_data = create_patient_level_split(
        data_list,
        config.split.train_ratio,
        config.split.val_ratio,
        config.split.test_ratio,
        random_seed=config.split.random_seed,
    )
    
    # Visualize a sample
    if len(data_list) > 0:
        sample_path = data_list[0]["image"]
        output_path = config.logging.log_dir / "sample_visualization.png"
        visualize_sample(sample_path, output_path)
