"""
PyTorch Dataset and DataLoader for 3D brain MRI.
Implements patient-level splits and deterministic data loading.
"""
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable, Union
import json
import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    RandAffined,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandCoarseDropoutd,
    RandGibbsNoised,
)
from monai.data import CacheDataset, PersistentDataset

from config import MRIConfig


logger = logging.getLogger(__name__)


class BrainMRIDataset(Dataset):
    """
    Dataset for 3D brain MRI volumes.
    
    Supports:
    - Multi-modal data (T1w, T2w, etc.)
    - Multi-field strength (3T, 7T)
    - Patient-level data organization
    - Deterministic loading
    - Optional data augmentation
    """
    
    def __init__(
        self,
        data_list: List[Dict[str, Union[str, Path]]],
        transform: Optional[Callable] = None,
        cache_data: bool = False,
    ):
        """
        Initialize dataset.
        
        Args:
            data_list: List of dictionaries with keys: 'image', 'subject', 'session', 'modality', etc.
            transform: Optional transform to apply
            cache_data: Whether to cache loaded data in memory (use for small datasets)
        """
        self.data_list = data_list
        self.transform = transform
        self.cache_data = cache_data
        
        if cache_data:
            self.cache = {}
        else:
            self.cache = None
        
        logger.info(f"Initialized dataset with {len(data_list)} samples")
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and return a single sample."""
        # Get data info
        data_info = self.data_list[idx].copy()
        
        # Check cache
        if self.cache is not None and idx in self.cache:
            data = self.cache[idx].copy()
        else:
            # Load image
            image_path = Path(data_info["image"])
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Load with nibabel for efficiency
            nib_img = nib.load(str(image_path))
            image_array = nib_img.get_fdata().astype(np.float32)
            
            # Add channel dimension if needed
            if image_array.ndim == 3:
                image_array = image_array[np.newaxis, ...]  # Add channel dim
            
            # Convert to tensor
            image_tensor = torch.from_numpy(image_array).float()
            
            # Create data dictionary
            data = {
                "image": image_tensor,
                "subject": data_info["subject"],
                "session": data_info["session"],
                "modality": data_info["modality"],
                "field_strength": data_info.get("field_strength", "unknown"),
                "image_path": str(image_path),
            }
            
            # Cache if enabled
            if self.cache is not None:
                self.cache[idx] = data.copy()
        
        # Apply transforms
        if self.transform is not None:
            # MONAI transforms expect dict with specific keys
            transform_input = {"image": data["image"]}
            transformed = self.transform(transform_input)
            data["image"] = transformed["image"]
        
        return data
    
    def get_subject_ids(self) -> List[str]:
        """Get list of unique subject IDs in dataset."""
        return sorted(list(set(item["subject"] for item in self.data_list)))
    
    def get_statistics(self) -> Dict[str, any]:
        """Compute dataset statistics."""
        subjects = self.get_subject_ids()
        
        sessions = set(item["session"] for item in self.data_list)
        modalities = set(item["modality"] for item in self.data_list)
        field_strengths = set(item.get("field_strength", "unknown") for item in self.data_list)
        
        return {
            "num_samples": len(self.data_list),
            "num_subjects": len(subjects),
            "subjects": subjects,
            "sessions": sorted(list(sessions)),
            "modalities": sorted(list(modalities)),
            "field_strengths": sorted(list(field_strengths)),
        }


class MultiModalBrainMRIDataset(Dataset):
    """
    Dataset that loads multiple modalities for each sample.
    Useful for multi-modal learning (e.g., T1w + T2w).
    """
    
    def __init__(
        self,
        data_list: List[Dict[str, Dict[str, Path]]],
        modalities: List[str],
        transform: Optional[Callable] = None,
    ):
        """
        Initialize multi-modal dataset.
        
        Args:
            data_list: List of dicts with structure:
                {
                    "subject": "sub-01",
                    "session": "ses-1",
                    "modalities": {
                        "T1w": Path("path/to/T1w.nii.gz"),
                        "T2w": Path("path/to/T2w.nii.gz"),
                    }
                }
            modalities: List of modality names to load
            transform: Optional transform
        """
        self.data_list = data_list
        self.modalities = modalities
        self.transform = transform
        
        logger.info(f"Initialized multi-modal dataset with {len(data_list)} samples, modalities: {modalities}")
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and return a multi-modal sample."""
        data_info = self.data_list[idx]
        
        # Load all modalities
        images = {}
        for modality in self.modalities:
            if modality in data_info["modalities"]:
                image_path = Path(data_info["modalities"][modality])
                
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")
                
                nib_img = nib.load(str(image_path))
                image_array = nib_img.get_fdata().astype(np.float32)
                
                # Add channel dimension
                if image_array.ndim == 3:
                    image_array = image_array[np.newaxis, ...]
                
                images[modality] = torch.from_numpy(image_array).float()
            else:
                logger.warning(f"Modality {modality} not found for {data_info['subject']}/{data_info['session']}")
                # Create empty placeholder
                images[modality] = torch.zeros((1, 64, 64, 64))  # Dummy shape
        
        # Stack modalities along channel dimension
        image_stack = torch.cat([images[mod] for mod in self.modalities], dim=0)
        
        # Create data dictionary
        data = {
            "image": image_stack,
            "subject": data_info["subject"],
            "session": data_info["session"],
        }
        
        # Apply transforms
        if self.transform is not None:
            transform_input = {"image": data["image"]}
            transformed = self.transform(transform_input)
            data["image"] = transformed["image"]
        
        return data


def build_augmentation_transforms(config: MRIConfig) -> Compose:
    """
    Build data augmentation pipeline for training.
    
    Args:
        config: MRI configuration
        
    Returns:
        Composed augmentation transforms
    """
    aug_config = config.augmentation
    
    if not aug_config.use_augmentation:
        return Compose([EnsureTyped(keys=["image"])])
    
    transforms_list = []
    
    # Random affine transformation
    if aug_config.random_affine_prob > 0:
        transforms_list.append(
            RandAffined(
                keys=["image"],
                prob=aug_config.random_affine_prob,
                rotate_range=aug_config.rotate_range,
                translate_range=aug_config.translate_range,
                scale_range=aug_config.scale_range,
                mode="bilinear",
                padding_mode="border",
            )
        )
    
    # Random flip
    if aug_config.random_flip_prob > 0:
        transforms_list.append(
            RandFlipd(
                keys=["image"],
                prob=aug_config.random_flip_prob,
                spatial_axis=aug_config.flip_axes,
            )
        )
    
    # Random intensity shift
    if aug_config.intensity_shift_prob > 0:
        transforms_list.append(
            RandShiftIntensityd(
                keys=["image"],
                prob=aug_config.intensity_shift_prob,
                offsets=aug_config.intensity_shift_offset,
            )
        )
    
    # Random intensity scale
    if aug_config.intensity_scale_prob > 0:
        transforms_list.append(
            RandScaleIntensityd(
                keys=["image"],
                prob=aug_config.intensity_scale_prob,
                factors=aug_config.intensity_scale_factor,
            )
        )
    
    # Random Gaussian noise
    if aug_config.gaussian_noise_prob > 0:
        transforms_list.append(
            RandGaussianNoised(
                keys=["image"],
                prob=aug_config.gaussian_noise_prob,
                std=aug_config.gaussian_noise_std,
            )
        )
    
    # Random Gaussian smooth
    if aug_config.gaussian_smooth_prob > 0:
        transforms_list.append(
            RandGaussianSmoothd(
                keys=["image"],
                prob=aug_config.gaussian_smooth_prob,
                sigma_x=aug_config.gaussian_smooth_sigma,
                sigma_y=aug_config.gaussian_smooth_sigma,
                sigma_z=aug_config.gaussian_smooth_sigma,
            )
        )
    
    # Random coarse dropout (cutout-style augmentation)
    if hasattr(aug_config, 'coarse_dropout_prob') and aug_config.coarse_dropout_prob > 0:
        transforms_list.append(
            RandCoarseDropoutd(
                keys=["image"],
                prob=getattr(aug_config, 'coarse_dropout_prob', 0.2),
                holes=6,
                spatial_size=(8, 8, 8),
                max_holes=10,
                max_spatial_size=(32, 32, 32),
            )
        )
    
    # Random Gibbs noise (MRI-specific artifact)
    if hasattr(aug_config, 'gibbs_noise_prob') and aug_config.gibbs_noise_prob > 0:
        transforms_list.append(
            RandGibbsNoised(
                keys=["image"],
                prob=getattr(aug_config, 'gibbs_noise_prob', 0.1),
                alpha=(0.0, 0.7),
            )
        )
    
    transforms_list.append(EnsureTyped(keys=["image"]))
    
    return Compose(transforms_list)


def create_data_loaders(
    config: MRIConfig,
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    use_persistent_cache: bool = False,
    use_memory_cache: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        config: MRI configuration
        train_data: Training data list
        val_data: Validation data list
        test_data: Test data list
        use_persistent_cache: Use persistent disk cache (recommended for large datasets)
        use_memory_cache: Use in-memory cache (recommended for small datasets)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Build augmentation for training
    train_transform = build_augmentation_transforms(config)
    
    # No augmentation for validation and test
    eval_transform = Compose([EnsureTyped(keys=["image"])])
    
    # Create datasets with optional caching
    if use_persistent_cache:
        logger.info("Using persistent disk cache for faster data loading")
        train_dataset = PersistentDataset(
            data=train_data,
            transform=train_transform,
            cache_dir=config.data.cache_dir / "train_cache",
        )
        val_dataset = PersistentDataset(
            data=val_data,
            transform=eval_transform,
            cache_dir=config.data.cache_dir / "val_cache",
        )
        test_dataset = PersistentDataset(
            data=test_data,
            transform=eval_transform,
            cache_dir=config.data.cache_dir / "test_cache",
        )
    elif use_memory_cache:
        logger.info("Using in-memory cache for faster data loading")
        train_dataset = CacheDataset(
            data=train_data,
            transform=train_transform,
            cache_rate=1.0,
            num_workers=config.training.num_workers,
        )
        val_dataset = CacheDataset(
            data=val_data,
            transform=eval_transform,
            cache_rate=1.0,
            num_workers=config.training.num_workers,
        )
        test_dataset = CacheDataset(
            data=test_data,
            transform=eval_transform,
            cache_rate=1.0,
            num_workers=config.training.num_workers,
        )
    else:
        # Standard datasets
        train_dataset = BrainMRIDataset(train_data, transform=train_transform)
        val_dataset = BrainMRIDataset(val_data, transform=eval_transform)
        test_dataset = BrainMRIDataset(test_data, transform=eval_transform)
    
    # Create data loaders with persistent workers for efficiency
    persistent_workers = config.training.num_workers > 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle_train,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=config.training.drop_last,
        prefetch_factor=config.training.prefetch_factor if config.training.num_workers > 0 else None,
        persistent_workers=persistent_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
    )
    
    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def compute_dataset_statistics(data_loader: DataLoader, max_samples: Optional[int] = None) -> Dict[str, float]:
    """
    Compute statistics across entire dataset.
    
    Args:
        data_loader: DataLoader to compute statistics from
        max_samples: Maximum number of samples to use (None for all)
        
    Returns:
        Dictionary of statistics
    """
    logger.info("Computing dataset statistics...")
    
    all_values = []
    shapes = []
    
    for i, batch in enumerate(data_loader):
        if max_samples is not None and i >= max_samples:
            break
        
        images = batch["image"]
        
        # Store shape
        shapes.append(tuple(images.shape[2:]))  # Spatial dimensions
        
        # Collect non-zero values
        for img in images:
            values = img[img != 0].cpu().numpy()
            if len(values) > 0:
                all_values.append(values)
    
    if len(all_values) == 0:
        logger.warning("No non-zero values found")
        return {}
    
    # Concatenate all values
    all_values = np.concatenate(all_values)
    
    # Compute statistics
    stats = {
        "mean": float(np.mean(all_values)),
        "std": float(np.std(all_values)),
        "min": float(np.min(all_values)),
        "max": float(np.max(all_values)),
        "median": float(np.median(all_values)),
        "p01": float(np.percentile(all_values, 1)),
        "p99": float(np.percentile(all_values, 99)),
        "num_samples": len(shapes),
        "unique_shapes": len(set(shapes)),
    }
    
    logger.info("Dataset statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    return stats


def deterministic_hash(data: str) -> str:
    """Create deterministic hash for reproducibility."""
    return hashlib.md5(data.encode()).hexdigest()


def save_data_split(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    output_path: Path,
) -> None:
    """
    Save data split to JSON file for reproducibility.
    
    Args:
        train_data: Training data list
        val_data: Validation data list
        test_data: Test data list
        output_path: Path to save JSON file
    """
    # Convert Path objects to strings for JSON serialization
    def convert_paths(data_list):
        converted = []
        for item in data_list:
            item_copy = item.copy()
            if "image" in item_copy:
                item_copy["image"] = str(item_copy["image"])
            converted.append(item_copy)
        return converted
    
    split_data = {
        "train": convert_paths(train_data),
        "val": convert_paths(val_data),
        "test": convert_paths(test_data),
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    logger.info(f"Saved data split to {output_path}")


def load_data_split(split_path: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load data split from JSON file.
    
    Args:
        split_path: Path to JSON file
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    with open(split_path, 'r') as f:
        split_data = json.load(f)
    
    # Convert string paths back to Path objects
    def convert_paths(data_list):
        converted = []
        for item in data_list:
            item_copy = item.copy()
            if "image" in item_copy:
                item_copy["image"] = Path(item_copy["image"])
            converted.append(item_copy)
        return converted
    
    train_data = convert_paths(split_data["train"])
    val_data = convert_paths(split_data["val"])
    test_data = convert_paths(split_data["test"])
    
    logger.info(f"Loaded data split from {split_path}")
    logger.info(f"  Train: {len(train_data)} samples")
    logger.info(f"  Val: {len(val_data)} samples")
    logger.info(f"  Test: {len(test_data)} samples")
    
    return train_data, val_data, test_data


if __name__ == "__main__":
    # Example usage
    from config import get_default_config
    import utils
    
    logging.basicConfig(level=logging.INFO)
    
    config = get_default_config()
    
    # Discover dataset
    data_list = utils.discover_dataset(config.data.data_root, config.data)
    
    # Create patient-level split
    train_data, val_data, test_data = utils.create_patient_level_split(
        data_list,
        config.split.train_ratio,
        config.split.val_ratio,
        config.split.test_ratio,
        random_seed=config.split.random_seed,
    )
    
    # Save split
    split_path = config.data.cache_dir / "data_split.json"
    save_data_split(train_data, val_data, test_data, split_path)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config, train_data, val_data, test_data
    )
    
    # Test loading a batch
    batch = next(iter(train_loader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Subjects in batch: {batch['subject']}")
