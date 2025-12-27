"""
Paired 3T→7T MRI Dataset for supervised GAN training.

Handles:
- Pairing 3T (ses-1) and 7T (ses-2) scans from same subject
- Patch extraction from full volumes
- Optional data augmentation
- Deterministic patch sampling for reproducibility
"""
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable, Union
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from monai.transforms import (
    Compose,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
)

logger = logging.getLogger(__name__)


class Paired3T7TDataset(Dataset):
    """
    Dataset for paired 3T → 7T MRI super-resolution.
    
    Extracts matching patches from paired 3T and 7T volumes.
    Assumes preprocessing (registration, normalization) already done.
    
    Args:
        data_pairs: List of dicts with 'input_3t' and 'target_7t' paths
        patch_size: Size of 3D patches (D, H, W)
        num_patches_per_volume: Number of random patches to extract per volume
        transform: Optional augmentation transforms
        deterministic: If True, use deterministic patch sampling
        random_seed: Seed for reproducibility
    """
    def __init__(
        self,
        data_pairs: List[Dict[str, Union[str, Path]]],
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        num_patches_per_volume: int = 10,
        transform: Optional[Callable] = None,
        deterministic: bool = True,
        random_seed: int = 42,
    ):
        self.data_pairs = data_pairs
        self.patch_size = patch_size
        self.num_patches_per_volume = num_patches_per_volume
        self.transform = transform
        self.deterministic = deterministic
        
        # Total number of patches
        self.total_patches = len(data_pairs) * num_patches_per_volume
        
        # For deterministic sampling
        if deterministic:
            self.rng = np.random.RandomState(random_seed)
        else:
            self.rng = np.random
        
        logger.info(f"Initialized Paired3T7TDataset:")
        logger.info(f"  Pairs: {len(data_pairs)}")
        logger.info(f"  Patches per volume: {num_patches_per_volume}")
        logger.info(f"  Total patches: {self.total_patches}")
        logger.info(f"  Patch size: {patch_size}")
    
    def __len__(self) -> int:
        return self.total_patches
    
    def _load_volume(self, path: Union[str, Path]) -> np.ndarray:
        """Load NIfTI volume and return as numpy array."""
        nib_img = nib.load(str(path))
        volume = nib_img.get_fdata().astype(np.float32)
        
        # Ensure 3D
        if volume.ndim == 4:
            volume = volume[..., 0]  # Take first volume if 4D
        
        return volume
    
    def _extract_random_patch(
        self,
        volume_3t: np.ndarray,
        volume_7t: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract matching random patch from 3T and 7T volumes.
        
        Args:
            volume_3t: 3T volume (D, H, W)
            volume_7t: 7T volume (D, H, W)
        
        Returns:
            patch_3t: 3T patch (D', H', W')
            patch_7t: 7T patch (D', H', W')
        """
        # Get volume shape
        vol_shape = volume_3t.shape
        patch_d, patch_h, patch_w = self.patch_size
        
        # Ensure patch fits in volume
        assert all(v >= p for v, p in zip(vol_shape, self.patch_size)), \
            f"Volume {vol_shape} too small for patch {self.patch_size}"
        
        # Random starting coordinates
        max_d = vol_shape[0] - patch_d
        max_h = vol_shape[1] - patch_h
        max_w = vol_shape[2] - patch_w
        
        start_d = self.rng.randint(0, max_d + 1) if max_d > 0 else 0
        start_h = self.rng.randint(0, max_h + 1) if max_h > 0 else 0
        start_w = self.rng.randint(0, max_w + 1) if max_w > 0 else 0
        
        # Extract patches
        patch_3t = volume_3t[
            start_d:start_d + patch_d,
            start_h:start_h + patch_h,
            start_w:start_w + patch_w,
        ]
        
        patch_7t = volume_7t[
            start_d:start_d + patch_d,
            start_h:start_h + patch_h,
            start_w:start_w + patch_w,
        ]
        
        return patch_3t, patch_7t
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a paired patch.
        
        Returns:
            Dictionary with keys:
                - 'input_3t': 3T patch (1, D, H, W)
                - 'target_7t': 7T patch (1, D, H, W)
                - 'subject': Subject ID
                - 'modality': Modality name
        """
        # Determine which volume pair this patch comes from
        pair_idx = idx // self.num_patches_per_volume
        
        # Get pair info
        pair = self.data_pairs[pair_idx]
        
        # Load volumes (cache could be added here for efficiency)
        volume_3t = self._load_volume(pair['input_3t'])
        volume_7t = self._load_volume(pair['target_7t'])
        
        # Extract random patch
        patch_3t, patch_7t = self._extract_random_patch(volume_3t, volume_7t)
        
        # Add channel dimension
        patch_3t = patch_3t[np.newaxis, ...]  # (1, D, H, W)
        patch_7t = patch_7t[np.newaxis, ...]  # (1, D, H, W)
        
        # Create data dict
        data = {
            'input_3t': torch.from_numpy(patch_3t),
            'target_7t': torch.from_numpy(patch_7t),
            'subject': pair.get('subject', f'pair_{pair_idx}'),
            'modality': pair.get('modality', 'T1w'),
        }
        
        # Apply transforms (augmentation)
        if self.transform is not None:
            data = self.transform(data)
        
        return data


def create_paired_data_list(
    data_list: List[Dict],
    modality: str = 'T1w',
) -> List[Dict[str, Path]]:
    """
    Create paired 3T→7T data list from discovered dataset.
    
    Args:
        data_list: Output from utils.discover_dataset()
        modality: Which modality to use (T1w or T2w)
    
    Returns:
        List of dicts with 'input_3t', 'target_7t', 'subject', 'modality'
    """
    # Group by subject
    by_subject = {}
    for item in data_list:
        subject = item['subject']
        session = item['session']
        mod = item['modality']
        
        if mod != modality:
            continue
        
        if subject not in by_subject:
            by_subject[subject] = {}
        
        by_subject[subject][session] = item['image']
    
    # Create pairs
    paired_data = []
    for subject, sessions in by_subject.items():
        if 'ses-1' in sessions and 'ses-2' in sessions:
            paired_data.append({
                'input_3t': sessions['ses-1'],  # 3T scan
                'target_7t': sessions['ses-2'],  # 7T scan
                'subject': subject,
                'modality': modality,
            })
        else:
            logger.warning(f"Subject {subject} missing 3T or 7T scan, skipping")
    
    logger.info(f"Created {len(paired_data)} paired samples for {modality}")
    return paired_data


def build_gan_augmentation_transforms(
    augmentation_prob: float = 0.5,
) -> Compose:
    """
    Build augmentation pipeline for GAN training.
    
    Note: We apply same augmentation to both 3T and 7T patches
    to preserve spatial correspondence.
    
    Args:
        augmentation_prob: Probability of applying each augmentation
    
    Returns:
        MONAI Compose transform
    """
    transforms = [
        # Random flip (left-right)
        RandFlipd(
            keys=['input_3t', 'target_7t'],
            prob=augmentation_prob,
            spatial_axis=0,
        ),
        
        # Random 90-degree rotation
        RandRotate90d(
            keys=['input_3t', 'target_7t'],
            prob=augmentation_prob,
            spatial_axes=(0, 1),
        ),
        
        # Small random affine (rotation + translation)
        RandAffined(
            keys=['input_3t', 'target_7t'],
            prob=augmentation_prob,
            rotate_range=(0.1, 0.1, 0.1),  # Small rotation
            translate_range=(5, 5, 5),     # Small translation
            mode='bilinear',
            padding_mode='border',
        ),
        
        # Intensity augmentation (only on 3T input)
        # Don't augment 7T target to preserve ground truth
        RandScaleIntensityd(
            keys=['input_3t'],
            factors=0.1,
            prob=augmentation_prob,
        ),
        
        RandShiftIntensityd(
            keys=['input_3t'],
            offsets=0.1,
            prob=augmentation_prob,
        ),
        
        RandGaussianNoised(
            keys=['input_3t'],
            prob=augmentation_prob * 0.5,  # Less frequent
            mean=0.0,
            std=0.01,
        ),
    ]
    
    return Compose(transforms)


def test_paired_dataset():
    """Test the paired dataset."""
    print("Testing Paired3T7TDataset...")
    
    # Create dummy data pairs
    # In practice, use create_paired_data_list() with real data
    dummy_pairs = []
    for i in range(3):
        dummy_pairs.append({
            'input_3t': f'dummy_3t_{i}.nii.gz',
            'target_7t': f'dummy_7t_{i}.nii.gz',
            'subject': f'sub-{i:02d}',
            'modality': 'T1w',
        })
    
    # Create dataset
    dataset = Paired3T7TDataset(
        data_pairs=dummy_pairs,
        patch_size=(64, 64, 64),
        num_patches_per_volume=5,
    )
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Expected: {len(dummy_pairs) * 5} = {len(dataset)}")
    
    print("✓ Paired dataset test passed!")


if __name__ == "__main__":
    test_paired_dataset()
