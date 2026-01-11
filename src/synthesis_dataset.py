"""
Data formulation for 3T-to-7T MRI synthesis learning.

Implements Blueprint Section 6 (Data Formulation for Learning):
- 6.1 Input-Target Construction: Paired 3T→7T loading
- 6.2 Patch vs Full-Volume Training: 64³ patch extraction
- 6.3 Multi-Contrast Usage: Optional T2 channel support (future)
- 6.4 Avoiding Data Leakage: Subject-level splitting with cross-validation
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

logger = logging.getLogger(__name__)


@dataclass
class PatchConfig:
    """Configuration for patch-based training."""
    
    patch_size: Tuple[int, int, int] = (64, 64, 64)
    patches_per_volume: int = 32  # Number of patches to sample per volume per epoch
    overlap_ratio: float = 0.5  # For inference tiling
    min_brain_fraction: float = 0.1  # Minimum fraction of patch that must be brain
    seed: int = 42


@dataclass  
class SplitConfig:
    """Configuration for subject-level data splitting."""
    
    n_folds: int = 10  # Number of CV folds (10 = LOOCV for 10 subjects)
    val_fold: int = 0  # Which fold to use for validation
    test_fold: int = 1  # Which fold to use for test (if different from val)
    use_loocv: bool = True  # Leave-one-out cross-validation
    seed: int = 42


class SubjectSplitter:
    """
    Subject-level train/val/test splitting with cross-validation support.
    
    Implements Blueprint Section 6.4: Avoiding Data Leakage.
    Ensures no patches from test subjects appear in training.
    """
    
    def __init__(self, config: SplitConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)
    
    def get_subjects_from_pairs(self, pairs: List[Dict]) -> List[str]:
        """Extract unique subject IDs from pairs list."""
        return sorted(set(p.get("subject", "unknown") for p in pairs))
    
    def create_folds(self, subjects: List[str]) -> List[List[str]]:
        """
        Create cross-validation folds from subject list.
        
        For N subjects and N folds (LOOCV), each fold contains 1 subject.
        """
        subjects = list(subjects)
        self._rng.shuffle(subjects)
        
        n_folds = min(self.config.n_folds, len(subjects))
        
        if self.config.use_loocv:
            # Leave-one-out: each subject is its own fold
            return [[s] for s in subjects]
        else:
            # K-fold: distribute subjects across folds
            folds = [[] for _ in range(n_folds)]
            for i, subject in enumerate(subjects):
                folds[i % n_folds].append(subject)
            return folds
    
    def get_split(
        self,
        pairs: List[Dict],
        val_fold: Optional[int] = None,
        test_fold: Optional[int] = None,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Get train/val/test split for given fold indices.
        
        Args:
            pairs: List of pair dictionaries with 'subject' key
            val_fold: Fold index to use for validation
            test_fold: Fold index to use for test (can be same as val)
            
        Returns:
            Tuple of (train_pairs, val_pairs, test_pairs)
        """
        subjects = self.get_subjects_from_pairs(pairs)
        folds = self.create_folds(subjects)
        
        val_fold = val_fold if val_fold is not None else self.config.val_fold
        test_fold = test_fold if test_fold is not None else self.config.test_fold
        
        # Determine which subjects go where
        val_subjects = set(folds[val_fold % len(folds)])
        test_subjects = set(folds[test_fold % len(folds)])
        
        # For same val/test fold, they overlap
        # For different folds, they're separate
        train_subjects = set(subjects) - val_subjects - test_subjects
        
        # Split pairs by subject
        train_pairs = [p for p in pairs if p.get("subject") in train_subjects]
        val_pairs = [p for p in pairs if p.get("subject") in val_subjects]
        test_pairs = [p for p in pairs if p.get("subject") in test_subjects]
        
        logger.info(f"Subject-level split (val_fold={val_fold}, test_fold={test_fold}):")
        logger.info(f"  Train: {len(train_pairs)} pairs ({len(train_subjects)} subjects)")
        logger.info(f"  Val: {len(val_pairs)} pairs ({len(val_subjects)} subjects)")
        logger.info(f"  Test: {len(test_pairs)} pairs ({len(test_subjects)} subjects)")
        
        return train_pairs, val_pairs, test_pairs
    
    def get_cv_splits(
        self,
        pairs: List[Dict],
    ) -> Iterator[Tuple[int, List[Dict], List[Dict], List[Dict]]]:
        """
        Generate all cross-validation splits.
        
        Yields:
            Tuple of (fold_index, train_pairs, val_pairs, test_pairs)
        """
        subjects = self.get_subjects_from_pairs(pairs)
        folds = self.create_folds(subjects)
        
        for fold_idx in range(len(folds)):
            train, val, test = self.get_split(
                pairs, 
                val_fold=fold_idx, 
                test_fold=fold_idx  # Same fold for val/test in LOOCV
            )
            yield fold_idx, train, val, test


class PairedPatchDataset(Dataset):
    """
    PyTorch Dataset for paired 3T-7T patch extraction.
    
    Implements Blueprint Section 6.1-6.2:
    - Loads preprocessed 3T (input) and 7T (target) volumes
    - Extracts random 3D patches from aligned locations
    - Brain-weighted sampling to focus on brain regions
    
    Each sample returns:
    - input: 3T T1w patch [1, D, H, W]
    - target: 7T T1w patch [1, D, H, W] at same coordinates
    - metadata: subject, coordinates, etc.
    """
    
    def __init__(
        self,
        pairs: List[Dict],
        config: Optional[PatchConfig] = None,
        augment: bool = False,
        cache_volumes: bool = True,
    ):
        """
        Initialize paired patch dataset.
        
        Args:
            pairs: List of dictionaries with keys:
                - subject: subject ID
                - input_3t: path to preprocessed 3T T1w
                - target_7t: path to preprocessed 7T T1w
                - input_3t_t2: (optional) path to 3T T2w
            config: Patch extraction configuration
            augment: Whether to apply data augmentation
            cache_volumes: Whether to cache loaded volumes in memory
        """
        self.pairs = pairs
        self.config = config or PatchConfig()
        self.augment = augment
        self.cache_volumes = cache_volumes
        
        # Volume cache
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
        
        # RNG for patch sampling
        self._rng = np.random.default_rng(self.config.seed)
        
        # Precompute valid patch centers for each volume
        self._valid_centers: Dict[str, np.ndarray] = {}
        
        # Total samples = pairs * patches_per_volume
        self._length = len(pairs) * self.config.patches_per_volume
        
        logger.info(
            f"Initialized PairedPatchDataset: {len(pairs)} pairs, "
            f"{self.config.patches_per_volume} patches/volume, "
            f"patch_size={self.config.patch_size}, "
            f"total samples={self._length}"
        )
    
    def __len__(self) -> int:
        return self._length
    
    def _load_volume(self, path: Path) -> np.ndarray:
        """Load a NIfTI volume."""
        nib_img = nib.load(str(path))
        return nib_img.get_fdata().astype(np.float32)
    
    def _get_cached_pair(self, pair_idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Get or load a cached volume pair."""
        pair = self.pairs[pair_idx]
        cache_key = pair.get("subject", str(pair_idx))
        
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return cached["input_3t"], cached["target_7t"], cached.get("mask")
        
        # Load volumes
        input_3t = self._load_volume(Path(pair["input_3t"]))
        target_7t = self._load_volume(Path(pair["target_7t"]))
        
        # Create brain mask from input (non-zero regions)
        mask = (input_3t > 0).astype(np.uint8)
        
        # Cache if enabled
        if self.cache_volumes:
            self._cache[cache_key] = {
                "input_3t": input_3t,
                "target_7t": target_7t,
                "mask": mask,
            }
        
        return input_3t, target_7t, mask
    
    def _get_valid_centers(
        self,
        mask: np.ndarray,
        pair_idx: int,
    ) -> np.ndarray:
        """
        Get valid patch center coordinates within the brain.
        
        Uses brain mask to ensure patches contain sufficient brain tissue.
        """
        cache_key = str(pair_idx)
        if cache_key in self._valid_centers:
            return self._valid_centers[cache_key]
        
        patch_size = np.array(self.config.patch_size)
        half_patch = patch_size // 2
        
        # Valid range for patch centers
        min_coords = half_patch
        max_coords = np.array(mask.shape) - half_patch - (patch_size % 2)
        
        # Find valid centers (where there's enough brain in the patch)
        # Sample candidates and filter
        n_candidates = 10000
        candidates = np.column_stack([
            self._rng.integers(min_coords[d], max_coords[d], size=n_candidates)
            for d in range(3)
        ])
        
        valid_centers = []
        for center in candidates:
            # Check brain fraction in this patch region
            slices = tuple(
                slice(center[d] - half_patch[d], center[d] + half_patch[d] + patch_size[d] % 2)
                for d in range(3)
            )
            patch_mask = mask[slices]
            brain_fraction = np.mean(patch_mask > 0)
            
            if brain_fraction >= self.config.min_brain_fraction:
                valid_centers.append(center)
        
        valid_centers = np.array(valid_centers) if valid_centers else candidates[:100]
        
        # Cache valid centers
        self._valid_centers[cache_key] = valid_centers
        
        return valid_centers
    
    def _extract_patch(
        self,
        volume: np.ndarray,
        center: np.ndarray,
    ) -> np.ndarray:
        """Extract a patch centered at given coordinates."""
        patch_size = np.array(self.config.patch_size)
        half_patch = patch_size // 2
        
        slices = tuple(
            slice(
                max(0, center[d] - half_patch[d]),
                min(volume.shape[d], center[d] + half_patch[d] + patch_size[d] % 2)
            )
            for d in range(3)
        )
        
        patch = volume[slices]
        
        # Pad if necessary (edge cases)
        if patch.shape != tuple(patch_size):
            padded = np.zeros(patch_size, dtype=patch.dtype)
            # Calculate offset for centering
            offset = [(patch_size[d] - patch.shape[d]) // 2 for d in range(3)]
            padded[
                offset[0]:offset[0]+patch.shape[0],
                offset[1]:offset[1]+patch.shape[1],
                offset[2]:offset[2]+patch.shape[2],
            ] = patch
            patch = padded
        
        return patch
    
    def _apply_augmentation(
        self,
        input_patch: np.ndarray,
        target_patch: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply paired data augmentation.
        Same transform applied to both input and target.
        """
        if not self.augment:
            return input_patch, target_patch
        
        # Random flip (same for both)
        for axis in range(3):
            if self._rng.random() < 0.5:
                input_patch = np.flip(input_patch, axis=axis).copy()
                target_patch = np.flip(target_patch, axis=axis).copy()
        
        # Random 90-degree rotations (same for both)
        k = self._rng.integers(0, 4)
        if k > 0:
            axes = self._rng.choice([0, 1, 2], size=2, replace=False)
            input_patch = np.rot90(input_patch, k=k, axes=tuple(axes)).copy()
            target_patch = np.rot90(target_patch, k=k, axes=tuple(axes)).copy()
        
        return input_patch, target_patch
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, Tuple]]:
        """
        Get a paired patch sample.
        
        Returns:
            Dictionary with:
                - input: [1, D, H, W] 3T patch tensor
                - target: [1, D, H, W] 7T patch tensor
                - subject: subject ID
                - center: patch center coordinates
        """
        # Determine which volume pair and patch index
        pair_idx = idx // self.config.patches_per_volume
        patch_idx = idx % self.config.patches_per_volume
        
        # Load volumes
        input_vol, target_vol, mask = self._get_cached_pair(pair_idx)
        
        # Get valid patch centers
        valid_centers = self._get_valid_centers(mask, pair_idx)
        
        # Select a center (deterministic based on patch_idx for reproducibility)
        center_idx = (patch_idx * 7919) % len(valid_centers)  # Prime for good distribution
        center = valid_centers[center_idx]
        
        # Add random jitter during training
        if self.augment:
            jitter = self._rng.integers(-5, 6, size=3)
            center = np.clip(
                center + jitter,
                self.config.patch_size[0] // 2,
                np.array(input_vol.shape) - np.array(self.config.patch_size) // 2 - 1
            )
        
        # Extract patches
        input_patch = self._extract_patch(input_vol, center)
        target_patch = self._extract_patch(target_vol, center)
        
        # Apply augmentation (same transform to both)
        input_patch, target_patch = self._apply_augmentation(input_patch, target_patch)
        
        # Convert to tensors with channel dimension
        input_tensor = torch.from_numpy(input_patch[np.newaxis, ...]).float()
        target_tensor = torch.from_numpy(target_patch[np.newaxis, ...]).float()
        
        return {
            "input": input_tensor,
            "target": target_tensor,
            "subject": self.pairs[pair_idx].get("subject", "unknown"),
            "center": tuple(center.tolist()),
            "pair_idx": pair_idx,
        }
    
    def get_full_volume(self, pair_idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get full volume pair (for inference/evaluation).
        
        Returns:
            Tuple of (input_volume, target_volume, metadata)
        """
        input_vol, target_vol, _ = self._get_cached_pair(pair_idx)
        
        input_tensor = torch.from_numpy(input_vol[np.newaxis, ...]).float()
        target_tensor = torch.from_numpy(target_vol[np.newaxis, ...]).float()
        
        return input_tensor, target_tensor, self.pairs[pair_idx]


class InferencePatchSampler:
    """
    Generate overlapping patches for full-volume inference with stitching.
    
    Implements Blueprint Section 6.6: Patch Aggregation at Inference.
    """
    
    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        overlap: float = 0.5,
    ):
        self.volume_shape = volume_shape
        self.patch_size = patch_size
        self.overlap = overlap
        
        # Calculate stride
        self.stride = tuple(int(p * (1 - overlap)) for p in patch_size)
        
        # Generate patch centers
        self.centers = self._generate_centers()
    
    def _generate_centers(self) -> List[Tuple[int, int, int]]:
        """Generate all patch center coordinates."""
        centers = []
        half_patch = [p // 2 for p in self.patch_size]
        
        for x in range(half_patch[0], self.volume_shape[0] - half_patch[0], self.stride[0]):
            for y in range(half_patch[1], self.volume_shape[1] - half_patch[1], self.stride[1]):
                for z in range(half_patch[2], self.volume_shape[2] - half_patch[2], self.stride[2]):
                    centers.append((x, y, z))
        
        return centers
    
    def __len__(self) -> int:
        return len(self.centers)
    
    def __iter__(self) -> Iterator[Tuple[int, int, int]]:
        return iter(self.centers)
    
    def stitch_patches(
        self,
        patches: List[np.ndarray],
        centers: List[Tuple[int, int, int]],
        blending: str = "average",
    ) -> np.ndarray:
        """
        Stitch patches back into a full volume.
        
        Args:
            patches: List of patch arrays
            centers: List of corresponding center coordinates
            blending: "average" or "gaussian" for overlap blending
        """
        output = np.zeros(self.volume_shape, dtype=np.float32)
        weights = np.zeros(self.volume_shape, dtype=np.float32)
        
        half_patch = [p // 2 for p in self.patch_size]
        
        # Create blending weight (cosine tapering at edges)
        if blending == "gaussian":
            from scipy.ndimage import gaussian_filter
            patch_weight = np.ones(self.patch_size, dtype=np.float32)
            patch_weight = gaussian_filter(patch_weight, sigma=3)
        else:
            patch_weight = np.ones(self.patch_size, dtype=np.float32)
        
        for patch, center in zip(patches, centers):
            slices = tuple(
                slice(
                    max(0, center[d] - half_patch[d]),
                    min(self.volume_shape[d], center[d] + half_patch[d] + self.patch_size[d] % 2)
                )
                for d in range(3)
            )
            
            # Handle edge patches
            patch_slices = tuple(
                slice(0, slices[d].stop - slices[d].start)
                for d in range(3)
            )
            
            output[slices] += patch[patch_slices] * patch_weight[patch_slices]
            weights[slices] += patch_weight[patch_slices]
        
        # Normalize by weights
        output = np.divide(output, weights, where=weights > 0)
        
        return output


def load_pairs_manifest(manifest_path: Path) -> List[Dict]:
    """Load pairs manifest from preprocessing output."""
    manifest_path = Path(manifest_path)
    
    if manifest_path.suffix == ".csv":
        import csv
        pairs = []
        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append(dict(row))
    else:  # .jsonl
        pairs = []
        with open(manifest_path) as f:
            for line in f:
                pairs.append(json.loads(line.strip()))
    
    logger.info(f"Loaded {len(pairs)} pairs from {manifest_path}")
    return pairs


def create_synthesis_dataloaders(
    pairs: List[Dict],
    config: Optional[PatchConfig] = None,
    split_config: Optional[SplitConfig] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    val_fold: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders for 3T-7T synthesis.
    
    Args:
        pairs: List of pair dictionaries from preprocessing
        config: Patch configuration
        split_config: Split configuration
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        val_fold: Which fold to use for validation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    config = config or PatchConfig()
    split_config = split_config or SplitConfig()
    
    # Create subject-level split
    splitter = SubjectSplitter(split_config)
    train_pairs, val_pairs, test_pairs = splitter.get_split(
        pairs, val_fold=val_fold, test_fold=val_fold
    )
    
    # Create datasets
    train_dataset = PairedPatchDataset(
        train_pairs, 
        config=config, 
        augment=True, 
        cache_volumes=True
    )
    val_dataset = PairedPatchDataset(
        val_pairs, 
        config=config, 
        augment=False, 
        cache_volumes=True
    )
    test_dataset = PairedPatchDataset(
        test_pairs, 
        config=config, 
        augment=False, 
        cache_volumes=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    logger.info(f"Created synthesis dataloaders:")
    logger.info(f"  Train: {len(train_dataset)} patches, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} patches, {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} patches, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load pairs from preprocessing output
    pairs_path = Path("derivatives/topobrain-preproc/pairs.csv")
    
    if pairs_path.exists():
        pairs = load_pairs_manifest(pairs_path)
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_synthesis_dataloaders(
            pairs,
            batch_size=4,
            val_fold=0,
        )
        
        # Test loading a batch
        batch = next(iter(train_loader))
        print(f"Input shape: {batch['input'].shape}")
        print(f"Target shape: {batch['target'].shape}")
        print(f"Subjects: {batch['subject']}")
    else:
        print(f"Pairs manifest not found: {pairs_path}")
        print("Run preprocessing first: python -m scripts.preprocess_bids")
