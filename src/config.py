"""
Configuration file for MRI preprocessing pipeline.
Centralizes all paths, hyperparameters, and experiment settings.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Optional, List


@dataclass
class DataConfig:
    """Data paths and organization."""
    # Root directory containing BIDS-formatted dataset
    data_root: Path = Path(r"d:\11PrabeshX\Projects\major_\Nifti")
    
    # Output directory for preprocessed data
    output_root: Path = Path(r"d:\11PrabeshX\Projects\major_\preprocessed")
    
    # Cache directory for intermediate results
    cache_dir: Path = Path(r"d:\11PrabeshX\Projects\major_\cache")
    
    # Number of subjects in dataset
    num_subjects: int = 10
    
    # Session identifiers for 3T and 7T
    session_3t: str = "ses-1"
    session_7t: str = "ses-2"
    
    # Modalities to process
    modalities: List[str] = field(default_factory=lambda: ["T1w", "T2w"])
    
    # File naming pattern
    file_pattern: str = "*_defaced.nii.gz"
    

@dataclass
class PreprocessingConfig:
    """Preprocessing pipeline parameters."""
    # Target orientation (RAS = Right-Anterior-Superior)
    target_orientation: str = "RAS"
    
    # Target spacing for isotropic resampling (None to skip)
    # Recommended: [1.0, 1.0, 1.0] for 1mm isotropic
    # For GAN: Set to None to preserve original resolution, or match lowest resolution
    target_spacing: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0)
    
    # Interpolation mode for resampling
    interpolation_mode: str = "trilinear"  # or "nearest" for labels
    
    # Bias field correction (N4ITK)
    # NOTE: For GAN training, consider setting to False to preserve fine anatomical details
    use_bias_correction: bool = False  # Set True for 7T data with strong bias
    n4_iterations: int = 20  # Reduced from 50 to preserve detail
    n4_convergence_threshold: float = 0.001
    
    # Skull stripping
    use_skull_stripping: bool = True
    brain_mask_pattern: str = "*brain_mask.nii.gz"  # If masks are available
    
    # Intensity normalization method
    normalization_method: str = "zscore"  # Options: "zscore", "minmax", "percentile"
    
    # For percentile normalization
    percentile_lower: float = 1.0
    percentile_upper: float = 99.0
    
    # Clipping values (to remove outliers before normalization)
    clip_values: bool = True
    clip_lower_percentile: float = 0.5
    clip_upper_percentile: float = 99.5
    
    # Padding/Cropping to fixed size (None to skip)
    target_size: Optional[Tuple[int, int, int]] = None  # e.g., (128, 128, 128)
    
    # Number of workers for parallel processing
    num_workers: int = 4
    

@dataclass
class DataSplitConfig:
    """Train/validation/test split configuration."""
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Split ratios (must sum to 1.0)
    train_ratio: float = 0.6  # 6 subjects
    val_ratio: float = 0.2    # 2 subjects
    test_ratio: float = 0.2   # 2 subjects
    
    # Patient-level split (no data leakage between sessions)
    patient_level_split: bool = True
    
    # K-fold cross-validation (set to None for single split)
    k_folds: Optional[int] = None  # e.g., 5 for 5-fold CV
    

@dataclass
class TrainingConfig:
    """Model training hyperparameters."""
    # Batch size
    batch_size: int = 2  # Small for 3D volumes
    
    # Number of workers for DataLoader
    num_workers: int = 4
    
    # Pin memory for faster GPU transfer
    pin_memory: bool = True
    
    # Prefetch factor
    prefetch_factor: int = 2
    
    # Shuffle training data
    shuffle_train: bool = True
    
    # Drop last incomplete batch
    drop_last: bool = False
    
    # Mixed precision training
    use_amp: bool = False  # Automatic Mixed Precision
    
    # Data caching
    use_persistent_cache: bool = False  # Disk cache
    use_memory_cache: bool = False      # RAM cache (good for small datasets)
    

@dataclass
class AugmentationConfig:
    """Data augmentation parameters (for training)."""
    # Enable augmentation
    use_augmentation: bool = True
    
    # Random affine transformation
    random_affine_prob: float = 0.5
    rotate_range: Tuple[float, float, float] = (0.1, 0.1, 0.1)  # radians
    translate_range: Tuple[float, float, float] = (10, 10, 10)  # pixels
    scale_range: Tuple[float, float, float] = (0.1, 0.1, 0.1)   # scale factor
    
    # Random flip
    random_flip_prob: float = 0.5
    flip_axes: Tuple[int, ...] = (0,)  # Flip left-right only
    
    # Random intensity shift
    intensity_shift_prob: float = 0.3
    intensity_shift_offset: float = 0.1
    
    # Random intensity scale
    intensity_scale_prob: float = 0.3
    intensity_scale_factor: float = 0.1
    
    # Random Gaussian noise
    gaussian_noise_prob: float = 0.2
    gaussian_noise_std: float = 0.01
    
    # Random Gaussian smooth
    gaussian_smooth_prob: float = 0.2
    gaussian_smooth_sigma: Tuple[float, float] = (0.5, 1.5)
    
    # Enhanced augmentation (optional)
    coarse_dropout_prob: float = 0.0  # Set > 0 to enable
    gibbs_noise_prob: float = 0.0     # Set > 0 to enable (MRI-specific artifact)
    

@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    # Log directory
    log_dir: Path = Path(r"d:\11PrabeshX\Projects\major_\logs")
    
    # Log level
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    # Save dataset statistics
    save_statistics: bool = True
    
    # Visualize samples
    save_visualizations: bool = True
    num_viz_samples: int = 3
    
    # TensorBoard logging
    use_tensorboard: bool = True
    tensorboard_dir: Optional[Path] = None  # If None, uses log_dir/tensorboard
    
    # Quality Control
    run_qc: bool = True
    qc_dir: Optional[Path] = None  # If None, uses log_dir/qc
    

@dataclass
class MRIConfig:
    """Master configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    split: DataSplitConfig = field(default_factory=DataSplitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.data.output_root.mkdir(parents=True, exist_ok=True)
        self.data.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logging.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default paths for tensorboard and qc if not specified
        if self.logging.tensorboard_dir is None:
            self.logging.tensorboard_dir = self.logging.log_dir / "tensorboard"
        if self.logging.qc_dir is None:
            self.logging.qc_dir = self.logging.log_dir / "qc"
        
        self.logging.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.logging.qc_dir.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # Check split ratios sum to 1.0
        total_ratio = self.split.train_ratio + self.split.val_ratio + self.split.test_ratio
        assert abs(total_ratio - 1.0) < 1e-6, f"Split ratios must sum to 1.0, got {total_ratio}"
        
        # Check data root exists
        assert self.data.data_root.exists(), f"Data root does not exist: {self.data.data_root}"
        
        # Check normalization method is valid
        valid_norms = ["zscore", "minmax", "percentile"]
        assert self.preprocessing.normalization_method in valid_norms, \
            f"Invalid normalization method: {self.preprocessing.normalization_method}"
        
        print("✓ Configuration validated successfully")


# Default configuration instance
def get_default_config() -> MRIConfig:
    """Get default configuration with recommended settings."""
    config = MRIConfig()
    config.validate()
    return config


# Example: Configuration for high-resolution processing
def get_highres_config() -> MRIConfig:
    """Configuration for high-resolution (0.5mm isotropic) processing."""
    config = MRIConfig()
    config.preprocessing.target_spacing = (0.5, 0.5, 0.5)
    config.preprocessing.target_size = (256, 256, 256)
    config.training.batch_size = 1  # Larger volumes require smaller batch
    config.validate()
    return config


# Example: Configuration for fast prototyping
def get_fast_config() -> MRIConfig:
    """Configuration for quick experimentation (lower resolution)."""
    config = MRIConfig()
    config.preprocessing.target_spacing = (2.0, 2.0, 2.0)
    config.preprocessing.target_size = (96, 96, 96)
    config.preprocessing.use_bias_correction = False  # Skip for speed
    config.training.batch_size = 4
    config.validate()
    return config


if __name__ == "__main__":
    # Example usage
    config = get_default_config()
    print(f"Data root: {config.data.data_root}")
    print(f"Output root: {config.data.output_root}")
    print(f"Target spacing: {config.preprocessing.target_spacing}")
    print(f"Train/Val/Test split: {config.split.train_ratio}/{config.split.val_ratio}/{config.split.test_ratio}")
