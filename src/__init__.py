"""
Core modules for MRI preprocessing and harmonization pipeline.
"""

from .config import get_default_config, get_highres_config, get_fast_config, MRIConfig
from .preprocessing import MRIPreprocessor
from .preprocess_pipeline import BIDSPreprocessingPipeline, PipelineConfig
from .bids import BIDSFile, discover_bids_files, create_3t_7t_pairs
from .dataset import (
    BrainMRIDataset,
    create_data_loaders,
    save_data_split,
    load_data_split,
    compute_dataset_statistics,
)
from .utils import (
    setup_logging,
    discover_dataset,
    create_patient_level_split,
    visualize_sample,
    visualize_batch,
    compute_and_save_statistics,
    verify_preprocessing,
    set_random_seeds,
)
from .harmonization import HistogramMatcher, IntensityHarmonizer
from .quality_control import QCMetrics, PreprocessingQC

__all__ = [
    # Config
    'get_default_config',
    'get_highres_config',
    'get_fast_config',
    'MRIConfig',
    # Preprocessing
    'MRIPreprocessor',
    'BIDSPreprocessingPipeline',
    'PipelineConfig',
    # Dataset
    'BrainMRIDataset',
    'create_data_loaders',
    'save_data_split',
    'load_data_split',
    'compute_dataset_statistics',
    # Utils
    'setup_logging',
    'discover_dataset',
    'create_patient_level_split',
    'visualize_sample',
    'visualize_batch',
    'compute_and_save_statistics',
    'verify_preprocessing',
    'set_random_seeds',
    # BIDS discovery
    'BIDSFile',
    'discover_bids_files',
    'create_3t_7t_pairs',
    # Harmonization
    'HistogramMatcher',
    'IntensityHarmonizer',
    # Quality Control
    'QCMetrics',
    'PreprocessingQC',
]
