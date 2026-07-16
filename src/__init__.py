"""
Core modules for the TopoBrain pipeline.

IMPORTS ARE GUARDED ON PURPOSE. The preprocessing / BIDS / harmonization / QC modules pull heavy,
optional dependencies (SimpleITK, ANTs, HD-BET, ...) that the TRAINING and EVALUATION paths never
use. Force-importing them here means a lean GPU environment (e.g. a CERN LCG view with only
torch/monai/nibabel/gudhi) cannot even `from src.model_cascaded import ...` -- the package init
crashes on `import SimpleITK` before the model is reached.

So each group below is wrapped in try/except: if its dependency is missing, those names are simply
unavailable, and the core training/eval modules still import cleanly. Nothing changes when the
full environment IS present.
"""
import logging as _logging

_log = _logging.getLogger(__name__)
_missing = []


def _try(groupname, importer):
    """Run an import group; on ImportError record it and continue instead of crashing."""
    try:
        importer()
    except ImportError as e:               # optional dependency absent -> degrade, don't die
        _missing.append(f"{groupname} ({e.name})")


# --- always-available core (torch/numpy/monai/nibabel/scipy only) --------------------------
from .config import get_default_config, get_highres_config, get_fast_config, MRIConfig
from .synthesis_dataset import (
    PairedPatchDataset, SubjectSplitter, InferencePatchSampler,
    PatchConfig, SplitConfig, load_pairs_manifest, create_synthesis_dataloaders,
)


# --- optional: heavy preprocessing / IO stacks ---------------------------------------------
def _imp_preprocessing():
    global MRIPreprocessor, BIDSPreprocessingPipeline, PipelineConfig
    from .preprocessing import MRIPreprocessor
    from .preprocess_pipeline import BIDSPreprocessingPipeline, PipelineConfig


def _imp_bids():
    global BIDSFile, discover_bids_files, create_3t_7t_pairs
    from .bids import BIDSFile, discover_bids_files, create_3t_7t_pairs


def _imp_dataset():
    global BrainMRIDataset, create_data_loaders, save_data_split, load_data_split, \
        compute_dataset_statistics
    from .dataset import (BrainMRIDataset, create_data_loaders, save_data_split,
                          load_data_split, compute_dataset_statistics)


def _imp_utils():
    global setup_logging, discover_dataset, create_patient_level_split, visualize_sample, \
        visualize_batch, compute_and_save_statistics, verify_preprocessing, set_random_seeds
    from .utils import (setup_logging, discover_dataset, create_patient_level_split,
                        visualize_sample, visualize_batch, compute_and_save_statistics,
                        verify_preprocessing, set_random_seeds)


def _imp_harmonization():
    global HistogramMatcher, IntensityHarmonizer
    from .harmonization import HistogramMatcher, IntensityHarmonizer


def _imp_qc():
    global QCMetrics, PreprocessingQC, AlignmentQC, MaskQualityValidator, MRIQCParser
    from .quality_control import (QCMetrics, PreprocessingQC, AlignmentQC,
                                  MaskQualityValidator, MRIQCParser)


for _name, _fn in (("preprocessing", _imp_preprocessing), ("bids", _imp_bids),
                   ("dataset", _imp_dataset), ("utils", _imp_utils),
                   ("harmonization", _imp_harmonization), ("quality_control", _imp_qc)):
    _try(_name, _fn)

if _missing:
    _log.debug("src: optional modules unavailable (missing deps): %s", ", ".join(_missing))

__all__ = [
    'get_default_config', 'get_highres_config', 'get_fast_config', 'MRIConfig',
    'PairedPatchDataset', 'SubjectSplitter', 'InferencePatchSampler',
    'PatchConfig', 'SplitConfig', 'load_pairs_manifest', 'create_synthesis_dataloaders',
    # the rest are available only when their optional dependencies are installed:
    'MRIPreprocessor', 'BIDSPreprocessingPipeline', 'PipelineConfig',
    'BrainMRIDataset', 'create_data_loaders', 'save_data_split', 'load_data_split',
    'compute_dataset_statistics',
    'setup_logging', 'discover_dataset', 'create_patient_level_split', 'visualize_sample',
    'visualize_batch', 'compute_and_save_statistics', 'verify_preprocessing', 'set_random_seeds',
    'BIDSFile', 'discover_bids_files', 'create_3t_7t_pairs',
    'HistogramMatcher', 'IntensityHarmonizer',
    'QCMetrics', 'PreprocessingQC', 'AlignmentQC', 'MaskQualityValidator', 'MRIQCParser',
]
