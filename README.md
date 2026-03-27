# Topo-Brain: 3T→7T MRI Super-Resolution with Diffusion Models

A complete pipeline for MRI preprocessing and 3T-to-7T super-resolution using anatomy-guided diffusion models. Includes brain extraction, preprocessing, and conditional diffusion training for generating high-field MRI images from low-field scans.

## 🎯 Features

### Preprocessing Pipeline
- **HD-BET Brain Extraction**: Deep learning-based skull stripping
- **N4 Bias Field Correction**: Optional intensity non-uniformity correction
- **Spatial Normalization**: RAS+ reorientation, isotropic resampling
- **Intensity Normalization**: Z-score, min-max, or percentile methods
- **Quality Control**: Automated QC metrics and outlier detection

### Diffusion Model (3T→7T Super-Resolution)
- **Anatomy-Guided U-Net**: 3D conditional U-Net with multi-task learning
- **Gaussian Diffusion**: DDPM-based training with 1000 timesteps
- **Multi-Task Learning**: Simultaneous denoising and segmentation
- **Paired Dataset**: Aligned 3T-7T pairs for supervised learning
- **Patient-Level Splits**: No data leakage between train/val/test
- **Advanced Augmentation**: MRI-specific augmentations (rotation, intensity)

### Production Features
- **Deterministic & Reproducible**: Fixed random seeds, saved splits
- **Config-Driven**: Flexible configuration with multiple presets
- **TensorBoard/Weights & Biases**: Real-time training monitoring
- **Mixed Precision Support**: AMP for faster training
- **EMA Model Tracking**: Exponential moving average for stable inference

## 📊 Dataset Structure

Your dataset follows the BIDS format:
- **10 subjects** (sub-01 to sub-10)
- **2 sessions per subject**:
  - `ses-1`: 3T scans
  - `ses-2`: 7T scans (~7T)
- **Modalities**: T1w and T2w

## 📁 Project Structure

```
Topo-Brain/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── pairs.csv                              # 3T-7T paired data manifest
│
├── configs/                               # 🔧 Configuration files
│   ├── dataset.yaml                       # Dataset configuration
│   ├── preprocess.yaml                    # Preprocessing config
│   └── train_diffusion.yaml               # Training configuration
│
├── docs/                                  # 📚 Documentation
│   ├── ARCHITECTURE.md                    # System architecture
│   ├── CHANGELOG.md                       # Version history
│   ├── IMPROVEMENTS_IMPLEMENTED.md        # Enhancement log
│   └── RUN_PREPROCESSING.md               # Preprocessing guide
│
├── src/                                   # 🐍 Core modules
│   ├── __init__.py
│   ├── config.py                          # Configuration management
│   ├── preprocessing.py                   # Preprocessing pipeline
│   ├── dataset.py                         # PyTorch Dataset classes
│   ├── synthesis_dataset.py               # Diffusion dataset loader
│   ├── model.py                           # Anatomy-Guided U-Net
│   ├── diffusion.py                       # Gaussian Diffusion logic
│   ├── utils.py                           # Utility functions
│   ├── harmonization.py                   # Intensity harmonization
│   └── quality_control.py                 # QC metrics & reports
│
├── models/                                # 🧠 Model checkpoints (saved during training)
│   └── __init__.py
│
├── scripts/                               # 🔧 Executable scripts
│   ├── generate_brain_masks.py            # HD-BET brain extraction
│   ├── preprocess_bids.py                 # BIDS preprocessing
│   ├── regenerate_pairs.py                # Create pairs.csv
│   ├── train_diffusion.py                 # Diffusion training script
│   ├── sample_diffusion.py                # Generate samples
│   ├── visualize_dataset.py               # Dataset visualization
│   └── example_pipeline.py                # Pipeline demo
│
├── notebooks/                             # 📓 Jupyter notebooks
│   └── visualization_demo.ipynb           # Interactive demo
│
└── tests/                                 # ✅ Unit tests
    ├── test_diffusion_logic.py            # Diffusion tests
    └── test_synthesis_dataset.py          # Dataset tests
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/prabeshx12/Topo-Brain.git
cd Topo-Brain

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install HD-BET for brain extraction
pip install HD-BET
```

### 2. Preprocessing Pipeline

#### BIDS Preprocessing CLI (recommended)
```bash
python -m scripts.preprocess_bids \
    --config configs/preprocess.yaml \
    --data-root /path/to/BIDS \
    --output-root /path/to/BIDS/derivatives/topobrain-preproc
```

#### Generate Brain Masks
```bash
python scripts/generate_brain_masks.py \
    --method hd-bet \
    --device cuda \
    --mode accurate \
    --data-root Nifti/
```

#### Preprocess MRI Data
```python
from src.config import get_default_config
from src.preprocessing import MRIPreprocessor
from src.utils import discover_dataset

# Load configuration
config = get_default_config()

# Discover dataset
data_list = discover_dataset(config.data.data_root, config.data)

# Preprocess
preprocessor = MRIPreprocessor(config.preprocessing)
for item in data_list:
preprocessor.preprocess_single(
        item['image'],
        output_path=config.data.output_root / f"{item['subject']}_{item['session']}_{item['modality']}_preprocessed.nii.gz"
    )
```

### 3. Diffusion Model Training (3T→7T Super-Resolution)

#### Create Dataset Pairs
```bash
python scripts/regenerate_pairs.py \
    --data-root preprocessed/ \
    --output pairs.csv \
    --modality T1w
```

#### Train Diffusion Model
```bash
python scripts/train_diffusion.py \
    --config configs/train_diffusion.yaml \
    --data-root preprocessed/
```

#### With Weights & Biases Tracking
```bash
python scripts/train_diffusion.py \
    --config configs/train_diffusion.yaml \
    --use-wandb \
    --wandb-project topobrain
```

#### Monitor Training
```bash
tensorboard --logdir logs/
```

### 4. Sample Generation
```bash
python scripts/sample_diffusion.py \
    --checkpoint models/best_model.pth \
    --input-3t preprocessed/sub-01_ses-1_T1w.nii.gz \
    --output generated_7t.nii.gz \
    --timesteps 1000
```

## ⚙️ Configuration

### Configuration Presets

```python
from src.config import get_default_config, get_highres_config, get_fast_config

# Default: No resampling, z-score norm, N4 disabled (preserves detail)
config = get_default_config()

# High-res: 0.5mm isotropic, 256³ volumes
config = get_highres_config()

# Fast: 2.0mm isotropic, 96³ volumes (for prototyping)
config = get_fast_config()
```

### Custom Configuration
```python
from src.config import MRIConfig

config = MRIConfig()

# Preprocessing
config.preprocessing.target_spacing = (1.0, 1.0, 1.0)
config.preprocessing.normalization_method = "zscore"
config.preprocessing.use_bias_correction = False  # Preserves anatomical detail

# Customize training
config.training.batch_size = 4
config.training.num_workers = 8

config.validate()
```

## 🔬 Pipeline Details

### HD-BET Brain Extraction
- Deep learning-based skull stripping (nnU-Net architecture)
- Automatic model download (~100MB from Zenodo)
- GPU acceleration supported
- Modes: `fast`, `accurate`

### Preprocessing Steps
1. **Brain Extraction**: HD-BET for accurate skull stripping
2. **Bias Field Correction**: Optional N4ITK (disabled by default to preserve detail)
3. **Spatial Transforms**: RAS+ reorientation, optional resampling
4. **Intensity Normalization**: Z-score (recommended for diffusion models)

### Diffusion Model Architecture
- **Backbone**: 3D U-Net with anatomy-guided conditioning
- **Conditioning**: 3T input concatenated with noisy 7T
- **Multi-Task**: Simultaneous denoising and tissue segmentation
- **Diffusion Process**: DDPM with 1000 timesteps
- **Noise Schedule**: Linear or cosine beta schedule
- **Loss**: L1 reconstruction + Perceptual loss + Segmentation loss
- **Training**: AdamW optimizer with EMA model tracking
- **Input**: 3D patches from paired 3T-7T MRI
- **Output**: High-quality synthetic 7T MRI

## � Data Augmentation

### Training Augmentation (Diffusion Model)
- Random 3D affine transforms (rotation, translation, scaling)
- Random flipping (L-R, A-P, S-I)
- Random intensity shifts and scaling
- Random Gaussian noise
- Random Gaussian blur

All configurable in [`src/synthesis_dataset.py`](src/synthesis_dataset.py)

## 🎯 Patient-Level Splitting

**Critical for medical imaging**: Ensures no data leakage!

```python
# All sessions from the same patient go to the same split
train_subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']
val_subjects = ['sub-07', 'sub-08']
test_subjects = ['sub-09', 'sub-10']

# This means both ses-1 AND ses-2 from sub-01 are in training set
```

The split is:
- **Deterministic**: Same random seed = same split
- **Saved**: JSON file for reproducibility
- **Verified**: Automatic leakage detection

## 📊 Dataset Statistics

The pipeline automatically computes and logs:
- Mean and standard deviation
- Min/max values
- Percentiles (1st, 99th)
- Median
- Shape distributions

Example output:
```
Dataset statistics:
  mean: 0.0234
  std: 0.9876
  min: -3.4567
  max: 4.5678
  median: 0.0123
```

## 🖼️ Visualization

Automatic visualization of:
- Preprocessing verification (before/after comparison)
- Sample batches from train/val/test sets
- Intensity distributions
- Orthogonal slices (sagittal, coronal, axial)

Saved to `logs/visualizations/`

## 🔧 Advanced Usage

### Multi-Modal Learning
```python
from dataset import MultiModalBrainMRIDataset

# Load T1w and T2w together
dataset = MultiModalBrainMRIDataset(
    data_list,
    modalities=["T1w", "T2w"],
    transform=transform,
)

# Each sample will have shape: (2, H, W, D)
```

### K-Fold Cross-Validation
```python
from utils import create_kfold_splits

folds = create_kfold_splits(data_list, k_folds=5, random_seed=42)

for fold_idx, (train_data, val_data) in enumerate(folds):
    print(f"Fold {fold_idx + 1}")
    # Train your model...
```

### Custom Transforms
```python
from monai.transforms import Compose, RandRotated

custom_transform = Compose([
    RandRotated(keys=["image"], range_x=0.5, prob=0.5),
    # Add your custom transforms...
])

dataset = BrainMRIDataset(data_list, transform=custom_transform)
```

## 📝 File Outputs

```
preprocessed/                          # Preprocessed volumes
  ├── sub-01_ses-1_T1w_preprocessed.nii.gz
  ├── sub-01_ses-1_T1w_preprocessed_metadata.json
  └── ...

pairs.csv                              # 3T-7T paired data manifest

models/                                # Training checkpoints
  ├── diffusion_model_iter_10000.pth
  ├── diffusion_model_iter_20000.pth
  ├── best_model.pth
  └── ema_model.pth

logs/                                  # Training logs
  ├── tensorboard/                     # TensorBoard logs
  ├── samples/                         # Generated samples during training
  └── training.log

results/                               # Evaluation outputs
  ├── generated_7T/
  └── metrics.json
```

## ⚠️ Important Notes

### HD-BET Installation
```bash
# Install HD-BET for brain extraction
pip install HD-BET

# Models auto-download on first use (~100MB)
```

### N4 Bias Correction
- **Disabled by default** to preserve anatomical detail
- Diffusion models can learn to handle bias fields
- Enable if needed: `config.preprocessing.use_bias_correction = True`

### Memory Requirements
- **GPU**: 16GB+ VRAM recommended for training (batch_size=4, patch=64³)
- **RAM**: 16GB+ for data loading
- Adjust batch size and patch size based on available memory

### Training Configuration
For training configuration options, see [`configs/train_diffusion.yaml`](configs/train_diffusion.yaml)

### Determinism
For full reproducibility:
```python
from utils import set_random_seeds
set_random_seeds(42)  # Sets seeds for random, numpy, torch
```

## 🐛 Troubleshooting

### GPU Out of Memory
```bash
# Reduce batch size
python scripts/train_diffusion.py --config configs/train_diffusion.yaml
# Edit batch_size in config file

# Use gradient accumulation for effective larger batch
# Edit accumulation_steps in config file
```

### Import Errors
```python
# Use correct import paths
from src.model import AnatomyGuidedUNet  # ✅
from src.diffusion import GaussianDiffusion  # ✅
```

### HD-BET Model Download Issues
```bash
# Manual download if auto-download fails
python -c "from HD_BET.checkpoint_download import maybe_download_parameters; maybe_download_parameters()"
```

## 📚 Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - System architecture overview
- [`docs/RUN_PREPROCESSING.md`](docs/RUN_PREPROCESSING.md) - Preprocessing guide
- [`docs/IMPROVEMENTS_IMPLEMENTED.md`](docs/IMPROVEMENTS_IMPLEMENTED.md) - Enhancement log
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) - Version history

## 🔗 References

- **MONAI**: https://monai.io/
- **HD-BET**: https://github.com/MIC-DKFZ/HD-BET
- **DDPM**: Ho et al. (2020) - Denoising Diffusion Probabilistic Models
- **3D U-Net**: Çiçek et al. (2016) - 3D U-Net: Learning Dense Volumetric Segmentation
- **BIDS Format**: https://bids.neuroimaging.io/

## 📄 License

MIT License - See LICENSE file for details

## ✨ Citation

If you use this code, please cite:
```bibtex
@software{topo_brain_2026,
  author = {Your Name},
  title = {Topo-Brain: 3T-to-7T MRI Super-Resolution with Diffusion Models},
  year = {2026},
  url = {https://github.com/prabeshx12/Topo-Brain}
}
```

---

**Dataset**: UNC Paired 3T-7T MRI Dataset  
**Interactive Demo**: [`notebooks/visualization_demo.ipynb`](notebooks/visualization_demo.ipynb)
