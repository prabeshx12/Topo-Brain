# 🧠 Topo-Brain: 3T→7T MRI Super-Resolution with GANs

**End-to-end pipeline for 3T-to-7T MRI super-resolution using 3D U-Net GANs**

Transform low-field (3T) brain MRI scans into high-field (7T) quality images using deep learning.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Colab](https://img.shields.io/badge/Colab-Ready-yellow)](https://colab.research.google.com/)

---

## 🚀 Quick Start

### **Google Colab (Recommended)**
1. Open `gan_training_colab_aligned.ipynb` in Colab
2. Mount your Google Drive with Aligned dataset
3. Run all cells → Start training immediately! 🎉

### **Local Setup**
```bash
git clone https://github.com/prabeshx12/Topo-Brain.git
cd Topo-Brain
pip install -r requirements.txt
python scripts/train_gan.py
```

---

## 🎯 Features

### **Preprocessing Pipeline**
✅ **HD-BET Skull Stripping** - Deep learning brain extraction  
✅ **N4 Bias Correction** - Optional intensity uniformity (disabled by default)  
✅ **Spatial Alignment** - Pre-registered dataset (256×304×308)  
✅ **Intensity Normalization** - Z-score on brain tissue  
✅ **Quality Control** - Automated visualizations and metrics  

### **GAN Architecture**
🔥 **3D U-Net Generator** - 5-level encoder-decoder with skip connections  
🔥 **3D PatchGAN Discriminator** - Multi-scale adversarial training  
🔥 **Paired Training** - Supervised learning with aligned 3T-7T pairs  
🔥 **Patient-Level Splits** - No data leakage (60/20/20 train/val/test)  
🔥 **Smart Augmentation** - MRI-specific transforms (rotation, intensity, noise)  

### **Production Ready**
⚡ **Colab Optimized** - Lightweight models for T4 GPU (32 base features)  
⚡ **Google Drive Integration** - Automatic data persistence  
⚡ **On-the-fly Normalization** - No preprocessing needed for quick start  
⚡ **Mixed Precision** - AMP for faster training  
⚡ **TensorBoard Logging** - Real-time monitoring  

---

## 📊 Dataset

**UNC 3T-7T Paired Brain MRI Dataset**

- **Subjects:** 10 (sub-01 to sub-10)
- **Sessions:** 
  - `ses-1`: 3T scans
  - `ses-2`: 7T scans
- **Modalities:** T1-weighted, T2-weighted
- **Format:** BIDS-compliant NIfTI (.nii.gz)
- **Spatial Alignment:** Pre-registered in `Aligned/` folder (256×304×308)

---

## 📁 Directory Structure

```
Topo-Brain/
├── 📓 Notebooks
│   ├── gan_training_colab_aligned.ipynb    ⭐ Main GAN training (Colab)
│   └── preprocessing_pipeline_colab.ipynb  ⭐ Preprocessing pipeline (Colab)
│
├── 📦 Source Code
│   ├── src/
│   │   ├── preprocessing.py                # N4, skull stripping, normalization
│   │   ├── config.py                       # Configuration management
│   │   ├── dataset.py                      # PyTorch datasets
│   │   ├── quality_control.py              # QC metrics
│   │   └── utils.py                        # Utilities
│   └── models/
│       ├── generator_unet3d.py             # 3D U-Net generator
│       ├── discriminator_patchgan3d.py     # 3D PatchGAN discriminator
│       └── paired_dataset.py               # Paired 3T-7T loader
│
├── 🔧 Scripts
│   ├── scripts/
│   │   ├── train_gan.py                    # Training script
│   │   ├── test_gan.py                     # Inference script
│   │   ├── eval_gan.py                     # Evaluation metrics
│   │   └── generate_brain_masks.py         # HD-BET wrapper
│   └── notebooks/
│       └── interactive_pipeline.ipynb      # Interactive analysis
│
├── 📊 Data (not in repo - too large)
│   ├── Nifti/                              # Raw BIDS data
│   ├── Aligned/                            # Pre-registered (use this!)
│   └── preprocessed/                       # Optional preprocessing output
│
├── 📝 Documentation
│   ├── docs/
│   │   ├── ARCHITECTURE.md
│   │   ├── GAN_README.md
│   │   └── CHANGELOG.md
│   ├── README.md                           # This file
│   ├── DIRECTORY_STRUCTURE.md              # Detailed structure
│   └── requirements.txt                    # Dependencies
│
└── 📂 Output (generated during runtime)
    ├── cache/                              # Temporary files
    ├── logs/                               # Training logs
    └── models/                             # Saved checkpoints
```

**See [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md) for complete details.**
│
├── src/                                   # 🐍 Core modules
│   ├── __init__.py
│   ├── config.py                          # Configuration management
│   ├── preprocessing.py                   # Preprocessing pipeline
│   ├── dataset.py                         # PyTorch Dataset classes
│   ├── utils.py                           # Utility functions
│   ├── harmonization.py                   # Intensity harmonization
│   └── quality_control.py                 # QC metrics & reports
│
├── models/                                # 🧠 GAN models
│   ├── __init__.py
│   ├── generator_unet3d.py                # 3D U-Net generator
│   ├── discriminator_patchgan3d.py        # PatchGAN discriminator
│   └── paired_dataset.py                  # 3T-7T paired dataset
│
├── scripts/                               # 🔧 Executable scripts
│   ├── generate_brain_masks.py            # HD-BET brain extraction
│   ├── train_gan.py                       # GAN training script
│   ├── eval_gan.py                        # GAN evaluation
│   ├── test_gan.py                        # Model testing
│   └── example_pipeline.py                # Pipeline demo
│
├── notebooks/                             # 📓 Jupyter notebooks
│   └── interactive_pipeline.ipynb         # Interactive demo
│
└── tests/                                 # ✅ Unit tests
    └── __init__.py
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
        item['path'],
        output_dir=config.data.output_root
    )
```

### 3. GAN Training (3T→7T Super-Resolution)

#### Create Dataset Splits
```python
from src.utils import create_patient_level_split
from models.paired_dataset import create_paired_data_list

# Patient-level split (no data leakage!)
train_data, val_data, test_data = create_patient_level_split(
    data_list,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    random_seed=42,
)

# Create 3T→7T pairs
train_pairs = create_paired_data_list(train_data, modality="T1w")
```

#### Train GAN
```bash
python scripts/train_gan.py \
    --data-root preprocessed/ \
    --output-dir checkpoints/baseline \
    --num-epochs 100 \
    --batch-size 2 \
    --patch-size 64 64 64 \
    --lambda-l1 100.0
```

#### Monitor Training
```bash
tensorboard --logdir checkpoints/baseline/logs
```

### 4. Evaluation
```bash
python scripts/eval_gan.py \
    --checkpoint checkpoints/baseline/best_generator.pth \
    --test-data preprocessed/ \
    --output-dir results/
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

# Customize splits
config.split.train_ratio = 0.7
config.split.val_ratio = 0.15
config.split.test_ratio = 0.15

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
4. **Intensity Normalization**: Z-score (recommended for GANs)

### GAN Architecture
- **Generator**: 3D U-Net with 5 levels, skip connections
- **Discriminator**: 3D PatchGAN (70×70×70 receptive field)
- **Loss**: L1 + Adversarial (λ_L1 = 100)
- **Training**: Adam optimizer, β1=0.5, β2=0.999
- **Input**: 64³ patches from 3T MRI
- **Output**: 64³ synthetic 7T MRI

## � Data Augmentation

### Training Augmentation (GAN)
- Random 3D affine transforms (rotation, translation, scaling)
- Random flipping (L-R, A-P, S-I)
- Random intensity shifts and scaling
- Random Gaussian noise
- Random Gaussian blur
- Optional: Gibbs ringing simulation

All configurable in [`models/paired_dataset.py`](models/paired_dataset.py)

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

checkpoints/                           # GAN training checkpoints
  ├── baseline/
  │   ├── generator_epoch_50.pth
  │   ├── discriminator_epoch_50.pth
  │   ├── best_generator.pth
  │   └── logs/                        # TensorBoard logs

results/                               # Evaluation outputs
  ├── generated_7T/
  ├── metrics.json
  └── visualizations/

cache/
  └── data_split.json                  # Reproducible splits

logs/
  ├── pipeline.log
  └── qc_reports/
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
- GANs can learn to handle bias fields
- Enable if needed: `config.preprocessing.use_bias_correction = True`

### Memory Requirements
- **GPU**: 8GB+ VRAM recommended for training (batch_size=2, patch=64³)
- **RAM**: 16GB+ for data loading
- Adjust batch size and patch size based on available memory

### Kaggle/Colab Usage
Use [`kaggle_preprocessing_notebook.ipynb`](kaggle_preprocessing_notebook.ipynb) for cloud preprocessing with free GPU

### Determinism
For full reproducibility:
```python
from utils import set_random_seeds
set_random_seeds(42)  # Sets seeds for random, numpy, torch
```

## 🐛 Troubleshooting

### GPU Out of Memory
```python
# Reduce batch size
python scripts/train_gan.py --batch-size 1

# Reduce patch size
python scripts/train_gan.py --patch-size 32 32 32

# Use CPU (slow but works)
python scripts/train_gan.py --device cpu
```

### Import Errors After Reorganization
```python
# Use new import paths
from src.config import get_default_config  # ✅
from config import get_default_config       # ❌ Old path
```

### HD-BET Model Download Issues
```bash
# Manual download if auto-download fails
python -c "from HD_BET.checkpoint_download import maybe_download_parameters; maybe_download_parameters()"
```

## 📚 Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - System architecture overview
- [`docs/GAN_README.md`](docs/GAN_README.md) - GAN model details
- [`docs/GAN_IMPLEMENTATION_SUMMARY.md`](docs/GAN_IMPLEMENTATION_SUMMARY.md) - Implementation notes
- [`docs/IMPROVEMENTS_IMPLEMENTED.md`](docs/IMPROVEMENTS_IMPLEMENTED.md) - Enhancement log

## 🔗 References

- **MONAI**: https://monai.io/
- **HD-BET**: https://github.com/MIC-DKFZ/HD-BET
- **Pix2Pix**: Isola et al. (2017) - Image-to-Image Translation with Conditional Adversarial Networks
- **3D U-Net**: Çiçek et al. (2016) - 3D U-Net: Learning Dense Volumetric Segmentation
- **BIDS Format**: https://bids.neuroimaging.io/

## 📄 License

MIT License - See LICENSE file for details

## ✨ Citation

If you use this code, please cite:
```bibtex
@software{topo_brain_2025,
  author = {Your Name},
  title = {Topo-Brain: 3T-to-7T MRI Super-Resolution with GANs},
  year = {2025},
  url = {https://github.com/prabeshx12/Topo-Brain}
}
```

---

**Dataset**: UNC Paired 3T-7T MRI Dataset  
**Interactive Demo**: [`notebooks/interactive_pipeline.ipynb`](notebooks/interactive_pipeline.ipynb)  
**Cloud Processing**: [`kaggle_preprocessing_notebook.ipynb`](kaggle_preprocessing_notebook.ipynb)
