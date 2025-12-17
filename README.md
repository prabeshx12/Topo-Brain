# Brain MRI Preprocessing and Data Loading Pipeline

A production-grade, deterministic preprocessing and data loading pipeline for 3D brain MRI using MONAI and PyTorch.

## 🎯 Features

### Core Features
- **Deterministic & Reproducible**: Patient-level splits with fixed random seeds
- **Comprehensive Preprocessing**:
  - Reorientation to RAS+ coordinate system
  - N4ITK bias field correction
  - Skull stripping (with automatic fallback)
  - Intensity normalization (z-score, min-max, percentile)
  - Optional isotropic resampling
- **Flexible Configuration**: Config-driven with multiple presets (default, high-res, fast)
- **Patient-Level Splits**: No data leakage between train/val/test sets
- **Multi-Field Strength Support**: Handles both 3T and 7T scans
- **Production-Ready**: Modular, extensible, well-documented code

### Advanced Features (NEW)
- **Quality Control (QC)**: Automated QC metrics (SNR, CNR, entropy), artifact detection, outlier identification, HTML reports
- **Intensity Harmonization**: Multi-scanner harmonization for 3T/7T data (histogram matching, z-score, quantile normalization)
- **Enhanced Augmentation**: MRI-specific augmentations (Gibbs ringing, coarse dropout)
- **TensorBoard Integration**: Real-time monitoring of training metrics, image logging, histogram tracking
- **Smart Caching**: Persistent disk cache and in-memory cache for faster data loading
- **Mixed Precision Support**: Automatic Mixed Precision (AMP) for faster training
- **Cross-Validation**: Built-in k-fold cross-validation support

## 📊 Dataset Structure

Your dataset follows the BIDS format:
- **10 subjects** (sub-01 to sub-10)
- **2 sessions per subject**:
  - `ses-1`: 3T scans
  - `ses-2`: 7T scans (~7T)
- **Modalities**: T1w and T2w

## 📁 Project Structure

```
major_/
├── config.py                      # Configuration management
├── preprocessing.py               # Preprocessing pipeline
├── dataset.py                     # PyTorch Dataset & DataLoader
├── utils.py                       # Utilities (splitting, visualization, logging)
├── quality_control.py             # QC metrics and reporting (NEW)
├── harmonization.py               # Intensity harmonization (NEW)
├── example_pipeline.py            # End-to-end example script
├── example_qc_harmonization.py    # QC and harmonization examples (NEW)
├── setup_validation.py            # Environment validation
├── generate_brain_masks.py        # Brain mask generation tool
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── IMPROVEMENTS.md                # Recommended enhancements
├── PROJECT_SUMMARY.md             # Executive summary
├── ARCHITECTURE.md                # Architecture overview
└── notebooks/
    └── interactive_pipeline.ipynb # Jupyter notebook demo
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from config import get_default_config
from preprocessing import MRIPreprocessor
from utils import discover_dataset, create_patient_level_split
from dataset import create_data_loaders

# Load configuration
config = get_default_config()

# Discover dataset
data_list = discover_dataset(config.data.data_root, config.data)

# Create patient-level split (no data leakage!)
train_data, val_data, test_data = create_patient_level_split(
    data_list,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    random_seed=42,
)

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(
    config, train_data, val_data, test_data
)

# Iterate through data
for batch in train_loader:
    images = batch["image"]  # Shape: (B, 1, H, W, D)
    subjects = batch["subject"]
    # Your training code here...
```

### 3. Run Complete Pipeline

```bash
# Run pipeline with preprocessing
python example_pipeline.py --preprocess --config default

# Run without preprocessing (using raw data)
python example_pipeline.py --config fast

# Available configs: default, highres, fast
```

### 4. Advanced Features

#### Quality Control (QC)
```bash
# Run quality control and harmonization examples
python example_qc_harmonization.py
```

```python
from quality_control import PreprocessingQC, QCMetrics

# Compute QC metrics for a single scan
metrics = QCMetrics.compute_metrics("path/to/scan.nii.gz")
print(f"SNR: {metrics['snr']:.2f}")
print(f"CNR: {metrics['cnr']:.2f}")

# Run comprehensive QC on multiple scans
qc = PreprocessingQC(output_dir="logs/qc")
qc_df = qc.run_qc(image_paths)
qc.generate_report(qc_df, "qc_report.html")
```

#### Intensity Harmonization (3T/7T)
```python
from harmonization import IntensityHarmonizer

# Create harmonizer
harmonizer = IntensityHarmonizer(method='histogram')  # or 'zscore', 'quantile'

# Fit on reference scans (e.g., 3T)
harmonizer.fit(scans_3t)

# Transform target scans (e.g., 7T)
harmonized_data = harmonizer.transform(scan_7t)

# Save harmonizer for later use
harmonizer.save("harmonizer.pkl")
```

#### TensorBoard Logging
```python
from utils import TensorBoardLogger

# Create logger
tb_logger = TensorBoardLogger(log_dir="logs/tensorboard")

# Log metrics
tb_logger.log_scalar("train/loss", loss_value, step)
tb_logger.log_image("train/predictions", image_tensor, step)
tb_logger.log_volume_slices("train/volume", volume_tensor, step)

# View in TensorBoard
# tensorboard --logdir=logs/tensorboard
```

#### Enhanced Data Caching
```python
# Use persistent disk cache (faster restarts)
train_loader, val_loader, test_loader = create_data_loaders(
    config, train_data, val_data, test_data,
    use_persistent_cache=True  # Caches to disk
)

# Use in-memory cache (faster iteration, good for small datasets)
train_loader, val_loader, test_loader = create_data_loaders(
    config, train_data, val_data, test_data,
    use_memory_cache=True  # Caches to RAM
)
```

#### Mixed Precision Training
```python
# Enable in config
config.training.use_amp = True

# In training loop
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        output = model(batch["image"])
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## ⚙️ Configuration Options

The pipeline supports three configuration presets:

### Default Configuration
- No resampling (preserves original resolution)
- Z-score normalization
- N4 bias correction enabled
- Batch size: 2

### High-Resolution Configuration
```python
from config import get_highres_config
config = get_highres_config()
# - Target spacing: 0.5mm isotropic
# - Target size: 256³
# - Batch size: 1
```

### Fast Configuration (for prototyping)
```python
from config import get_fast_config
config = get_fast_config()
# - Target spacing: 2.0mm isotropic
# - Target size: 96³
# - No bias correction
# - Batch size: 4
```

### Custom Configuration
```python
from config import MRIConfig

config = MRIConfig()

# Customize preprocessing
config.preprocessing.target_spacing = (1.0, 1.0, 1.0)
config.preprocessing.normalization_method = "percentile"
config.preprocessing.use_bias_correction = True

# Customize splits
config.split.train_ratio = 0.7
config.split.val_ratio = 0.15
config.split.test_ratio = 0.15

# Customize training
config.training.batch_size = 4
config.training.num_workers = 8

config.validate()
```

## 🔬 Preprocessing Pipeline Details

### Step 1: Bias Field Correction
Uses **N4ITK algorithm** (SimpleITK implementation) for intensity non-uniformity correction:
- Adaptive for both 3T and 7T data
- Automatic masking via Otsu thresholding
- Configurable iterations and convergence

### Step 2: Skull Stripping
- **Priority**: Uses pre-computed brain masks if available
- **Fallback**: Simple thresholding + morphological operations
- **Recommendation**: For production, generate masks using:
  - HD-BET
  - SynthStrip
  - FSL BET
  - ANTs

### Step 3: Spatial Transforms
- **Reorientation**: Converts to RAS+ orientation
- **Resampling**: Optional isotropic spacing
- **Padding/Cropping**: Optional fixed size

### Step 4: Intensity Normalization
Three methods available:
- **Z-score**: `(x - mean) / std` (recommended for MRI)
- **Min-Max**: `(x - min) / (max - min)`
- **Percentile**: Robust to outliers

## 📈 Data Augmentation

Training augmentation (enabled by default):
- Random affine transformations (rotation, translation, scaling)
- Random flipping (left-right)
- Random intensity shifts and scaling
- Random Gaussian noise
- Random Gaussian smoothing

All configurable via `config.augmentation`.

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

Running the pipeline creates:
```
preprocessed/          # Preprocessed NIfTI files
  ├── sub-01_ses-1_T1w_preprocessed.nii.gz
  ├── sub-01_ses-1_T1w_preprocessed_metadata.json
  └── ...

cache/
  └── data_split.json  # Reproducible split information

logs/
  ├── pipeline.log     # Detailed execution log
  ├── statistics/
  │   ├── train_statistics.json
  │   ├── val_statistics.json
  │   └── test_statistics.json
  └── visualizations/
      ├── train/
      ├── val/
      └── test/
```

## ⚠️ Important Notes

### Skull Stripping
The current implementation uses **simple thresholding** as a fallback. For production use:

1. **Generate proper brain masks** using specialized tools:
   ```bash
   # Example with HD-BET
   hd-bet -i input.nii.gz -o output.nii.gz
   
   # Example with SynthStrip (FreeSurfer)
   mri_synthstrip -i input.nii.gz -o output.nii.gz -m mask.nii.gz
   ```

2. Place masks following this naming convention:
   ```
   sub-01/ses-1/anat/sub-01_ses-1_T1w_brain_mask.nii.gz
   ```

### Memory Usage
- 3D volumes can be large (especially 7T high-res)
- Adjust `batch_size` based on available GPU memory
- Use `num_workers > 0` for faster data loading
- Consider `cache_data=True` for small datasets

### Determinism
For full reproducibility:
```python
from utils import set_random_seeds
set_random_seeds(42)  # Sets seeds for random, numpy, torch
```

## 🐛 Troubleshooting

### Issue: Out of Memory
```python
# Reduce batch size
config.training.batch_size = 1

# Reduce target size
config.preprocessing.target_size = (96, 96, 96)

# Disable caching
dataset = BrainMRIDataset(data_list, cache_data=False)
```

### Issue: Slow Data Loading
```python
# Increase workers
config.training.num_workers = 8

# Enable prefetching
config.training.prefetch_factor = 4

# Pin memory for GPU
config.training.pin_memory = True
```

### Issue: N4 Correction Fails
```python
# Disable bias correction
config.preprocessing.use_bias_correction = False

# Or adjust parameters
config.preprocessing.n4_iterations = 30
```

## 📚 References

- **MONAI**: https://monai.io/
- **BIDS**: https://bids.neuroimaging.io/
- **N4ITK**: Tustison et al. (2010) IEEE TMI
- **SimpleITK**: https://simpleitk.org/

## 🤝 Contributing

Suggestions for improvement:
1. **Better skull stripping**: Integrate HD-BET or SynthStrip
2. **Registration**: Add inter-subject/inter-session registration
3. **Quality control**: Automated QC metrics and outlier detection
4. **Performance**: Multi-GPU support for preprocessing
5. **Formats**: Support DICOM input

## 📄 License

This pipeline is provided as-is for research and educational purposes.

## ✨ Acknowledgments

Dataset: UNC Paired 3T-7T dataset (Chen et al.)

---

Example notebook: `notebooks/interactive_pipeline.ipynb`
