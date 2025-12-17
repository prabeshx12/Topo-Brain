# 🚀 Topo-Brain: Complete Workflow Guide

**3T → 7T MRI Super-Resolution with Baseline 3D GAN**

This guide provides step-by-step instructions for running the complete pipeline from raw data to trained model evaluation.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Step 1: Preprocessing](#step-1-preprocessing)
5. [Step 2: GAN Training](#step-2-gan-training)
6. [Step 3: Evaluation](#step-3-evaluation)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

---

## Prerequisites

### Required
- **Python**: 3.8+
- **CUDA**: 11.0+ (for GPU training)
- **RAM**: 16GB+ recommended
- **Storage**: ~10GB for preprocessed data + checkpoints

### Dataset Structure
Your data should follow BIDS format:
```
Nifti/
├── sub-01/
│   ├── ses-1/          # 3T scans
│   │   └── anat/
│   │       ├── sub-01_ses-1_T1w.json
│   │       └── sub-01_ses-1_T1w.nii.gz (or .nii)
│   └── ses-2/          # 7T scans
│       └── anat/
│           ├── sub-01_ses-2_T1w.json
│           └── sub-01_ses-2_T1w.nii.gz
├── sub-02/
│   └── ...
└── sub-10/
    └── ...
```

**Key Points:**
- **Session 1 (ses-1)**: 3T scans (input)
- **Session 2 (ses-2)**: 7T scans (target)
- 10 subjects total
- T1w and T2w modalities supported

---

## Environment Setup

### 1. Clone/Navigate to Project
```powershell
cd D:\11PrabeshX\Projects\major_
```

### 2. Install Dependencies
```powershell
# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install monai nibabel numpy pandas matplotlib scikit-image scipy SimpleITK

# Optional: Install development tools
pip install tensorboard ipykernel jupyter
```

### 3. Verify Installation
```powershell
python setup_validation.py
```

**Expected Output:**
```
✓ Python version: 3.x.x
✓ PyTorch: 2.x.x
✓ CUDA available: True
✓ MONAI: 1.x.x
✓ All dependencies satisfied
```

---

## Data Preparation

### Check Your Data Structure
```powershell
# List your data
ls Nifti/

# Should show: sub-01, sub-02, ..., sub-10
```

### Verify Data Completeness
The pipeline will automatically:
- Discover all subjects and sessions
- Pair 3T (ses-1) with 7T (ses-2)
- Create patient-level splits (train/val/test)

---

## Step 1: Preprocessing

Preprocessing applies standardized transformations to prepare raw MRI for training.

### What It Does
1. **Reorientation**: Aligns to RAS+ coordinate system
2. **Bias Field Correction**: Removes intensity inhomogeneity (N4ITK)
3. **Skull Stripping**: Removes non-brain tissue
4. **Normalization**: Z-score intensity normalization
5. **Optional Resampling**: Isotropic spacing (configurable)

### Quick Start (Default Settings)
```powershell
python example_pipeline.py --preprocess
```

### Custom Configuration
```powershell
# High-resolution preprocessing (1mm isotropic)
python example_pipeline.py --preprocess --config highres

# Fast preprocessing (skip bias correction)
python example_pipeline.py --preprocess --config fast
```

### Expected Output
```
[INFO] Discovering dataset...
[INFO] Found 10 subjects with 20 sessions
[INFO] Creating patient-level split...
[INFO]   Train: 6 subjects (12 scans)
[INFO]   Val: 2 subjects (4 scans)
[INFO]   Test: 2 subjects (4 scans)
[INFO] Starting preprocessing...
[INFO] Processing sub-01_ses-1_T1w... ✓
[INFO] Processing sub-01_ses-2_T1w... ✓
...
[INFO] Preprocessing complete!
[INFO] Preprocessed data saved to: preprocessed/
```

### Output Structure
```
preprocessed/
├── sub-01_ses-1_T1w_preprocessed.nii.gz
├── sub-01_ses-2_T1w_preprocessed.nii.gz
├── ...
└── preprocessing_report.json
```

### Verify Preprocessing
```powershell
python -c "from utils import verify_preprocessing; verify_preprocessing('preprocessed/')"
```

---

## Step 2: GAN Training

Train the 3D U-Net GAN for paired 3T → 7T super-resolution.

### Architecture Overview
- **Generator**: 3D U-Net (58.8M parameters)
  - Encoder: 4 levels (32→64→128→256→512 channels)
  - Decoder: 4 levels with skip connections
  - InstanceNorm3d, LeakyReLU activations
  
- **Discriminator**: 3D PatchGAN (11.0M parameters)
  - 4 convolutional layers with stride-2 downsampling
  - Outputs spatial map of real/fake predictions

- **Loss**: L1 reconstruction + adversarial (LSGAN)
  - L_G = 100.0 × L1 + 1.0 × adversarial

### Quick Start (Default Settings)
```powershell
python train_gan.py
```

### Recommended Settings for Small Dataset
```powershell
python train_gan.py `
    --epochs 200 `
    --batch_size 4 `
    --patch_size 64 `
    --lr_g 0.0001 `
    --lr_d 0.0001 `
    --l1_weight 100.0 `
    --adv_weight 1.0 `
    --modality T1w `
    --device cuda
```

### Training Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 4 | Batch size (reduce if OOM) |
| `--patch_size` | 64 | 3D patch size (64³ voxels) |
| `--lr_g` | 2e-4 | Generator learning rate |
| `--lr_d` | 2e-4 | Discriminator learning rate |
| `--l1_weight` | 100.0 | L1 loss weight |
| `--adv_weight` | 1.0 | Adversarial loss weight |
| `--modality` | T1w | Modality (T1w or T2w) |
| `--augment` | True | Enable augmentation |
| `--mixed_precision` | True | Use AMP for faster training |

### During Training

**Expected Console Output:**
```
Epoch [1/200]
Train: 100%|██████████| 12/12 [02:34<00:00]
  G Loss: 45.234, D Loss: 0.823, L1: 0.452, Adv: 0.823
  
Validation:
  L1: 0.389, PSNR: 24.56 dB

[INFO] Saved checkpoint: checkpoints/epoch_001.pth
[INFO] New best model (PSNR: 24.56 dB)
```

**Monitor Progress:**
```powershell
# Option 1: TensorBoard (if installed)
tensorboard --logdir logs/

# Option 2: Check checkpoint directory
ls checkpoints/
```

### Output Files
```
checkpoints/
├── best_model.pth         # Best model (highest val PSNR)
├── latest_model.pth       # Most recent checkpoint
├── epoch_001.pth          # Per-epoch checkpoints
├── epoch_002.pth
└── ...

logs/
├── train.log              # Training log
└── events.out.tfevents.*  # TensorBoard logs

visualizations/
├── epoch_001_val.png      # Validation visualizations
├── epoch_002_val.png
└── ...
```

### Checkpoints Contain
- Generator and discriminator weights
- Optimizer states
- Training history (losses, metrics)
- Configuration used
- Random seeds for reproducibility

### Training Tips

**If GPU Memory Issues:**
```powershell
# Reduce batch size
python train_gan.py --batch_size 2

# Reduce patch size
python train_gan.py --patch_size 48

# Disable mixed precision
python train_gan.py --no_mixed_precision
```

**For Better Results (requires more time):**
```powershell
python train_gan.py --epochs 500 --early_stopping_patience 50
```

**Resume Training:**
```powershell
python train_gan.py --resume checkpoints/latest_model.pth
```

---

## Step 3: Evaluation

Comprehensive evaluation of trained GAN on validation/test set.

### What It Does
1. **Quantitative Metrics**: PSNR, SSIM, L1 error
2. **Visual Inspection**: Multi-view slice comparisons
3. **Overfitting Checks**: Training/validation loss curves
4. **Anatomical Consistency**: Histogram analysis, outlier detection
5. **Full-Volume Inference**: Sliding window with overlap

### Quick Start
```powershell
# Evaluate on validation set
python eval_gan.py --checkpoint checkpoints/best_model.pth --split val

# Evaluate on test set
python eval_gan.py --checkpoint checkpoints/best_model.pth --split test
```

### Recommended Evaluation
```powershell
python eval_gan.py `
    --checkpoint checkpoints/best_model.pth `
    --split val `
    --save_volumes `
    --patch_size 64 `
    --overlap 0.5 `
    --device cuda
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | Required | Path to trained model |
| `--split` | val | Split to evaluate (train/val/test) |
| `--save_volumes` | False | Save generated 7T volumes |
| `--patch_size` | 64 | Patch size for sliding window |
| `--overlap` | 0.5 | Overlap ratio (0.0-1.0) |
| `--modality` | T1w | Modality to evaluate |

### Expected Output

**Console:**
```
[INFO] Loading checkpoint: checkpoints/best_model.pth
[INFO] Generator loaded (epoch 145)
[INFO] Evaluating 2 subjects from val split

Subject 1/2
[INFO] Evaluating: sub-03
[INFO] Running sliding window inference on sub-03_ses-1_T1w_preprocessed.nii.gz
[INFO]   Processed 64 patches
[INFO] Computing metrics...
[INFO] Metrics Summary for sub-03:
[INFO]   PSNR (3T vs 7T): 23.45 dB
[INFO]   PSNR (Gen vs 7T): 26.78 dB
[INFO]   PSNR Improvement: 3.33 dB
[INFO]   SSIM (3T vs 7T): 0.7234
[INFO]   SSIM (Gen vs 7T): 0.8012
[INFO]   SSIM Improvement: 0.0778

Subject 2/2
[INFO] Evaluating: sub-07
...

AGGREGATE METRICS
================================================================================
Number of subjects: 2
PSNR (3T vs 7T):     23.56 ± 0.89 dB
PSNR (Gen vs 7T):    26.95 ± 1.12 dB
PSNR Improvement:    3.39 ± 0.45 dB

SSIM (3T vs 7T):     0.7189 ± 0.0234
SSIM (Gen vs 7T):    0.7998 ± 0.0189
SSIM Improvement:    0.0809 ± 0.0078
================================================================================
```

### Output Files
```
evaluation/
├── metrics/
│   ├── metrics_summary.csv              # Per-subject metrics
│   └── aggregate_statistics.csv         # Mean, std, min, max
├── visualizations/
│   ├── sub-03_slices.png               # Multi-view comparison
│   ├── sub-03_histograms.png           # Intensity distributions
│   ├── sub-07_slices.png
│   └── sub-07_histograms.png
├── plots/
│   └── training_curves.png             # Loss curves + overfitting check
├── generated_volumes/                   # (if --save_volumes)
│   ├── sub-03_generated_7T.nii.gz
│   └── sub-07_generated_7T.nii.gz
└── logs/
    └── evaluation.log
```

### Understanding Metrics

**PSNR (Peak Signal-to-Noise Ratio)**
- Measures reconstruction quality (higher is better)
- Typical range: 20-35 dB
- Improvement of 2-4 dB is good for small datasets

**SSIM (Structural Similarity Index)**
- Measures perceptual similarity (0-1, higher is better)
- >0.8 is good, >0.9 is excellent
- More robust than PSNR for medical images

**Interpretation:**
- **Positive improvement**: GAN is learning useful features
- **Negative improvement**: GAN may be overfitting or data misaligned
- **Small dataset caveat**: With only 10 subjects, expect modest improvements

### Visualizations

**Slice Comparisons (`*_slices.png`):**
- 3 views × 3 images = 9 panels
- Rows: Axial, Coronal, Sagittal
- Columns: 3T Input, Generated 7T, Real 7T

**Histograms (`*_histograms.png`):**
- Intensity distributions for 3T, Generated, Real 7T
- Shows mean and std
- Useful for detecting intensity shifts

**Training Curves (`training_curves.png`):**
- Generator loss, Discriminator loss, L1 loss, PSNR
- **Warning overlay** if validation loss increases

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```powershell
# Reduce batch size
python train_gan.py --batch_size 2

# Reduce patch size
python train_gan.py --patch_size 48

# Use CPU (slower)
python train_gan.py --device cpu
```

#### 2. No Preprocessed Data Found
```
ERROR: No preprocessed data found!
```

**Solution:**
```powershell
# Run preprocessing first
python example_pipeline.py --preprocess
```

#### 3. Checkpoint Not Found
```
ERROR: Checkpoint not found: checkpoints/best_model.pth
```

**Solution:**
```powershell
# Check available checkpoints
ls checkpoints/

# Use correct path
python eval_gan.py --checkpoint checkpoints/latest_model.pth --split val
```

#### 4. Model Not Improving
**Symptoms:** PSNR stays constant or decreases

**Possible Causes:**
- Dataset too small (only 10 subjects)
- Learning rate too high/low
- Data not properly aligned
- Overfitting

**Solutions:**
```powershell
# Try lower learning rate
python train_gan.py --lr_g 0.0001 --lr_d 0.0001

# Increase L1 weight
python train_gan.py --l1_weight 200.0

# More augmentation
python train_gan.py --augment

# Check data alignment
python -c "from utils import verify_preprocessing; verify_preprocessing('preprocessed/')"
```

#### 5. Preprocessing Fails
```
ERROR: Skull stripping failed
```

**Solution:**
```powershell
# Use fast config (skips some steps)
python example_pipeline.py --preprocess --config fast

# Or manually disable skull stripping in config.py
# Set: preprocessing.skull_strip = False
```

---

## Advanced Usage

### Quality Control & Harmonization

```powershell
# Run QC on preprocessed data
python example_qc_harmonization.py --qc

# Harmonize 3T and 7T intensities
python example_qc_harmonization.py --harmonize

# Both
python example_qc_harmonization.py --qc --harmonize
```

**Outputs:**
- `qc_reports/`: HTML reports with metrics and artifact detection
- `harmonized/`: Intensity-harmonized volumes

### Cross-Validation

```python
# In your training script
from utils import create_kfold_splits

folds = create_kfold_splits(data_list, k=5, random_seed=42)

for fold_idx, (train_data, val_data) in enumerate(folds):
    print(f"Training fold {fold_idx+1}/5")
    # Train model on this fold
```

### Custom Configuration

Edit `config.py`:
```python
# Modify preprocessing settings
config.preprocessing.bias_correction = False
config.preprocessing.target_spacing = (1.5, 1.5, 1.5)

# Modify data augmentation
config.augmentation.rotation_range = 20
config.augmentation.flip_probability = 0.5
```

### TensorBoard Monitoring

```powershell
# Start TensorBoard server
tensorboard --logdir logs/ --port 6006

# Open browser
start http://localhost:6006
```

View:
- Scalar metrics (losses, PSNR)
- Images (input, generated, target)
- Histograms (model weights, activations)

### Batch Inference on New Data

```python
from eval_gan import GANEvaluator
import torch

evaluator = GANEvaluator(
    checkpoint_path="checkpoints/best_model.pth",
    output_dir="inference_output",
    device=torch.device("cuda"),
)

# Infer on new 3T scan
generated_7t = evaluator.infer_full_volume("path/to/new_3T_scan.nii.gz")
```

---

## Complete Workflow Summary

```powershell
# 1. Verify environment
python setup_validation.py

# 2. Preprocess data
python example_pipeline.py --preprocess

# 3. Train GAN (adjust epochs as needed)
python train_gan.py --epochs 200 --batch_size 4

# 4. Evaluate on validation set
python eval_gan.py --checkpoint checkpoints/best_model.pth --split val --save_volumes

# 5. Evaluate on test set (final results)
python eval_gan.py --checkpoint checkpoints/best_model.pth --split test --save_volumes

# 6. (Optional) Monitor with TensorBoard
tensorboard --logdir logs/
```

---

## Expected Timeline

| Step | Time (GPU) | Time (CPU) |
|------|------------|------------|
| **Preprocessing** | 10-20 min | 30-60 min |
| **Training (100 epochs)** | 2-4 hours | 12-24 hours |
| **Evaluation** | 5-10 min | 15-30 min |

**Total**: 2.5-5 hours on GPU, 12-25 hours on CPU

---

## Success Criteria

✅ **Preprocessing**: All 20 scans processed without errors  
✅ **Training**: Validation PSNR improves over baseline  
✅ **Evaluation**: PSNR improvement 2-4 dB, SSIM improvement 0.05-0.10  
✅ **Visual**: Generated images look sharper than 3T input  
✅ **Consistency**: No major anatomical distortions or artifacts  

---

## Next Steps

After baseline GAN is working:
1. **Experiment with architectures**: Add perceptual loss, topology loss
2. **Try diffusion models**: DDPM, DDIM for super-resolution
3. **Larger datasets**: Collect more paired 3T/7T data
4. **Advanced metrics**: Add LPIPS, FID for better evaluation
5. **Clinical validation**: Test on held-out clinical cases

---

## Support

- **Documentation**: See [GAN_README.md](GAN_README.md) for architecture details
- **Issues**: Check [GitHub Issues](https://github.com/prabeshx12/Topo-Brain/issues)
- **Logs**: Check `logs/` directory for detailed error messages

---

## Citation

If you use this code, please cite:
```
@misc{topo-brain-2025,
  title={Topo-Brain: 3T to 7T MRI Super-Resolution with 3D GANs},
  author={Your Name},
  year={2025},
  url={https://github.com/prabeshx12/Topo-Brain}
}
```

---

**Good luck with your research! 🧠✨**
