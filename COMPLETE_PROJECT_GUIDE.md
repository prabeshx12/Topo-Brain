# Topo-Brain: Complete Project Guide
## Topology-Preserving 3T-to-7T MRI Enhancement Using Diffusion Models

**Author**: Research Team  
**Project Start**: 2024  
**Last Updated**: Latest Session  
**Status**: Training & Evaluation Phase

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Goals & Motivation](#project-goals--motivation)
3. [Dataset Overview](#dataset-overview)
4. [Complete System Architecture](#complete-system-architecture)
5. [Phase 1: Preprocessing Pipeline](#phase-1-preprocessing-pipeline)
6. [Phase 2: Data Preparation for Training](#phase-2-data-preparation-for-training)
7. [Phase 3: Diffusion Model Architecture](#phase-3-diffusion-model-architecture)
8. [Phase 4: Training Pipeline](#phase-4-training-pipeline)
9. [Phase 5: Inference & Sampling](#phase-5-inference--sampling)
10. [Phase 6: Model Extensions & Future Work](#phase-6-model-extensions--future-work)
11. [Implementation Details & Code Structure](#implementation-details--code-structure)
12. [Configuration Reference](#configuration-reference)
13. [Running the Complete Pipeline](#running-the-complete-pipeline)
14. [Troubleshooting & Common Issues](#troubleshooting--common-issues)

---

## Executive Summary

Topo-Brain is a complete MRI enhancement pipeline that transforms 3T brain MRI scans to 7T quality using conditional diffusion models. The system is designed to:

- **Preprocess** BIDS-organized brain MRI datasets (3T and 7T pairs)
- **Extract & prepare** paired training data with anatomical conservation
- **Train** a conditional U-Net diffusion model for 3T→7T synthesis
- **Generate** high-quality 7T-like images from 3T inputs
- **Validate** topology preservation through tissue classification
- **Enable** clinical applications (early AD detection when 7T unavailable)

### Key Innovation: Topology-Preserving Conditioning

Unlike naive super-resolution, Topo-Brain explicitly conditions generation on:
1. **3T anatomy** (via U-Net concatenation)
2. **Tissue constraints** (via multi-task segmentation losses)
3. **Diffusion noise schedule** (DDPM with 1000 timesteps)

This ensures enhanced 7T-quality images maintain biological validity.

---

## Project Goals & Motivation

### Medical Motivation

**Problem**: 7T MRI scanners provide superior brain imaging but are:
- Expensive (>$10M)
- Rare (only ~100 worldwide)
- Unavailable for most research institutions

**Solution**: Train a diffusion model to synthesize 7T-quality images from widely-available 3T scans, enabling:
- Better visualization of subtle brain pathology
- Early detection of Alzheimer's disease biomarkers
- Research on 7T-equivalent data without expensive hardware

### Technical Motivation

**Why Diffusion Models?**

| Approach | Strengths | Weaknesses |
|----------|-----------|-----------|
| **GANs (v1)** | Fast generation | Training instability, mode collapse, topology blurring |
| **Diffusion** | Stable training, quality, topology control | Slower sampling, higher training cost |
| **VAEs** | Stable, interpretable latents | Blurry outputs, limited reconstruction |

Diffusion chosen for medical imaging's demand for **anatomical fidelity**.

### Clinical Targets

- **Early Alzheimer's Detection**: Identify hippocampal atrophy earlier via 7T-quality gray matter contrast
- **White Matter Pathology**: Better visualization of demyelination and lesions
- **Stroke Assessment**: Enhanced tissue characterization in acute stroke

---

## Dataset Overview

### Physical Organization (BIDS Format)

```
data/
├── sub-01/
│   ├── ses-1/          # 3T session
│   │   └── anat/
│   │       ├── sub-01_ses-1_T1w.nii.gz
│   │       └── sub-01_ses-1_T2w.nii.gz
│   └── ses-2/          # 7T session
│       └── anat/
│           ├── sub-01_ses-2_T1w.nii.gz
│           └── sub-01_ses-2_T2w.nii.gz
├── sub-02/
│   └── ...
└── sub-10/
    └── ...

# Additional derivatives (optional)
derivatives/
└── Aligned/            # Pre-registered 3T→7T space (if available)
    └── sub-XX/aligned_T1w.nii.gz  # FreeSurfer outputs, etc.
```

### Data Statistics

| Property | Value |
|----------|-------|
| **Total Subjects** | 10 |
| **Scans per Subject** | 2 (3T + 7T) |
| **Total Volumes** | 40 (20 pairs) |
| **Primary Modality** | T1w (also T2w available) |
| **Spatial Resolution** | Variable (native 3T/7T resolutions) |
| **Brain Mask Source** | FreeSurfer aseg outputs (derivatives/Aligned folder) |
| **Data Format** | NIfTI (.nii.gz) |

### Pre-Registration Status

- **Status**: Pre-registered (3T images aligned to 7T space)
- **Tool Used**: Likely FSL/ANTs + manual verification
- **Impact**: Enables direct patch-by-patch pairing without registration during training

### Data Splits (Patient-Level)

```
Training Set:    sub-01, sub-02, sub-03, sub-04, sub-05, sub-06 (12 volumes)
Validation Set:  sub-07, sub-08 (4 volumes)
Testing Set:     sub-09, sub-10 (4 volumes)
Total:           20 volumes (10 pairs × 2 modalities)
```

**Critical Design**: No patient overlap between splits → prevents data leakage

---

## Complete System Architecture

### High-Level Pipeline Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                          PREPROCESSING PHASE                          │
│                    (Weeks 1-2 of project)                            │
└──────────────────────────────────────────────────────────────────────┘
    │
    ├─ STEP 1: Brain Extraction (HD-BET or manual masks)
    │   Input: BIDS data (raw 3T/7T)
    │   Output: Brain-only volumes
    │
    ├─ STEP 2: Bias Field Correction (N4ITK, optional)
    │   Output: Corrected intensity fields
    │
    ├─ STEP 3: Spatial Reorientation (RAS+)
    │   Output: Consistent anatomy orientation
    │
    ├─ STEP 4: Isotropic Resampling (optional)
    │   Output: Uniform voxel spacing
    │
    ├─ STEP 5: Intensity Normalization (percentile-based)
    │   Output: Normalized to [-1, 1] range
    │
    └─ STEP 6: QC & Visualization
        Output: QC reports, outlier flags


┌──────────────────────────────────────────────────────────────────────┐
│                      DATA PREPARATION PHASE                           │
│                    (Weeks 2-3 of project)                            │
└──────────────────────────────────────────────────────────────────────┘
    │
    ├─ STEP 1: Pair Matching (3T-7T alignment verification)
    │   Output: pairs.csv manifest
    │
    ├─ STEP 2: Subject-Level Splitting
    │   Output: train/val/test split info
    │
    ├─ STEP 3: Patch Extraction & Augmentation (on-the-fly)
    │   Output: SynthesisDataset with 64³ patches
    │
    └─ STEP 4: Dataset Statistics & Sanity Checks
        Output: Dataset summary, shape verification


┌──────────────────────────────────────────────────────────────────────┐
│                      TRAINING PHASE                                   │
│                    (Weeks 3-8+ of project)                           │
└──────────────────────────────────────────────────────────────────────┘
    │
    ├─ STEP 1: Initialize Model (AnatomyGuidedUNet)
    │   Architecture: 3D U-Net with dual decoders
    │   Features: [32, 64, 128, 256] channels
    │
    ├─ STEP 2: Initialize Diffusion (GaussianDiffusion)
    │   Timesteps: 1000
    │   Schedule: Linear or Cosine beta schedule
    │
    ├─ STEP 3: Setup Optimizer & LR Scheduler
    │   Optimizer: Adam (lr=1e-4, β1=0.9, β2=0.999)
    │   Scheduler: Warmup (optional)
    │
    ├─ STEP 4: Main Training Loop (N iterations)
    │   For each batch:
    │     a) Sample random timestep t
    │     b) Add noise to 7T target: x_t = √(α̅_t) * x_0 + √(1-α̅_t) * ε
    │     c) Predict noise & tissue labels
    │     d) Compute loss: L_denoise + λ_seg * L_seg + λ_percep * L_percep
    │     e) Backprop & update weights
    │     f) Update EMA model every N steps
    │
    ├─ STEP 5: Periodic Evaluation
    │   Every N iterations:
    │     a) Save checkpoint (model + optimizer + EMA)
    │     b) Generate samples on val set
    │     c) Compute metrics (PSNR, SSIM, FID)
    │     d) Log to TensorBoard/W&B
    │
    └─ STEP 6: Training Completion
        Output: Trained model checkpoints, best model


┌──────────────────────────────────────────────────────────────────────┐
│                      INFERENCE PHASE                                  │
│                    (Weeks 8+ of project)                             │
└──────────────────────────────────────────────────────────────────────┘
    │
    ├─ STEP 1: Load Trained Model & Checkpoint
    │   Use: EMA model for inference
    │
    ├─ STEP 2: Full-Volume Inference (from 3T input)
    │   Process: Sliding window with 64³ patches
    │   Overlap: 50% overlap for smooth blending
    │
    ├─ STEP 3: Reverse Diffusion Sampling
    │   For t = 1000 down to 1:
    │     a) Predict noise ε̂_t using model
    │     b) Update x_{t-1} using DDPM sampling
    │     c) Optional: Classifier-free guidance
    │
    ├─ STEP 4: Post-Processing
    │   Operations:
    │     a) Clamp output to [-1, 1]
    │     b) Blend overlapping patches (Gaussian weights)
    │     c) Inverse normalize intensity
    │     d) Apply optional sharpening
    │
    ├─ STEP 5: Evaluation & Comparison
    │   Metrics vs. real 7T:
    │     - PSNR (Peak Signal-to-Noise Ratio)
    │     - SSIM (Structural Similarity Index)
    │     - FID (Fréchet Inception Distance)
    │     - Topology metrics (via segmentation)
    │
    └─ STEP 6: Visualization & Reporting
        Output: Enhanced images, metrics, visualizations


┌──────────────────────────────────────────────────────────────────────┐
│                      VALIDATION PHASE                                 │
│                    (Ongoing)                                          │
└──────────────────────────────────────────────────────────────────────┘
    │
    ├─ Quantitative Metrics
    │   - Image quality (PSNR, SSIM, FID)
    │   - Tissue preservation (Dice coefficient)
    │   - Artifact metrics (ring artifacts, blur)
    │
    ├─ Qualitative Assessment
    │   - Visual inspection by radiologists
    │   - Anatomical fidelity checks
    │   - Artifact detection
    │
    └─ Clinical Validation
        - Feature extraction consistency
        - Diagnostic accuracy preservation
```

---

## Phase 1: Preprocessing Pipeline

### 1.1 Overview

The preprocessing pipeline transforms raw BIDS-organized MRI data into clean, normalized volumes suitable for machine learning. This phase is critical because:

1. **Quality Control**: Removes corrupted or outlier scans
2. **Consistency**: Ensures all volumes have compatible orientations and intensities
3. **Comparability**: Makes 3T and 7T scans directly comparable
4. **Efficiency**: Reduces training data complexity

### 1.2 Step-by-Step Preprocessing

#### **Step 1: Data Discovery & BIDS Parsing**

**Code**: `src/bids.py`

```python
# Discover all BIDS files in the dataset
from src.bids import discover_bids_files

bids_files = discover_bids_files(
    data_root="/path/to/data",
    subjects=["sub-01", "sub-02", ...],
    sessions=["ses-1", "ses-2"],
    modalities=["T1w", "T2w"]
)

# Output structure:
# [
#   {"subject": "sub-01", "session": "ses-1", "modality": "T1w", 
#    "image_path": "...", "mask_path": "...", ...},
#   ...
# ]
```

**Purpose**: Parse BIDS filenames to extract metadata (subject ID, session, modality)

**Key Files**:
- Input: Raw NIfTI files from BIDS
- Output: List of file metadata dictionaries

---

#### **Step 2: Brain Extraction (Skull Stripping)**

**Code**: `scripts/generate_brain_masks.py`

**Methods Supported**:

| Method | Speed | Accuracy | Requirements | Notes |
|--------|-------|----------|--------------|-------|
| **HD-BET** | ~5 sec/vol | 95%+ | GPU, `antspyx` | Recommended for T1w |
| **SynthStrip** | ~10 sec/vol | 93%+ | GPU, FreeSurfer | Good for T2w |
| **FSL BET** | ~20 sec/vol | 85% | CPU only | Fallback option |
| **Manual masks** | N/A | 100% | Pre-computed | Use if available |

**Current Approach**: Use pre-computed masks from FreeSurfer aseg derivatives

```python
# Load brain mask (from FreeSurfer or HD-BET)
mask = nib.load("derive/Aligned/sub-XX/aparc+aseg.nii").get_fdata()
brain_mask = mask > 0  # Binary mask

# Apply to volume
brain_only = volume * brain_mask
```

**Output**: Brain-only volumes stored as:
```
derivatives/topobrain-preproc/
├── sub-01_ses-1_T1w_brain.nii.gz
├── sub-01_ses-2_T1w_brain.nii.gz
└── ...
```

---

#### **Step 3: Bias Field Correction (Optional)**

**Code**: `src/preprocessing.py` → `N4BiasFieldCorrection`

**Purpose**: Remove intensity non-uniformity (especially pronounced in 7T)

**Method**: N4ITK (Tustison et al., 2010)

```python
from src.preprocessing import N4BiasFieldCorrection

n4 = N4BiasFieldCorrection(
    iterations=50,
    convergence_threshold=0.001
)

# Apply to volume
corrected = n4(brain_volume)
```

**When to use**:
- ✅ **Always for 7T** (strong bias fields at high field strength)
- ⚠️ **Optional for 3T** (moderate bias fields)
- ❌ **Skip if preserving fine texture** (for GAN training)

**Configuration** (in `config.py`):
```yaml
bias_correction:
  enabled: false  # Set to true for 7T data
  n4_iterations: 20
  n4_convergence_threshold: 0.001
```

---

#### **Step 4: Reorientation to RAS+**

**Code**: MONAI `Orientationd` transform

**Purpose**: Ensure consistent anatomical orientation (Right-Anterior-Superior)

```python
from monai.transforms import Orientationd

# Convert to RAS+
orient_transform = Orientationd(keys=["image"], axcodes="RAS")
```

**Standard Formats**:
- **RAS+** (neuroimaging standard): Right, Anterior, Superior
- **LPS** (DICOM default): Left, Posterior, Superior

**Output**: All volumes guaranteed to have:
- X-axis pointing Right
- Y-axis pointing Anterior
- Z-axis pointing Superior

---

#### **Step 5: Isotropic Resampling (Optional)**

**Code**: `src/preprocessing.py` → `ImageResampler`

**Purpose**: Convert to uniform voxel spacing (e.g., 1mm³)

```python
# 3T typical: 1.2 × 1.0 × 1.0 mm
# 7T typical: 0.8 × 0.8 × 0.8 mm
# Target: 1.0 × 1.0 × 1.0 mm (isotropic)

from nibabel.processing import resample_to_output
resampled = resample_to_output(img, voxel_sizes=[1.0, 1.0, 1.0])
```

**Configuration**:
```yaml
resample:
  target_spacing: [1.0, 1.0, 1.0]  # None to skip
  interpolation: "trilinear"        # or "nearest" for masks
```

**Trade-offs**:
- ✅ Simplifies patch extraction
- ❌ Can blur high-res 7T data
- ⚠️ **For GAN**: Keep original resolution, don't resample

---

#### **Step 6: Intensity Normalization**

**Code**: `src/preprocessing.py` → `IntensityNormalization`

**Critical for Diffusion Models**: Ensures stable noise schedules

**Methods**:

| Method | Formula | Range | Use Case |
|--------|---------|-------|----------|
| **Z-score** | (x - μ) / σ | (-∞, +∞) | **Recommended** |
| **Min-Max** | (x - min) / (max - min) | [0, 1] | When min/max known |
| **Percentile** | Clip then z-score | [-1, 1] | **For diffusion** |

**Implementation** (Percentile-based for Diffusion):

```python
# 1. Compute percentiles from brain voxels only
p_low = np.percentile(brain_voxels, 0.5)   # P0.5
p_high = np.percentile(brain_voxels, 99.5) # P99.5

# 2. Clip outliers
clipped = np.clip(volume, p_low, p_high)

# 3. Z-score normalize
normalized = (clipped - clipped.mean()) / clipped.std()

# 4. Scale to [-1, 1]
final = 2 * (normalized - normalized.min()) / (normalized.max() - normalized.min()) - 1
```

**Diffusion-Specific Normalization** (in `src/diffusion.py`):

```python
# Store normalization statistics for inference
normalization_stats = {
    "method": "percentile",
    "p_low": p_low,
    "p_high": p_high,
    "mean": mean,
    "std": std,
    "background_value": -1  # Background set to -1 for masking
}
```

**Configuration**:
```yaml
normalization:
  method: "percentile"
  percentile_lower: 0.5
  percentile_upper: 99.5
  clip_lower_percentile: 0.5
  clip_upper_percentile: 99.5
```

---

#### **Step 7: Optional Padding/Cropping**

**Purpose**: Standardize volume dimensions for batching

```python
# Pad to fixed size [128, 128, 128]
target_size = (128, 128, 128)
padded = np.pad(volume, 
                ((0, target_size[0]-volume.shape[0]), ...), 
                mode='constant', constant_values=-1)
```

**Decision**:
- ✅ **Use**: If training on full volumes (memory-efficient)
- ❌ **Skip**: If using patch-based training (adaptive padding)

---

#### **Step 8: Quality Control & Outlier Detection**

**Code**: `src/quality_control.py`

**QC Metrics Computed**:

| Metric | Calculation | What it detects |
|--------|-------------|-----------------|
| **SNR** | Signal / Noise | Acquisition quality |
| **CNR** | (Signal - Background) / Noise | Tissue contrast |
| **Entropy** | -Σ p(x) log p(x) | Image complexity |
| **Mean intensity** | Average brain voxel intensity | Calibration issues |
| **Std deviation** | Intensity variability | Noise estimate |

**Outlier Detection** (Z-score):
```python
import numpy as np
stats = compute_dataset_statistics(volumes)
z_scores = np.abs((metrics - stats['mean']) / stats['std'])
outliers = z_scores > 3  # 3σ threshold
```

**Output**: QC report in HTML format
```
derivatives/topobrain-preproc/qc/
├── qc_report.html
├── outliers.json
└── metrics.csv
```

---

### 1.3 Complete Preprocessing Script

**Script**: `scripts/preprocess_bids.py`

```bash
# Run preprocessing
python scripts/preprocess_bids.py \
  --data-root /path/to/data \
  --config configs/preprocess.yaml \
  --output-root derivatives/topobrain-preproc \
  --num-workers 4
```

**Configuration** (`configs/preprocess.yaml`):
```yaml
data:
  data_root: /path/to/data
  output_root: derivatives/topobrain-preproc
  num_subjects: 10
  modalities: ["T1w", "T2w"]
  session_3t: "ses-1"
  session_7t: "ses-2"

preprocessing:
  target_orientation: "RAS"
  target_spacing: [1.0, 1.0, 1.0]  # None to skip
  use_bias_correction: false        # Set true for 7T
  use_skull_stripping: true
  normalization_method: "percentile"
  percentile_lower: 0.5
  percentile_upper: 99.5

quality_control:
  enabled: true
  max_samples: 5
```

**Output Structure**:
```
derivatives/topobrain-preproc/
├── sub-01_ses-1_T1w_desc-preproc.nii.gz
├── sub-01_ses-2_T1w_desc-preproc.nii.gz
├── ...
├── manifest.csv                    # List of all preprocessed volumes
└── qc/
    ├── qc_report.html
    ├── metrics.csv
    └── outliers.json
```

---

## Phase 2: Data Preparation for Training

### 2.1 Overview

Once preprocessing is complete, we prepare paired training data for the diffusion model:

1. **Pair Matching**: Verify 3T-7T alignment
2. **Subject Splitting**: Create train/val/test splits (patient-level)
3. **Patch Extraction**: Create 64³ patches with augmentation
4. **Dataset Loading**: Implement efficient PyTorch DataLoaders

### 2.2 Step 1: Generate Pairs Manifest

**Script**: `scripts/regenerate_pairs.py`

**Purpose**: Create `pairs.csv` file linking 3T input to 7T target

```python
import pandas as pd

pairs_data = []
for subject in subjects:
    for modality in ["T1w", "T2w"]:
        pairs_data.append({
            "subject": subject,
            "modality": modality,
            "input_3t": f"derivatives/topobrain-preproc/{subject}_ses-1_{modality}_desc-preproc.nii.gz",
            "target_7t": f"derivatives/topobrain-preproc/{subject}_ses-2_{modality}_desc-preproc.nii.gz",
            "mask": f"derivatives/Aligned/{subject}/aparc+aseg.nii.gz",
            "split": "train"  # or "val" or "test"
        })

df = pd.DataFrame(pairs_data)
df.to_csv("pairs.csv", index=False)
```

**Output**: `pairs.csv`
```
subject,modality,input_3t,target_7t,mask,split
sub-01,T1w,derivatives/...,derivatives/...,derivatives/...,train
sub-01,T2w,derivatives/...,derivatives/...,derivatives/...,train
...
```

---

### 2.3 Step 2: Subject-Level Train/Val/Test Splitting

**Code**: `src/synthesis_dataset.py` → `SubjectSplitter`

**Key Requirement**: **NO patient overlap between splits**

```python
from src.synthesis_dataset import SubjectSplitter, SplitConfig

splitter = SubjectSplitter(
    SplitConfig(
        n_folds=10,
        val_fold=8,
        test_fold=9,
        use_loocv=True,
        seed=42
    )
)

pairs = pd.read_csv("pairs.csv").to_dict('records')
train_pairs, val_pairs, test_pairs = splitter.get_split(pairs)

# Output:
# Train: 12 volumes (sub-01...sub-06, T1w & T2w)
# Val:    4 volumes (sub-07...sub-08, T1w & T2w)
# Test:   4 volumes (sub-09...sub-10, T1w & T2w)
```

**Why Patient-Level Splitting?**

| Approach | Data Leakage | Realistic? | Use Case |
|----------|--------------|-----------|----------|
| **Volume-level** | ✅ YES (same patient in train+test) | ❌ Not realistic | Avoid! |
| **Patient-level** | ❌ NO | ✅ Realistic | **Recommended** |

---

### 2.4 Step 3: Patch Extraction & Augmentation

**Code**: `src/synthesis_dataset.py` → `SynthesisDataset`

**Training Process**:

```python
# For each training iteration:
# 1. Sample a random pair from training set
# 2. Extract 32 random 64³ patches from the pair
# 3. Apply augmentations
# 4. Return batch

from src.synthesis_dataset import SynthesisDataset, PatchConfig

dataset = SynthesisDataset(
    pairs=train_pairs,
    patch_config=PatchConfig(
        patch_size=(64, 64, 64),
        patches_per_volume=32,
        min_brain_fraction=0.1
    ),
    augment=True  # Enable during training
)

# Augmentations applied:
# - Random rotation (±15 degrees)
# - Random flip (left-right)
# - Random intensity shift (±0.1)
# - Random Gaussian noise (σ=0.02)
```

**Data Dictionary** (per batch item):
```python
{
    "x_3t": torch.Tensor([1, 64, 64, 64]),      # 3T patch (input)
    "x_7t": torch.Tensor([1, 64, 64, 64]),      # 7T patch (target)
    "mask": torch.Tensor([1, 64, 64, 64]),      # Brain mask
    "seg": torch.Tensor([1, 64, 64, 64]),       # Tissue segmentation (0-3)
    "subject": "sub-01",                         # Subject ID (for tracking)
    "coordinates": (x, y, z)                     # Patch location in full volume
}
```

**Loss Function (Multi-Task)** during training:

```python
# Forward pass
noise_pred, seg_pred = model(x_3t_noisy, x_3t_clean, t)

# Losses
L_denoise = F.l1_loss(noise_pred, noise)
L_seg = F.cross_entropy(seg_pred, seg_gt)
L_percep = perceptual_loss(model_denoise(x_3t_noisy), x_7t_clean)

# Total
L_total = L_denoise + 0.1 * L_seg + 0.01 * L_percep
```

---

### 2.5 Step 4: DataLoader Setup

```python
from torch.utils.data import DataLoader
from src.synthesis_dataset import create_synthesis_dataloaders

train_loader, val_loader, test_loader = create_synthesis_dataloaders(
    pairs=pairs,
    batch_size=4,
    patches_per_volume=32,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2
)

# Usage:
for batch in train_loader:
    x_3t = batch["x_3t"]        # [B, 1, 64, 64, 64]
    x_7t = batch["x_7t"]        # [B, 1, 64, 64, 64]
    mask = batch["mask"]        # [B, 1, 64, 64, 64]
    
    # Forward pass...
    loss = model(x_3t, x_7t, mask)
    loss.backward()
```

---

## Phase 3: Diffusion Model Architecture

### 3.1 Overview

The diffusion model consists of two main components:

1. **AnatomyGuidedUNet** (`src/model.py`): The neural network
2. **GaussianDiffusion** (`src/diffusion.py`): The diffusion process

### 3.2 AnatomyGuidedUNet Architecture

**File**: `src/model.py`

#### **Design Principles**

1. **Conditional Input**: Concatenate noisy 7T + clean 3T
2. **Multi-Task Learning**: Simultaneous denoising + segmentation
3. **Time Embedding**: Inject timestep information at every layer
4. **Anatomy Preservation**: Multi-level skip connections
5. **Efficiency**: Compact enough for 64³ patches on 1 GPU

#### **Architecture Diagram**

```
Input: [x_7t_noisy (1ch) | x_3t_clean (1ch)] → concatenate → 2 channels
                          ↓
                   Initial Conv3d(2→32)
                          ↓
        ┌─── ENCODER (Downsampling) ───┐
        │  32 → 64 → 128 → 256         │
        │  ResnetBlock + TimeEmb        │
        │  ConvTranspose3d(stride=2)    │
        │  Save skip connections        │
        └───────────────────────────────┘
                 ↓
        ┌─── BOTTLENECK ───┐
        │  ResnetBlock      │
        │  SelfAttention3D  │ (global context)
        │  ResnetBlock      │
        └───────────────────┘
                 ↓
        ┌────────┴────────┐
        ↓                  ↓
    DENOISING          SEGMENTATION
    DECODER             DECODER
    256→128→64→32       256→128→64→32
    Conv1x1→1ch         Conv1x1→4-class
    (pred noise)        (tissue logits)
```

#### **Key Components**

##### **1. Sinusoidal Time Embedding**

```python
class SinusoidalPosEmb(nn.Module):
    def forward(self, x):
        # x shape: [B]  (timestep indices)
        # output shape: [B, dim]
        
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]  # [B, half_dim]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # [B, dim]
        return emb
```

**Output**: Positional encoding of timestep (e.g., {sin(ωt), cos(ωt)})

**Used in**: ResnetBlock to modulate features with time-dependent information

##### **2. ResnetBlock with Time Conditioning**

```python
class ResnetBlock(nn.Module):
    def forward(self, x, time_emb=None):
        h = self.block1(x)  # Conv → GroupNorm → SiLU
        
        if time_emb is not None:
            # Project time embedding to feature dimension
            time_feat = self.mlp(time_emb)  # [B, D] → [B, C]
            # Add to spatial features (broadcast)
            h = h + time_feat[:, :, None, None, None]
        
        h = self.block2(h)
        return h + self.res_conv(x)  # Skip connection
```

**Key Insight**: Time embedding is **added** (not concatenated) to preserve spatial dimensions

##### **3. Self-Attention Module**

```python
class SelfAttention3D(nn.Module):
    def forward(self, x):
        # x: [B, C, D, H, W]
        # 1. Project to Q, K, V
        qkv = self.qkv(self.norm(x))  # [B, 3C, D, H, W]
        q, k, v = qkv.split(C, dim=1)
        
        # 2. Reshape for attention
        # [B, C, D*H*W] → attention
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        
        # 3. Apply attention
        out = (v @ attn.transpose(-2, -1))
        return x + self.proj(out)
```

**Used at**: Bottleneck only (full volume attention too expensive)

**Purpose**: Capture long-range dependencies between brain regions

---

#### **Model Configuration**

```python
model = AnatomyGuidedUNet(
    in_channels=1,          # Noisy target (7T) channel
    cond_channels=1,        # Conditioning input (3T) channel
    out_channels=1,         # Output noise prediction
    num_classes=3,          # Tissue classes (GM, WM, CSF) + background
    features=[32, 64, 128, 256],  # Channel progression
    use_attention=False     # Set to True for fine-tuning
)

# Total parameters: ~22M (manageable for 1 GPU)
```

---

#### **Forward Pass**

```python
def forward(self, x_noisy, x_cond, t):
    """
    Args:
        x_noisy: [B, 1, D, H, W] - Noisy 7T data at timestep t
        x_cond:  [B, 1, D, H, W] - Clean 3T conditioning
        t:       [B] - Timestep indices (0 to 999)
    
    Returns:
        noise_pred:  [B, 1, D, H, W] - Predicted noise ε̂
        seg_logits:  [B, 4, D, H, W] - Tissue logits (4 classes)
    """
    # 1. Concatenate inputs
    x = torch.cat([x_noisy, x_cond], dim=1)  # [B, 2, D, H, W]
    
    # 2. Time embedding
    t_emb = self.time_mlp(t)  # [B, dim]
    
    # 3. Initial convolution
    h = self.inc(x)  # [B, 32, D, H, W]
    skip_connections = [h]
    
    # 4. Encoder
    for down_block, downsample in self.downs:
        h = down_block(h, t_emb)  # ResnetBlock
        skip_connections.append(h)
        h = downsample(h)  # Reduce spatial dims
    
    # 5. Bottleneck
    h = self.mid_block1(h, t_emb)
    h = self.mid_attn(h)
    h = self.mid_block2(h, t_emb)
    
    # 6. Denoising decoder
    noise_pred = self._decode_denoising(h, skip_connections, t_emb)
    
    # 7. Segmentation decoder (multi-task)
    seg_logits = self._decode_segmentation(h, skip_connections, t_emb)
    
    return noise_pred, seg_logits
```

---

### 3.3 GaussianDiffusion: The Noise Schedule

**File**: `src/diffusion.py`

#### **Diffusion Process Overview**

The diffusion process has two directions:

| Direction | Name | Formula | Purpose |
|-----------|------|---------|---------|
| **Forward** | Noise addition | $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ | Add noise to data |
| **Reverse** | Denoising | $x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z$ | Gradually denoise |

where:
- $x_0$ = original clean 7T image
- $x_t$ = noisy image at timestep t
- $\epsilon$ = standard Gaussian noise
- $\bar{\alpha}_t$ = cumulative product of alphas
- $\mu_\theta$ = learned mean (our model)

#### **Forward Process (Training)**

```python
def q_sample(x_0, t, noise):
    """
    Add noise to x_0 at timestep t.
    x_t = sqrt(alpha_cumprod[t]) * x_0 + sqrt(1 - alpha_cumprod[t]) * noise
    """
    sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    
    x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    return x_t
```

#### **Training Objective**

```python
def p_losses(self, x_3t, x_7t, t, loss_type="l1"):
    """
    Compute training loss.
    
    Args:
        x_3t: [B, 1, D, H, W] - Clean 3T conditioning
        x_7t: [B, 1, D, H, W] - Clean 7T target
        t:    [B] - Timesteps
    """
    # 1. Sample random noise
    noise = torch.randn_like(x_7t)
    
    # 2. Add noise to target
    x_t = self.q_sample(x_7t, t, noise)
    
    # 3. Predict noise + segmentation
    noise_pred, seg_logits = model(x_t, x_3t, t)
    
    # 4. Compute losses
    if loss_type == "l1":
        loss_denoise = F.l1_loss(noise_pred, noise)
    else:
        loss_denoise = F.mse_loss(noise_pred, noise)
    
    loss_seg = F.cross_entropy(seg_logits, target_seg)
    loss_percep = perceptual_loss(x_t, x_7t)
    
    # 5. Weighted sum
    loss = (
        loss_denoise + 
        0.1 * loss_seg + 
        0.01 * loss_percep
    )
    
    return loss
```

#### **Reverse Process (Inference)**

```python
@torch.no_grad()
def sample(self, x_3t, shape):
    """
    Generate 7T image from 3T input.
    Start from pure noise, denoise step by step.
    """
    device = x_3t.device
    
    # 1. Start with pure Gaussian noise
    img = torch.randn(shape, device=device)
    
    # 2. Iterative denoising (t: 999 → 0)
    for t in reversed(range(self.num_timesteps)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        
        # 3. Predict noise ε̂
        noise_pred, _ = model(img, x_3t, t_batch)
        
        # 4. Update x_{t-1}
        posterior_variance_t = extract(self.posterior_variance, t_batch, img.shape)
        posterior_mean_coeff1_t = extract(self.posterior_mean_coeff1, t_batch, img.shape)
        posterior_mean_coeff2_t = extract(self.posterior_mean_coeff2, t_batch, img.shape)
        
        mean = (
            posterior_mean_coeff1_t * img +
            posterior_mean_coeff2_t * noise_pred
        )
        
        if t > 0:
            noise = torch.randn_like(img)
            img = mean + torch.sqrt(posterior_variance_t) * noise
        else:
            img = mean
    
    return img
```

#### **Noise Schedules**

**Linear Schedule** (default):

```python
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps)
```

**Cosine Schedule** (smoother, better results):

```python
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)
```

**Configuration** (`configs/train_diffusion.yaml`):
```yaml
diffusion:
  timesteps: 1000
  schedule: "cosine"  # or "linear"
  beta_start: 0.0001
  beta_end: 0.02
```

---

### 3.4 Loss Functions

#### **1. Denoising Loss (L1 or MSE)**

```python
# Predict noise, minimize difference from true noise
loss_denoise = F.l1_loss(noise_pred, noise)  # L1 recommended for medical
# or
loss_denoise = F.mse_loss(noise_pred, noise)  # L2
```

#### **2. Segmentation Loss (Cross-Entropy)**

```python
# Multi-class segmentation (GM, WM, CSF, background)
loss_seg = F.cross_entropy(seg_logits, target_seg)
```

**Purpose**: Force model to learn tissue anatomy without explicitly modifying outputs

#### **3. Perceptual Loss (Optional)**

```python
class PerceptualLoss(nn.Module):
    def forward(self, x, y):
        # Extract features from frozen VGG16
        feat_x = self.feature_extractor(x)
        feat_y = self.feature_extractor(y)
        # MSE in feature space
        return F.mse_loss(feat_x, feat_y)
```

**When to use**:
- ✅ To improve texture and fine details
- ❌ Can bias toward VGG-learned features (less relevant for medical)
- ⚠️ Set `lambda_percep=0.01` (low weight)

#### **4. Total Loss**

```python
lambda_seg = 0.1
lambda_percep = 0.01

loss_total = (
    loss_denoise + 
    lambda_seg * loss_seg + 
    lambda_percep * loss_percep
)
```

---

## Phase 4: Training Pipeline

### 4.1 Overview

This phase trains the diffusion model on paired 3T-7T data. A single training run:

- **Duration**: 100,000-500,000 iterations (~1-4 GPU-days)
- **Batch size**: 4 patches per GPU
- **Learning rate**: 1e-4 with optional warmup
- **Checkpointing**: Every 1000 iterations

### 4.2 Training Configuration

**File**: `configs/train_diffusion.yaml`

```yaml
dataset:
  pairs_csv: "pairs.csv"
  batch_size: 4
  num_workers: 4
  prefetch_factor: 2
  persistent_workers: true

model:
  in_channels: 1
  cond_channels: 1
  out_channels: 1
  num_classes: 3
  features: [32, 64, 128, 256]
  use_attention: false

diffusion:
  timesteps: 1000
  schedule: "cosine"  # or "linear"
  beta_end: 0.02

training:
  n_iters: 400000           # Total iterations
  lr: 1.0e-4
  beta1: 0.9                # Adam β1
  beta2: 0.999              # Adam β2
  weight_decay: 0.0001
  gradient_clip_norm: 1.0
  
  # Warmup (optional)
  warmup_steps: 5000
  warmup_init_lr: 1.0e-5
  
  # EMA (Exponential Moving Average)
  ema_decay: 0.9999
  ema_update_step: 1        # Update every N steps
  
  # Mixed precision
  use_amp: true
  
  # Checkpointing
  save_freq: 1000           # Save every N iterations
  eval_freq: 2000           # Evaluate every N iterations
  sample_freq: 2000         # Generate samples every N iterations
  
  # Losses
  loss_type: "l1"           # l1 or l2
  lambda_seg: 0.1
  lambda_percep: 0.01
```

---

### 4.3 Training Loop

**Script**: `scripts/train_diffusion.py`

```python
def train():
    # Setup
    config = load_config("configs/train_diffusion.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model
    model = AnatomyGuidedUNet(**config.model).to(device)
    diffusion = GaussianDiffusion(**config.diffusion)
    ema_model = copy.deepcopy(model)
    
    # Optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), 
                                  lr=config.training.lr,
                                  betas=(config.training.beta1, 
                                        config.training.beta2),
                                  weight_decay=config.training.weight_decay)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=config.training.n_iters
    )
    
    # Data
    train_loader, val_loader, _ = create_synthesis_dataloaders(
        pairs=load_pairs_manifest("pairs.csv"),
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers
    )
    train_iter = cycle(train_loader)
    
    # Logging
    tb_logger = TensorBoardLogger("logs")
    scaler = torch.cuda.amp.GradScaler()
    
    # Main loop
    for iteration in range(config.training.n_iters):
        batch = next(train_iter)
        x_3t = batch["x_3t"].to(device)
        x_7t = batch["x_7t"].to(device)
        mask = batch["mask"].to(device)
        
        # Forward pass
        with torch.cuda.amp.autocast():
            t = torch.randint(0, config.diffusion.timesteps, 
                            (x_3t.shape[0],), device=device)
            loss = diffusion.p_losses(
                model, x_3t, x_7t, t, 
                loss_type=config.training.loss_type,
                lambda_seg=config.training.lambda_seg,
                lambda_percep=config.training.lambda_percep
            )
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        if config.training.gradient_clip_norm:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config.training.gradient_clip_norm
            )
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Update EMA
        ema.step_ema(ema_model, model)
        
        # Logging
        if iteration % 100 == 0:
            tb_logger.log_scalar("loss/total", loss.item(), iteration)
            tb_logger.log_scalar("lr", scheduler.get_last_lr()[0], iteration)
        
        # Evaluation
        if iteration % config.training.eval_freq == 0:
            metrics = evaluate(model, val_loader, device)
            tb_logger.log_scalars(metrics, iteration)
        
        # Sampling
        if iteration % config.training.sample_freq == 0:
            samples = sample_batch(ema_model, val_loader, device)
            tb_logger.log_images("samples", samples, iteration)
        
        # Checkpointing
        if iteration % config.training.save_freq == 0:
            checkpoint_path = f"checkpoints/model_iter_{iteration}.pt"
            torch.save({
                "iteration": iteration,
                "model_state": model.state_dict(),
                "ema_model_state": ema_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config
            }, checkpoint_path)

train()
```

---

### 4.4 Metrics & Monitoring

**TensorBoard Logging**:

```python
# Scalar metrics
tb_logger.log_scalar("loss/total", loss.item(), iteration)
tb_logger.log_scalar("loss/denoise", loss_denoise.item(), iteration)
tb_logger.log_scalar("loss/seg", loss_seg.item(), iteration)
tb_logger.log_scalar("loss/percep", loss_percep.item(), iteration)
tb_logger.log_scalar("lr", lr, iteration)
tb_logger.log_scalar("ema_decay", ema.beta, iteration)

# Image metrics on validation set
psnr = compute_psnr(pred_7t, true_7t)
ssim = compute_ssim(pred_7t, true_7t)
tb_logger.log_scalar("metrics/psnr", psnr, iteration)
tb_logger.log_scalar("metrics/ssim", ssim, iteration)

# Sample montages
samples = sample_batch(ema_model, val_batch, num_samples=10)
tb_logger.log_images("samples/generated_7t", samples, iteration)
```

**View in TensorBoard**:

```bash
tensorboard --logdir=logs --port=6006
# Open http://localhost:6006 in browser
```

---

### 4.5 Checkpointing & Resuming

```python
# Save checkpoint
checkpoint = {
    "iteration": iteration,
    "model_state_dict": model.state_dict(),
    "ema_model_state_dict": ema_model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "config": config,
    "rng_state": torch.get_rng_state(),
    "cuda_rng_state": torch.cuda.get_rng_state()
}
torch.save(checkpoint, f"checkpoints/ckpt_iter_{iteration:06d}.pt")

# Resume from checkpoint
checkpoint = torch.load("checkpoints/ckpt_iter_100000.pt")
model.load_state_dict(checkpoint["model_state_dict"])
ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
start_iteration = checkpoint["iteration"]
```

---

## Phase 5: Inference & Sampling

### 5.1 Overview

After training completes, generate enhanced 7T-like images from new 3T inputs.

**Two modes**:

| Mode | Use Case | Speed | Quality |
|------|----------|-------|---------|
| **DDPM (50 steps)** | Research, analysis | ~30 sec/volume | Good |
| **DDIM (10 steps)** | Clinical, real-time | ~5 sec/volume | Slightly lower |

### 5.2 Inference Setup

```python
# Load trained model
checkpoint = torch.load("checkpoints/best_model.pt")
model = AnatomyGuidedUNet(...)
model.load_state_dict(checkpoint["ema_model_state_dict"])
model.eval().to(device)
diffusion = GaussianDiffusion(...)
```

### 5.3 Full-Volume Inference (Patch-Based Tiling)

**Problem**: Full 3D volumes (~256³) too large for GPU memory

**Solution**: Sliding window patch extraction

```python
def infer_full_volume(model, x_3t_vol, patch_size=64, overlap=0.5):
    """
    Generate 7T-like image from full 3T volume.
    
    Args:
        x_3t_vol: [D, H, W] - Full 3T volume
        patch_size: 64
        overlap: 0.5 (50% overlap)
    
    Returns:
        x_7t_synth: [D, H, W] - Synthesized 7T volume
    """
    D, H, W = x_3t_vol.shape
    stride = int(patch_size * (1 - overlap))
    
    # Initialize output volume
    x_7t_synth = np.zeros_like(x_3t_vol)
    weight_map = np.zeros_like(x_3t_vol)
    
    # Sliding window
    for d in range(0, D - patch_size + 1, stride):
        for h in range(0, H - patch_size + 1, stride):
            for w in range(0, W - patch_size + 1, stride):
                # Extract patch
                patch_3t = x_3t_vol[d:d+patch_size, h:h+patch_size, w:w+patch_size]
                patch_3t_t = torch.from_numpy(patch_3t).unsqueeze(0).unsqueeze(0).to(device)
                
                # Generate 7T patch
                with torch.no_grad():
                    patch_7t = diffusion.sample(model, patch_3t_t)
                
                patch_7t_np = patch_7t.squeeze().cpu().numpy()
                
                # Blend with Gaussian weights (smooth transitions)
                gaussian_weight = create_gaussian_window(patch_size)
                x_7t_synth[d:d+patch_size, h:h+patch_size, w:w+patch_size] += \
                    patch_7t_np * gaussian_weight
                weight_map[d:d+patch_size, h:h+patch_size, w:w+patch_size] += \
                    gaussian_weight
    
    # Normalize by weight map
    x_7t_synth /= weight_map + 1e-6
    
    return x_7t_synth
```

### 5.4 DDIM Fast Sampling

```python
def sample_ddim(model, x_3t, num_steps=50, eta=0.0):
    """
    DDIM sampling: faster than DDPM.
    
    Args:
        num_steps: Number of denoising steps (50-100 typical)
        eta: 0.0 for deterministic, 1.0 for stochastic
    """
    device = x_3t.device
    seq = np.linspace(0, 999, num_steps)
    
    x = torch.randn_like(x_3t)
    
    for i, t_step in enumerate(reversed(seq)):
        t = torch.full((x_3t.shape[0],), t_step, device=device, dtype=torch.long)
        
        with torch.no_grad():
            noise_pred, _ = model(x, x_3t, t)
        
        # DDIM update rule
        alpha_t = alphas_cumprod[t_step]
        alpha_t_prev = alphas_cumprod[t_step - 1] if i < len(seq) - 1 else alphas_cumprod[0]
        
        sigma_t = eta * sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        
        x = sqrt(alpha_t_prev / alpha_t) * x + \
            sqrt(1 - alpha_t_prev - sigma_t**2) * noise_pred + \
            sigma_t * torch.randn_like(x)
    
    return x
```

---

### 5.5 Post-Processing

```python
def post_process(x_synth, norm_params):
    """
    Reverse normalization to original intensity scale.
    """
    # Clamp to valid range
    x_synth = torch.clamp(x_synth, -1, 1)
    
    # Inverse percentile normalization
    p_low = norm_params["p_low"]
    p_high = norm_params["p_high"]
    mean = norm_params["mean"]
    std = norm_params["std"]
    
    # Reverse z-score
    x_denorm = (x_synth * std) + mean
    
    # Reverse percentile clipping
    x_orig_scale = np.clip(x_denorm, p_low, p_high)
    
    return x_orig_scale
```

---

### 5.6 Inference Script

```bash
python scripts/sample_diffusion.py \
  --checkpoint checkpoints/best_model.pt \
  --input-3t data/sub-XX_ses-1_T1w_desc-preproc.nii.gz \
  --output output/sub-XX_ses-1_to_7t_synth.nii.gz \
  --num-inference-steps 50 \
  --device cuda
```

---

## Phase 6: Model Extensions & Future Work

### 6.1 Topology-Preserving Loss (Future)

**Goal**: Enforce anatomically valid tissue boundaries

```python
# Use FreeSurfer aseg as supervision
from src.topology_loss import TopologyLoss

topology_loss = TopologyLoss(
    tissue_classes=3,  # GM, WM, CSF
    boundary_penalty=1.0
)

# During training:
seg_pred = model_seg_head(features)
loss_topo = topology_loss(seg_pred, aseg_gt)

loss_total = loss_denoise + 0.1 * loss_seg + 0.5 * loss_topo
```

### 6.2 Multi-Modal Guidance (Future)

**Extend to include T2-weighted data**:

```python
# Current: [x_7t_noisy (T1) | x_3t_clean (T1)] → 2 channels
# Future:  [x_7t_noisy (T1) | x_3t_clean (T1) | x_3t_clean (T2)] → 3 channels

model_multimodal = AnatomyGuidedUNet(
    in_channels=1,        # Noisy target (7T T1w)
    cond_channels=2,      # 3T T1w + T2w conditions
    out_channels=1
)
```

### 6.3 ADNI Extension (Future)

**Apply to real clinical data**:

```python
# ADNI dataset: Single 3T scan (no 7T available)
# Strategy: Transfer learning from healthy subjects

1. Pre-train on healthy subjects (current)
2. Fine-tune on ADNI 3T data (optional weak supervision)
3. Generate 7T-like ADNI images
4. Extract biomarkers (hippocampal volume, GM/WM ratio)
5. Predictive modeling for AD/MCI/CN classification
```

### 6.4 Clinical Validation (Future)

**Radiologist review protocol**:

```
1. Blind evaluation: Compare synthetic 7T vs. real 7T
   - Anatomical accuracy score (1-5)
   - Artifact presence (none/mild/moderate/severe)
   - Clinician confidence (1-5)

2. Feature extraction consistency:
   - Measure hippocampal volume on real vs. synthetic 7T
   - Compare corpus callosum thickness
   - Assess cortical thickness maps

3. Disease detection:
   - Train AD detector on real 7T scans
   - Test on synthetic 7T scans
   - Compare performance
```

### 6.5 Model Variants (Future)

| Variant | Use Case | Modifications |
|---------|----------|---|
| **Lightweight** | Mobile/edge deployment | Reduce features to [16, 32, 64, 128] |
| **Attention-Heavy** | Enhanced detail | Add attention at all decoder levels |
| **Recurrent** | Sequential refinement | Iterative denoising with feedback |
| **MultiScale** | Hierarchical synthesis | Coarse-to-fine generation |

---

## Implementation Details & Code Structure

### 7.1 Project Directory Tree

```
Topo-Brain/
│
├── README.md                                  # Quick start guide
├── COMPLETE_PROJECT_GUIDE.md                 # This document
├── requirements.txt                          # Python dependencies
│
├── configs/                                  # Configuration files (YAML)
│   ├── dataset.yaml                          # Dataset paths & organization
│   ├── preprocess.yaml                       # Preprocessing options
│   ├── train_diffusion.yaml                  # Training hyperparameters
│   └── inference.yaml                        # Inference settings
│
├── data/                                     # BIDS-formatted raw data
│   ├── sub-01/
│   │   ├── ses-1/ (3T)
│   │   └── ses-2/ (7T)
│   ├── sub-02/
│   └── ...
│
├── derivatives/                              # Preprocessed & outputs
│   ├── topobrain-preproc/
│   │   ├── sub-01_ses-1_T1w_desc-preproc.nii.gz
│   │   ├── manifest.csv
│   │   └── qc/
│   │
│   ├── Aligned/                             # Pre-registered (if available)
│   │   └── FreeSurfer aseg outputs
│   │
│   └── inference/                           # Synthetic 7T outputs
│       └── sub-XX_synthetic_7t.nii.gz
│
├── src/                                      # Core Python modules
│   ├── __init__.py
│   ├── bids.py                              # BIDS parsing utilities
│   ├── config.py                            # Configuration dataclasses
│   ├── preprocessing.py                     # N4, resampling, normalization
│   ├── preprocess_pipeline.py               # End-to-end pipeline
│   ├── dataset.py                           # PyTorch Dataset classes
│   ├── synthesis_dataset.py                 # Paired 3T-7T dataset
│   ├── model.py                             # AnatomyGuidedUNet
│   ├── diffusion.py                         # GaussianDiffusion
│   ├── utils.py                             # Helper functions
│   ├── harmonization.py                     # Intensity harmonization
│   ├── quality_control.py                   # QC metrics
│   └── topology_loss.py       [Future]      # Topology enforcement
│
├── scripts/                                  # Executable scripts
│   ├── generate_brain_masks.py              # HD-BET skull stripping
│   ├── preprocess_bids.py                   # Preprocessing pipeline
│   ├── regenerate_pairs.py                  # Create pairs.csv
│   ├── train_diffusion.py                   # Main training script
│   ├── sample_diffusion.py                  # Inference script
│   ├── visualize_dataset.py                 # Data visualization
│   ├── test_gan.py                          # GAN testing  [Legacy]
│   └── example_pipeline.py                  # Complete demo
│
├── models/                                   # Saved checkpoints
│   ├── best_model.pt                       # Best checkpoint
│   ├── latest_model.pt
│   └── archive/
│       └── model_iter_100000.pt
│
├── logs/                                     # TensorBoard logs
│   └── runs/
│       └── 20240115_234500/
│           ├── events.out.tfevents...
│           └── checkpoints/
│
├── notebooks/                                # Jupyter notebooks
│   ├── gan_training_notebook.ipynb          # Training workflow
│   └── visualization_demo.ipynb             # Results visualization
│
├── tests/                                    # Test suite
│   ├── test_diffusion_logic.py             # Diffusion tests
│   ├── test_synthesis_dataset.py           # Dataset tests
│   └── __init__.py
│
├── docs/                                     # Documentation
│   ├── ARCHITECTURE.md                      # System design
│   ├── FULL_PROJECT_REPORT.md              # Development log
│   ├── RUN_PREPROCESSING.md                 # Preprocessing guide
│   ├── CHANGELOG.md                         # Version history
│   └── blueprint_extracted.txt              # Research blueprint
│
└── pairs.csv                                 # 3T-7T pairing manifest
```

---

### 7.2 Key Classes & Functions

#### **src/model.py**: Neural Network Architecture

```python
SinusoidalPosEmb(dim)               # Time embedding layer
Block(in_ch, out_ch, groups=8)      # Basic conv block
ResnetBlock(in_ch, out_ch, time_emb_dim)  # Residual block with time
SelfAttention3D(channels, num_heads=4)    # Global attention
AnatomyGuidedUNet(...)              # Main model
  ├── forward_full(x_noisy, x_cond, t)   # Complete forward pass
  ├── forward_denoising(h, skip_connections, t_emb)  # Denoising decoder
  └── forward_segmentation(h, skip_connections, t_emb)  # Seg decoder
```

#### **src/diffusion.py**: Noise Scheduling & Sampling

```python
linear_beta_schedule(timesteps)     # Linear noise schedule
cosine_beta_schedule(timesteps)     # Cosine noise schedule
GaussianDiffusion(...)              # Main diffusion class
  ├── q_sample(x_0, t, noise)       # Add noise (training)
  ├── p_losses(model, x_3t, x_7t, t)  # Compute loss
  ├── sample(model, x_3t, shape)    # Generate (inference)
  └── sample_ddim(model, x_3t, steps)  # Fast sampling
PerceptualLoss()                    # VGG-based feature loss
```

#### **src/dataset.py**: Data Loading

```python
BrainMRIDataset(data_list, transform)  # Generic MRI dataset
MultiModalBrainMRIDataset(...)      # Multiple modalities
create_data_loaders(...)             # Utility function
```

#### **src/synthesis_dataset.py**: Training Data

```python
PatchConfig                         # Patch extraction settings
SplitConfig                         # Cross-validation configuration
SubjectSplitter.get_split()        # Train/val/test splitting
SynthesisDataset(pairs, patch_config, augment)  # Main dataset class
create_synthesis_dataloaders(...)   # Create PyTorch loaders
```

#### **src/preprocessing.py**: Preprocessing Operations

```python
N4BiasFieldCorrection               # Bias field correction
ImageResampler                      # Isotropic resampling
IntensityNormalization              # Percentile normalization
```

#### **src/bids.py**: BIDS Utilities

```python
BIDSFile                            # Represents one BIDS file
discover_bids_files()               # Find all BIDS files
get_bids_dict()                     # Parse filename
```

---

### 7.3 Configuration System

**All settings centralized in `configs/` YAML files**:

```yaml
dataset:
  data_root: /path/to/data
  output_root: derivatives/topobrain-preproc
  num_subjects: 10
  modalities: ["T1w", "T2w"]

model:
  in_channels: 1
  features: [32, 64, 128, 256]

training:
  n_iters: 400000
  lr: 1.0e-4
  batch_size: 4

diffusion:
  timesteps: 1000
  schedule: "cosine"
```

**Load in Python**:

```python
import yaml
with open("configs/train_diffusion.yaml") as f:
    config = yaml.safe_load(f)
```

---

## Configuration Reference

### 7.4 Dataset Configuration

**File**: `configs/dataset.yaml`

```yaml
data:
  # Root directory containing BIDS-formatted dataset
  data_root: /path/to/data
  
  # Output directory for preprocessed data
  output_root: derivatives/topobrain-preproc
  
  # Cache directory for intermediate results
  cache_dir: cache
  
  # Dataset statistics
  num_subjects: 10
  session_3t: "ses-1"
  session_7t: "ses-2"
  modalities: ["T1w", "T2w"]
  
data_split:
  # Cross-validation configuration
  random_seed: 42
  train_ratio: 0.6    # 6 subjects
  val_ratio: 0.2      # 2 subjects
  test_ratio: 0.2     # 2 subjects
  use_loocv: true
```

---

### 7.5 Preprocessing Configuration

**File**: `configs/preprocess.yaml`

```yaml
preprocessing:
  # Orientation
  target_orientation: "RAS"
  
  # Resampling
  target_spacing: [1.0, 1.0, 1.0]  # None to skip
  interpolation_mode: "trilinear"
  
  # Bias correction
  use_bias_correction: false  # Set true for 7T
  n4_iterations: 20
  n4_convergence_threshold: 0.001
  
  # Skull stripping
  use_skull_stripping: true
  brain_mask_pattern: "*brain_mask.nii.gz"
  
  # Intensity normalization
  normalization_method: "percentile"
  percentile_lower: 0.5
  percentile_upper: 99.5
  clip_lower_percentile: 0.5
  clip_upper_percentile: 99.5
  
  # Padding/cropping
  target_size: null  # E.g., [128, 128, 128]
  
  # Processing
  num_workers: 4
```

---

### 7.6 Training Configuration

**File**: `configs/train_diffusion.yaml`

```yaml
dataset:
  pairs_csv: "pairs.csv"
  batch_size: 4
  num_workers: 4
  prefetch_factor: 2
  persistent_workers: true

model:
  in_channels: 1
  cond_channels: 1
  out_channels: 1
  num_classes: 3
  features: [32, 64, 128, 256]
  use_attention: false

diffusion:
  timesteps: 1000
  schedule: "cosine"  # or "linear"
  beta_start: 0.0001
  beta_end: 0.02

training:
  n_iters: 400000
  lr: 1.0e-4
  beta1: 0.9
  beta2: 0.999
  weight_decay: 0.0001
  gradient_clip_norm: 1.0
  
  warmup_steps: 5000
  warmup_init_lr: 1.0e-5
  
  ema_decay: 0.9999
  ema_update_step: 1
  
  use_amp: true
  
  save_freq: 1000
  eval_freq: 2000
  sample_freq: 2000
  
  loss_type: "l1"
  lambda_seg: 0.1
  lambda_percep: 0.001
```

---

## Running the Complete Pipeline

### 8.1 Quick Start (5 steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess BIDS data
python scripts/preprocess_bids.py \
  --data-root /path/to/data \
  --config configs/preprocess.yaml

# 3. Generate pairs.csv
python scripts/regenerate_pairs.py \
  --preproc-root derivatives/topobrain-preproc \
  --output pairs.csv

# 4. Train diffusion model
python scripts/train_diffusion.py \
  --config configs/train_diffusion.yaml \
  --output models/

# 5. Generate samples
python scripts/sample_diffusion.py \
  --checkpoint models/best_model.pt \
  --input-3t data/sub-XX_ses-1_T1w_desc-preproc.nii.gz \
  --output derivatives/inference/sub-XX_synthetic_7t.nii.gz
```

---

### 8.2 Detailed Workflow

#### **Step 1: Set Up Environment**

```bash
# Create virtual environment
python -m venv topobrain_env
source topobrain_env/bin/activate  # On Windows: topobrain_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

#### **Step 2: Preprocessing**

```bash
# Edit configs/preprocess.yaml with your data paths
# Then run:

python scripts/preprocess_bids.py \
  --data-root /data/BIDSdata \
  --config configs/preprocess.yaml \
  --output-root derivatives/topobrain-preproc \
  --num-workers 4

# Monitor QC report
open derivatives/topobrain-preproc/qc/qc_report.html
```

#### **Step 3: Data Preparation**

```bash
python scripts/regenerate_pairs.py \
  --preproc-root derivatives/topobrain-preproc \
  --session-3t ses-1 \
  --session-7t ses-2 \
  --output pairs.csv

# Verify pairs.csv was created
head -5 pairs.csv
```

#### **Step 4: Training**

```bash
# Edit configs/train_diffusion.yaml (set paths, hyperparameters)

# Dry run (test first batch)
python scripts/train_diffusion.py \
  --config configs/train_diffusion.yaml \
  --output models/ \
  --dry-run

# Full training
python scripts/train_diffusion.py \
  --config configs/train_diffusion.yaml \
  --output models/ \
  --use-wandb

# Monitor in TensorBoard
tensorboard --logdir=logs --port=6006
# Open http://localhost:6006
```

#### **Step 5: Inference**

```bash
# Generate synthetic 7T from 3T
python scripts/sample_diffusion.py \
  --checkpoint models/best_model.pt \
  --input-3t derivatives/topobrain-preproc/sub-XX_ses-1_T1w_desc-preproc.nii.gz \
  --output derivatives/inference/sub-XX_synthetic_7t.nii.gz \
  --num-inference-steps 50 \
  --device cuda

# Visualize results
python scripts/visualize_dataset.py \
  --input-3t derivatives/topobrain-preproc/sub-XX_ses-1_T1w_desc-preproc.nii.gz \
  --input-7t-synthetic derivatives/inference/sub-XX_synthetic_7t.nii.gz \
  --input-7t-real derivatives/topobrain-preproc/sub-XX_ses-2_T1w_desc-preproc.nii.gz \
  --output derivatives/inference/comparison_XX.png
```

---

## Troubleshooting & Common Issues

### 9.1 Memory Issues

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:

```python
# 1. Reduce batch size in config
batch_size: 2  # Default 4

# 2. Reduce patch size
patch_size: 48  # Default 64

# 3. Reduce model size
features: [16, 32, 64, 128]  # Default [32, 64, 128, 256]

# 4. Enable gradient checkpointing (slower but memory-efficient)
use_gradient_checkpointing: true

# 5. Reduce number of workers
num_workers: 0  # Default 4
```

---

### 9.2 Data Loading Issues

**Problem**: `FileNotFoundError: Image not found`

**Solutions**:

```bash
# 1. Verify preprocessing completed
ls derivatives/topobrain-preproc/*.nii.gz | wc -l  # Should be 20

# 2. Check pairs.csv paths
head pairs.csv
# Verify paths exist:
test -f $(head -2 pairs.csv | tail -1 | cut -d, -f3)

# 3. Verify BIDS structure
ls -la data/sub-01/ses-1/anat/
```

---

### 9.3 Training Issues

**Problem**: Loss not decreasing (NaN or Inf)

**Solutions**:

```python
# 1. Check data normalization
assert x_3t.min() >= -1.0 and x_3t.max() <= 1.0
assert x_7t.min() >= -1.0 and x_7t.max() <= 1.0

# 2. Reduce learning rate
lr: 1.0e-5  # From 1.0e-4

# 3. Enable gradient clipping
gradient_clip_norm: 1.0

# 4. Check diffusion schedule
beta_end: 0.01  # Lower value

# 5. Reduce loss weights
lambda_seg: 0.05  # From 0.1
lambda_percep: 0.001  # From 0.01
```

---

### 9.4 Inference Issues

**Problem**: Generated image has artifacts or poor quality

**Solutions**:

```python
# 1. Use best checkpoint (not latest)
checkpoint = "models/best_model.pt"

# 2. Increase inference steps
num_inference_steps: 100  # From 50

# 3. Adjust DDIM eta (for stochasticity)
eta: 0.0  # Deterministic (default)
eta: 1.0  # More stochastic but potentially better quality

# 4. Apply post-processing
sample = torch.clamp(sample, -1, 1)
sample = apply_bilateral_filter(sample)  # Optional denoising

# 5. Check conditioning strength (use classifier-free guidance)
guidance_scale: 1.5  # 1.0 = no guidance
```

---

### 9.5 Preprocessing Issues

**Problem**: Brain masks look wrong

**Solutions**:

```bash
# 1. Use manual inspection
python scripts/visualize_dataset.py --show-masks

# 2. If using HD-BET, verify installation
python -c "from hd_bet.run import run_hd_bet"

# 3. Fall back to simple thresholding
use_skull_stripping: false
# Or use FreeSurfer aseg if available

# 4. Manually review and provide masks
# Place in derivatives/Aligned/{subject}/brainmask.nii.gz
```

---

## Advanced Topics

### 10.1 Distributed Training (Multi-GPU)

```bash
# Using PyTorch DistributedDataParallel
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  scripts/train_diffusion.py \
  --config configs/train_diffusion.yaml \
  --distributed
```

---

### 10.2 Mixed Precision Training

```yaml
# The model already supports AMP:
training:
  use_amp: true  # Automatic Mixed Precision

# Expected speedup: 1.5-2x
# Memory reduction: ~30%
```

---

### 10.3 Custom Loss Functions

Extend the diffusion loss:

```python
# In src/diffusion.py
class CustomDiffusion(GaussianDiffusion):
    def p_losses(self, model, x_3t, x_7t, t, loss_type="l1"):
        # Add your custom losses here
        noise_pred, seg_logits = model(...)
        
        loss_denoise = F.l1_loss(noise_pred, noise)
        loss_seg = F.cross_entropy(seg_logits, seg_gt)
        
        # Custom losses
        loss_contrastive = contrastive_loss(features)
        loss_boundary = boundary_loss(seg_logits)
        
        return loss_denoise + 0.1 * loss_seg + \
               0.05 * loss_contrastive + 0.02 * loss_boundary
```

---

### 10.4 Classifier-Free Guidance

```python
# Generate with guidance
def sample_with_guidance(model, x_3t, guidance_scale=1.5, num_steps=50):
    """
    Guidance scale > 1: Condition more strongly on x_3t
    Guidance scale = 1: No guidance (default)
    Guidance scale < 1: Condition less strongly
    """
    x = torch.randn_like(x_3t)
    
    for t in reversed(range(num_steps)):
        # Predict with conditioning
        noise_cond, _ = model(x, x_3t, t)
        
        # Predict without conditioning
        noise_uncond, _ = model(x, torch.zeros_like(x_3t), t)
        
        # Blend
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        
        # Update x
        x = update_step(x, noise_pred, t)
    
    return x
```

---

## Final Summary

### Key Achievements

✅ **Complete preprocessing pipeline** from raw BIDS data
✅ **AnatomyGuidedUNet** with multi-task learning
✅ **Stable training** using DDPM with EMA smoothing
✅ **Efficient inference** with patch-based tiling
✅ **Comprehensive documentation** for reproducibility
✅ **Quality control** and evaluation framework

### Performance Metrics (Typical)

| Metric | Typical Value |
|--------|---|
| **PSNR** | 28-32 dB |
| **SSIM** | 0.85-0.92 |
| **Training time** | 1-4 days (1-4 GPU-days) |
| **Inference time** | 30-60 sec/volume (50 steps) |
| **Model size** | 22M parameters |

### Next Steps

1. **Clinical validation** with radiologist review
2. **Feature extraction consistency** testing
3. **ADNI deployment** for AD detection
4. **Topology-preserving loss** implementation
5. **Multi-modal guidance** with T2 data

---

## References & Resources

### Papers

- **Diffusion**: Ho et al., DDPM (2020), Song et al., DDIM (2020)
- **Vision Transformers**: Dosovitskiy et al., ViT (2021)
- **Medical Imaging**: The dataset follows BIDS (Gorgolewski et al., 2016)

### Official Documentation

- PyTorch: https://pytorch.org/docs/
- MONAI: https://monai.io/
- Nibabel: https://nipy.org/nibabel/
- TensorBoard: https://www.tensorflow.org/tensorboard

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Status**: Complete Project Documentation

For questions or updates, refer to the project repository's CHANGELOG.md and FULL_PROJECT_REPORT.md.
