# Topo-Brain: Preprocessing & Diffusion Model Pipeline - Architecture Overview

This document describes the complete architecture from preprocessing through diffusion model training.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CONFIGURATION LAYER                              │
│                            (config.py)                                   │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │   Default    │  │   HighRes    │  │     Fast     │                  │
│  │   Config     │  │   Config     │  │   Config     │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA DISCOVERY LAYER                              │
│                            (utils.py)                                    │
│                                                                          │
│  Input: BIDS Dataset (Nifti/)                                          │
│  ┌───────────────────────────────────────────────┐                     │
│  │  sub-01/  sub-02/  ...  sub-10/               │                     │
│  │    ├── ses-1/ (3T)                            │                     │
│  │    │   └── anat/                              │                     │
│  │    │       ├── T1w_defaced.nii.gz            │                     │
│  │    │       └── T2w_defaced.nii.gz            │                     │
│  │    └── ses-2/ (7T)                            │                     │
│  │        └── anat/                              │                     │
│  │            ├── T1w_defaced.nii.gz            │                     │
│  │            └── T2w_defaced.nii.gz            │                     │
│  └───────────────────────────────────────────────┘                     │
│                                                                          │
│  Output: data_list = [                                                  │
│    {image: Path, subject: str, session: str, modality: str, ...},      │
│    ...                                                                   │
│  ]                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PATIENT-LEVEL SPLITTING                             │
│                            (utils.py)                                    │
│                                                                          │
│  ┌─────────────────────────────────────────────┐                       │
│  │  NO DATA LEAKAGE GUARANTEE                   │                       │
│  │                                               │                       │
│  │  Train: sub-01, sub-02, sub-03, sub-04,     │                       │
│  │         sub-05, sub-06 (both sessions)      │                       │
│  │                                               │                       │
│  │  Val:   sub-07, sub-08 (both sessions)      │                       │
│  │                                               │                       │
│  │  Test:  sub-09, sub-10 (both sessions)      │                       │
│  └─────────────────────────────────────────────┘                       │
│                                                                          │
│  Saved to: cache/data_split.json (for reproducibility)                 │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING PIPELINE                              │
│                        (preprocessing.py)                                │
│                                                                          │
│  For each volume:                                                       │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ 1. Load NIfTI                                                  │     │
│  │    ↓                                                           │     │
│  │ 2. N4 Bias Field Correction (SimpleITK)                       │     │
│  │    ├─ Removes intensity non-uniformity                        │     │
│  │    └─ Critical for 7T data                                    │     │
│  │    ↓                                                           │     │
│  │ 3. Skull Stripping                                            │     │
│  │    ├─ Uses pre-computed masks if available                    │     │
│  │    └─ Fallback: Simple thresholding (⚠️ not production-ready) │     │
│  │    ↓                                                           │     │
│  │ 4. Reorientation to RAS+ (MONAI)                             │     │
│  │    └─ Standard neuroimaging orientation                       │     │
│  │    ↓                                                           │     │
│  │ 5. Optional: Isotropic Resampling                            │     │
│  │    └─ E.g., resample to 1mm³ or 0.5mm³                       │     │
│  │    ↓                                                           │     │
│  │ 6. Intensity Normalization                                    │     │
│  │    ├─ Z-score: (x - μ) / σ  [recommended]                    │     │
│  │    ├─ Min-Max: (x - min) / (max - min)                       │     │
│  │    └─ Percentile-based                                        │     │
│  │    ↓                                                           │     │
│  │ 7. Optional: Pad/Crop to Fixed Size                          │     │
│  │    └─ E.g., resize to 128³ or 256³                           │     │
│  │    ↓                                                           │     │
│  │ 8. Save Preprocessed Volume + Metadata                        │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  Output: preprocessed/*.nii.gz + metadata.json                         │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      3T-7T PAIR GENERATION                               │
│                     (regenerate_pairs.py)                                │
│                                                                          │
│  Input: Preprocessed 3T and 7T volumes                                 │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ 1. Match 3T and 7T scans by subject/modality                  │     │
│  │    ↓                                                           │     │
│  │ 2. Create pairs.csv manifest                                  │     │
│  │    └─ Columns: input_3t, target_7t, subject, modality        │     │
│  │    ↓                                                           │     │
│  │ 3. Patient-level train/val/test split                        │     │
│  │    └─ No data leakage between splits                          │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  Output: pairs.csv                                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      SYNTHESIS DATASET LOADER                            │
│                     (synthesis_dataset.py)                               │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │  SynthesisDataset                                            │       │
│  │  ├─ Loads paired 3T-7T volumes from pairs.csv              │       │
│  │  ├─ Random 3D patch extraction                              │       │
│  │  ├─ Applies data augmentation (training only)               │       │
│  │  └─ Returns: {x_3t, x_7t, mask, subject}                   │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                          │                                              │
│                          ▼                                              │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │  Data Augmentation (Training Only)                           │       │
│  │  ├─ Random Affine (rotation, translation, scaling)          │       │
│  │  ├─ Random Flip (left-right)                                │       │
│  │  ├─ Random Intensity Shift & Scale                          │       │
│  │  └─ Random Gaussian Noise                                   │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  Output: train_loader, val_loader, test_loader                         │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      DIFFUSION MODEL ARCHITECTURE                        │
│                     (model.py + diffusion.py)                            │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │  AnatomyGuidedUNet (model.py)                               │       │
│  │  ├─ Input: Concatenate(noisy_7t, clean_3t)                 │       │
│  │  ├─ Encoder: [32, 64, 128, 256] features                   │       │
│  │  ├─ Decoder (Denoising): Mirror encoder                     │       │
│  │  ├─ Decoder (Segmentation): Tissue classification           │       │
│  │  └─ Multi-Task: Denoise 7T + Predict tissue masks          │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                          │                                              │
│                          ▼                                              │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │  GaussianDiffusion (diffusion.py)                           │       │
│  │  ├─ Forward: Add noise to 7T (q(x_t | x_0))               │       │
│  │  ├─ Reverse: Denoise step-by-step (p(x_{t-1} | x_t))      │       │
│  │  ├─ Loss: L1 + Perceptual + Segmentation                   │       │
│  │  └─ Timesteps: 1000 (linear/cosine schedule)               │       │
│  └─────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      TRAINING LOOP                                       │
│                     (train_diffusion.py)                                 │
│                                                                          │
│  for iteration in range(n_iters):                                      │
│      batch = next(train_loader)                                        │
│      x_3t, x_7t, mask = batch['x_3t'], batch['x_7t'], batch['mask']   │
│                                                                          │
│      # Sample random timestep                                           │
│      t = random.randint(0, 1000)                                       │
│                                                                          │
│      # Compute loss (denoising + segmentation)                          │
│      loss_dict = diffusion.compute_loss(x_3t, x_7t, mask, t)          │
│                                                                          │
│      # Backprop & optimize                                              │
│      optimizer.zero_grad()                                              │
│      loss_dict['total'].backward()                                      │
│      optimizer.step()                                                   │
│                                                                          │
│      # Update EMA model                                                 │
│      ema.step_ema(ema_model, model)                                    │
│                                                                          │
│      # Periodic saving & sampling                                       │
│      if iteration % save_freq == 0:                                    │
│          save_checkpoint(model, ema_model, optimizer)                  │
│          generate_samples(ema_model, val_batch)                        │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      INFERENCE & SAMPLING                                │
│                     (sample_diffusion.py)                                │
│                                                                          │
│  Input: 3T MRI volume + trained model                                  │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ 1. Load trained EMA model                                     │     │
│  │    ↓                                                           │     │
│  │ 2. Start with random noise x_T ~ N(0, I)                     │     │
│  │    ↓                                                           │     │
│  │ 3. Iterative denoising (t = 1000 → 0)                        │     │
│  │    └─ x_{t-1} = denoise(x_t, x_3t, t)                        │     │
│  │    ↓                                                           │     │
│  │ 4. Final output: x_0 (synthetic 7T)                          │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  Output: Synthetic 7T MRI volume                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. config.py
- Centralized configuration
- Multiple presets (default, highres, fast)
- All parameters in one place

### 2. preprocessing.py
- N4BiasFieldCorrection class (SimpleITK)
- SkullStripping class (with fallback)
- IntensityNormalization class (3 methods)
- MRIPreprocessor (orchestrates all steps)

### 3. synthesis_dataset.py
- SynthesisDataset (paired 3T-7T loading)
- Random patch extraction
- Conditional augmentation
- Patient-level data splitting

### 4. model.py
- AnatomyGuidedUNet (3D conditional U-Net)
- Multi-task learning (denoising + segmentation)
- Residual blocks for stable training
- Dual decoder architecture

### 5. diffusion.py
- GaussianDiffusion (DDPM implementation)
- Forward diffusion process (noise addition)
- Reverse diffusion process (denoising)
- Multi-component loss functions
- Perceptual loss (VGG-based)

### 6. train_diffusion.py
- Main training loop
- EMA model tracking
- TensorBoard/W&B logging
- Checkpoint management
- Periodic sampling

### 7. sample_diffusion.py
- Inference pipeline
- DDPM/DDIM sampling
- Full volume reconstruction
- Quality metrics computation

## Data Flow

```
Raw BIDS Data
    ↓
Discover & Parse
    ↓
Preprocessing (N4, Skull Strip, Normalize)
    ↓
Create 3T-7T Pairs (pairs.csv)
    ↓
SynthesisDataset (Patch Extraction + Augmentation)
    ↓
Diffusion Training (Iterative Denoising)
    ↓
Trained Model
    ↓
Sampling (Generate 7T from 3T)
```

## File Locations

```
Input:  Nifti/**/*_defaced.nii.gz
Output: preprocessed/**/*_preprocessed.nii.gz
Pairs:  pairs.csv
Models: models/*.pth
Logs:   logs/**/* (TensorBoard, samples, training logs)
```

## Critical Features

✅ **Patient-Level Splitting** - Prevents data leakage
✅ **Conditional Diffusion** - 3T guides 7T generation
✅ **Multi-Task Learning** - Denoising + tissue segmentation
✅ **EMA Tracking** - Stable inference model
✅ **Production-Ready** - Error handling, logging, validation
✅ **Flexible Configuration** - YAML-based settings
✅ **Modular** - Easy to customize and extend
