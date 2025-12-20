# Brain MRI Preprocessing Pipeline - Architecture Overview

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
│                      PYTORCH DATASET & DATALOADER                        │
│                            (dataset.py)                                  │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │  BrainMRIDataset                                             │       │
│  │  ├─ Loads preprocessed volumes                              │       │
│  │  ├─ Applies data augmentation (training only)               │       │
│  │  ├─ Returns: {image: Tensor, subject: str, ...}            │       │
│  │  └─ Optional: In-memory caching                             │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                          │                                              │
│                          ▼                                              │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │  Data Augmentation (Training Only)                           │       │
│  │  ├─ Random Affine (rotation, translation, scaling)          │       │
│  │  ├─ Random Flip (left-right)                                │       │
│  │  ├─ Random Intensity Shift & Scale                          │       │
│  │  ├─ Random Gaussian Noise                                   │       │
│  │  └─ Random Gaussian Smoothing                               │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                          │                                              │
│                          ▼                                              │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │  PyTorch DataLoader                                          │       │
│  │  ├─ Batch size: 2-4 (3D volumes are large)                 │       │
│  │  ├─ Multi-worker loading (4-8 workers)                      │       │
│  │  ├─ Pin memory for GPU transfer                             │       │
│  │  └─ Prefetching for faster loading                          │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  Output: train_loader, val_loader, test_loader                         │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      MONITORING & VISUALIZATION                          │
│                            (utils.py)                                    │
│                                                                          │
│  ┌────────────────────┐  ┌────────────────────┐  ┌─────────────────┐  │
│  │  Statistics        │  │  Visualizations    │  │  Logging        │  │
│  │  ├─ Mean/Std       │  │  ├─ Orthogonal    │  │  ├─ Pipeline    │  │
│  │  ├─ Min/Max        │  │  │   slices        │  │  │   logs       │  │
│  │  ├─ Percentiles    │  │  ├─ Intensity      │  │  ├─ Statistics │  │
│  │  └─ Shape info     │  │  │   histograms    │  │  └─ Errors     │  │
│  └────────────────────┘  │  └─ Before/After   │  └─────────────────┘  │
│                          │    preprocessing     │                       │
│                          └────────────────────┘                        │
│                                                                          │
│  Saved to: logs/ directory                                             │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         YOUR MODEL TRAINING                              │
│                                                                          │
│  for epoch in range(num_epochs):                                       │
│      for batch in train_loader:                                        │
│          images = batch['image']  # (B, 1, H, W, D)                    │
│          # Your model forward pass                                      │
│          outputs = model(images)                                        │
│          # Compute loss, backprop, optimize                             │
│          ...                                                             │
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

### 3. dataset.py
- BrainMRIDataset (single modality)
- MultiModalBrainMRIDataset (T1w + T2w)
- build_augmentation_transforms()
- create_data_loaders()

### 4. utils.py
- discover_dataset() - BIDS parsing
- create_patient_level_split() - No leakage!
- create_kfold_splits() - Cross-validation
- visualize_sample() - Visualization
- compute_dataset_statistics() - Analytics

### 5. example_pipeline.py
- End-to-end demonstration
- Command-line interface
- Complete workflow

## Data Flow

```
Raw BIDS Data
    ↓
Discover & Parse
    ↓
Patient-Level Split (Train/Val/Test)
    ↓
Preprocessing (N4, Skull Strip, Normalize)
    ↓
PyTorch Dataset
    ↓
DataLoader (with Augmentation)
    ↓
Your Model
```

## File Locations

```
Input:  Nifti/**/*_defaced.nii.gz
Output: preprocessed/**/*_preprocessed.nii.gz
Cache:  cache/data_split.json
Logs:   logs/pipeline.log, logs/statistics/, logs/visualizations/
```

## Critical Features

✅ **Patient-Level Splitting** - Prevents data leakage
✅ **Deterministic** - Fixed random seeds, reproducible
✅ **Medical Imaging Best Practices** - RAS orientation, N4 correction
✅ **Production-Ready** - Error handling, logging, validation
✅ **Modular** - Easy to customize and extend
✅ **Well-Documented** - Extensive docstrings and examples
