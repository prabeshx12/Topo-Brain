# Quick Reference: Advanced Features

A quick guide to using the new production-ready features.

## 📋 Table of Contents

1. [Quality Control (QC)](#quality-control-qc)
2. [Intensity Harmonization](#intensity-harmonization)
3. [TensorBoard Logging](#tensorboard-logging)
4. [Data Caching](#data-caching)
5. [Mixed Precision Training](#mixed-precision-training)
6. [Cross-Validation](#cross-validation)
7. [Enhanced Augmentation](#enhanced-augmentation)

---

## Quality Control (QC)

### Quick Start
```python
from quality_control import QCMetrics, PreprocessingQC

# Single scan QC
metrics = QCMetrics.compute_metrics("scan.nii.gz")
print(f"SNR: {metrics['snr']:.2f}, CNR: {metrics['cnr']:.2f}")

# Batch QC with report
qc = PreprocessingQC(output_dir="logs/qc")
qc_df = qc.run_qc(image_paths)
qc.generate_report(qc_df, "qc_report.html")
```

### Metrics Computed
- **SNR**: Signal-to-Noise Ratio (higher is better)
- **CNR**: Contrast-to-Noise Ratio (higher is better)
- **Entropy**: Image complexity measure
- **Artifacts**: Detection of ringing, ghosting, motion
- **Outliers**: Statistical outlier detection (IQR + z-score)

### When to Use
- After preprocessing to validate quality
- Before training to remove bad scans
- For dataset quality assessment

---

## Intensity Harmonization

### Quick Start
```python
from harmonization import IntensityHarmonizer

# Create and fit harmonizer
harmonizer = IntensityHarmonizer(method='histogram')
harmonizer.fit(reference_scans)  # e.g., 3T scans

# Transform target scans
harmonized = harmonizer.transform("7t_scan.nii.gz")

# Save for later
harmonizer.save("harmonizer.pkl")
```

### Methods Available
1. **Histogram Matching**: Matches intensity distributions exactly
2. **Z-score**: Standardizes mean and std across scans
3. **Quantile**: Aligns intensity percentiles

### When to Use
- Training on mixed 3T/7T data
- Multi-site studies with different scanners
- Reducing batch effects

### Workflow
```python
# 1. Separate by field strength
scans_3t = [d['image'] for d in data if d['session'] == 'ses-1']
scans_7t = [d['image'] for d in data if d['session'] == 'ses-2']

# 2. Fit on reference (3T)
harmonizer = IntensityHarmonizer(method='histogram')
harmonizer.fit(scans_3t)

# 3. Transform target (7T)
for scan_7t in scans_7t:
    harmonized = harmonizer.transform(scan_7t)
    # Use harmonized scan for training
```

---

## TensorBoard Logging

### Quick Start
```python
from utils import TensorBoardLogger

# Create logger
logger = TensorBoardLogger(log_dir="logs/tensorboard")

# Log metrics
logger.log_scalar("train/loss", loss, step)
logger.log_scalar("train/accuracy", acc, step)

# Log images
logger.log_image("train/input", image_tensor, step)

# Log 3D volumes (as slice montage)
logger.log_volume_slices("train/volume", volume_4d, step)

# Log histograms
logger.log_histogram("model/weights", weight_tensor, step)

# Close when done
logger.close()
```

### View Logs
```bash
tensorboard --logdir=logs/tensorboard
# Open browser to http://localhost:6006
```

### Best Practices
```python
# In training loop
for epoch in range(num_epochs):
    for i, batch in enumerate(train_loader):
        # Training step
        loss = train_step(batch)
        
        # Log every N iterations
        global_step = epoch * len(train_loader) + i
        if global_step % 10 == 0:
            logger.log_scalar("train/loss", loss, global_step)
        
        # Log images less frequently
        if global_step % 100 == 0:
            logger.log_batch_statistics(batch, "train/batch", global_step)
```

---

## Data Caching

### Persistent Cache (Disk)
```python
# Enable in config
config.training.use_persistent_cache = True

# Or pass to create_data_loaders
train_loader, val_loader, test_loader = create_data_loaders(
    config, train_data, val_data, test_data,
    use_persistent_cache=True
)
```

**Benefits:**
- Caches preprocessed data to disk
- Faster startup after first run (10-50x speedup)
- Survives program restarts
- Good for: Large datasets, slow preprocessing

**Trade-offs:**
- Requires disk space (same size as preprocessed data)
- First run is slower (building cache)

### Memory Cache (RAM)
```python
# Enable in config
config.training.use_memory_cache = True

# Or pass to create_data_loaders
train_loader, val_loader, test_loader = create_data_loaders(
    config, train_data, val_data, test_data,
    use_memory_cache=True
)
```

**Benefits:**
- Caches preprocessed data in RAM
- Fastest iteration speed
- Good for: Small datasets that fit in memory

**Trade-offs:**
- Requires sufficient RAM
- Cache lost on program restart
- Not suitable for large datasets

### When to Use What

| Dataset Size | RAM Available | Recommendation |
|--------------|---------------|----------------|
| < 10 GB      | > 32 GB       | Memory cache   |
| 10-100 GB    | > 64 GB       | Persistent cache |
| > 100 GB     | Any           | No cache (or persistent) |

---

## Mixed Precision Training

### Quick Start
```python
# Enable in config
config.training.use_amp = True

# In training loop
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():  # Auto mixed precision
        output = model(batch["image"])
        loss = criterion(output, target)
    
    # Scaled backprop
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Benefits
- 2-3x faster training on modern GPUs
- Reduced memory usage (~50%)
- No accuracy loss (usually)

### Requirements
- NVIDIA GPU with Tensor Cores (RTX series, A100, etc.)
- PyTorch 1.6+
- CUDA 10.0+

### Compatibility
- Works with: ResNet, U-Net, Transformers
- May not work with: Some custom layers, very old models

---

## Cross-Validation

### Quick Start
```python
from utils import run_kfold_cross_validation, create_kfold_splits

# Create k-fold splits
splits = create_kfold_splits(data_list, k=5, random_seed=42)

# Run CV (manual)
for fold_idx, (train_data, val_data) in enumerate(splits):
    print(f"Training fold {fold_idx + 1}/5")
    # Train model on this fold
    
# Or use wrapper (advanced)
# See utils.py for run_kfold_cross_validation()
```

### Configuration
```python
from utils import create_kfold_splits

# 5-fold CV
splits = create_kfold_splits(data_list, k=5, random_seed=42)

# 10-fold CV
splits = create_kfold_splits(data_list, k=10, random_seed=42)
```

### Best Practices
- Use k=5 for small datasets (< 50 subjects)
- Use k=10 for medium datasets (50-200 subjects)
- Always use patient-level splitting (automatic in our implementation)
- Report mean ± std across folds

---

## Enhanced Augmentation

### MRI-Specific Augmentations

#### Gibbs Ringing Artifact
```python
# Enable in config
config.augmentation.gibbs_noise_prob = 0.3

# Simulates Gibbs ringing (common MRI artifact)
# Helps model become robust to acquisition artifacts
```

#### Coarse Dropout
```python
# Enable in config
config.augmentation.coarse_dropout_prob = 0.2

# Randomly drops out regions of the volume
# Improves robustness to missing/corrupted data
```

### Full Augmentation Pipeline
```python
from config import AugmentationConfig

aug_config = AugmentationConfig(
    # Spatial
    affine_prob=0.5,
    flip_prob=0.5,
    
    # Intensity
    intensity_shift_prob=0.3,
    intensity_scale_prob=0.3,
    
    # Noise
    gaussian_noise_prob=0.2,
    gaussian_smooth_prob=0.2,
    
    # MRI-specific (NEW)
    gibbs_noise_prob=0.3,
    coarse_dropout_prob=0.2,
)
```

### Recommendations
- **Light augmentation**: Set all probs to 0.2-0.3
- **Standard augmentation**: Set all probs to 0.3-0.5 (default)
- **Heavy augmentation**: Set all probs to 0.5-0.7 (for very small datasets)
- **No augmentation**: Set all probs to 0.0 (for validation/testing)

---

## Configuration Examples

### High-Performance Setup
```python
from config import MRIConfig

config = MRIConfig()

# Enable all performance features
config.training.use_amp = True
config.training.use_persistent_cache = True
config.training.batch_size = 8  # Larger batch with AMP
config.training.num_workers = 8

# Moderate augmentation
config.augmentation.gibbs_noise_prob = 0.3
config.augmentation.coarse_dropout_prob = 0.2

# TensorBoard logging
config.logging.use_tensorboard = True
```

### Production QC Setup
```python
from config import MRIConfig

config = MRIConfig()

# Enable QC
config.logging.run_qc = True
config.logging.qc_dir = Path("logs/qc")

# Conservative preprocessing
config.preprocessing.use_bias_correction = True
config.preprocessing.normalization_method = "zscore"

# No augmentation for QC
config.augmentation.affine_prob = 0.0
config.augmentation.flip_prob = 0.0
```

### Small Dataset Setup
```python
from config import MRIConfig

config = MRIConfig()

# Memory cache (dataset fits in RAM)
config.training.use_memory_cache = True

# Heavy augmentation (small data)
config.augmentation.affine_prob = 0.7
config.augmentation.flip_prob = 0.7
config.augmentation.gibbs_noise_prob = 0.5

# K-fold CV
# Use create_kfold_splits() instead of create_patient_level_split()
```

---

## Troubleshooting

### QC Reports Show Many Outliers
- Review QC report visualizations
- Check preprocessing parameters
- Consider adjusting thresholds in `PreprocessingQC`
- Some outliers are normal (check if they're actually bad)

### Harmonization Not Working
- Ensure reference scans are high quality
- Try different harmonization methods
- Check if intensity ranges are reasonable
- Visualize before/after histograms

### TensorBoard Not Starting
- Check if `tensorboard` is installed: `pip install tensorboard`
- Verify log directory exists and is writable
- Try: `tensorboard --logdir=logs/tensorboard --bind_all`

### Caching Issues
- Clear cache: Delete contents of `cache_dir`
- Check disk space (persistent cache)
- Check RAM usage (memory cache)
- Disable cache if problems persist

### Mixed Precision Errors
- Check GPU compatibility (needs Tensor Cores)
- Update PyTorch: `pip install --upgrade torch`
- Some layers may not support AMP (disable if needed)
- Check for NaN/Inf values in gradients

---

## Example Workflows

### Workflow 1: Standard Training
```bash
# 1. Validate setup
python setup_validation.py

# 2. Preprocess with QC
python example_pipeline.py --preprocess --config default

# 3. Check QC results
python example_qc_harmonization.py

# 4. Train with monitoring
# (enable TensorBoard in config, use training script)
```

### Workflow 2: Multi-Scanner Data
```bash
# 1. Preprocess both scanners
python example_pipeline.py --preprocess

# 2. Run harmonization
python example_qc_harmonization.py

# 3. Train on harmonized data
# (use harmonized scans in training)
```

### Workflow 3: Cross-Validation
```python
from utils import create_kfold_splits

# 1. Create CV splits
data_list = discover_dataset(config.data.data_root, config.data)
splits = create_kfold_splits(data_list, k=5)

# 2. Train each fold
results = []
for fold, (train_data, val_data) in enumerate(splits):
    # Create loaders
    train_loader, val_loader, _ = create_data_loaders(
        config, train_data, val_data, []
    )
    
    # Train model
    metrics = train_model(train_loader, val_loader)
    results.append(metrics)

# 3. Report results
print(f"Mean accuracy: {np.mean(results):.3f} ± {np.std(results):.3f}")
```

---

## Further Reading

- [IMPROVEMENTS.md](IMPROVEMENTS.md) - Additional enhancement ideas
- [CHANGELOG.md](CHANGELOG.md) - Detailed change history
- [README.md](README.md) - Full documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture

---

**Questions?** Check the main README or examine the example scripts!
