# Installation and Testing Guide

Complete guide for installing dependencies and testing the new features.

## 📦 Installation

### Step 1: Update Dependencies

```bash
# Navigate to project directory
cd d:\11PrabeshX\Projects\major_

# Install/upgrade new dependencies
pip install --upgrade scikit-learn tensorboard scikit-image

# Or install all requirements
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
# Run validation script
python setup_validation.py
```

Expected output should show all packages installed, including:
- ✓ scikit-learn (>= 1.3.0)
- ✓ tensorboard (>= 2.13.0)
- ✓ scikit-image (>= 0.21.0)

---

## 🧪 Testing New Features

### Test 1: Quality Control

**Purpose**: Verify QC metrics and reporting work correctly

```bash
# Run QC example (requires preprocessed data)
python example_qc_harmonization.py
```

**Expected behavior**:
1. Discovers preprocessed scans
2. Computes QC metrics (SNR, CNR, entropy)
3. Generates HTML report at `logs/qc/qc_report.html`
4. Prints summary statistics

**Validation**:
- Check `logs/qc/qc_report.html` exists and opens in browser
- Verify metrics are reasonable (SNR > 0, CNR > 0)
- Check for outliers in report

**If no preprocessed data**:
```bash
# First preprocess data
python example_pipeline.py --preprocess --config default
# Then run QC
python example_qc_harmonization.py
```

---

### Test 2: Intensity Harmonization

**Purpose**: Verify harmonization across 3T/7T scans

```bash
# Run harmonization example (included in QC script)
python example_qc_harmonization.py
```

**Expected behavior**:
1. Separates 3T (ses-1) and 7T (ses-2) scans
2. Fits harmonizers on 3T data
3. Transforms 7T data
4. Saves harmonized outputs to `Nifti/preprocessed/`
5. Prints before/after statistics

**Validation**:
- Check harmonized files exist: `harmonized_histogram_sample.nii.gz`, etc.
- Verify intensity statistics change (mean/std should be closer to 3T)
- Harmonizer models saved: `cache/harmonizer_*.pkl`

---

### Test 3: TensorBoard Logging

**Purpose**: Verify TensorBoard integration works

```python
# Create test script: test_tensorboard.py
from utils import TensorBoardLogger
import torch
import numpy as np

logger = TensorBoardLogger(log_dir="logs/tensorboard_test")

# Test scalar logging
for i in range(10):
    logger.log_scalar("test/loss", np.random.rand(), i)
    logger.log_scalar("test/accuracy", np.random.rand(), i)

# Test image logging
test_image = torch.rand(1, 128, 128)
logger.log_image("test/sample_image", test_image, 0)

# Test histogram
test_tensor = torch.randn(1000)
logger.log_histogram("test/weights", test_tensor, 0)

logger.close()
print("✓ TensorBoard test complete")
print("Run: tensorboard --logdir=logs/tensorboard_test")
```

**Run**:
```bash
python test_tensorboard.py
tensorboard --logdir=logs/tensorboard_test
```

**Validation**:
- Open http://localhost:6006
- Verify scalars plot correctly
- Check image appears
- Verify histogram shows distribution

---

### Test 4: Data Caching

**Purpose**: Verify caching speeds up data loading

```python
# Create test script: test_caching.py
from config import get_default_config
from utils import discover_dataset, create_patient_level_split
from dataset import create_data_loaders
import time

config = get_default_config()

# Discover data
data_list = discover_dataset(config.data.preprocessed_root, config.data)
train, val, test = create_patient_level_split(data_list)

# Test 1: No cache
print("Testing without cache...")
start = time.time()
loaders = create_data_loaders(config, train[:5], val[:2], test[:2])
for batch in loaders[0]:
    pass  # Iterate once
no_cache_time = time.time() - start
print(f"No cache: {no_cache_time:.2f}s")

# Test 2: Persistent cache (first run)
print("\nTesting persistent cache (first run)...")
start = time.time()
loaders = create_data_loaders(
    config, train[:5], val[:2], test[:2],
    use_persistent_cache=True
)
for batch in loaders[0]:
    pass
cache_first_time = time.time() - start
print(f"Persistent cache (first run): {cache_first_time:.2f}s")

# Test 3: Persistent cache (second run)
print("\nTesting persistent cache (second run)...")
start = time.time()
loaders = create_data_loaders(
    config, train[:5], val[:2], test[:2],
    use_persistent_cache=True
)
for batch in loaders[0]:
    pass
cache_second_time = time.time() - start
print(f"Persistent cache (second run): {cache_second_time:.2f}s")

print(f"\n✓ Speedup: {cache_first_time/cache_second_time:.1f}x")
```

**Run**:
```bash
python test_caching.py
```

**Expected output**:
```
No cache: 15.3s
Persistent cache (first run): 16.8s
Persistent cache (second run): 0.9s
✓ Speedup: 18.7x
```

**Validation**:
- Second run should be much faster (10-50x)
- Cache files appear in `cache/persistent/`

---

### Test 5: Enhanced Augmentation

**Purpose**: Verify new augmentations work

```python
# Create test script: test_augmentation.py
from config import get_default_config
from dataset import build_augmentation_transforms
import torch

config = get_default_config()

# Enable new augmentations
config.augmentation.gibbs_noise_prob = 1.0  # Always apply
config.augmentation.coarse_dropout_prob = 1.0

# Build transforms
transforms = build_augmentation_transforms(config.augmentation)

# Test on dummy data
dummy_data = {
    "image": torch.rand(1, 96, 96, 96),
}

print("Applying augmentation...")
augmented = transforms(dummy_data)

print(f"Original shape: {dummy_data['image'].shape}")
print(f"Augmented shape: {augmented['image'].shape}")
print(f"✓ Augmentation test passed")
```

**Run**:
```bash
python test_augmentation.py
```

**Validation**:
- Script runs without errors
- Shapes match
- If you visualize, you should see Gibbs ringing and dropout effects

---

### Test 6: Mixed Precision (GPU Required)

**Purpose**: Verify AMP works on GPU

```python
# Create test script: test_amp.py (requires GPU)
import torch
from torch.cuda.amp import autocast, GradScaler

if not torch.cuda.is_available():
    print("⚠️ GPU not available, skipping AMP test")
    exit()

device = torch.device("cuda")
model = torch.nn.Conv3d(1, 16, 3).to(device)
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

# Dummy batch
batch = torch.rand(2, 1, 32, 32, 32).to(device)
target = torch.rand(2, 16, 30, 30, 30).to(device)

print("Testing mixed precision training...")
for i in range(5):
    with autocast():
        output = model(batch)
        loss = ((output - target) ** 2).mean()
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    
    print(f"Step {i+1}: loss={loss.item():.4f}")

print("✓ AMP test passed")
```

**Run**:
```bash
python test_amp.py
```

**Validation**:
- Script completes without errors
- Loss decreases or stays stable
- No NaN values

---

## 🔍 Troubleshooting

### Issue: QC Script Fails

**Error**: `No preprocessed data found`

**Solution**:
```bash
# Preprocess data first
python example_pipeline.py --preprocess --config default
# Then run QC
python example_qc_harmonization.py
```

---

### Issue: TensorBoard Not Found

**Error**: `ModuleNotFoundError: No module named 'tensorboard'`

**Solution**:
```bash
pip install tensorboard
```

---

### Issue: Caching Not Faster

**Possible causes**:
1. First run (cache building) - Run twice to test
2. Cache directory not writable - Check permissions
3. SSD vs HDD - Caching works best with SSD

**Solution**:
```bash
# Clear cache and retry
rm -rf cache/persistent/*  # Or delete folder in Windows
python test_caching.py
```

---

### Issue: Harmonization Errors

**Error**: `ValueError: No reference scans found`

**Solution**:
- Ensure you have both ses-1 and ses-2 data preprocessed
- Check data discovery is finding scans correctly

---

### Issue: AMP Produces NaN

**Symptoms**: Loss becomes NaN during training

**Solution**:
1. Reduce learning rate
2. Use gradient clipping
3. Check for numerical stability in model
4. Disable AMP if problem persists: `config.training.use_amp = False`

---

## ✅ Validation Checklist

After testing, verify:

- [ ] All dependencies installed correctly
- [ ] QC reports generate successfully
- [ ] Harmonization outputs reasonable results
- [ ] TensorBoard logs appear and are viewable
- [ ] Caching provides speedup (second run faster)
- [ ] Enhanced augmentations run without errors
- [ ] (GPU) Mixed precision training works
- [ ] Existing code still works unchanged
- [ ] Documentation is clear and helpful

---

## 📊 Performance Benchmarks

Expected performance on typical hardware:

| Feature | Metric | Value |
|---------|--------|-------|
| QC | Time per scan | 0.5-2s |
| Harmonization | Time per scan | 1-3s |
| Persistent Cache | Speedup (2nd run) | 10-50x |
| Memory Cache | Speedup | 50-100x |
| AMP (GPU) | Training speedup | 2-3x |
| AMP (GPU) | Memory reduction | ~50% |

---

## 🎯 Next Steps

After successful testing:

1. **Review Documentation**
   - [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Feature guides
   - [IMPROVEMENTS_IMPLEMENTED.md](IMPROVEMENTS_IMPLEMENTED.md) - What changed
   - [CHANGELOG.md](CHANGELOG.md) - Version history

2. **Configure for Your Use Case**
   - Update `config.py` with your preferred settings
   - Enable features you need
   - Disable features you don't need

3. **Run Full Pipeline**
   ```bash
   # Complete workflow
   python example_pipeline.py --preprocess --config default
   python example_qc_harmonization.py
   # Then start training
   ```

4. **Integrate with Your Model**
   - Use data loaders in your training script
   - Add TensorBoard logging
   - Enable caching for performance
   - Use AMP if you have GPU

---

## 🆘 Getting Help

### Check Documentation
1. [README.md](README.md) - Main docs
2. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick guides
3. [IMPROVEMENTS_IMPLEMENTED.md](IMPROVEMENTS_IMPLEMENTED.md) - Feature details

### Run Examples
- `example_pipeline.py` - Basic workflow
- `example_qc_harmonization.py` - Advanced features
- `notebooks/interactive_pipeline.ipynb` - Interactive tutorial

### Common Commands
```bash
# Validate environment
python setup_validation.py

# Full pipeline
python example_pipeline.py --preprocess --config default

# QC and harmonization
python example_qc_harmonization.py

# View logs
tensorboard --logdir=logs/tensorboard
```

---

**Installation complete!** 🎉

You now have a production-ready brain MRI preprocessing pipeline with:
- ✅ Quality control
- ✅ Multi-scanner harmonization
- ✅ High-performance caching
- ✅ Professional monitoring
- ✅ Enhanced augmentation
- ✅ Mixed precision support

**Start processing your data!** 🚀
