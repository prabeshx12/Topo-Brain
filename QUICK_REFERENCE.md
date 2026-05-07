# Topo-Brain: Quick Reference Guide

## TL;DR - Complete Pipeline in 5 Commands

```bash
# 1. PREPROCESS: Clean raw BIDS data
python scripts/preprocess_bids.py --data-root /path/to/data --config configs/preprocess.yaml

# 2. PREPARE: Create 3T-7T pairs
python scripts/regenerate_pairs.py --preproc-root derivatives/topobrain-preproc --output pairs.csv

# 3. TRAIN: Learn to synthesize 7T from 3T
python scripts/train_diffusion.py --config configs/train_diffusion.yaml --output models/

# 4. GENERATE: Create synthetic 7T images
python scripts/sample_diffusion.py --checkpoint models/best_model.pt \
  --input-3t derivatives/topobrain-preproc/sub-XX_ses-1_T1w_desc-preproc.nii.gz \
  --output derivatives/inference/sub-XX_synthetic_7t.nii.gz

# 5. VISUALIZE: Compare results
python scripts/visualize_dataset.py --input-3t <3t_path> --input-7t-synthetic <synth_path> --input-7t-real <real_path>
```

---

## Project at a Glance

```
RAW BIDS DATA
     ↓
[PREPROCESSING: Skull strip, normalize, resample]
     ↓
PREPROCESSED VOLUMES (normalized, [-1, 1])
     ↓
[PAIRING: Match 3T-7T by subject]
     ↓
pairs.csv (20 paired volumes)
     ↓
[DATASET LOADING: Create 64³ patches, augment]
     ↓
TRAINING SET (12 volumes / 200+ patches)
VALIDATION SET (4 volumes / 70+ patches)
     ↓
[TRAINING: 400K iterations, diffusion model]
     ↓
TRAINED MODEL (22M parameters, EMA checkpoint)
     ↓
[INFERENCE: Sliding window, 50 denoising steps]
     ↓
SYNTHETIC 7T IMAGE (same resolution as input 3T)
     ↓
[EVALUATION: Compare to real 7T with PSNR/SSIM]
     ↓
CLINICAL APPLICATIONS
```

---

## Key Files to Know

| File | Purpose | When to modify |
|------|---------|---|
| `configs/train_diffusion.yaml` | Training hyperparameters | Tuning performance |
| `src/model.py` | U-Net architecture | Changing model size |
| `src/diffusion.py` | Loss functions & sampling | Adding topology loss |
| `scripts/train_diffusion.py` | Main training loop | Debugging training |
| `pairs.csv` | 3T-7T pairing manifest | Fixing data paths |
| `models/best_model.pt` | Trained checkpoint | For inference |

---

## Important Default Values

```python
# Architecture
Channels: [32, 64, 128, 256]
Patch Size: 64³
Total Parameters: ~22M

# Training
Learning Rate: 1e-4
Batch Size: 4 (patches)
Iterations: 400,000
Training Time: 1-4 GPU-days

# Diffusion
Timesteps: 1000
Schedule: cosine (smoother than linear)
Noise Adaptation: Standard Gaussian

# Losses (weighted composite)
Denoising Loss:      1.0 × L1(ε̂, ε)
Segmentation Loss:   0.1 × CrossEntropy(seg_pred, seg_gt)
Perceptual Loss:     0.01 × LPIPS (VGG features)

# Inference
Steps (DDPM):  50  (≈30 sec/volume)
Steps (DDIM):  10  (≈5 sec/volume)
Patch Overlap: 50% (smooth blending with Gaussian weights)
```

---

## Data Flow Diagram

```
BIDS RAW DATA (40 volumes: 10 subjects × 2 sessions)
    │
    ├─ 3T Session (ses-1): sub-01...sub-10
    └─ 7T Session (ses-2): sub-01...sub-10
    
        ↓ [PREPROCESSING]
        
NORMALIZED [-1, 1] (40 volumes)
    │
    ├─ 3T Volumes: {32, 64, 128, 256}³ pixels
    └─ 7T Volumes: {32, 64, 128, 256}³ pixels

        ↓ [PAIRING & SPLITTING]
        
pairs.csv (20 pairs)
    │
    ├─ TRAIN: 12 volumes (sub-01...sub-06)
    ├─ VAL:    4 volumes (sub-07...sub-08)
    └─ TEST:   4 volumes (sub-09...sub-10)
    
        ↓ [PATCH EXTRACTION]
        
TRAINING PATCHES: 32/volume × 12 volumes = 384 patches
    │
    ├─ Patch Pool: 64³, minimum 10% brain fraction
    └─ Augmentations: Rotation, flip, intensity shift
    
        ↓ [BATCH CREATION]
        
TRAINING BATCH: {x_3t, x_7t, mask, seg}
    Shape: {4, 1, 64, 64, 64} (4 patches per batch)
    
        ↓ [MODEL]
        
ANATOMY-GUIDED U-NET
    Input:  Concat(noisy_7t, clean_3t) [B, 2, 64³]
    Encode: 32 → 64 → 128 → 256 channels
    Decode: 256 → 128 → 64 → 32 channels (2 decoders)
    Output: noise_pred [B, 1, 64³] + seg_logits [B, 4, 64³]
    
        ↓ [TRAINING LOOP: 400,000 iterations]
        
SAVED CHECKPOINT
    └─ model_state, ema_model_state, optimizer_state
    └─ config, iteration number
    
        ↓ [INFERENCE]
        
INPUT: 3T Full Volume [256³ pixels]
    │
    ├─ Sliding Window (64³ patches, 50% overlap)
    ├─ Diffusion Sampling (50 steps DDPM or 10 steps DDIM)
    ├─ Patch-wise Gaussian Blending
    └─ Clamping [-1, 1] + inverse normalization
    
OUTPUT: Synthetic 7T [256³ pixels]
    │
    └─ PSNR: 28-32 dB, SSIM: 0.85-0.92 (vs real 7T)
```

---

## Essential Equations

### Forward Diffusion (Training)
$$x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1-\bar{\alpha}_t} \, \epsilon$$

Where:
- $x_0$ = original clean 7T image
- $x_t$ = noisy image at timestep $t$
- $\epsilon \sim \mathcal{N}(0, I)$ = random Gaussian noise
- $\bar{\alpha}_t$ = cumulative product of alphas (noise schedule)

### Reverse Diffusion (Inference)
$$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z$$

Where:
- $\mu_\theta$ = learned mean (from model)
- $\sigma_t$ = posterior variance
- $z \sim \mathcal{N}(0, I)$ = random noise

### Training Loss
$$L = L_{denoise} + \lambda_{seg} \, L_{seg} + \lambda_{percep} \, L_{percep}$$

### Noise Prediction (Model Output)
$$\hat{\epsilon} = \text{UNet}(x_t, x_3t, t, \text{encoder_features})$$

---

## Model Architecture Summary

```
╔════════════════════════════════════════════════════════════╗
║                   AnatomyGuidedUNet (22M params)           ║
╠════════════════════════════════════════════════════════════╣
║                                                             ║
║  INPUT: [x_7t_noisy (1ch) | x_3t_clean (1ch)]  [B, 2, 64³]║
║                     ↓                                       ║
║         Initial Conv3d: Ch 2 → 32                          ║
║                     ↓                                       ║
║  ┌─────────────────────────────────────────┐               ║
║  │ ENCODER (Downsampling)                   │               ║
║  │ ├─ Level 0: ResnetBlock + TimeEmb (32)  │               ║
║  │ ├─ Level 1: ResnetBlock + TimeEmb (64)  │               ║
║  │ ├─ Level 2: ResnetBlock + TimeEmb (128) │               ║
║  │ └─ Level 3: ResnetBlock + TimeEmb (256) │               ║
║  │   [Save skip connections at each level] │               ║
║  └─────────────────────────────────────────┘               ║
║                     ↓                                       ║
║  ┌─────────────────────────────────────────┐               ║
║  │ BOTTLENECK (Global Context)              │               ║
║  │ ├─ ResnetBlock + TimeEmb (256)          │               ║
║  │ ├─ SelfAttention3D (256)        [*]     │               ║
║  │ └─ ResnetBlock + TimeEmb (256)          │               ║
║  │    [*] Only at bottleneck (cost savings)│               ║
║  └─────────────────────────────────────────┘               ║
║                     ↓                                       ║
║              ┌──────────────┐                              ║
║              ↓              ↓                               ║
║    ┌──────────────────┐  ┌──────────────────┐             ║
║    │ DENOISING        │  │ SEGMENTATION     │             ║
║    │ DECODER          │  │ DECODER          │             ║
║    ├─ Level 3: 256   │  ├─ Level 3: 256   │             ║
║    ├─ Level 2: 128   │  ├─ Level 2: 128   │             ║
║    ├─ Level 1: 64    │  ├─ Level 1: 64    │             ║
║    ├─ Level 0: 32    │  ├─ Level 0: 32    │             ║
║    └─ Conv1x1: 1ch   │  └─ Conv1x1: 4-cls │             ║
║    (predicted noise) │  (tissue logits)    │             ║
║    ↓                 │  ↓                  │             ║
║    OUTPUT            │  OUTPUT             │             ║
║    [B, 1, 64³]       │  [B, 4, 64³]       │             ║
║    ε̂_t (noise)       │  logits (seg)      │             ║
║                      └──────────────────┘             ║
║                                                             ║
╚════════════════════════════════════════════════════════════╝
```

---

## Training Checklist

- [ ] Raw BIDS data organized (`data/sub-XX/ses-{1,2}/anat/*.nii.gz`)
- [ ] FreeSurfer brain masks available (or run HD-BET)
- [ ] Preprocessing completed (`derivatives/topobrain-preproc/`)
- [ ] `pairs.csv` generated (20 rows, correct file paths)
- [ ] GPU available with ≥8GB VRAM
- [ ] Config files edited with correct paths
- [ ] Test dry-run: `--dry-run` flag
- [ ] TensorBoard running for monitoring
- [ ] Checkpoints saving to `models/`
- [ ] Training loss decreasing within first 1000 iterations

---

## Inference Checklist

- [ ] Best checkpoint identified (lowest val loss)
- [ ] Input 3T volume in correct format (normalized [-1, 1])
- [ ] Output directory writable
- [ ] GPU available (or use CPU with `--device cpu`)
- [ ] Inference script: `sample_diffusion.py`
- [ ] Generated file saved and verified (≠ NaN, shape correct)
- [ ] Post-processing applied (optional)
- [ ] Visualization comparison created

---

## Performance Optimization Tips

### Training Speed

```python
# ✅ Enable mixed precision (2-3x speedup)
use_amp: true

# ✅ Use DDPand persistent workers
persistent_workers: true
num_workers: 4

# ✅ Gradient accumulation (if want larger effective batch)
accumulate_grad_batches: 2

# ⚠️ Avoid expensive operations
use_attention: false  # False by default (only bottleneck)
lambda_percep: 0.001  # Keep perceptual loss minimal
```

### Inference Speed

```python
# ✅ Use DDIM (10-20 steps) if speed critical
num_inference_steps: 10  # vs 50 for DDPM

# ✅ Process at original resolution (no padding)
skip_padding: true

# ⚠️ Sliding window efficiency
patch_size: 64
overlap: 0.5  # Balance between quality and speed
```

---

## Debugging Matrix

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| CUDA OOM | Batch/patch too large | Reduce batch_size, patch_size |
| Loss = NaN | Data not normalized | Verify x ∈ [-1, 1] |
| Loss plateaus | LR too high | Reduce lr to 1e-5 |
| Poor image quality | Insufficient training | Increase n_iters to 500k+ |
| Artifacts in output | Model overfitting | Add regularization, increase training data |
| Slow inference | CPU instead of GPU | Check device selection |
| Memory leak | DataLoader issue | persistent_workers=False |

---

## Most Common Tasks

### 1. Change Model Size
```yaml
# In configs/train_diffusion.yaml
model:
  features: [16, 32, 64, 128]  # Smaller
  # or
  features: [64, 128, 256, 512]  # Larger
```

### 2. Adjust Training Duration
```yaml
training:
  n_iters: 200000  # Shorter: ~12 hours
  # or
  n_iters: 500000  # Longer: ~48 hours
```

### 3. Use Different Loss Weights
```yaml
training:
  lambda_seg: 0.05      # Less segmentation guidance
  lambda_percep: 0.001  # Minimal perceptual loss
```

### 4. Change Inference Quality/Speed
```bash
# Fast, good quality
python scripts/sample_diffusion.py --num-inference-steps 50

# Very fast, lower quality
python scripts/sample_diffusion.py --num-inference-steps 10

# Slow, best quality
python scripts/sample_diffusion.py --num-inference-steps 100
```

### 5. Resume Training from Checkpoint
```bash
python scripts/train_diffusion.py \
  --config configs/train_diffusion.yaml \
  --resume models/checkpoint_iter_100000.pt
```

---

## File Sizes & Storage

| Component | Typical Size | Notes |
|-----------|--------------|-------|
| Raw BIDS (10 subjects) | ~10 GB | 40 volumes, 256³ each |
| Preprocessed | ~5 GB | Normalized, compressed |
| Checkpoints (1) | ~100 MB | Model + optimizer state |
| Checkpoint history (500 saves) | ~50 GB | Keep only best + latest |
| TensorBoard logs | ~500 MB | Full training history |
| **Total (with history)** | ~65 GB | |

---

## Recommended Hardware

| Component | Minimum | Recommended | Ideal |
|-----------|---------|-------------|-------|
| **GPU** | 8 GB VRAM | 16 GB VRAM | 32+ GB (A40/A100) |
| **CPU** | 4 cores | 8 cores | 16+ cores |
| **RAM** | 16 GB | 32 GB | 64+ GB |
| **Storage** | 100 GB | 200 GB | 500+ GB SSD |
| **Training Time** | 4 GPU-days | 2 GPU-days | <1 GPU-day |

---

## Contact & Troubleshooting

See `COMPLETE_PROJECT_GUIDE.md` for:
- Detailed architecture explanations
- Phase-by-phase workflow
- Full configuration reference
- Advanced topics (multi-GPU, custom losses, etc.)
- Extended troubleshooting guide

See `docs/FULL_PROJECT_REPORT.md` for:
- Development history
- Phase-by-phase decisions
- Bug fixes and improvements
- Metric investigations

---

**Quick Reference v1.0**  
For full documentation, see COMPLETE_PROJECT_GUIDE.md
