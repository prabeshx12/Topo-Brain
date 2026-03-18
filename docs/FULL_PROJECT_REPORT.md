# Topo-Brain: Full Project Report

**Topology-Preserving 3T-to-7T MRI Enhancement Using Conditional Diffusion Models**

> This document is a comprehensive account of every phase, decision, architectural change, bug fix, and engineering step taken throughout the development of Topo-Brain — from the initial catastrophic training failure through to the current state of topology-aware fine-tuning preparation.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [Phase 1: Catastrophic Training Failure & Emergency Fixes](#3-phase-1-catastrophic-training-failure--emergency-fixes)
4. [Phase 2: Training to 166k Iterations](#4-phase-2-training-to-166k-iterations)
5. [Phase 3: Checkpoint Evaluation Infrastructure](#5-phase-3-checkpoint-evaluation-infrastructure)
6. [Phase 4: Metric Consistency Investigation](#6-phase-4-metric-consistency-investigation)
7. [Phase 5: Full-Volume Inference Pipeline](#7-phase-5-full-volume-inference-pipeline)
8. [Phase 6: PSNR Investigation & Output Clamping](#8-phase-6-psnr-investigation--output-clamping)
9. [Phase 7: Visualization Fixes](#9-phase-7-visualization-fixes)
10. [Phase 8: Deterministic Checkpoint Evaluation](#10-phase-8-deterministic-checkpoint-evaluation)
11. [Phase 9: Topology Loss Integration with FreeSurfer aseg](#11-phase-9-topology-loss-integration-with-freesurfer-aseg)
12. [Current State & Metrics Summary](#12-current-state--metrics-summary)
13. [File-by-File Change Log](#13-file-by-file-change-log)
14. [Appendix: Configuration Reference](#14-appendix-configuration-reference)

---

## 1. Project Overview

### 1.1 Goal

Build a conditional diffusion model that enhances 3T MRI brain scans to 7T-quality while **preserving anatomical topology** (grey matter/white matter/CSF boundaries). The model should be clinically useful for early Alzheimer's disease detection where 7T scanners are unavailable.

### 1.2 Data

- **10 paired subjects** — each has a 3T scan (ses-1) and a 7T scan (ses-2)
- **Modalities**: T1-weighted (primary), T2-weighted (optional secondary channel)
- **Format**: BIDS-compliant NIfTI files, pre-registered (3T aligned to 7T space)
- **FreeSurfer outputs**: `aparc+aseg.nii` and `aseg.nii` available per subject in the Aligned folder
- **Storage**: CERNBox EOS filesystem (`/eos/user/p/ppokhrel/`)

### 1.3 Training Setup

- **Hardware**: CERN SWAN (Jupyter) with GPU
- **Framework**: PyTorch with custom DDPM implementation
- **Normalization**: Percentile-based diffusion normalization (P0.5/P99.5 → [-1, 1], background = -1)
- **Patch-based**: 64³ patches extracted from brain regions

---

## 2. Architecture Deep Dive

### 2.1 AnatomyGuidedUNet (`src/model.py`)

A dual-decoder 3D U-Net designed for multi-task learning:

```
Input: [noisy_target (1ch) | 3T_condition (1ch)] → concat → 2 channels
                           ↓
                    Initial Conv3d (2 → 32)
                           ↓
              ┌─── Encoder (Downsampling) ───┐
              │  32 → 64 → 128 → 256         │
              │  Each level:                   │
              │    ResnetBlock + TimeEmb       │
              │    Conv3d(stride=2) downsample │
              │    Skip connections saved      │
              └───────────────────────────────┘
                           ↓
              ┌─── Bottleneck ───┐
              │  ResnetBlock     │
              │  SelfAttention3D │  (Global Context Module)
              │  ResnetBlock     │
              └──────────────────┘
                    ↓           ↓
         ┌─── Denoising ───┐  ┌─── Segmentation ───┐
         │    Decoder       │  │    Decoder          │
         │  256→128→64→32   │  │  256→128→64→32      │
         │  + Skip Concat   │  │  + Skip Concat      │
         │  Conv1x1 → 1ch   │  │  Conv1x1 → 4 class  │
         │  (noise pred)    │  │  (tissue logits)     │
         └──────────────────┘  └─────────────────────┘
```

**Key Design Decisions:**

| Component | Choice | Reason |
|-----------|--------|--------|
| Normalization | GroupNorm (8 groups) | Stable at batch_size=2 (InstanceNorm was unstable) |
| Time embedding | Sinusoidal → MLP (dim→4dim→dim) | Standard DDPM practice, GELU activation |
| Attention | SelfAttention3D at bottleneck only | Full volume attention is too expensive at higher resolutions |
| Skip connections | Concatenation (not addition) | Preserves more spatial information for medical imaging |
| Features | [32, 64, 128, 256] | Balance between capacity and GPU memory for 64³ patches |
| Dual decoder | Shared encoder, separate decoising/seg decoders | Seg head provides anatomical supervision without interfering with denoising gradients |
| Dropout | 0.1 | Light regularization for 10-subject dataset |

**The `forward_full` method** is the actual forward pass (assigned to `forward`). It:
1. Concatenates noisy image + 3T condition
2. Runs shared encoder, saving skip connections to a list
3. Runs two **independent** decoder paths using the **same** skip connections (indexed, not popped)
4. Returns `{"prediction": noise_pred, "segmentation": seg_logits}`

### 2.2 GaussianDiffusion (`src/diffusion.py`)

Standard DDPM (Denoising Diffusion Probabilistic Model) with multi-task extensions:

- **Timesteps**: 200 (last 15 excluded → effective 185)
- **Beta schedule**: Cosine (more stable than linear for medical imaging)
- **Objective**: Predict noise (ε-prediction)
- **Loss type**: L1 (smoother gradients than L2 for medical images)

**Forward pass computes 4 loss terms:**

```
total_loss = loss_diff + λ_pixel * loss_pixel + λ_percep * loss_vgg + λ_topo * loss_topo
```

| Loss | What it does | Weight |
|------|-------------|--------|
| `loss_diff` | L1 between predicted and actual noise | 1.0 (always) |
| `loss_pixel` | L1 between reconstructed x₀ and ground truth | 1.0 |
| `loss_percep` | VGG-16 feature matching (2D slices through depth) | 0.3 |
| `loss_topo` | Multi-scale edge-aware segmentation loss | 0.2 |

### 2.3 Topology Loss (`src/topology_loss.py`)

**MultiScaleTopologyLoss** wrapping **EdgeAwareTopologyLoss** at 3 scales [1.0, 0.5, 0.25]:

```
EdgeAwareTopologyLoss:
  1. Weighted Cross-Entropy (class weights: BG=0.5, CSF=2.0, GM=1.5, WM=1.0)
  2. 3D Sobel edge detection on ground truth mask
  3. Edge-weighted CE (boundary voxels get 2x weight)
  4. Boundary Dice loss (predicted edges vs GT edges)
  
  final = weighted_CE + 0.5 * boundary_Dice

MultiScaleTopologyLoss:
  Computes EdgeAwareTopologyLoss at 3 resolutions:
  - Scale 1.0: Full resolution (weight = 1.0)
  - Scale 0.5: Half resolution (weight = 0.71)
  - Scale 0.25: Quarter resolution (weight = 0.5)
  Normalized by total weight sum
```

**Why multi-scale?** Fine-scale catches GM/WM boundary sharpness. Coarse-scale catches large structural features (ventricle shape, overall brain morphology). Together they ensure both local and global anatomy is preserved.

### 2.4 Perceptual Loss (`src/diffusion.py` → `PerceptualLoss`)

Uses frozen VGG-16 (first 16 layers) to compare deep features:
- 3D volumes are reshaped to 2D: `[B, 1, D, H, W] → [B*D, 3, H, W]` (1ch repeated to 3ch)
- Normalized to ImageNet stats before VGG
- MSE on feature maps, clamped to max=10.0 to prevent explosion

### 2.5 Dataset (`src/synthesis_dataset.py` → `PairedPatchDataset`)

Handles the 3T→7T paired data loading:

1. **Reads pairs CSV** with columns: `subject, modality, input_3t, target_7t, input_3t_t2, tissue_mask_path`
2. **Loads volumes** (already preprocessed to [-1, 1])
3. **Loads tissue mask**: `pair.get("tissue_mask_path") or pair.get("mask")` — falls back to binary brain mask `(input_3t > -0.95)` if no mask column exists
4. **Extracts 64³ patches** from brain regions (≥10% brain fraction required)
5. **Returns dict**: `{input, target, seg, subject, center, pair_idx}`

**Critical detail**: The `seg` tensor is what gets passed to topology loss. If `tissue_mask_path` is absent from CSV, `seg` is either binary (0/1) or all-zeros, making topology loss useless.

### 2.6 Preprocessing (`src/preprocessing.py`)

- **N4 bias field correction** (SimpleITK)
- **Skull stripping** (SynthStrip or HD-BET)
- **Registration** (3T → 7T space, already done externally)
- **Normalization**: `method: diffusion` — Percentile-based:
  - Compute P0.5 and P99.5 within brain mask
  - Clip to [P0.5, P99.5]
  - Scale to [0, 1] then shift to [-1, 1]
  - Background voxels set to -1.0

---

## 3. Phase 1: Catastrophic Training Failure & Emergency Fixes

### 3.1 The Problem

Initial training produced **catastrophically bad results**:
- **SSIM = -0.13** (negative! worse than random noise)
- **PSNR = 0.52 dB** (essentially zero signal quality)

The model was producing garbage — pure noise or constant-value outputs, not brain images.

### 3.2 Root Cause Analysis

A comprehensive code audit revealed **5 critical bugs** that were compounding:

#### Bug 1: Timestep Gating on Auxiliary Losses

```python
# BROKEN: Only computed pixel/percep/topo loss at low timesteps
t_gate = (t < self.num_timesteps // 4).float()  # Only t < 50
loss_pixel = t_gate * F.l1_loss(x_recon, x_start)
```

**Why it was wrong**: At high noise levels (t > 50), `x_recon` is very noisy, but the model still needs gradient signal to learn denoising at those timesteps. By gating off 75% of timesteps, the auxiliary losses provided almost no learning signal.

**Fix**: Removed timestep gating entirely. All losses apply at all timesteps:
```python
t_gate = torch.ones_like(t).float()  # ALL timesteps
```

#### Bug 2: Tanh Clamping in `predict_start_from_noise`

```python
# BROKEN: Hard tanh squash killed gradients
x_0 = torch.tanh(sqrt_recip * x_t - sqrt_recipm1 * noise)
```

**Why it was wrong**: `tanh` saturates at ±1, creating vanishing gradients for any prediction even slightly outside [-1, 1]. During early training, predictions are wildly off — tanh kills the gradient signal needed to correct them. This is the classic "dead neuron" problem but applied to the entire reconstruction path.

**Fix**: Soft clamp with margin for gradient flow:
```python
x_0 = torch.clamp(sqrt_recip * x_t - sqrt_recipm1 * noise, min=-2.0, max=2.0)
```
Allows gradients to flow for predictions in [-2, 2] while still preventing numerical explosion.

#### Bug 3: Loss Weight Imbalance

```python
# BROKEN: Perceptual loss overwhelmed everything
lambda_pixel: 0.05   # Almost zero
lambda_percep: 1.0   # Way too high
lambda_topo: 0.3     # Too high for early training
```

**Why it was wrong**: VGG perceptual loss at weight 1.0 dominated the gradient, but VGG features are designed for natural images, not brain MRI. The pixel loss at 0.05 was too weak to drive basic reconstruction. The topology loss at 0.3 was too high before the seg head was trained.

**Fix**: Rebalanced per blueprint requirements:
```yaml
lambda_pixel: 1.0    # PRIMARY loss for reconstruction
lambda_percep: 0.3   # Moderate texture guidance
lambda_topo: 0.2     # Gentle anatomy supervision
```

#### Bug 4: No Curriculum Learning (All Losses from Step 0)

All 4 losses were active from iteration 0, creating conflicting gradients before the model learned basic denoising.

**Fix**: 4-stage curriculum:
```
Stage 1 (0-10k):     Diffusion + 50% pixel only (warm-up)
Stage 2 (10k-50k):   Diffusion + full pixel loss
Stage 3 (50k-100k):  Add perceptual loss (with 10k warmup)
Stage 4 (100k+):     Add topology loss (with 25k warmup)
```

#### Bug 5: Cosine Schedule Numerical Instability

The cosine beta schedule at `T=200` produces extreme coefficient values at the final timesteps:

```
t=195: sqrt_recip_alphas_cumprod = 1247.5
t=199: sqrt_recip_alphas_cumprod = 4713.9
```

When the model samples these timesteps, `predict_start_from_noise` multiplies by 4000+, causing loss spikes of 10⁵.

**Fix**: Exclude last 15 timesteps from training:
```python
max_safe_timestep = len(self.betas) - 15  # Use t ∈ [0, 185) only
t = torch.randint(0, max_safe_timestep, (b,), device=...).long()
```

Also added coefficient clamping as defense-in-depth:
```python
sqrt_recip = torch.clamp(sqrt_recip, max=10.0)
sqrt_recipm1 = torch.clamp(sqrt_recipm1, max=10.0)
```

And loss spike detection:
```python
if loss_pixel > 10.0:
    loss_pixel = torch.clamp(loss_pixel, max=5.0)
```

### 3.3 Additional Stabilization

- **Added gradient clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
- **Added noise prediction clamping**: `noise_pred = torch.clamp(noise_pred, -5.0, 5.0)` (noise is N(0,1), so ±5σ is safe)
- **GroupNorm instead of InstanceNorm**: More stable at batch_size=2
- **SiLU activation** throughout (instead of ReLU): Smoother gradients

---

## 4. Phase 2: Training to 166k Iterations

After applying all Phase 1 fixes, the model was trained from scratch to 166,000 iterations:

### Training Progress

| Iteration | SSIM | PSNR (dB) | Stage | Notes |
|-----------|------|-----------|-------|-------|
| 0 | -0.13 | 0.52 | 1 | Before fixes |
| 10,000 | ~0.15 | ~8 | 1→2 | Basic structure emerging |
| 50,000 | ~0.30 | ~12 | 2→3 | Reasonable brain shapes |
| 100,000 | ~0.45 | ~15 | 3→4 | Good quality, percep active |
| 136,000 | 0.44 | 15.68 | 4 | Checkpoint evaluation metric |
| 141,000 | **0.5142** | **17.60** | 4 | **Best checkpoint** |
| 166,000 | ~0.48 | ~16 | 4 | Slight decline (overfitting?) |

### What Each Stage Did

- **Stage 1 (0→10k)**: Model learned basic denoising — outputting something vaguely brain-shaped instead of noise
- **Stage 2 (10k→50k)**: Pixel loss drove voxel-level accuracy, brain structures became recognizable
- **Stage 3 (50k→100k)**: Perceptual loss sharpened textures and added anatomical detail
- **Stage 4 (100k→166k)**: Topology loss was active BUT was supervising against **zero/binary masks** (the CSV had no `tissue_mask_path` column) — effectively providing no useful anatomical signal

### Key Realization

The topology loss during Stage 4 was **training against zeros**. The `seg` tensor from the dataset was either all-zeros or a simple binary brain mask (0=background, 1=brain), not the intended 4-class tissue segmentation (BG/CSF/GM/WM). This meant the topology loss was:
- Not providing any real boundary supervision
- The segmentation head learned to predict "everything is background" 
- **66,000 iterations of wasted/harmful topology signal** (100k→166k)

This is why we later decided to resume from checkpoint 100k instead of 141k for fine-tuning with real masks.

---

## 5. Phase 3: Checkpoint Evaluation Infrastructure

### 5.1 Problem

167 checkpoints (saved every 1,000 steps) needed evaluation to find the best one. Manual evaluation was impractical.

### 5.2 Script: `scripts/evaluate_all_checkpoints.py`

Created an automated evaluator that:
1. Scans checkpoint directory for all `checkpoint_*.pt` files
2. For each checkpoint: loads model → extracts center 64³ crop → runs inference → computes SSIM + PSNR
3. Saves results to CSV
4. Reports best checkpoint

**Development process**: This script went through ~12 iterations of bug fixes:
- Device mismatch errors (model on GPU, data on CPU)
- State dict key mismatches (missing `seg_outc` keys when num_classes changed)
- Memory leaks from not clearing GPU cache between checkpoints
- Normalization inconsistencies (see Phase 4)

### 5.3 Results

Best checkpoint by center-crop SSIM: **iteration 141,000** (SSIM=0.5142, PSNR=17.60 dB)

---

## 6. Phase 4: Metric Consistency Investigation

### 6.1 Problem

Different evaluation scripts gave **wildly different** SSIM/PSNR numbers for the same checkpoint:
- `evaluate_all_checkpoints.py`: SSIM ≈ 0.51
- `sample_diffusion.py`: SSIM ≈ 0.44
- Manual calculation: SSIM ≈ 0.38

### 6.2 Root Causes

1. **Different patch locations**: `evaluate_all_checkpoints.py` used random patches, `sample_diffusion.py` used center 64³ crop. Random patches could sample from easier brain regions.

2. **Different normalization before metric computation**: Some scripts normalized [-1,1] → [0,1] before SSIM, others didn't. SSIM is sensitive to the data range — computing on [-1,1] vs [0,1] gives different numbers.

3. **No output clamping**: The diffusion model could output values outside [-1,1], distorting metrics.

### 6.3 Fix

Standardized across ALL evaluation scripts:
```python
# 1. Clamp output to valid range
output = torch.clamp(output, -1.0, 1.0)

# 2. Normalize to [0, 1] for metrics
output_01 = (output + 1) / 2
target_01 = (target + 1) / 2

# 3. Compute metrics with data_range=1.0
ssim = structural_similarity(output_01, target_01, data_range=1.0)
psnr = peak_signal_noise_ratio(output_01, target_01, data_range=1.0)
```

Applied to: `evaluate_all_checkpoints.py`, `sample_diffusion.py`, `evaluate_full_volume.py`, `find_best_checkpoint.py`

---

## 7. Phase 5: Full-Volume Inference Pipeline

### 7.1 Problem

All evaluation so far was on isolated 64³ patches. We needed **full-volume inference** to assess clinical quality, but the model only processes 64³ patches.

### 7.2 Script: `scripts/evaluate_full_volume.py`

Implemented **tiled patch inference** with blending:

```
Full Volume (e.g., 256×256×256)
    │
    ├── Split into overlapping 64³ patches (stride = 32, overlap = 50%)
    │
    ├── Run each patch through the diffusion model
    │
    ├── Blend overlapping regions using Tukey window weights
    │   (smooth tapering at edges, full weight in center)
    │
    └── Reassemble into full volume
```

### 7.3 Issues Encountered & Fixes

#### Issue 1: Grid Artifacts (Overlap = 16)

With only 16-voxel overlap (25%), visible grid lines appeared at patch boundaries because the blending window was too narrow.

**Fix**: Increased overlap to 32 (50% of patch size).

#### Issue 2: Hanning Window Artifacts

The Hanning window drops to zero at edges, creating visible seams.

**Fix**: Switched to **Tukey window** (α=0.5) — flat top with smooth edges:
```python
def _tukey_window_1d(size, alpha=0.5):
    # Flat center region (weight=1.0) with cosine taper at edges
```

#### Issue 3: Background Noise Tanking Metrics

Full-volume SSIM was **0.10** — terrible! But brain-only SSIM was 0.85. The background (air/skull) contributed random noise that destroyed the aggregate metric.

**Fix**: Added brain masking for metrics computation:
```python
# Compute metrics at 3 levels:
# 1. Full masked  — brain voxels only from full volume (SSIM ~0.85)
# 2. Brain bbox   — tight crop around brain (SSIM ~0.68)
# 3. Center 64³   — same as patch evaluation (SSIM ~0.52)
```

#### Issue 4: Inconsistent Visualization Scaling

Each panel (input/target/output) was auto-scaled independently, making comparison meaningless.

**Fix**: Shared `vmin/vmax` across all display panels:
```python
vmin = min(input_slice.min(), target_slice.min(), output_slice.min())
vmax = max(input_slice.max(), target_slice.max(), output_slice.max())
```

### 7.4 Final Full-Volume Results (Checkpoint 141k)

| Metric Region | SSIM | PSNR (dB) |
|---------------|------|-----------|
| Full masked | 0.85 | ~16 |
| Brain bounding box | 0.68 | ~15 |
| Center 64³ crop | 0.52 | ~17 |

---

## 8. Phase 6: PSNR Investigation & Output Clamping

### 8.1 Problem

PSNR at 14-17 dB seemed low. Was it a bug or genuine model performance?

### 8.2 Investigation

Added pre/post-inference intensity diagnostics:
```python
print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
print(f"Target range: [{target.min():.3f}, {target.max():.3f}]")
print(f"Values outside [-1,1]: {(output.abs() > 1.0).sum()} voxels")
```

Found: ~0.1% of output voxels were outside [-1, 1], e.g., values of 1.05 or -1.12. These out-of-range values disproportionately hurt PSNR.

### 8.3 Fix

Added output clamping to the diffusion sampling loop (`src/diffusion.py`, `p_sample_loop`):
```python
# Before return, clamp to data range
img = torch.clamp(img, -1.0, 1.0)
```

**Result**: Only ~0.5 dB improvement. PSNR of 14-17 dB is **genuine** performance for this dataset size (10 subjects) and model capacity. For context, many 3T→7T synthesis papers report 20-25 dB, but with larger datasets (50-200 subjects).

### 8.4 Why PSNR is Low but SSIM is Decent

- **PSNR** is L2-based (mean squared error). A few high-error voxels (e.g., at tissue boundaries) tank the entire metric.
- **SSIM** is structural — it measures luminance, contrast, and structure locally. Brain structures are well-preserved even if absolute intensity accuracy varies.
- With 10 subjects, the model hasn't seen enough anatomical variation to achieve pixel-perfect 7T reconstruction.

---

## 9. Phase 7: Visualization Fixes

### 9.1 Problem

Generated comparison images (input/target/output) looked misleadingly different because matplotlib auto-scaled each panel independently.

### 9.2 Fix

All visualization code updated to use shared display range:
```python
# Compute shared range across all images
all_slices = [input_slice, target_slice, output_slice]
vmin = min(s.min() for s in all_slices)
vmax = max(s.max() for s in all_slices)

# Plot with consistent scaling
ax.imshow(slice, cmap='gray', vmin=vmin, vmax=vmax)
```

This ensures brightness differences in the visualization reflect actual intensity differences, not display artifacts.

---

## 10. Phase 8: Deterministic Checkpoint Evaluation

### 10.1 Problem

`evaluate_all_checkpoints.py` used **random** patch sampling, making results non-reproducible. Running the same script twice could give different "best" checkpoints.

### 10.2 Script: `scripts/find_best_checkpoint.py`

Created a deterministic evaluator using **8 fixed anatomical locations** per subject:

```
1. Center of volume
2. Superior (top of brain)
3. Inferior (bottom of brain)
4. Anterior (front)
5. Posterior (back)
6. Left hemisphere
7. Right hemisphere
8. Brain centroid (center of mass of brain mask)
```

**Why these 8 locations?** They cover diverse anatomical regions — cortical surface, deep structures, ventricles, cerebellum. Each presents different challenges (boundaries, textures, fluid-filled spaces). Using fixed locations ensures:
- Perfect reproducibility across runs
- Fair comparison between checkpoints
- Coverage of clinically important regions

### 10.3 Results

Confirmed checkpoint 141,000 as best with deterministic evaluation:
- Mean SSIM across all patches/subjects: 0.5142
- Mean PSNR: 17.60 dB

---

## 11. Phase 9: Topology Loss Integration with FreeSurfer aseg

### 11.1 The Critical Discovery

The topology loss had been **training against zeros/binary masks** for all 66k iterations of Stage 4 (100k→166k). The CSV had no `tissue_mask_path` column, so `synthesis_dataset.py` fell back to:
```python
mask = (input_3t > -0.95).astype(np.uint8)  # Binary: 0=background, 1=brain
```

This binary mask was passed as `seg` — but the topology loss expected 4 classes (0=BG, 1=CSF, 2=GM, 3=WM). The segmentation head learned to predict class 0 everywhere because the "target" was mostly zeros.

### 11.2 FreeSurfer aseg Files

FreeSurfer's `aseg` (automatic segmentation) provides voxel-level anatomical labels:
- `aseg.nii`: Basic subcortical segmentation (~40 labels)
- `aparc+aseg.nii`: Full cortical + subcortical parcellation (~100+ labels)

**We chose `aparc+aseg.nii`** because it includes cortical parcellation labels (1000-2999) that map to grey matter, giving full cortical coverage. Plain `aseg.nii` only labels the cortical ribbon as "cortex" (labels 3/42) without the detailed parcellation.

### 11.3 Label Mapping (`scripts/preprocess_masks.py`)

FreeSurfer uses ~100+ label IDs. We map to 4 classes:

```python
FS_MAPPING = {
    0: 0,   # Background → BG
    
    # CSF (Class 1) — Ventricles and CSF spaces
    4: 1,   # Left-Lateral-Ventricle
    14: 1,  # 3rd-Ventricle
    15: 1,  # 4th-Ventricle
    43: 1,  # Right-Lateral-Ventricle
    44: 1,  # Right-Inf-Lat-Ventricle
    72: 1,  # 5th-Ventricle
    
    # White Matter (Class 3)
    2: 3,   # Left-Cerebral-White-Matter
    41: 3,  # Right-Cerebral-White-Matter
    
    # Grey Matter (Class 2) — Cortex + subcortical
    3: 2,   # Left-Cerebral-Cortex
    42: 2,  # Right-Cerebral-Cortex
    10: 2,  # Left-Thalamus
    11: 2,  # Left-Caudate
    12: 2,  # Left-Putamen
    13: 2,  # Left-Pallidum
    17: 2,  # Left-Hippocampus
    18: 2,  # Left-Amygdala
    # ... (right hemisphere mirrors)
    16: 2,  # Brainstem
}

# CRITICAL: aparc+aseg has cortical parcellation labels 1000-2999
# These ALL map to GM (Class 2)
out[(data >= 1000) & (data < 3000)] = 2
```

**Expected class distribution** (healthy brain):
- BG: ~40-55% (depends on FOV)
- CSF: ~3-8%
- GM: ~20-30% (with aparc labels providing full cortical coverage)
- WM: ~15-25%

### 11.4 Script Rewrite: `preprocess_masks.py`

The original script searched for asegs via `--data-root` rglob, which didn't match the actual file layout. The aseg files are in a **separate** `Aligned` folder tree:

```
/eos/user/p/ppokhrel/Untitled Folder 1/Aligned/
  ├── sub-01/ses-2/anat/aparc+aseg.nii   ← aseg files here
  ├── sub-02/ses-2/anat/aparc+aseg.nii
  └── ...

/eos/home-i04/p/ppokhrel/Untitled Folder 1/tissue_masks/
  ├── sub-01/ses-2/anat/sub-01_ses-2_desc-preproc_T1w.nii.gz  ← preprocessed data here
  └── ...
```

**Rewritten script** with:
- `--aligned-root`: Points to the Aligned folder
- `--data-root`: Resolves relative CSV paths for preprocessed data
- `--output-dir`: Where to save mapped masks
- `--output-csv`: Writes updated CSV with `tissue_mask_path` column
- `--dry-run`: Finds asegs without processing

**Path derivation logic:**
```python
# CSV has: target_7t = "sub-01/ses-2/anat/sub-01_ses-2_desc-preproc_T1w.nii.gz"
# Extract parent dir: "sub-01/ses-2/anat"
# Construct aseg path: aligned_root / "sub-01/ses-2/anat" / "aparc+aseg.nii"
```

**Processing per subject:**
1. Find `aparc+aseg.nii` (preferred) or `aseg.nii` in the derived path
2. Load aseg and target_7t NIfTI files
3. Check if resampling needed (different geometry/affine)
4. Resample aseg → target geometry using nearest-neighbor interpolation
5. Map FreeSurfer labels → 4 classes via `FS_MAPPING` + aparc range handling
6. Save as `*_tissue_seg.nii.gz` (uint8)
7. Update CSV row with absolute path to saved mask

### 11.5 Why Resume from 100k, Not 141k

| Factor | Checkpoint 100k | Checkpoint 141k |
|--------|-----------------|-----------------|
| Denoising head | Good (SSIM ~0.47) | Best (SSIM ~0.51) |
| Segmentation head | **Clean** — topology loss never applied | 41k iterations trained on all-zeros |
| Topology warmup | Will ramp 100k→125k naturally | Already past warmup window |
| Risk | Low — fresh start for seg head | Must unlearn "predict all background" |

The 141k checkpoint's segmentation head spent 41,000 iterations learning to predict class 0 (background) everywhere. With real masks, it would need to unlearn this — and because the topology warmup is already past at 141k, the full λ_topo=0.2 would hit the corrupted seg head instantly, risking instability.

At 100k, the topology loss hasn't been applied yet. The 25k warmup (100k→125k) provides a gentle ramp for the seg head to learn real tissue classification from scratch.

### 11.6 Pipeline Flow After Fix

```
aparc+aseg.nii (FreeSurfer)
       ↓
preprocess_masks.py (resample + map → 4 classes)
       ↓
*_tissue_seg.nii.gz → tissue_mask_path column in CSV
       ↓
synthesis_dataset.py loads via pair.get("tissue_mask_path")
       ↓
Returned as 'seg' tensor in __getitem__
       ↓
train_diffusion.py: seg_target = batch['seg']
       ↓
diffusion.forward(... seg_target, lambda_topo=0.2)
       ↓
_compute_topology_loss(seg_pred, seg_target)
       ↓
MultiScaleTopologyLoss(seg_logits, 4-class mask)
       ↓
Real gradient signal → seg head learns tissue boundaries
```

---

## 12. Current State & Metrics Summary

### 12.1 Best Results So Far (Checkpoint 141k, without topology)

| Metric | Center 64³ | Brain BBox | Full Masked |
|--------|-----------|------------|-------------|
| SSIM | 0.52 | 0.68 | 0.85 |
| PSNR | 17.60 dB | ~15 dB | ~16 dB |

### 12.2 What's Next

1. **Run `preprocess_masks.py`** on CERNBox to create 4-class tissue masks from `aparc+aseg.nii`
2. **Resume training from checkpoint 100k** with the updated CSV containing `tissue_mask_path`
3. **Monitor `loss_topo`** — should be non-zero and decreasing after 125k
4. **Evaluate new checkpoints** with `find_best_checkpoint.py` after sufficient training
5. **Full-volume evaluation** to compare boundary quality with/without topology supervision

### 12.3 Expected Improvements from Topology Integration

- **Sharper GM/WM boundaries** — the edge-aware loss specifically upweights boundary voxels
- **Better ventricular definition** — CSF class weight 2.0 prioritizes these structures
- **Global consistency** — multi-scale loss ensures large structures (ventricles, cortical ribbon) maintain correct shape
- **Potentially higher SSIM in brain regions** — topology regularization prevents the model from "smearing" boundaries to minimize pixel loss

---

## 13. File-by-File Change Log

### Modified Files

| File | Changes | Why |
|------|---------|-----|
| `src/model.py` | Dual decoder (`forward_full`), GroupNorm, SiLU, SinusoidalPosEmb, SelfAttention3D, skip concat | Architecture upgrade for multi-task learning + training stability |
| `src/diffusion.py` | Removed timestep gating, soft clamp instead of tanh, cosine schedule fix (exclude last 15), noise pred clamping, loss spike detection, output clamp in `p_sample_loop`, PerceptualLoss with max cap | Training collapse fixes + stability |
| `src/topology_loss.py` | New file: EdgeAwareTopologyLoss + MultiScaleTopologyLoss | Anatomy-preserving supervision |
| `src/synthesis_dataset.py` | `tissue_mask_path` loading, fallback to binary mask, seg tensor return | Dataset support for topology supervision |
| `configs/train_diffusion.yaml` | 4-stage curriculum, rebalanced loss weights, 200 timesteps, cosine schedule, pairs_csv updated to v2 | Training stability + topology integration |
| `scripts/train_diffusion.py` | Curriculum stage logic, seg_target routing, loss weight warmup, robust checkpoint loading, WandB support | Training infrastructure |
| `scripts/preprocess_masks.py` | Complete rewrite: `--aligned-root`, find_aseg(), FreeSurfer mapping, resample, CSV update | Real tissue mask generation |

### New Files Created

| File | Purpose |
|------|---------|
| `scripts/evaluate_all_checkpoints.py` | Batch checkpoint evaluation (random patches) |
| `scripts/evaluate_full_volume.py` | Full-volume tiled inference with Tukey blending |
| `scripts/find_best_checkpoint.py` | Deterministic 8-location checkpoint evaluation |
| `scripts/download_top_checkpoints.py` | Zip top-N checkpoints from evaluation CSV |
| `scripts/download_checkpoint.py` | Zip specific checkpoint(s) by iteration number |
| `src/topology_loss.py` | Multi-scale edge-aware topology loss module |

---

## 14. Appendix: Configuration Reference

### Training Config (`configs/train_diffusion.yaml`)

```yaml
dataset:
  pairs_csv: "...pairs_with_tissue_masks_v2.csv"  # Updated with tissue_mask_path
  batch_size: 2
  patch_size: [64, 64, 64]
  num_workers: 4
  use_t2: false

model:
  in_channels: 1       # Target (noisy 7T)
  cond_channels: 1     # Condition (3T)
  out_channels: 1      # Predicted noise
  num_classes: 4       # BG/CSF/GM/WM
  features: [32, 64, 128, 256]
  dropout: 0.1
  use_attention: true

diffusion:
  timesteps: 200       # Last 15 excluded automatically
  beta_schedule: cosine
  loss_type: l1

training:
  lr: 1e-4
  weight_decay: 1e-4
  grad_clip: 1.0
  n_iters: 200000
  save_freq: 1000
  ema_decay: 0.9999
  ema_start: 2000

stages:
  stage1_end: 10000    # Diffusion + 50% pixel
  stage2_end: 50000    # Full pixel
  stage3_end: 100000   # + Perceptual
  # Stage 4: + Topology (warmup 25k)

loss_weights:
  lambda_pixel: 1.0
  lambda_percep: 0.3
  lambda_topo: 0.2
  topo_warmup_steps: 25000
  percep_warmup_steps: 10000
```

### Critical Parameters Explained

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `timesteps: 200` | Not 1000 | Faster training/inference, sufficient for medical imaging (less diversity than natural images) |
| `beta_schedule: cosine` | Not linear | More uniform noise levels across timesteps, better for medical images with subtle contrast |
| `loss_type: l1` | Not l2 | L1 is less sensitive to outliers, produces sharper outputs |
| `batch_size: 2` | Not larger | GPU memory constraint for 64³ 3D patches |
| `ema_decay: 0.9999` | Standard | Exponential moving average of model weights for smoother inference |
| `features: [32,64,128,256]` | 4 levels | Balance between model capacity and memory. Each level halves spatial dims. |
| `dropout: 0.1` | Light | Prevents overfitting on 10-subject dataset without hurting capacity |
| `grad_clip: 1.0` | Standard | Prevents gradient explosion from diffusion instabilities |
| `lambda_topo: 0.2` | Conservative | Topology loss should guide, not dominate — primary goal is image quality |
| `topo_warmup_steps: 25000` | Gradual | Seg head needs time to learn before full-strength supervision |

---

*Report generated: February 2026*
*Project: Topo-Brain (prabeshx12/Topo-Brain, branch: feat/branch-new-pipeline)*
