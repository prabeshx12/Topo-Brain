# Topo-Brain: Complete Technical Training Report
# Everything From Data to Model — Every Decision Explained

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [The Dataset](#2-the-dataset)
3. [Preprocessing Pipeline](#3-preprocessing-pipeline)
4. [Segmentation Masks](#4-segmentation-masks)
5. [Model Architecture](#5-model-architecture)
6. [The Diffusion Process](#6-the-diffusion-process)
7. [Loss Functions](#7-loss-functions)
8. [Training Curriculum (4 Stages)](#8-training-curriculum-4-stages)
9. [Hyperparameters and Configuration](#9-hyperparameters-and-configuration)
10. [Numerical Stability Fixes](#10-numerical-stability-fixes)
11. [Problems Encountered and Fixes Applied](#11-problems-encountered-and-fixes-applied)
12. [Current State and What Comes Next](#12-current-state-and-what-comes-next)

---

## 1. Project Goal

**What are we trying to do?**

We are building a deep learning model that takes a 3T MRI brain scan as input and generates a synthetic 7T MRI brain scan as output.

**Why does this matter?**

- 7T (7 Tesla) MRI scanners produce much higher resolution, sharper images than standard 3T scanners
- 7T scanners cost $10–30 million and there are only ~100 worldwide vs ~20,000 3T scanners
- 7T images reveal brain structures important for early Alzheimer's detection — hippocampal subfields, cortical layer thinning, microbleeds — that are invisible on 3T
- If we can make a 3T scan look like a 7T scan using software, every hospital with a 3T scanner can benefit from 7T-level analysis without buying a 7T scanner

**The key novelty — Topology Preservation:**

Earlier work on this problem (GANs, basic CNNs) produced visually sharp images but often broke brain anatomy — creating disconnected white matter regions, hallucinating structures, or losing small features like the hippocampus. Our approach adds an explicit anatomical constraint during training using a "topology loss" that forces the model to preserve the correct boundaries between brain tissues (white matter, grey matter, CSF). This is what makes our project different from simply applying an existing image enhancement model.

**The downstream validation goal:**

Once the synthesis model is trained, we validate it clinically by showing that synthetic 7T images improve Alzheimer's Disease classification accuracy compared to raw 3T images — proving the synthesis is not just visually nice but diagnostically useful.

---

## 2. The Dataset

### 2.1 The UNC 3T/7T Paired Dataset

We use paired MRI data from the University of North Carolina (UNC). "Paired" means the same subject was scanned at both 3T and 7T, so we have a ground truth to train against.

- **Subjects**: 10 subjects (small dataset — this is a proof-of-concept)
- **Modalities**: T1-weighted (T1w) MRI. T2-weighted (T2w) is available but not used in the current main training run
- **Sessions**: Each subject has session 1 (3T scan) and session 2 (7T scan)
- **Format**: BIDS (Brain Imaging Data Structure) — the standard in neuroimaging. Files are named like `sub-01_ses-1_T1w.nii.gz`

### 2.2 The Pairs Manifest (pairs_new.csv)

A CSV file that maps each 3T scan to its corresponding 7T scan for each subject:

```
subject | input_3t (3T scan path) | target_7t (7T scan path) | input_3t_t2 | seg (segmentation mask)
```

This is the "recipe" the dataloader uses to know which files to load together during training.

### 2.3 Why Only 10 Subjects?

10 paired 3T/7T subjects is extremely small for deep learning. We handle this through:
- **Patch-based training**: Instead of training on whole brains (which would overfit immediately), we cut each brain into 64x64x64 voxel patches. With 32 patches per volume, 10 subjects gives ~320 patch pairs — still small but workable
- **Data augmentation**: Random flips, rotations during training
- **Leave-One-Out Cross-Validation (LOOCV)**: With 10 subjects we use LOOCV — in a full run, train on 9, validate on 1, and rotate through all subjects. In practice, the current training run uses `val_fold=0` and `test_fold=1` (two different subjects held out), meaning **8 subjects train, 1 validates, 1 tests** — running all 10 folds would require 10 separate training runs
- **EMA (Exponential Moving Average)**: Smooths weight updates to prevent overfitting to single batches

---

## 3. Preprocessing Pipeline

Before any training happens, every MRI scan goes through a standardized preprocessing pipeline. This is critical — raw scanner data has inconsistent orientations, resolutions, intensity scales, and artifacts.

### Step 1: N4 Bias Field Correction

MRI scanners produce a "bias field" — a smooth, low-frequency intensity variation across the image caused by radio-frequency field inhomogeneity. A voxel in the centre of the brain appears brighter than the same tissue at the edge just because of scanner physics, not actual tissue differences.

**How it's corrected**: N4ITK algorithm (from SimpleITK). It estimates and removes this smooth background distortion iteratively. After N4, white matter has consistent intensity across the whole brain.

**Why it matters**: Without N4, the model would see different intensity values for the same tissue depending on where in the brain it is — making tissue classification unreliable and making the synthesis task harder.

### Step 2: Skull Stripping (Brain Extraction)

The raw MRI includes the skull, scalp, and surrounding air. We only care about brain tissue.

**How**: Uses existing brain masks from the dataset (or ANTs/FSL brain extraction tools). The result is a brain-only volume where everything outside the brain is set to 0.

**Why it matters**: The skull has very different intensity from brain tissue. Including it would confuse the model and waste capacity on non-brain voxels.

### Step 3: Registration (Alignment)

For paired data, the 3T and 7T scans of the same subject were acquired on different days on different scanners. The brain is in a slightly different position and orientation in each scan. We register (align) them so every voxel in the 3T image corresponds to the same anatomical location in the 7T image.

**Method**: Rigid registration (translation + rotation only, no stretching). ANTs or FSL FLIRT.

**Why it matters**: If the images are misaligned, the model is trying to learn to map a voxel in one location of the 3T brain to a voxel in a completely different location of the 7T brain — the training signal becomes noise.

### Step 4: Intensity Normalization

After N4 correction, intensities are normalized to the range **[-1, 1]** using the "diffusion normalization" method:

```
1. Find the 99th percentile of brain voxel intensities (to exclude outliers)
2. Divide all values by this percentile to get [0, 1]
3. Scale to [-1, 1]: value = value * 2 - 1
4. Background (non-brain) voxels are set to -1.0
```

**Why [-1, 1] and not [0, 1] or Z-score?**
- Diffusion models are trained with Gaussian noise added to data. The noise is N(0,1) — centred at 0. If data is centred at 0 (i.e., [-1, 1] range), the signal-to-noise ratio at each timestep is well-calibrated
- At timestep t=0 (no noise), data is in [-1, 1]. At t=T (full noise), data is N(0,1). The transition is smooth
- This is standard practice in all DDPM implementations

### Step 5: Patch Extraction

Training on whole 64×64×64 volumes is impractical (GPU memory). Instead, we extract 64×64×64 cubic patches from each volume.

**Smart patch sampling**: Not all patches are useful. A patch from the corner of the skull strip (all background, no brain) teaches the model nothing. We use a minimum brain fraction threshold (10%) — a patch must contain at least 10% brain voxels to be used. This is enforced using the brain mask.

**32 patches per volume per epoch**: Each subject contributes 32 random valid patches per training iteration.

---

## 4. Segmentation Masks

The segmentation masks are used as the supervision signal for the Topology Loss. They tell the model which tissue type each voxel belongs to.

### 4.1 Source: FreeSurfer from the UNC Dataset

The UNC dataset comes with FreeSurfer outputs — FreeSurfer is the gold-standard neuroscience tool for brain segmentation. It was run on each subject's 7T scan and produces files called `aparc+aseg.nii.gz` or `aseg.nii.gz`.

FreeSurfer assigns hundreds of labels (e.g., label 17 = left hippocampus, labels 1000-3000 = cortical parcellations). We only need 4 coarse tissue classes for our topology loss.

### 4.2 Label Remapping (preprocess_masks.py)

The FreeSurfer labels are collapsed into 4 classes using a mapping table (FS_MAPPING):

| FreeSurfer Label IDs | Tissue | Our Class ID |
|---|---|---|
| 0 | Background (outside brain) | 0 |
| 4, 14, 15, 43, 44, 72... | CSF (ventricles, etc.) | 1 |
| 3, 42, 10-18, 49-58, 1000-3000 | Grey Matter (cortex + subcortical) | 2 |
| 2, 41 | White Matter | 3 |

### 4.3 Registration to 7T Space

After remapping, the segmentation mask is resampled to exactly match the 7T target image geometry using nearest-neighbour interpolation (nearest-neighbour is critical — bilinear would create non-integer label values, which are meaningless for class labels).

### 4.4 Why the Mask is on the 7T Image

We want the model to produce synthetic 7T images that have correct tissue anatomy as seen in real 7T scans. So the ground-truth segmentation used for supervision comes from the 7T scan — not the 3T. This is the correct design: we are telling the model "your output should have this tissue structure" using 7T-derived labels.

---

## 5. Model Architecture

The model is called **AnatomyGuidedUNet**. It is a 3D UNet enhanced with time embeddings, self-attention, and a dual-decoder for multi-task learning.

### 5.1 Overview

```
TRAINING:
  x_0       = real clean 7T patch          (ground truth — never fed to model directly)
  noise     = random Gaussian noise
  x_t       = q_sample(x_0, t, noise)      = noisy version of 7T at timestep t
  Model in  = [x_t, clean_3T]  → [B, 2, 64, 64, 64]
  Model out = [predicted noise ε, tissue segmentation logits]

INFERENCE (no real 7T exists):
  Model in  = [pure Gaussian noise, clean_3T]  → iteratively denoised → synthetic 7T
```

**Key point**: During training the model never sees a clean 7T — only noise-corrupted versions of it. The 7T is the target we corrupt and ask the model to reconstruct by predicting the noise. During inference there is no real 7T at all; we start from pure random noise and denoise guided by the 3T.

### 5.2 Component 1: Sinusoidal Time Embedding

The diffusion model adds different amounts of noise at different timesteps (t=0 to t=185). The model must know *how much noise* was added in order to predict the correct amount to remove.

The timestep `t` (an integer) is converted to a continuous embedding vector using sinusoidal positional encoding — the same idea used in Transformers to encode word positions.

```
For each dimension d in [0, dim/2]:
    embedding[2d]   = sin(t / 10000^(2d/dim))
    embedding[2d+1] = cos(t / 10000^(2d/dim))
```

This creates a unique fingerprint for each timestep. The embedding is then passed through two linear layers (MLP) to project it to the right size. This time embedding is **injected into every ResNet block** in the model so every layer knows what timestep it's operating at.

### 5.3 Component 2: Conditioning via Concatenation

The 3T image is the conditioning signal — it tells the model what the underlying brain structure looks like before the 7T enhancement. The clean 3T patch is concatenated channel-wise with the noisy 7T patch:

```
[noisy_7T: B×1×64×64×64] + [clean_3T: B×1×64×64×64] → [B×2×64×64×64]
```

This is the simplest and most stable conditioning mechanism. The model sees both channels simultaneously from the first convolution and can use the 3T structural information at every layer to guide the denoising.

**Why concatenation and not cross-attention?**
Concatenation is proven more stable for this type of paired-image conditioning where the input and condition have the same spatial dimensions. Cross-attention is better for conditioning on text or other non-spatial signals.

### 5.4 Component 3: The 3D UNet Encoder (Downsampling Path)

Feature channels: [32, 64, 128, 256]

The encoder progressively downsamples the input while increasing the number of feature channels (to capture more abstract features):

```
Input: [B, 2, 64, 64, 64]
  → Inc Conv: [B, 32, 64, 64, 64]   (Initial feature extraction)
  → ResBlock + Downsample: [B, 64, 32, 32, 32]
  → ResBlock + Downsample: [B, 128, 16, 16, 16]
  → ResBlock + Downsample: [B, 256, 8, 8, 8]
```

**ResNet Block** (the basic building unit):
```
x → Conv3d(3x3x3) → GroupNorm → SiLU activation
  → Add time embedding (broadcast to spatial dims)
  → Conv3d(3x3x3) → GroupNorm → SiLU
  → Skip connection (1x1 conv if channels change) + residual add
```

**GroupNorm instead of InstanceNorm** (as stated in the code comment): GroupNorm divides the channels into groups and normalises within each group, per sample. InstanceNorm normalises across all spatial positions per channel per sample — it can be unstable when feature maps are small (e.g., the 8×8×8 bottleneck has only 512 positions). GroupNorm is more stable at small spatial sizes. It also has no dependency on batch size at all, making it the standard choice for 3D medical imaging where batch sizes are small and volumes are large.

**SiLU (Swish) activation**: `x * sigmoid(x)` — smooth, non-monotonic, empirically better than ReLU for generative models.

**Downsampling**: Strided convolution with stride=2, not MaxPool. Strided conv is learnable (can learn optimal downsampling) and preserves gradient flow better.

### 5.5 Component 4: Bottleneck with Self-Attention

At the deepest level (8×8×8 spatial, 256 channels), the model applies:

```
ResBlock → Self-Attention 3D → ResBlock
```

**Self-Attention 3D**: Allows every voxel in the bottleneck to attend to every other voxel. This captures global context — e.g., the model can learn that the left hippocampus and right hippocampus should be symmetric, or that white matter in one region implies white matter in structurally connected regions.

**Implementation detail**: The Q, K, V projections are 1×1×1 convolutions. The attention score is `QK^T / sqrt(d_head)`. A numerical stability fix was added: before the softmax, we subtract the max value from each row (`attn = attn - attn.max(dim=-1, keepdim=True).values`). This prevents overflow in float16 (which clips at 65,504 — exp(11) already exceeds this).

With 4 attention heads and 256 channels: each head has 64 dimensions. The 8×8×8 spatial grid has 512 positions, so the attention matrix is [B, 4, 512, 512] — feasible in memory.

### 5.6 Component 5: Dual Decoder (Multi-Task Architecture)

From the bottleneck, the model splits into **two separate decoder paths** that share the same encoder features (skip connections):

**Decoder 1 — Denoising Path:**
Reconstructs the predicted noise from [B, 256, 8, 8, 8] back to [B, 1, 64, 64, 64].

**Decoder 2 — Segmentation Path:**
Simultaneously predicts tissue class labels at every voxel: [B, 4, 64, 64, 64] (4 classes: BG, CSF, GM, WM).

Both decoders use:
```
TransposeConv3d (stride=2) → Concat with skip from encoder → ResBlock
```

**Why two decoders sharing one encoder?**

This is multi-task learning. The encoder is forced to learn features that are useful for *both* noise prediction (image quality) *and* tissue classification (anatomy). This means the encoder's internal representation must capture both texture/noise structure AND semantic tissue information. The result: the denoising path benefits from anatomy-aware features, and the synthesis output implicitly respects tissue boundaries even when we only look at the denoising output.

**Skip Connections (Concatenation):**
Skip connections go from each encoder level to the corresponding decoder level. They carry fine spatial detail that would be lost during downsampling. We use concatenation (not addition) — the concatenated tensor has doubled channels, which are then reduced by the ResBlock. Addition would blend the skip and upsampled features, which can cause gradient conflicts. Concatenation lets the ResBlock learn how to combine them optimally.

### 5.7 Final Output Heads

**Denoising head**: `Conv3d(32 channels → 1 channel, kernel=1)` — predicts the noise at each voxel

**Segmentation head**: `Conv3d(32 channels → 4 channels, kernel=1)` — predicts raw logits for 4 tissue classes. No softmax here (applied inside the loss function).

### 5.8 Parameter Count

With features=[32, 64, 128, 256] and a dual decoder:

- Approximately **14–15 million parameters** (the 256-channel bottleneck with two ResBlocks alone accounts for ~7M; each decoder adds ~2M)
- Designed to fit in ~24GB GPU memory (RTX 4090) with batch size 8, patch size 64³, AMP enabled

---

## 6. The Diffusion Process

### 6.1 What is a Diffusion Model?

A diffusion model is a generative model trained to reverse a noise-adding process.

**Forward Process (adding noise):**
Starting from a clean 7T image `x_0`, we progressively add Gaussian noise over T timesteps:
```
x_1 = sqrt(α_1) * x_0 + sqrt(1-α_1) * ε       (a little noise)
x_2 = sqrt(α_2) * x_0 + sqrt(1-α_2) * ε       (more noise)
...
x_T ≈ ε   (pure noise — the original image is gone)
```
where ε ~ N(0, I) is standard Gaussian noise.

The key formula is the closed-form `q_sample`:
```
x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
```
where `ᾱ_t = product of (1 - β_s) for s=1..t`.

This means we can directly compute the noisy version at *any* timestep without running all previous steps.

**Reverse Process (removing noise — what the model learns):**
The model learns to predict `ε` (the noise that was added) given `x_t` and `t`. During inference, we start from pure noise and iteratively remove a little noise at each step, guided by the 3T conditioning image.

### 6.2 Why Diffusion Instead of GAN?

| GAN | Diffusion |
|---|---|
| Mode collapse — forgets some outputs | No mode collapse — samples from full distribution |
| Training instability — discriminator-generator balance | Stable training — single objective (noise prediction) |
| Can hallucinate plausible but anatomically wrong structures | Can be constrained with anatomy losses |
| Training collapses silently | Loss is always meaningful (noise prediction MSE) |

The specific failure mode of GANs for brain MRI: they produce sharp-looking images but often create disconnected white matter blobs or missing hippocampi because the discriminator only checks "does this look real?" not "is the anatomy correct?"

### 6.3 Beta Schedule: Cosine

The noise schedule controls how much noise is added at each step.

**Linear schedule**: Adds noise uniformly across steps. Problem: the first few and last few steps add too little / too much noise relative to what the model can learn.

**Cosine schedule** (what we use):
```
ᾱ_t = cos((t/T + s) / (1 + s) * π/2)^2    where s=0.008
```

The cosine curve means:
- Early steps (low t): noise is added slowly — model learns fine details
- Middle steps: noise is added at a steady rate
- Late steps (high t): noise is added quickly toward pure noise

This gives more balanced training signal across all timesteps.

**Critical fix — last 15 timesteps excluded:**
At T=200, the cosine schedule produces extreme coefficient values near t=185-200 (the `sqrt(1/ᾱ_t)` term exceeds 4000x). This caused catastrophic loss spikes when these timesteps were sampled. We restrict training to timesteps 0–184 (`t = randint(0, 185)` gives integers 0 to 184 inclusive — 185 values, max is 184 not 185). Inference also uses the same range for consistency.

### 6.4 Forward Process (q_sample)

Used during training to add noise to the clean 7T patch:
```python
x_noisy = sqrt(ᾱ_t) * x_start + sqrt(1 - ᾱ_t) * noise
```
where `noise = randn_like(x_start)`.

This is the "corrupted" input the model sees during training.

### 6.5 Reverse Process During Inference (p_sample_loop)

Starting from pure Gaussian noise, iteratively denoise:
```
For t = 185 down to 0:
    x_{t-1} = model_mean(x_t, t, conditioning_3T) + noise (if t > 0)
```

At each step, the model predicts the noise, we compute x_0 from that, then compute the posterior mean x_{t-1}. The final image at t=0 is the synthetic 7T.

### 6.6 Predict x_0 from Noise

During training, we need to also compute the pixel loss (MSE between predicted image and real 7T). We reconstruct x_0 from the predicted noise:
```
x_0 = sqrt(1/ᾱ_t) * x_t - sqrt(1/ᾱ_t - 1) * noise_pred
```

The coefficients `sqrt(1/ᾱ_t)` and `sqrt(1/ᾱ_t - 1)` are clamped to max=10.0 to prevent explosion at high timesteps. The reconstructed x_0 is further clamped to [-2, 2] (wider than the data range of [-1, 1] to allow gradient flow at boundary cases).

---

## 7. Loss Functions

The total loss is a weighted combination of 4 components:

```
L_total = L_diff + λ_pixel * L_pixel + λ_percep * L_percep + λ_topo * L_topo
```

### 7.1 Diffusion Loss (L_diff) — Always Active, Weight=1.0

The primary diffusion objective: predict the noise that was added.

```
L_diff = L1(noise_pred, actual_noise)
```

We use L1 (Mean Absolute Error) rather than L2 (MSE) because:
- L1 is more robust to outliers — a few voxels with very wrong predictions don't dominate the loss
- L1 encourages sharper outputs; L2 tends toward blurry averages

The noise prediction is clamped to [-10, 10] before computing the loss to prevent catastrophic outliers from exploding the gradient.

### 7.2 Pixel Loss (L_pixel) — Active from Start, λ=1.0

Direct image reconstruction loss:
```
L_pixel = L1(x_recon, x_start)    clamped to max=5.0
```

Where `x_recon` is the image reconstructed from the predicted noise (using `predict_start_from_noise`).

**Why is this needed?** The diffusion loss trains the model to predict noise, but doesn't directly penalize bad image quality. The pixel loss creates a direct signal: "the reconstructed image should match the ground truth 7T image." This dramatically speeds convergence and prevents the model from predicting noise correctly but producing garbage images.

**Why clamp at 5.0?** In early training, when predictions are random, the pixel loss can be very large (reconstructed values all wrong). Clamping prevents a single bad batch from creating gradient explosions that destabilize training.

### 7.3 Perceptual Loss (L_percep) — Active from Stage 3, λ=0.25

Human perception doesn't notice pixel-level differences — we notice whether textures, edges, and structural patterns look correct. The perceptual loss uses a pre-trained VGG16 network to measure feature-level similarity.

```
L_percep = MSE(VGG16_features(x_recon), VGG16_features(x_start))
```

**Implementation details:**
- VGG16 is frozen (not trained) — we only use it for feature extraction
- VGG16 expects 2D RGB images. We handle 3D by sampling every 8th depth slice (stride=max(1, D//8)) and treating depth as a batch dimension
- The 1-channel grayscale MRI is replicated to 3 channels
- Input is mapped from [-1,1] to [0,1] then normalized to ImageNet statistics (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
- We use VGG16 feature layers 0–15 (indices 0 to 15) — this covers the first two full convolutional blocks and the first two convolutions of the third block (not "4 blocks" — the 4th block is not included). These capture low-to-mid level texture and edge features
- Loss is clamped to max=10.0

**Why add this at all?** Pixel loss (L1) alone produces blurry images. The model minimizes the average error, which means averaging over the distribution of plausible images → blur. Perceptual loss penalizes incorrect textures and edge sharpness, pushing the model toward images that look structurally correct even at a feature level.

### 7.4 Topology Loss (L_topo) — Active from Stage 4, λ=0.2

This is the core novel contribution. It ensures the synthetic 7T image has correct brain tissue boundaries.

The topology loss is computed on the **segmentation head output** (not the denoising head). It has two components:

#### 7.4.1 Weighted Cross-Entropy (CE) Component

```
L_CE = CrossEntropy(seg_pred_logits, tissue_labels, class_weights=[0.5, 2.0, 1.5, 1.0])
```

Class weights: Background=0.5 (dominant, downweighted), CSF=2.0 (rare, upweighted), GM=1.5, WM=1.0. This compensates for class imbalance — background voxels would otherwise overwhelm the loss.

The CE loss is further weighted per-voxel by an edge weight:
```
edge_weight[voxel] = 1.0 + (2.0 - 1.0) * is_edge[voxel]
                   = 1.0 for interior voxels
                   = 2.0 for boundary voxels
```

Boundary voxels are detected using 3D Sobel filters applied to the one-hot encoded target mask. This forces the model to pay double attention to tissue boundary voxels — exactly where topology errors occur.

#### 7.4.2 Boundary Dice Loss (Differentiable Edge Matching)

```
L_boundary = 1 - (2 * intersection + smooth) / (union + smooth)
```

Where:
- `edge_map`: binary edge map from target mask (detected using one-hot + Sobel)
- `pred_edges`: **soft** edge map from predicted probabilities (using Sobel on softmax probabilities)
- `smooth = 1.0`: Laplace smoothing prevents division-by-zero and gradient spikes

**Critical design decision — why "soft" edges for predictions?**

The naive approach would be: `argmax(seg_pred)` → binary prediction → detect edges. But `argmax` has zero gradient (it's not differentiable). The model would receive no gradient signal from the boundary loss.

Instead, we apply Sobel filters directly on the **softmax probabilities** (which are continuous and differentiable):
```
For each class c:
    prob_c = softmax(pred_logits)[:, c, :, :, :]
    gradient_c = sqrt(Sobel_x(prob_c)^2 + Sobel_y(prob_c)^2 + Sobel_z(prob_c)^2 + ε)

pred_edges = max over classes(gradient_c)
pred_edges = sigmoid(pred_edges * 5.0 - 2.5)  # normalize to [0,1]
```

The sigmoid normalization ensures `pred_edges` is always in [0,1] even for uniform predictions (minimum value ~0.076 from sigmoid(-2.5)). This guarantees the loss is never exactly zero.

#### 7.4.3 Multi-Scale Topology Loss

The topology loss is applied at 3 spatial scales: 1.0 (full), 0.5 (half), 0.25 (quarter). At smaller scales, the predictions and targets are downsampled. This ensures topology is correct at both fine-grained and coarse anatomical levels:
- Scale 1.0: individual voxel boundaries
- Scale 0.5: local structural consistency (gyri/sulci)
- Scale 0.25: global connectivity (white matter tracts)

Total multi-scale loss:
```
L_topo = (1.0 * L_at_scale_1.0 + 0.707 * L_at_scale_0.5 + 0.5 * L_at_scale_0.25) / (1.0 + 0.707 + 0.5)
```
(weights are sqrt(scale) — finer scales get more weight).

#### 7.4.4 Final Topology Loss Formula
```
L_topo = L_edge_weighted_CE + 0.5 * L_boundary_Dice
```

---

## 8. Training Curriculum (4 Stages)

Training uses a 4-stage progressive loss curriculum. The idea: start with the most stable loss (diffusion), then gradually add more complex losses as the model stabilizes. This prevents the model from being overwhelmed by contradictory gradient signals in early training.

### Stage 1: Steps 0 → 3,000

**Active losses**: Diffusion (full) + Pixel (half strength)
```
λ_pixel = 0.5,  λ_percep = 0.0,  λ_topo = 0.0
```

**Purpose**: Get the model to learn the basic image reconstruction task. The model starts from random weights and needs to first understand "reconstruct image from noise." Adding perceptual or topology losses here would be meaningless (the predictions are garbage initially, and those losses would provide contradictory signals).

The pixel loss is at half strength to avoid overpowering the diffusion signal in very early training.

### Stage 2: Steps 3,000 → 12,000

**Active losses**: Diffusion + Pixel (full strength)
```
λ_pixel = 1.0,  λ_percep = 0.0,  λ_topo = 0.0
```

**Purpose**: Model has learned basic reconstruction. Now we push hard on image quality with full pixel loss. By step 12,000, the model should be producing recognizable brain structures (SSIM ~0.3-0.5 expected).

### Stage 3: Steps 12,000 → 30,000

**Active losses**: Diffusion + Pixel + Perceptual (gradually increasing)
```
λ_pixel = 1.0
λ_percep = 0.25 * (step - 12000) / 8000     (ramps from 0 to 0.25 over 8,000 steps)
λ_topo = 0.0
```

**Purpose**: Add texture and edge detail guidance. VGG perceptual features push the model to generate sharp, realistic-looking structures. The gradual warm-up prevents VGG loss spikes from destabilizing the already-trained pixel reconstruction.

By step 30,000: SSIM should be ~0.6-0.75.

### Stage 4: Steps 30,000 → 150,000

**Active losses**: All four, topology ramping up
```
λ_pixel = 1.0,  λ_percep = 0.25
λ_topo = 0.2 * (step - 30000) / 25000     (ramps from 0 to 0.2 over 25,000 steps)
→ λ_topo = 0.2 (full) from step 55,000 onward
```

**Purpose**: The most important stage. Now that the model produces reasonable images (it has learned what brain tissue looks like), we add the topology constraint to ensure anatomical boundaries are correct. The ramp-up prevents the topology loss from overwhelming the image quality losses.

A learning rate decay is applied at step 120,000:
```
LR: 1e-4 → 2e-5    (80% reduction for fine-grained precision in final training)
```

### Training Duration

Total: 150,000 steps. At batch size 8 and ~1 second/step on a GPU: approximately 40 hours of training.

---

## 9. Hyperparameters and Configuration

| Parameter | Value | Why |
|---|---|---|
| Patch size | 64×64×64 | GPU memory limit; 64^3 = 262,144 voxels per sample |
| Batch size | 8 | RTX 4090 / similar; uses AMP for memory efficiency |
| Diffusion timesteps | 200 | Blueprint requirement; 200 gives fast convergence while maintaining quality |
| Usable timesteps | 185 (0-184) | Last 15 excluded due to cosine schedule instability |
| Beta schedule | Cosine | More uniform training signal than linear |
| Noise objective | pred_noise | Predict the noise ε (not x_0 directly) |
| Loss type | L1 | More robust than L2, produces sharper outputs |
| Feature channels | [32, 64, 128, 256] | Balances capacity vs memory |
| Self-attention | Enabled (bottleneck only) | Global context without O(N^2) cost at full resolution |
| Optimizer | AdamW | Adam + L2 weight regularization (weight_decay=1e-4) |
| Learning rate | 1e-4 | Standard for DDPM; decays to 2e-5 at step 120k |
| AdamW epsilon | 1e-5 | Raised from default 1e-8 for AMP stability |
| Gradient clip | 1.0 | Prevents gradient explosion; conservative (blueprint says 5.0) |
| EMA decay | 0.9999 | Very slow average — weights used for inference |
| EMA start | Step 2,000 | Don't average random-initialized weights |
| AMP GradScaler init_scale | 4096 (2^12) | Lower than default 2^16 — prevents float16 overflow with 3D medical volumes |
| LOOCV | Enabled | Leave-One-Out Cross-Validation; n_folds=10, val_fold=0 |
| Patches per volume | 32 | Balances coverage vs training speed |
| Min brain fraction | 0.10 | At least 10% brain voxels per patch |

---

## 10. Numerical Stability Fixes

Training deep learning models with float16 (AMP/mixed precision) on large 3D volumes introduces unique numerical challenges. These were systematically identified and fixed.

### 10.1 float16 Overflow in Loss Sums

**Problem**: float16 can only represent values up to 65,504. A 64×64×64 patch has 262,144 voxels. If the average cross-entropy loss per voxel is 0.3, the total sum before taking the mean is 262,144 × 0.3 = 78,643 — which **overflows float16** before the mean is computed.

**Fix**: Force float32 for all loss summation operations, even when running under AMP:
```python
loss_ce.float().mean()      # explicit float32 mean
edge_map_f32 = curr_edge_map.float()
pred_edges_f32 = pred_edges.float()
intersection = (pred_edges_f32 * edge_map_f32).sum()  # float32 sum
```

### 10.2 Cosine Schedule Instability at High Timesteps

**Problem**: Near t=200, the cosine schedule produces `sqrt(1/ᾱ_t)` values exceeding 4000. When used in `predict_start_from_noise`, this amplifies any noise in the prediction by 4000×, causing catastrophic loss spikes (loss reaching 10^6).

**Fix**:
1. Restrict training to t ∈ [0, 184]: `t = randint(0, 185)`
2. Clamp the recip coefficients: `sqrt_recip = clamp(sqrt_recip, max=10.0)`
3. Clamp reconstructed x_0: `x_recon = clamp(x_recon, -2.0, 2.0)`
4. Apply same restriction during inference (p_sample_loop)

### 10.3 Self-Attention Softmax Overflow

**Problem**: In float16, the attention logits `QK^T / sqrt(d)` can be large. `exp(large_value)` overflows float16 → softmax produces NaN.

**Fix**: Subtract the row maximum before softmax (numerically stable softmax):
```python
attn = attn - attn.max(dim=-1, keepdim=True).values
attn = attn.softmax(dim=-1)
```
This is mathematically equivalent (softmax is shift-invariant) but prevents overflow.

### 10.4 argmax Has Zero Gradient (Topology Loss)

**Problem**: The original boundary Dice loss computed edges on `argmax(pred_logits)` — the hard predicted class map. `argmax` is not differentiable (gradient is zero everywhere). The entire boundary loss component provided no gradient signal to the model — it was silently broken.

**Fix**: Apply Sobel filters to the **softmax probability maps** (which are differentiable) to get soft edges, then apply sigmoid normalization. Full gradient flows back through softmax → conv weights.

### 10.5 NaN Guard

As a final safety net, all four loss components are checked before backward:
```python
if not isfinite(loss):
    print(f"WARNING: {name} loss is non-finite, zeroing for this step")
    return zeros(requires_grad=True)  # connected to graph
```

Additionally, if any gradient is NaN/Inf after backward, the optimizer step is skipped entirely for that batch. This prevents NaN from corrupting the model weights.

### 10.6 Logit Clamping

Before cross-entropy, prediction logits are clamped:
```python
pred_logits = clamp(pred_logits, -50.0, 50.0)
```
Prevents extremely confident wrong predictions from creating infinite CE loss.

---

## 11. Problems Encountered and Fixes Applied

### Problem 1: Training Produced Garbage (SSIM = -0.13, PSNR = 0.52 dB)

**Root causes (discovered via blueprint audit):**

1. **Timestep gating**: The original code multiplied all auxiliary losses by `(t < 400).float()` — meaning for 60% of training steps (when t ≥ 400), the model received zero gradient from pixel loss and perceptual loss. The model only learned to predict noise, not reconstruct images.

2. **Tanh clamping**: `x_recon = tanh(x_recon) * 1.05` was applied, destroying intensity scales and causing anti-correlation with the target.

3. **1000 timesteps**: Too many timesteps for this dataset size. Blueprint specifies 200.

4. **Missing Stage 1 pixel loss**: The original Stage 1 had no pixel loss at all — the first 10,000 steps taught the model nothing about image reconstruction.

5. **Underweighted losses**: λ_pixel=0.25, λ_percep=0.1 were too low. Blueprint requires λ_pixel=1.0.

**Fixes**: Complete rewrite of loss computation, removal of timestep gating, reduction to 200 timesteps, curriculum restructured with pixel loss from step 0.

### Problem 2: Zero Gradient from Topology Boundary Loss

**Root cause**: `argmax` used for predicted edge detection — not differentiable.

**Fix**: Soft Sobel edges on probability maps (described in Section 10.4).

### Problem 3: float16 Overflow in Topology Loss

**Root cause**: All loss computations were in AMP float16. Sums over 262,144 voxels overflowed.

**Fix**: Explicit `.float()` casts before every summation. The entire topology loss is wrapped in `torch.autocast(enabled=False)` to force float32.

### Problem 4: Segmentation Head Returning Zeros

**Root cause**: The original `forward()` method had a bug — it returned `torch.zeros(...)` as the segmentation output (placeholder that was never removed). The segmentation head's weights were never trained.

**Fix**: `forward = forward_full` overrides the broken forward method with the correct `forward_full` that runs both decoders properly.

### Problem 5: FreeSurfer Label Remapping Bug

**Root cause**: The `num_classes` in the model config was set to 3 (from an old config) but the masks had 4 classes (0=BG, 1=CSF, 2=GM, 3=WM). This caused the Cross-Entropy loss to crash with "Target out of range" errors, or silently use wrong class weights.

**Fix**: `num_classes=4` in config, matching the actual mask labels. The checkpoint loading code was also made robust to handle segmentation head shape mismatches when resuming from old checkpoints.

### Problem 6: Out-of-Range Segmentation Labels

**Root cause**: FreeSurfer labels that were NOT in the FS_MAPPING table were silently left as their original value (e.g., label 77, 85, etc.). These would cause cross-entropy to crash.

**Fix**: Training script checks `seg_target` at every step:
```python
invalid_mask = (seg_target < 0) | (seg_target >= num_classes)
seg_target[invalid_mask] = 0   # remap to background
```
Logs a warning when this happens so we can identify problem subjects.

### Problem 7: EMA Applied Before Model Stabilizes

**Problem**: EMA was applied from step 0. In the first 2000 steps, the model weights are random and changing rapidly. Averaging random weights into the EMA model contaminated it.

**Fix**: EMA starts at step 2000 (`ema_start: 2000`).

---

## 12. Current State and What Comes Next

### 12.1 Where We Are

- **Branch**: `feat/train-with-masks`
- **Stage**: Training in progress (the curriculum-based training is designed to run for 150,000 steps)
- **All critical bugs fixed**: The model now trains with correct gradients from all 4 loss components

### 12.2 What the Training Looks Like in Practice

At each step:
1. Load a batch of 8 patch pairs (3T + 7T + segmentation mask)
2. Sample random timestep t ∈ [0, 184] for each sample
3. Add appropriate amount of noise to the 7T patches using `q_sample`
4. Pass [noisy_7T, clean_3T] through the model → predicted_noise + segmentation_logits
5. Compute L_diff, L_pixel (after reconstructing x_0), L_percep (VGG features), L_topo (on segmentation)
6. Combine with stage-appropriate weights
7. Backward pass, gradient check (skip if NaN), gradient clip at 1.0
8. AdamW update + EMA update
9. Log to TensorBoard / W&B every 50 steps
10. Save checkpoint every 5,000 steps

### 12.3 What the Checkpoints Contain

Each checkpoint file saved every 5,000 steps contains:
- Model weights (the UNet)
- EMA model weights (used for inference — smoother, better than raw model weights)
- Optimizer state (AdamW momentum buffers)
- GradScaler state (AMP scaling factor)
- Current step number

### 12.4 Expected Milestones

| Step | Expected SSIM | Expected Behaviour |
|---|---|---|
| 3,000 | 0.1-0.2 | Basic brain shape visible, very blurry |
| 12,000 | 0.4-0.5 | WM/GM distinction starting to appear |
| 30,000 | 0.6-0.75 | Textural details (sulci, gyri) improving |
| 55,000 | 0.75-0.85 | Topology loss fully active, tissue boundaries sharpening |
| 120,000 | 0.85-0.90 | LR decays, final refinement |
| 150,000 | >0.88 | Target: SSIM > 0.88, PSNR > 28 dB |

### 12.5 What Comes After Training

1. **Evaluation**: Run the trained model on held-out test subjects. Compute SSIM, PSNR, NMSE, LPIPS (image quality) and Dice scores for WM/GM/CSF (anatomy fidelity).

2. **Ablation study**: Retrain without topology loss → compare Dice scores. This proves the topology loss contributes.

3. **AD Downstream Validation**: Run inference on ADNI 3T scans → generate synthetic 7T → train 4 AD classifiers (3T-only, synthetic-7T-only, 3T+synthetic fusion, generic-enhancement baseline) → compare diagnostic accuracy.

### 12.6 Files Quick Reference

| File | Purpose |
|---|---|
| `src/model.py` | AnatomyGuidedUNet architecture |
| `src/diffusion.py` | GaussianDiffusion — forward/reverse process, all loss functions |
| `src/topology_loss.py` | EdgeAwareTopologyLoss, MultiScaleTopologyLoss |
| `src/synthesis_dataset.py` | Dataset, patch sampling, data loading |
| `src/preprocessing.py` | N4 correction, normalization pipeline |
| `scripts/train_diffusion.py` | Training loop, curriculum, checkpoint saving |
| `scripts/preprocess_masks.py` | FreeSurfer label remapping to 4 classes |
| `configs/train_diffusion.yaml` | All hyperparameters |
| `pairs_new copy.csv` | Subject→file path mapping with seg column |

---

*Report written: March 2026*
*Project: Topology-Preserving 3T-to-7T MRI Enhancement for Early Alzheimer's Detection*
*Architecture: Anatomy-Guided Conditional Diffusion Model (DDPM) with Multi-Task Segmentation Head*
