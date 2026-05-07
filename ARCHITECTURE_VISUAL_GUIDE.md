# Topo-Brain: Visual Architecture & Concepts

## Complete Pipeline Visualization

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                         TOPO-BRAIN PIPELINE                               ║
╚═══════════════════════════════════════════════════════════════════════════╝

                    ┌─────────────────────────────────┐
                    │   PHASE 0: RAW BIDS DATA        │
                    │  (10 subjects × 2 sessions)     │
                    │  40 volumes total               │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
            ┌──────────────┐          ┌──────────────┐
            │ 3T Sessions  │          │ 7T Sessions  │
            │ (ses-1)      │          │ (ses-2)      │
            │ 10 volumes   │          │ 10 volumes   │
            └──────────────┴──────────┬──────────────┘
                    all 20 volumes used for training
                                   │
                    ┌──────────────▼──────────────┐
                    │  PHASE 1: PREPROCESSING     │
                    │  ✓ Brain extraction         │
                    │  ✓ Bias correction (opt)    │
                    │  ✓ Reorientation (RAS+)     │
                    │  ✓ Normalization [-1, 1]    │
                    │  ✓ QC & outlier detection   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  PHASE 2: DATA PREPARATION          │
                    │  ✓ Pair matching & verification     │
                    │  ✓ Patient-level splitting          │
                    │    - Train: 12 volumes (sub-1..6)   │
                    │    - Val:    4 volumes (sub-7..8)   │
                    │    - Test:   4 volumes (sub-9..10)  │
                    │  ✓ Patch extraction (64³ patches)   │
                    │  ✓ Augmentation (on-the-fly)        │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  PHASE 3: MODEL ARCHITECTURE        │
                    │  AnatomyGuidedUNet (22M params)     │
                    │  ├─ Dual-decoder U-Net              │
                    │  ├─ Denoising decoder               │
                    │  ├─ Segmentation decoder            │
                    │  ├─ Sinusoidal time embeddings      │
                    │  └─ Self-attention at bottleneck    │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  PHASE 4: TRAINING (400K iters)     │
                    │  ✓ Noise scheduler (cosine)         │
                    │  ✓ Multi-task loss                  │
                    │      - Denoising (L1)               │
                    │      - Segmentation (CE)            │
                    │      - Perceptual (LPIPS)           │
                    │  ✓ EMA model tracking               │
                    │  ✓ Checkpointing & evaluation       │
                    │  ✓ TensorBoard logging              │
                    │  Result: best_model.pt (~100MB)     │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  PHASE 5: INFERENCE                 │
                    │  Input: 3T volume (any size)        │
                    │  ✓ Sliding window patching          │
                    │  ✓ Diffusion sampling (50 steps)    │
                    │  ✓ Gaussian blending                │
                    │  ✓ Inverse normalization            │
                    │  Output: Synthetic 7T volume        │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  PHASE 6: EVALUATION                │
                    │  ✓ PSNR, SSIM comparison           │
                    │  ✓ Tissue segmentation validation   │
                    │  ✓ Clinical feature extraction      │
                    │  ✓ Radiologist review (future)      │
                    └──────────────────────────────────────┘
```

---

## Data Transformation Flow

### Input → Preprocessing → Normalized Volume

```
Raw 3T T1w Volume                 After Preprocessing
     │                                   │
     │ native resolution                 │ isotropic 1mm³
     │ arbitrary intensity range         │ normalized [-1, 1]
     │ RPI orientation (varies)          │ RAS+ orientation
     │ with skull                        │ brain only
     │                                   │
     ▼                                   ▼
  ┌─────┐                         ┌──────────┐
  │ ███ │                         │ ↑        │
  │█████│  Brain: 3T@1.2×1.0×1.0  │ 220 voxels
  │ ███ │  Intensity: [100, 4000]  │ ← Z-axis
  └─────┘  Shape: (220, 256, 220) │ Intensity: [-1, 1]
                                   │ Shape: (200, 200, 180)
                                   └──────────┘
```

### Paired 3T-7T Training Samples

```
TRAINING BATCH (B=4, after augmentation):

┌────────────────────────────────────────────────────────────┐
│ Batch Index │ 3T Input │ 7T Target │ Tissue Mask │ Subject │
├────────────────────────────────────────────────────────────┤
│ [0]         │ [1,64³]  │ [1,64³]   │ [1,64³]     │ sub-02  │
│ [1]         │ [1,64³]  │ [1,64³]   │ [1,64³]     │ sub-05  │
│ [2]         │ [1,64³]  │ [1,64³]   │ [1,64³]     │ sub-01  │
│ [3]         │ [1,64³]  │ [1,64³]   │ [1,64³]     │ sub-04  │
└────────────────────────────────────────────────────────────┘
  Stack to: [4, 1, 64, 64, 64] for efficient GPU processing
```

---

## Model Internal Architecture

### Encoding Phase: Feature Extraction

```
Input x: [B=4, 2, 64, 64, 64]
         (noisy 7T + clean 3T concatenated)

    ▼ Initial Conv3d(2→32, kernel=3)

x₀: [4, 32, 64, 64, 64]  ──→ Save as skip[0]
    └─ ResnetBlock + TimeEmb
    └─ Conv3d(stride=2)  downsampling

    ▼

x₁: [4, 64, 32, 32, 32]  ──→ Save as skip[1]
    └─ ResnetBlock + TimeEmb
    └─ Conv3d(stride=2)  downsampling

    ▼

x₂: [4, 128, 16, 16, 16]  ──→ Save as skip[2]
    └─ ResnetBlock + TimeEmb
    └─ Conv3d(stride=2)  downsampling

    ▼

x₃: [4, 256, 8, 8, 8]  ──→ Save as skip[3]
    └─ ResnetBlock + TimeEmb
    └─ Conv3d(stride=2)  downsampling

    ▼

BOTTLENECK: [4, 256, 4, 4, 4]
    ├─ ResnetBlock + TimeEmb
    ├─ SelfAttention3D (global context for entire brain patch)
    └─ ResnetBlock + TimeEmb
```

### Decoding Phase: Feature Refinement (Dual Decoders)

```
From Bottleneck [4, 256, 4, 4, 4], TWO independent decoders:

═══════════════════════════════════════════════════════════════

DECODER 1: DENOISING (predict noise ε̂)

    Upsample: [4, 256, 4,4,4] → [4, 128, 8,8,8]
    Concat  : + skip[3] from encoder [4, 256, 8,8,8]
             = [4, 256, 8,8,8]
    ResnetBlock + TimeEmb + Conv1x1(256→128)
    
    ▼
    
    Upsample: [4, 128, 8,8,8] → [4, 64, 16,16,16]
    Concat  : + skip[2] [4, 128, 16,16,16]
             = [4, 128, 16,16,16]
    ResnetBlock + TimeEmb + Conv1x1(128→64)
    
    ▼
    
    Upsample: [4, 64, 16,16,16] → [4, 32, 32,32,32]
    Concat  : + skip[1] [4, 64, 32,32,32]
             = [4, 64, 32,32,32]
    ResnetBlock + TimeEmb + Conv1x1(64→32)
    
    ▼
    
    Upsample: [4, 32, 32,32,32] → [4, 32, 64,64,64]
    Concat  : + skip[0] [4, 32, 64,64,64]
             = [4, 32, 64,64,64]
    ResnetBlock + TimeEmb + Conv1x1(32→1)
    
    ▼
    
    OUTPUT: [4, 1, 64, 64, 64]  ← Predicted noise ε̂

═══════════════════════════════════════════════════════════════

DECODER 2: SEGMENTATION (predict tissue masks)

    [Same structure as Decoder 1, but:]
    ├─ Uses SAME skip connections as Decoder 1
    ├─ Final Conv1x1(32→4) for tissue classes:
    │  ├─ Class 0: Gray Matter (GM)
    │  ├─ Class 1: White Matter (WM)
    │  ├─ Class 2: Cerebrospinal Fluid (CSF)
    │  └─ Class 3: Background
    │
    └─ OUTPUT: [4, 4, 64, 64, 64]  ← Tissue logits
```

### Key Design Decision: Why Dual Decoders?

```
Problem: Multi-task learning can cause conflicts

Solution: Shared Encoder + Dual Decoders

┌─────────────────────────────────────────┐
│ SHARED ENCODER (all gradients flow here) │
│ - Learns robust, anatomy-aware features │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
   DECODER 1             DECODER 2
   (Denoising)           (Segmentation)
   Independent           Independent
   loss: L1              loss: CE
   
Benefits:
✓ Denoising & segmentation gradients don't interfere
✓ Shared encoder learns richer features
✓ Anatomical structure enforced via segmentation
✓ No need to choose which decoder's gradients win
```

---

## Training Recipe: How the Model Learns

### One Training Step

```
Step 1: FORWARD PASS
────────────────────

Input batch from training set:
  x_3t_clean: [B=4, 1, 64, 64, 64] ─┐
  x_7t_clean: [B=4, 1, 64, 64, 64] ─├─ Fetch from DataLoader
  seg_target: [B=4, 1, 64, 64, 64] ─┘

Sample random timestep t:
  t ~ Uniform(1, 1000)  e.g., t=347

Add noise to target (forward diffusion):
  noise ~ N(0, I)
  √α_bar[347] ≈ 0.45, √(1-α_bar[347]) ≈ 0.89
  x_t = 0.45 * x_7t_clean + 0.89 * noise

Concatenate inputs:
  x_cond = [x_t, x_3t_clean]  ([B, 2, 64³])

Feed to model:
  ε̂, seg_logits = model(x_t, x_3t_clean, t)
  
  ε̂:        Model's prediction of noise [B, 1, 64³]
  seg_logits: Model prediction of tissue [B, 4, 64³]


Step 2: COMPUTE LOSSES
──────────────────────

Denoising loss (primary objective):
  L_denoise = ||ε̂ - noise||₁
            = mean absolute difference
            
Segmentation loss (anatomical constraint):
  L_seg     = CrossEntropy(seg_logits, seg_target)
            = classification loss for tissue types
            
Perceptual loss (fine details):
  L_percep  = ||VGG(x_t) - VGG(x_7t_clean)||₂
            = feature matching in VGG space
            
Total loss (weighted):
  L_total   = L_denoise 
              + 0.1 * L_seg 
              + 0.01 * L_percep


Step 3: BACKWARD PASS
─────────────────────

Compute gradients via backpropagation:
  dL/dθ ← Automatic differentiation

Clip gradients (prevent exploding):
  ||dL/dθ|| > 1.0 → clip to norm 1.0

Update weights via Adam optimizer:
  θ_new = θ_old - lr * (m/(√v + ε))
  
  where:
    lr=1e-4 (learning rate)
    m=momentum (exponential moving avg of gradients)
    v=velocity (exponential moving avg of squared gradients)


Step 4: EXPONENTIAL MOVING AVERAGE
──────────────────────────────────

Update EMA model (for inference stability):
  θ_ema_new = 0.9999 * θ_ema_old + 0.0001 * θ_new
  
  → EMA is much more stable than latest weights
  → Always use EMA model for inference


Step 5: LOGGING & CHECKPOINTING
───────────────────────────────

Every 100 steps:
  Log to TensorBoard:
    - train/loss = 0.245
    - train/loss_denoise = 0.235
    - train/loss_seg = 0.008
    - train/loss_percep = 0.002
  
Every 2000 steps:
  - Generate validation samples
  - Compute PSNR/SSIM
  - Save checkpoint

Result:
  Iteration 347 completed in 0.23 seconds
  Memory used: 7.2 GB / 8.0 GB available
```

---

## Inference Pipeline: From 3T to Synthetic 7T

### Full-Volume Inference (Slide Window Approach)

```
INPUT: 3T Volume [D=192, H=256, W=256] (too large for GPU)

Step 1: TILE DECOMPOSITION
──────────────────────────

Patch size: 64³
Overlap: 50% (stride = 32)

Visual (2D cross-section):

     Original volume (256×256)
     ┌───────────────────────┐
     │  ●●●  ●●●  ●●●  ●●●  │
     │  ●●●  ●●●  ●●●  ●●●  │
     │  ●●●  ●●●  ●●●  ●●●  │
     │  ─────────────────── │ ← Patch #1 [0:64, 0:64]
     │  ●●●  ●●●  ●●●  ●●●  │
     │  ●●●  ●●●  ●●●  ●●●  │
     │  ─────────────────── │ ← Patch #2 [32:96, 0:64] (overlaps!)
     │  ●●●  ●●●  ●●●  ●●●  │
     │  ●●●  ●●●  ●●●  ●●●  │
     └───────────────────────┘

Result: (192-64)/32 + 1 = 5 tiles in each dimension
Total patches: 5×5×5 = 125 patches


Step 2: BATCH PROCESSING
────────────────────────

Process patches in batches of B=4:
  Loop iteration 1: Process patches [0-3]
  Loop iteration 2: Process patches [4-7]
  ...
  Loop iteration 32: Process patches [124]

For each batch:
  x_3t_patch: [B, 1, 64, 64, 64]
  ↓ Feed to model
  x_7t_patch: [B, 1, 64, 64, 64] (generated)


Step 3: DIFFUSION SAMPLING (for each patch)
─────────────────────────────────────────────

Start with pure noise:
  x_999 ~ N(0, I) [B, 1, 64, 64, 64]

Iterative denoising (reverse diffusion):
  For t = 999 down to 1:
    
    ε̂ = model(x_t, x_3t_patch, t)
    
    x_{t-1} = (1/√α_t) * (x_t - (1-α_t)/√(1-ᾱ_t) * ε̂)
              + σ_t * z
    
    where z ~ N(0, I) for t>1, z=0 for t=1

After 50 iterations: x_0 ≈ synthetic 7T patch


Step 4: PATCH BLENDING (Gaussian Weighting)
──────────────────────

Overlapping patches need smooth transitions.

Create Gaussian window:
  w(x,y,z) = exp(-3 * ((x-32)² + (y-32)² + (z-32)²) / 32²)
  
  → Weight = 1.0 at center
  → Weight = 0.0 at edges
  → Smooth transition in overlap region

For each patch position (i,j,k):
  output[i*32:(i*32+64), ...] += patch_output * window


Step 5: NORMALIZATION
─────────────────────

After blending, normalize by accumulated weights:
  output /= (accumulated_weight_map + eps)
  
  → Each voxel receives contribution from 1-8 overlapping patches
  → Normalize by total weight
  → Result: smooth, artifact-free full volume


Step 6: INVERSE NORMALIZATION
──────────────────────────────

Convert from [-1, 1] to original intensity scale:

Stored normalization stats during preprocessing:
  p_low = 250, p_high = 3500 (percentile values)
  mean = 1200, std = 800 (z-score statistics)

Reverse normalization:
  x_denorm = x_clipped * std + mean
  x_orig_scale = np.clip(x_denorm, p_low, p_high)

OUTPUT: Synthetic 7T Volume [192, 256, 256]
        Same shape as input 3T
        Similar intensity distribution to real 7T
        Ready for clinical use!


Step 7: POST-PROCESSING (Optional)
───────────────────────────────────

Optional enhancements:
  - Bilateral filtering (edge-preserving denoising)
  - Sharpening (enhance fine details)
  - Histogram matching (match real 7T intensity distribution)
```

---

## Loss Functions Explained

### 1. Denoising Loss (L1)

```
During training at timestep t:

True noise:     ε ~ N(0, I)
Predicted:      ε̂ = model(x_t, x_3t, t)

Loss = mean(|ε̂ - ε|)

Why L1 instead of L2?
  L1 is more robust to outliers
  Better for medical imaging (fewer artifacts)

Why predict noise instead of image?
  Historical: Early diffusion work predicted x₀ directly
  But predicting ε works better in practice
  Also called "ε-objective" or "velocity parameterization"

Effect:
  At t=0 (clean):   Model learns fine details
  At t=500:         Model learns coarse structure
  At t=999 (noisy): Model learns denoising from chaos
```

### 2. Segmentation Loss (Cross-Entropy)

```
During training, simultaneously predict tissue masks:

Prediction:  seg_logits [B, 4, 64³] (4 tissue classes)
Target:      seg_target  [B, 1, 64³] (class indices 0-3)

Loss = CrossEntropy(seg_logits, seg_target)
     = -1/N * Σ target[i] * log(softmax(logits[i]))

Tissue classes:
  Class 0: Gray Matter (GM)
  Class 1: White Matter (WM)
  Class 2: Cerebrospinal Fluid (CSF)
  Class 3: Background

Why include segmentation loss?
  ✓ Anatomically grounds the model
  ✓ Prevents generation of invalid tissue boundaries
  ✓ Soft constraint: not explicitly modifying denoised output
  ✓ Encoder learns to respect tissue anatomy

Weight: λ_seg = 0.1
  (10× smaller than denoising loss)
```

### 3. Perceptual Loss (VGG Features)

```
Perceptual loss matches high-level features:

Features x: VGG_layers(x_t)              [B, C, H, W]
Features y: VGG_layers(x_7t_clean)       [B, C, H, W]

Loss = mean((Features_x - Features_y)²)

Why perceptual loss?
  ✓ Captures texture and fine details
  ✓ Humans perceive feature similarity better than pixel similarity
  ✗ Can induce VGG-bias (not ideal for medical imaging)

How it's computed:
  - Load pre-trained VGG16 (frozen, no gradient updates)
  - Convert 3D volumes to 2D for VGG (slice by slice)
  - Extract layers 1-16 features
  - Compute MSE in feature space

Weight: λ_percep = 0.01
  (100× smaller than denoising loss)
  Keep minimal to avoid distorting medical features
```

### 4. Total Training Loss

```
L_total = L_denoise 
          + 0.1 * L_seg 
          + 0.01 * L_percep

Example values per iteration:
  L_denoise = 0.235 ← Main contributor
  L_seg     = 0.008 × 0.1 = 0.0008 ← Guides anatomy
  L_percep  = 0.002 × 0.01 = 0.00002 ← Polishes details
  ──────────────────────────────
  L_total   = 0.2358

Loss schedule:
  Iterations 1-5K:    L_Denoise decreases steeply
  Iterations 5K-50K:  L_total continues to decrease
  Iterations 50K-400K: Gradual refinement, EMA accumulation
```

---

## Noise Schedule: The Heartbeat of Diffusion

```
Cosine Schedule (used in Topo-Brain):

β_t = 1 - α_t        (per-step noise increase)
ᾱ_t = Π α_{1..t}     (cumulative product)
√ᾱ_t = signal strength at timestep t
√(1-ᾱ_t) = noise strength at timestep t

Visualization:
┌─────────────────────────────────────────────────────┐
│ Signal vs Noise Strength                            │
│                                                      │
│ 1.0 │ √ᾱ_t (signal)                                │
│     │●●●●              signal strong at early steps  │
│     │    ●●●●           smooth transition             │
│ 0.5 │        ●●●●                                   │
│     │            ●●●●                               │
│     │                ●●●●   √(1-ᾱ_t) (noise)       │
│ 0.0 │                    ─────────────────►         │
│     └─────────────────────────────────────────────  │
│       0     250      500     750     999  (timestep) │
│                                                      │
│ Key property:                                       │
│  - At t=0:   signal=1.0, noise=0.0 (clean image)   │
│  - At t=999: signal≈0.0, noise≈1.0 (pure noise)    │
│  - Cosine: Smooth, empirically better than linear  │
│                                                      │
└─────────────────────────────────────────────────────┘

Why cosine schedule vs linear?
  Linear:  β increases uniformly → harsh transitions
  Cosine:  β increases smoothly → natural progression
           Better recent empirical results
```

---

## Key Differences Between Denoising & Segmentation Decoders

```
                    DENOISING          SEGMENTATION
                    ─────────────────  ──────────────────
Output Channels     1                  4 (tissue classes)
Output Type         Noise prediction   Class probabilities
Loss Function       L1 (regression)    Cross-Entropy (classification)
Output Range        [-∞, +∞]           [0, 1] (after softmax)
Post-processing     No argmax          argmax → integer labels
Clinical relevance  Image quality      Tissue validity
Weight in training  1.0                0.1 (auxiliary)
Optimization goal   Minimize pixel     Enforce anatomical
                    error              boundaries

Visualization:

DENOISING DECODER OUTPUT:
  ε̂[b=0, :, :, :]  (predicted noise)
  ┌──────────────────────────┐
  │ ░░░░░░░░░░░░░░░░░░░░░░░░│
  │ ░ -0.15 +0.02 -0.08 ░  │  Range: [-1, +1]
  │ ░ +0.01 -0.12 +0.19 ░  │  (Continuous values)
  │ ░ -0.07 +0.04 +0.06 ░  │
  │ ░░░░░░░░░░░░░░░░░░░░░░░░│
  └──────────────────────────┘

SEGMENTATION DECODER OUTPUT:
  seg_logits[b=0, :, :, :]  (before softmax)
  Class 0 (GM):              Class 1 (WM):
  ┌──────────────┐          ┌──────────────┐
  │ ░░░░░░░░░░░░│          │ ░░░░░░░░░░░░│
  │ ░ +2.1 +1.8 ░│          │ ░ -0.5 +0.2 ░│
  │ ░ +1.9 +2.0 ░│          │ ░ +0.1 -0.3 ░│
  │ ░░░░░░░░░░░░│          │ ░░░░░░░░░░░░│
  └──────────────┘          └──────────────┘
  Class 2 (CSF):            Class 3 (BG):
  ┌──────────────┐          ┌──────────────┐
  │ ░░░░░░░░░░░░│          │ ░░░░░░░░░░░░│
  │ ░ -1.2 -0.8 ░│          │ ░ -0.4 -0.6 ░│
  │ ░ -0.9 -1.1 ░│          │ ░ -0.3 -0.5 ░│
  │ ░░░░░░░░░░░░│          │ ░░░░░░░░░░░░│
  └──────────────┘          └──────────────┘

After argmax:
  ┌──────────────┐
  │ 0 0 0 0 0 0  │
  │ 0 0 0 0 0 0  │  (Discrete labels: 0, 1, 2, or 3)
  │ 0 0 0 0 1 1  │
  │ 0 0 1 1 1 2  │
  │ 2 2 2 2 2 2  │
  │ 3 3 3 3 3 3  │
  └──────────────┘
```

---

## Metrics for Evaluation

### During Training (Logged to TensorBoard)

```
LEARNING CURVES:
┌──────────────────────────────────────────┐
│                                          │
│    Total Loss                            │
│    ╲╲╲╲╲  ← Steep drop (initial training)
│         ╲╲╲╲╲ ← Gradual decrease       │
│               ╲╲╲ ← Plateau (convergence)
│                  ╲ ← EMA smoothing     │
│                                          │
│ ┴─────┴─────┴─────┴─────┴─────┴        │
│ 0     50K   100K  200K  300K  400K iters│
│                                          │
└──────────────────────────────────────────┘

Healthy indicators:
✓ Loss decreases monotonically for first 100K iters
✓ Loss plateaus as training progresses
✓ EMA model loss is smoother than raw loss
✓ No sudden spikes (indicate unstable gradients)
```

### On Validation Set

```
PSNR (Peak Signal-to-Noise Ratio)
  Measures pixel-level reconstruction quality
  Higher is better
  Typical: 28-32 dB (medical imaging standard)
  Formula: PSNR = 20 * log10(MAX_VALUE / RMSE)
           
  ├─ 10 dB: Terrible (strong artifacts)
  ├─ 20 dB: Poor (visible degradation)
  ├─ 25 dB: Acceptable (for compression)
  ├─ 30 dB: Good (high quality)
  └─ 40+ dB: Excellent (nearly lossless)

SSIM (Structural Similarity Index)
  Measures perceptual similarity (humans see it better)
  Range: [-1, 1]; 1 = identical
  Typical: 0.85-0.92
  Better than PSNR for medical imaging
  
  ├─ 0.5-0.7: Low similarity
  ├─ 0.7-0.85: Moderate similarity
  ├─ 0.85-0.95: High similarity
  └─ 0.95+: Near-identical

FID (Fréchet Inception Distance)
  Measures distribution-level similarity
  Lower is better
  Captures realism better than pixel metrics
  Used for high-level quality assessment
```

---

## Training Dynamics Graph

```
Iteration: 1                               Iteration: 400,000
┌─────────────────────────────────────┐
│ Model starts random                 │ Model well-trained
│ L_total ≈ 3.2                       │ L_total ≈ 0.18
│ L_denoise = 3.15                    │ L_denoise = 0.17
│ L_seg = 0.03                        │ L_seg = 0.008
│ L_percep = 0.02                     │ L_percep = 0.002
│                                     │
│ Large losses: random predictions    │ Small losses: good predictions
│                                     │
│ Gradient norm: 2.4 (clipped to 1.0)│ Gradient norm: 0.3
│ → Explosive gradients              │ → Stable gradients
│                                     │
│ Model updates: Large               │ Model updates: Small
│    Δθ ≈ 0.0001 × [(large grad)]   │    Δθ ≈ 0.0001 × [(small grad)]
│     ≈ 0.0024 per param             │     ≈ 0.00003 per param
└─────────────────────────────────────┘

Learning trajectory:

Phase 1 (0-50K iters): Steep descent
  → Model learns basic denoising
  → Loss drops from 3.2 → 0.6
  → Visible improvements each checkpoint

Phase 2 (50K-200K iters): Gradual improvement
  → Model learns fine details
  → Loss drops from 0.6 → 0.25
  → PSNR improves: 20 → 28 dB
  → SSIM improves: 0.65 → 0.85

Phase 3 (200K-400K iters): Refinement
  → Loss drops from 0.25 → 0.18
  → Diminishing returns (law of diminishing returns)
  → PSNR: 28 → 30 dB (small improvement)
  → SSIM: 0.85 → 0.89 (plateauing)

Phase 4 (400K+ iters): Convergence
  → Loss stable around 0.18
  → No significant improvements
  → Stop training (early stopping)
```

---

## File I/O During Training

```
Training session creates/modifies:

models/
├── best_model.pt              ← Save when val_loss improves
├── latest_model.pt            ← Always latest (overwrite)
└── checkpoints/
    ├── ckpt_iter_1000.pt
    ├── ckpt_iter_2000.pt
    ...
    └── ckpt_iter_400000.pt     ← Every save_freq=1000 iters

logs/
├── 20240115_234500/
│   ├── events.out.tfevents.xxxx  ← TensorBoard logs
│   ├── scalars.csv               ← Metrics export
│   ├── train_loss.npy            ← Numpy arrays
│   └── validation_psnr.npy

Configuration snapshots:
├── config_snapshot_iter_0.yaml
├── config_snapshot_iter_100000.yaml
└── config_snapshot_best.yaml

Size estimates:
  Model weights:        ~100 MB
  Optimizer state:      ~100 MB (same size as model)
  Full checkpoint:      ~200-300 MB
  5 checkpoints saved:  ~1-1.5 GB
  TensorBoard logs:     ~500 MB
  Total per run:        ~2-3 GB
```

---

**Visual Architecture Reference v1.0**

For implementation details, see: COMPLETE_PROJECT_GUIDE.md
For quick start, see: QUICK_REFERENCE.md
