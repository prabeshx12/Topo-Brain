# 3T → 7T MRI Super-Resolution GAN

Baseline GAN implementation for supervised 3T to 7T brain MRI enhancement.

## Architecture

### Generator: 3D U-Net
- **Input**: 3T MRI patch (1, D, H, W)
- **Output**: Generated 7T MRI patch (1, D, H, W)
- **Features**:
  - Encoder-decoder with skip connections
  - 3D convolutions throughout
  - InstanceNorm for stability (better than BatchNorm for small batches)
  - LeakyReLU activations
  - Preserves spatial dimensions

### Discriminator: 3D PatchGAN
- **Input**: Real or fake 7T MRI patch
- **Output**: Spatial map of real/fake predictions
- **Features**:
  - Fully convolutional
  - Operates on local 3D patches
  - Strided convolutions for downsampling
  - InstanceNorm (except first layer)

## Loss Functions

### Generator Loss
```
L_G = λ_L1 * L1(fake_7T, real_7T) + λ_adv * L_adv(D(fake_7T), real)
```

Where:
- **L1 Loss**: Pixel-wise reconstruction loss
- **Adversarial Loss**: LSGAN or BCE loss to fool discriminator
- **λ_L1**: Weight for L1 loss (default: 100.0)
- **λ_adv**: Weight for adversarial loss (default: 1.0)

### Discriminator Loss
```
L_D = 0.5 * [L_adv(D(real_7T), real) + L_adv(D(fake_7T), fake)]
```

## Training

### Quick Start

```bash
# Basic training (CPU/GPU auto-detected)
python train_gan.py

# With custom parameters
python train_gan.py \
    --epochs 100 \
    --batch_size 2 \
    --lr_g 2e-4 \
    --lr_d 2e-4 \
    --lambda_l1 100.0 \
    --patch_size 64 \
    --num_patches 10 \
    --modality T1w

# With mixed precision (requires GPU with Tensor Cores)
python train_gan.py --use_amp

# Force CPU
python train_gan.py --device cpu
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | int | 100 | Number of training epochs |
| `--batch_size` | int | 2 | Batch size (keep small for 3D) |
| `--lr_g` | float | 2e-4 | Generator learning rate |
| `--lr_d` | float | 2e-4 | Discriminator learning rate |
| `--lambda_l1` | float | 100.0 | L1 loss weight |
| `--patch_size` | int | 64 | Size of 3D patches (D=H=W) |
| `--num_patches` | int | 10 | Patches per volume |
| `--use_amp` | flag | False | Enable mixed precision |
| `--modality` | str | T1w | T1w or T2w |
| `--device` | str | cuda | cuda or cpu |
| `--seed` | int | 42 | Random seed |

## Data Requirements

### Preprocessing
The training script assumes **preprocessed** data exists. Run preprocessing first:

```bash
python example_pipeline.py --preprocess --config default
```

### Data Structure
The script expects:
- **3T scans**: `ses-1` (session 1)
- **7T scans**: `ses-2` (session 2)
- **Same subjects** in both sessions (paired data)
- **Preprocessed** (registered, normalized, skull-stripped)

### Patient-Level Splitting
Data is automatically split at the **patient level**:
- Train: 60% of subjects (6 subjects)
- Val: 20% of subjects (2 subjects)
- Test: 20% of subjects (2 subjects)

**No data leakage** - all sessions from same subject stay in same split.

## Patch-Based Training

### Why Patches?
- **Memory**: Full 3D volumes too large for GPU
- **Augmentation**: More training samples from limited data
- **Local features**: PatchGAN operates on local structure

### Patch Extraction
- Random patches extracted from each volume
- Same location used for 3T and 7T (paired patches)
- Deterministic sampling for reproducibility
- Configurable patch size (default: 64³)

## Outputs

### Checkpoints
Saved to `checkpoints/`:
- `checkpoint_epoch_XXX.pth` - Regular checkpoints (every 5 epochs)
- `best_model.pth` - Best model based on validation L1 loss

Each checkpoint contains:
- Generator and discriminator weights
- Optimizer states
- Training history
- Configuration

### Visualizations
Saved to `visualizations/`:
- `epoch_XXX.png` - Side-by-side comparison of:
  - Input 3T (left)
  - Generated 7T (middle)
  - Real 7T ground truth (right)

### Logs
Saved to `logs/`:
- Training progress
- Loss curves
- Validation metrics

## Training Progress

### Console Output
```
Epoch [1/100] Iter [10/50] G_loss: 0.8234 (L1: 0.0078, Adv: 0.3456) D_loss: 0.4567 (Real: 0.2234, Fake: 0.2333)
...
Validation - L1: 0.0065, PSNR: 28.45 dB
Saved checkpoint: checkpoints/checkpoint_epoch_005.pth
Saved visualization: visualizations/epoch_005.png
```

### Metrics Tracked
- **Generator**:
  - Total loss (L1 + adversarial)
  - L1 reconstruction loss
  - Adversarial loss
- **Discriminator**:
  - Total loss
  - Real prediction loss
  - Fake prediction loss
- **Validation**:
  - L1 loss
  - PSNR (Peak Signal-to-Noise Ratio)

## Stability Features

### Gradient Clipping
- Clips gradients to prevent explosion
- Default: 1.0 (configurable)

### InstanceNorm
- More stable than BatchNorm for small batches
- Normalizes per-instance rather than per-batch

### LSGAN Loss
- Least Squares GAN (default)
- More stable than original GAN loss
- Alternative: BCE loss (can change via config)

### Mixed Precision (Optional)
- Faster training on modern GPUs
- Reduces memory usage
- Uses PyTorch AMP (Automatic Mixed Precision)

## Model Architecture Details

### Generator (3D U-Net)
```python
UNet3DGenerator(
    in_channels=1,        # Grayscale MRI
    out_channels=1,       # Grayscale output
    base_features=32,     # Base number of features
    num_levels=4,         # Encoder/decoder depth
    norm_type="instance", # Normalization type
)
```

**Parameters**: ~1-5M (depends on config)

**Receptive field**: Large (covers significant context)

### Discriminator (PatchGAN)
```python
PatchGANDiscriminator3D(
    in_channels=1,
    base_features=64,
    num_layers=3,
    norm_type="instance",
)
```

**Parameters**: ~500K-2M (depends on config)

**Output**: Spatial map (not single scalar)

## Augmentation

### Spatial Augmentations
Applied to **both** 3T and 7T patches (preserves correspondence):
- Random flip (left-right)
- Random 90° rotation
- Small affine transformations (rotation + translation)

### Intensity Augmentations
Applied **only** to 3T input (preserves 7T ground truth):
- Random intensity scaling
- Random intensity shifting
- Random Gaussian noise

### Configuration
```python
gan_config.use_augmentation = True
gan_config.augmentation_prob = 0.5  # 50% chance per augmentation
```

## Testing the Models

### Test Generator
```bash
python models/generator_unet3d.py
```

Expected output:
```
Testing UNet3DGenerator...
Total parameters: 1,234,567
Input shape: torch.Size([2, 1, 64, 64, 64])
Output shape: torch.Size([2, 1, 64, 64, 64])
✓ Generator test passed!
```

### Test Discriminator
```bash
python models/discriminator_patchgan3d.py
```

Expected output:
```
Testing PatchGANDiscriminator3D...
Total parameters: 987,654
Input shape: torch.Size([2, 1, 64, 64, 64])
Output shape: torch.Size([2, 1, 8, 8, 8])
✓ Discriminator test passed!
```

## Limitations & Future Work

### Current Limitations
- **No topology preservation** - This is a baseline
- **No diffusion** - Classical GAN only
- **Limited data** - Only 10 subjects
- **Patch-based** - May miss global context

### Potential Extensions
1. **Topology loss** - Preserve anatomical structure
2. **Diffusion models** - Alternative to GAN
3. **Perceptual loss** - Use pretrained features
4. **Progressive training** - Start with small patches
5. **Ensemble** - Multiple generators
6. **Self-supervised pretraining** - Use unlabeled data

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch_size` (try 1)
- Reduce `--patch_size` (try 32 or 48)
- Enable `--use_amp` (if GPU supports it)

### Training Instability
- Reduce learning rates (`--lr_g 1e-4 --lr_d 1e-4`)
- Increase L1 weight (`--lambda_l1 200.0`)
- Check data normalization

### Poor Results
- Train longer (`--epochs 200`)
- Increase patch size (`--patch_size 96`)
- Check data quality (preprocessing)
- Verify 3T-7T registration

### No GPU Detected
- Check CUDA installation: `torch.cuda.is_available()`
- Force CPU: `--device cpu`
- Install GPU PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

## Citation

If you use this code, please cite:

```bibtex
@misc{3t7t_gan_baseline,
  title={Baseline GAN for 3T to 7T Brain MRI Super-Resolution},
  author={Your Name},
  year={2024},
}
```

## License

Research prototype - use for academic purposes.

---

**Questions?** Check the main [README.md](../README.md) or examine the code comments.
