# QUICK START GUIDE: Topo-Brain 3T→7T MRI Enhancement

**Ready to train in 5 steps!** 🚀

---

## 📋 PRE-FLIGHT CHECKLIST

- [ ] Python 3.8+ installed
- [ ] CUDA-capable GPU (8GB+ VRAM recommended)
- [ ] 50GB+ free disk space
- [ ] Dataset with paired 3T-7T MRI scans

---

## ⚡ 5-MINUTE SETUP

### Step 1: Install Environment

```bash
# Navigate to project
cd d:\11PrabeshX\Projects\major_

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install all dependencies
pip install -r requirements.txt

# Install topology library (REQUIRED)
pip install gudhi
# OR if above fails:
# conda install -c conda-forge gudhi

# Install HD-BET for skull stripping
cd HD-BET
pip install -e .
cd ..
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import gudhi; print('✓ GUDHI installed')"
```

---

## 🎯 CRITICAL FIRST STEP: REGISTRATION

**⚠️ DO NOT SKIP THIS!** Your training will fail without proper registration.

```bash
# Register all 3T-7T pairs
python scripts/register_dataset.py \
    --input preprocessed \
    --output preprocessed_registered \
    --registration-type rigid

# This will:
# 1. Align each 3T volume to its corresponding 7T
# 2. Validate alignment quality
# 3. Create checkerboard visualizations
# 4. Generate registration_report.json

# Check the report:
# - Good: correlation > 0.7
# - Review: < 0.7
```

**Inspect Results:**
1. Open `preprocessed_registered/registration_report.json`
2. Check correlation values for all subjects
3. Visually inspect checkerboard images
4. If any subject has correlation < 0.5, investigate or exclude

---

## 🚀 TRAINING PIPELINE

### Option A: Quick Test (Recommended First)

Start with a short run to verify everything works:

```bash
# Test Stage 1 only (10 epochs)
python scripts/train_gan_enhanced.py \
    --epochs 10 \
    --batch-size 2 \
    --lr-g 2e-4 \
    --lr-d 5e-5 \
    --data-root preprocessed_registered \
    --checkpoint-dir checkpoints_test

# Expected output:
# - Training starts without errors
# - D_loss stays around 0.3-0.7 (NOT near 0!)
# - PSNR gradually increases
# - Visualizations saved every epoch
```

**What to watch for:**
- ✅ D_loss in range [0.2, 0.8] - balanced training
- ❌ D_loss < 0.1 - discriminator winning (check your registration!)
- ❌ D_loss > 1.5 - generator winning (very rare)
- ✅ PSNR > 15 dB after 10 epochs
- ✅ No NaN/Inf errors

### Option B: Full Training (Production)

Once test works, run full pipeline:

```bash
# Full 3-stage training
python scripts/run_full_pipeline.py \
    --mode train \
    --data-root preprocessed_registered \
    --checkpoint-dir checkpoints_full \
    --epochs-stage1 100 \
    --epochs-stage2 50 \
    --epochs-stage3 100 \
    --batch-size 2

# Estimated time on T4 GPU: 90-130 hours
```

### Option C: Stage-by-Stage (More Control)

```bash
# Stage 1: Enhanced GAN (most critical)
python scripts/train_gan_enhanced.py \
    --epochs 100 \
    --batch-size 2 \
    --lr-g 2e-4 \
    --lr-d 5e-5 \
    --data-root preprocessed_registered \
    --checkpoint-dir checkpoints_stage1

# Wait for completion, check best PSNR

# Stage 2: PH Refiner
python scripts/run_full_pipeline.py \
    --mode train \
    --stage 2 \
    --epochs-stage2 50 \
    --checkpoint-stage1 checkpoints_stage1/best_model.pth

# Stage 3: Latent Diffusion
python scripts/run_full_pipeline.py \
    --mode train \
    --stage 3 \
    --epochs-stage3 100 \
    --checkpoint-stage2 checkpoints_stage2/ph_corrector.pth
```

---

## 📊 MONITORING TRAINING

### TensorBoard (Real-time)
```bash
tensorboard --logdir logs_enhanced
# Open http://localhost:6006
```

### Key Metrics to Monitor

**Stage 1 (GAN):**
| Metric | Target Range | Concern If... |
|--------|--------------|---------------|
| D_loss | 0.3 - 0.7 | <0.1 or >1.5 |
| G_loss | 10 - 50 | Diverging |
| Val PSNR | 20+ (early), 25-30 (final) | <15 dB |
| L1 Loss | Decreasing | Plateaus early |

**Stage 2 (PH Refiner):**
| Metric | Expected |
|--------|----------|
| Refinement Loss | Decreasing |
| Topology improvement | >0 (positive) |

**Stage 3 (Latent Diffusion):**
| Metric | Expected |
|--------|----------|
| VAE Reconstruction | <0.1 L1 loss |
| Diffusion Loss | Stable, gradually decreasing |

---

## ⚠️ TROUBLESHOOTING

### Problem: "ModuleNotFoundError: No module named 'gudhi'"
```bash
pip install gudhi
# If fails, try:
conda install -c conda-forge gudhi
```

### Problem: "CUDA out of memory"
**Solutions:**
1. Reduce batch size: `--batch-size 1`
2. Reduce patch size in config: `patch_size = (48, 48, 48)`
3. Enable gradient checkpointing (advanced)

### Problem: D_loss drops to near 0 immediately
**Causes:**
1. ❌ Data not registered (most common!)
2. ❌ Bad data (mismatched 3T-7T pairs)

**Fix:**
1. Re-run registration: `python scripts/register_dataset.py`
2. Check correlation in registration_report.json
3. Verify data alignment manually

### Problem: PSNR stuck at ~10 dB
**Causes:**
1. Data misalignment
2. Wrong intensity normalization
3. Learning rate too high/low

**Debug:**
```bash
# Check a few samples visually
python -c "
import nibabel as nib
import matplotlib.pyplot as plt

# Load registered pair
img_3t = nib.load('preprocessed_registered/sub-01/ses-1/anat/sub-01_ses-1_T1w_registered.nii.gz')
img_7t = nib.load('preprocessed_registered/sub-01/ses-2/anat/sub-01_ses-2_T1w_preprocessed.nii.gz')

data_3t = img_3t.get_fdata()
data_7t = img_7t.get_fdata()

# Check shapes match
print(f'3T shape: {data_3t.shape}')
print(f'7T shape: {data_7t.shape}')
assert data_3t.shape == data_7t.shape, 'Shape mismatch!'

# Visual check - middle slice
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(data_3t[data_3t.shape[0]//2], cmap='gray')
plt.title('3T')
plt.subplot(1, 2, 2)
plt.imshow(data_7t[data_7t.shape[0]//2], cmap='gray')
plt.title('7T')
plt.savefig('alignment_check.png')
print('Saved alignment_check.png - inspect visually!')
"
```

### Problem: Training very slow
**Optimizations:**
1. Enable AMP (mixed precision): `config.use_amp = True`
2. Reduce topology loss frequency: `topology_frequency = 20`
3. Use fewer slices for perceptual loss: `perceptual_num_slices = 3`
4. Check data loading isn't bottleneck: `num_workers = 4`

---

## ✅ SUCCESS INDICATORS

After training, you should see:

**Stage 1 (100 epochs):**
- ✅ Best validation PSNR: 25-30 dB
- ✅ D_loss final: 0.3-0.7 range
- ✅ Visualizations show improved detail over input 3T
- ✅ No checkerboard artifacts

**Stage 2 (+50 epochs):**
- ✅ PSNR improvement: +2-4 dB over Stage 1
- ✅ Topology consistency improved
- ✅ Smoother anatomical boundaries

**Stage 3 (+100 epochs):**
- ✅ Final PSNR: 30-35 dB
- ✅ Visual quality close to real 7T
- ✅ No obvious artifacts

---

## 📁 OUTPUT STRUCTURE

After training, you'll have:

```
checkpoints_full/
├── stage1/
│   ├── best_model.pth              # Best Stage 1 GAN
│   ├── checkpoint_epoch_100.pth    # Final checkpoint
│   └── training_history.json
├── stage2/
│   ├── ph_corrector.pth            # PH refinement model
│   └── training_history.json
├── stage3/
│   ├── vae.pth                     # VAE encoder/decoder
│   ├── diffusion_unet.pth          # Diffusion model
│   └── training_history.json
└── complete_pipeline.pth           # All-in-one checkpoint

visualizations_enhanced/
├── epoch_001/
│   ├── sample_0_input.png
│   ├── sample_0_output.png
│   ├── sample_0_target.png
│   └── ...
└── ...

logs_enhanced/
└── tensorboard logs
```

---

## 🎓 FOR THESIS/PROJECT PRESENTATION

### What You Can Claim:
✅ Implemented complete 3-stage pipeline  
✅ Novel topology-aware architecture using persistent homology  
✅ State-of-the-art latent diffusion refinement  
✅ Fixed critical GAN training issues (discriminator dominance)  
✅ Production-quality modular codebase  

### What You CANNOT Claim (with n=10):
❌ Statistically significant improvement  
❌ Generalization to unseen subjects  
❌ Clinical validation  
❌ State-of-the-art performance claims  

### Honest Limitations to State:
1. "Dataset size (n=10) insufficient for rigorous validation"
2. "Proof-of-concept implementation requiring larger-scale study"
3. "Future work: validation on n≥100 dataset"
4. "Focus: engineering contributions and novel methodology"

---

## 🚨 BEFORE YOU START TRAINING

**CRITICAL CHECKLIST:**
- [ ] Registration complete with all subjects >0.5 correlation
- [ ] Visually inspected at least 5 random checkerboard images
- [ ] Verified shapes match between 3T-7T pairs
- [ ] Tested data loading works (no crashes)
- [ ] GPU has sufficient memory (run quick test)
- [ ] Have time/compute budget (100+ GPU hours)
- [ ] Backed up your preprocessed data

**If all checked, you're ready to train!** 🎉

---

## 📞 HELP & SUPPORT

### Quick Diagnostics

```bash
# Check system
python -c "
import torch
import sys
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Test model creation
python -c "
import sys
sys.path.append('.')
from models.generator_unet3d import UNet3DGenerator
from models.discriminator_patchgan3d import PatchGANDiscriminator3D
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g = UNet3DGenerator().to(device)
d = PatchGANDiscriminator3D(use_spectral_norm=True).to(device)
print(f'✓ Generator params: {sum(p.numel() for p in g.parameters()):,}')
print(f'✓ Discriminator params: {sum(p.numel() for p in d.parameters()):,}')
print('✓ Models created successfully!')
"
```

### Common Issues Database

| Error | Solution |
|-------|----------|
| "RuntimeError: CUDA out of memory" | Reduce batch size or patch size |
| "ImportError: cannot import name 'gudhi'" | `pip install gudhi` or use conda |
| "FileNotFoundError: *.nii.gz" | Check data paths in config.py |
| "ValueError: shapes mismatch" | Re-run registration |
| "RuntimeError: Expected 5D tensor" | Check batch dimensions |

---

## 🎯 REALISTIC TIMELINE

**With n=10 (current):**
- Setup: 1 hour
- Registration: 2 hours
- Quick test (10 epochs): 3-4 hours
- Full training (100 epochs): 30-40 hours GPU time

**With n=100 (recommended):**
- Setup: 1 hour
- Registration: 20 hours
- Full 3-stage training: 90-130 hours GPU time
- Evaluation & analysis: 1 week
- **Total: 2-3 weeks full-time work**

---

## 🎓 THESIS DELIVERABLES

1. **Code Repository:** ✅ Complete and documented
2. **Trained Models:** ⏳ Pending training
3. **Evaluation Results:** ⏳ Pending training
4. **Thesis Chapters:** Use PROJECT_COMPLETION_SUMMARY.md as outline
5. **Presentation Slides:** Focus on architecture diagrams and methodology
6. **Demo:** Can demo inference after Stage 1 training

---

**You're all set! Good luck with your training!** 🚀

**Next Command to Run:**
```bash
python scripts/register_dataset.py --input preprocessed --output preprocessed_registered
```
