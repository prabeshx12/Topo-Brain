# Google Colab Setup Guide

## ✅ Ready to Use!

The notebook has been corrected and is ready for Google Colab training.

## 📁 Files Cleaned Up

**Removed:**
- ❌ `gan_training_notebook.ipynb` (duplicate in root)
- ❌ `new/` folder (contained old notebooks and 5GB+ of tar archives)
- ❌ All `__pycache__/` directories

**Current Structure:**
```
major_/
├── notebooks/
│   ├── colab_training.ipynb         ← USE THIS for Google Colab
│   └── interactive_pipeline.ipynb   ← Local Jupyter notebook
├── scripts/                          ← Training scripts
├── src/                              ← Core modules
├── models/                           ← Model architectures
├── HD-BET/                           ← Skull stripping tool
├── requirements.txt                  ← Dependencies
├── QUICKSTART.md                     ← Local training guide
└── COLAB_SETUP.md                    ← This file
```

## 🔧 Fixed Issues in Notebook

1. **Corrected `discover_dataset()` call:**
   - ❌ Was: `discover_dataset(data_root, modality='T1w', session_pattern='ses-*')`
   - ✅ Now: `discover_dataset(data_root, config.data)`

2. **Verified all imports are correct**
3. **Removed unnecessary code**
4. **All paths use Google Drive correctly**

## 🚀 How to Use

### Step 1: Upload Data to Google Drive
```
MyDrive/
└── Topo-Brain-Data/
    └── Nifti/
        ├── sub-01/
        ├── sub-02/
        └── ...
```

### Step 2: Upload Notebook to Colab
1. Open Google Colab
2. File → Upload Notebook
3. Select: `notebooks/colab_training.ipynb`

### Step 3: Run All Cells
- Just click "Runtime" → "Run all"
- Everything is automated!

## ⏱️ Expected Timeline

| Step | Time | GPU |
|------|------|-----|
| Setup | 10 min | No |
| Preprocessing | 1-2 hours | No |
| Registration | 30-60 min | No |
| Quick Test | 2-3 hours | T4 |
| Full Training | 35-45 hours | T4 |

## ⚠️ Important Notes

1. **Colab Free Tier Limits:**
   - 12-hour session timeout
   - May need to resume training 3-4 times
   - All progress saved to Google Drive

2. **Colab Pro (Recommended):**
   - Longer sessions (24 hours)
   - Priority GPU access
   - Can complete training in 2-3 sessions

3. **GPU Memory:**
   - T4 GPU: 16GB (sufficient)
   - A100 GPU: 40GB (if available, even better)
   - Batch size 2 works on T4

## 📊 What Gets Saved to Google Drive

```
MyDrive/Topo-Brain-Data/
├── Nifti/                           # Your input data
├── preprocessed/                    # After Step 2
├── preprocessed_registered/         # After Step 3
├── checkpoints/                     # Model checkpoints
├── visualizations/                  # Training images
├── logs/                            # TensorBoard logs
├── registration_report.json         # Registration quality
└── training_curves.png              # Training plots
```

## ✅ Verification Checklist

Before starting full training:
- [ ] Data uploaded to correct Google Drive path
- [ ] Notebook runs without errors in setup cells
- [ ] Registration report shows correlation > 0.5
- [ ] Quick test (10 epochs) completes successfully
- [ ] D_loss stays in 0.3-0.7 range (not <0.1)
- [ ] PSNR shows gradual improvement

## 🆘 Common Issues

### "Data not found"
**Fix:** Verify exact path is `MyDrive/Topo-Brain-Data/Nifti/`

### "ModuleNotFoundError: gudhi"
**Fix:** Run the installation cell again, or use: `!conda install -c conda-forge gudhi -y`

### "Session disconnected"
**Fix:** Just re-run all cells. Training will resume from last checkpoint.

### "CUDA out of memory"
**Fix:** Change `--batch-size 2` to `--batch-size 1`

## 📞 Need Help?

1. Check error messages in notebook output
2. Review [QUICKSTART.md](QUICKSTART.md) for detailed troubleshooting
3. Verify registration quality visually
4. Check TensorBoard logs

---

**Everything is ready! Just open the notebook in Colab and hit "Run all"!** 🎉
