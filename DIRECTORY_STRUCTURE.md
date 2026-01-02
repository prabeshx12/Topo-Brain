# 📂 Topo-Brain Project Directory Structure

**Clean, organized structure for 3T→7T MRI super-resolution using GANs**

Last Updated: January 2, 2026

---

## 📁 Root Directory Layout

```
Topo-Brain/
├── 📓 Notebooks (Colab/Jupyter)
│   ├── gan_training_colab_aligned.ipynb    # Main GAN training notebook (Colab)
│   └── preprocessing_pipeline_colab.ipynb  # Preprocessing pipeline (Colab)
│
├── 📦 Core Source Code
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config.py                       # Configuration management
│   │   ├── preprocessing.py                # MRI preprocessing (N4, skull stripping, normalization)
│   │   ├── dataset.py                      # PyTorch dataset classes
│   │   ├── harmonization.py                # Harmonization utilities
│   │   ├── quality_control.py              # QC metrics and visualization
│   │   └── utils.py                        # Utility functions
│   │
│   └── models/
│       ├── __init__.py
│       ├── generator_unet3d.py             # 3D U-Net generator
│       ├── discriminator_patchgan3d.py     # 3D PatchGAN discriminator
│       └── paired_dataset.py               # Paired 3T-7T dataset loader
│
├── 🔧 Scripts
│   ├── scripts/
│   │   ├── train_gan.py                    # GAN training script
│   │   ├── test_gan.py                     # GAN testing/inference
│   │   ├── eval_gan.py                     # GAN evaluation metrics
│   │   ├── example_pipeline.py             # Complete pipeline example
│   │   └── generate_brain_masks.py         # HD-BET brain mask generation
│   │
│   └── notebooks/
│       └── interactive_pipeline.ipynb      # Interactive analysis notebook
│
├── 📊 Data Directories
│   ├── Nifti/                              # Raw BIDS dataset (10 subjects, 3T+7T)
│   │   ├── dataset_description.json
│   │   ├── README.md
│   │   └── sub-0X/
│   │       ├── ses-1/                      # 3T session
│   │       │   └── anat/
│   │       └── ses-2/                      # 7T session
│   │           └── anat/
│   │
│   ├── Aligned/                            # Pre-registered aligned data (256×304×308)
│   │   └── sub-0X/
│   │       ├── ses-1/
│   │       │   └── *_registered.nii.gz
│   │       └── ses-2/
│   │           └── *_registered.nii.gz
│   │
│   ├── preprocessed/                       # Preprocessed data (optional)
│   │   └── sub-0X/
│   │       └── *_preprocessed.nii.gz
│   │
│   └── preprocessed_test/                  # Test preprocessing output
│
├── 🧪 Testing
│   └── tests/
│       └── __init__.py
│
├── 📝 Documentation
│   ├── docs/
│   │   ├── ARCHITECTURE.md                 # System architecture
│   │   ├── GAN_README.md                   # GAN implementation guide
│   │   ├── GAN_IMPLEMENTATION_SUMMARY.md   # Implementation details
│   │   ├── IMPROVEMENTS_IMPLEMENTED.md     # Changelog
│   │   └── CHANGELOG.md                    # Version history
│   │
│   ├── README.md                           # Main project README
│   ├── DIRECTORY_STRUCTURE.md              # This file
│   └── requirements.txt                    # Python dependencies
│
├── 📂 Output Directories
│   ├── cache/                              # Temporary cache files
│   ├── logs/                               # Training logs
│   │   ├── qc/                             # Quality control logs
│   │   └── tensorboard/                    # TensorBoard logs
│   │
│   └── models/                             # Saved model checkpoints
│
├── 🔧 Development Tools
│   ├── HD-BET/                             # Skull stripping tool (submodule)
│   ├── .git/                               # Git version control
│   ├── .gitignore                          # Git ignore rules
│   └── venv/                               # Virtual environment (local)
│
└── ⚙️ Configuration
    └── .vscode/                            # VS Code settings (optional)
```

---

## 📊 Data Flow

```
1. Raw Data (Nifti/)
   ↓
2. Registration (→ Aligned/)           [Already done]
   ↓
3. Preprocessing (→ preprocessed/)     [Optional: N4, skull stripping, normalization]
   ↓
4. Training (GAN models)               [gan_training_colab_aligned.ipynb]
   ↓
5. Inference & Evaluation              [test_gan.py, eval_gan.py]
   ↓
6. Results & Visualizations            [logs/, cache/]
```

---

## 🎯 Key Files for Users

### **For Training:**
- **Main Notebook:** `gan_training_colab_aligned.ipynb`
  - Ready-to-use Colab notebook
  - Works with Aligned dataset (pre-registered)
  - On-the-fly normalization included

### **For Preprocessing:**
- **Preprocessing Notebook:** `preprocessing_pipeline_colab.ipynb`
  - HD-BET skull stripping
  - Intensity normalization
  - Quality control visualizations
  - Saves to Google Drive

### **For Development:**
- **Training Script:** `scripts/train_gan.py`
- **Config:** `src/config.py`
- **Models:** `models/generator_unet3d.py`, `models/discriminator_patchgan3d.py`

---

## 📦 Data Specifications

### **Aligned Dataset** (Recommended for Training)
- **Location:** `Aligned/`
- **Format:** NIfTI (.nii.gz)
- **Dimensions:** 256 × 304 × 308 (all images)
- **Voxel Size:** ~0.65mm isotropic
- **Naming:** `*_defaced_registered.nii.gz`
- **Modalities:** T1w, T2w
- **Sessions:** ses-1 (3T), ses-2 (7T)
- **Status:** ✅ Already registered (spatial alignment done)

### **Preprocessed Dataset** (Optional)
- **Location:** `preprocessed/` or `Aligned_Preprocessed/`
- **Additional Processing:**
  - ❌ N4 Bias Correction (disabled by default)
  - ✅ Skull Stripping (HD-BET)
  - ✅ Z-score Normalization
- **Naming:** `*_preprocessed.nii.gz`

---

## 🗑️ Removed Files (Cleanup)

These files were removed as they're no longer needed:

### **Diagnostic Scripts:**
- ❌ `check_image_sizes.py` → Replaced by preprocessing notebook
- ❌ `check_registered_sizes.py` → Replaced by preprocessing notebook
- ❌ `check_aligned_sizes.py` → Replaced by preprocessing notebook
- ❌ `visualize_aligned_images.py` → Integrated into preprocessing notebook

### **Old Documentation:**
- ❌ `QUICKSTART_GAN_FIXES.md` → Superseded by updated README
- ❌ `REORGANIZATION_SUMMARY.md` → Superseded by this file

### **Deprecated Notebooks:**
- ❌ `gan_training_notebook.ipynb` → Replaced by `gan_training_colab_aligned.ipynb`

### **Unused Modules:**
- ❌ `src/registration.py` → Registration already done in Aligned dataset

### **Generated Outputs:**
- ❌ `aligned_images_visualization.png` → Temporary diagnostic output
- ❌ `aligned_images_histograms.png` → Temporary diagnostic output

---

## 🚀 Quick Start Guide

### **1. For Google Colab Users:**

```python
# Clone the repository
!git clone https://github.com/prabeshx12/Topo-Brain.git
%cd Topo-Brain

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Open gan_training_colab_aligned.ipynb
# Update data path to your Google Drive location
# Run all cells
```

### **2. For Local Development:**

```bash
# Clone repository
git clone https://github.com/prabeshx12/Topo-Brain.git
cd Topo-Brain

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run training
python scripts/train_gan.py
```

---

## 📝 Notes

### **Data Not Included in Repository:**
- `Nifti/` → Raw BIDS dataset (too large for git)
- `Aligned/` → Aligned dataset (too large for git)
- `preprocessed/` → Preprocessed data (generated)
- `cache/`, `logs/`, `models/` → Output directories (generated)

### **Google Drive Structure for Colab:**
```
MyDrive/
└── Topo-Brain/
    ├── Aligned/                    # Upload this
    ├── Aligned_Preprocessed/       # Generated by preprocessing notebook
    └── (Notebooks clone to /content/)
```

### **Version Control:**
- `.gitignore` configured to exclude data directories
- Only source code, notebooks, and docs are tracked
- HD-BET can be cloned automatically by notebooks

---

## 🔄 Maintenance

### **Adding New Features:**
1. Create feature branch
2. Add code to appropriate directory
3. Update tests in `tests/`
4. Update documentation
5. Create pull request

### **Updating Documentation:**
- **Architecture changes:** Update `docs/ARCHITECTURE.md`
- **API changes:** Update `docs/GAN_README.md`
- **Directory structure:** Update this file
- **Changelog:** Add to `docs/CHANGELOG.md`

---

## 📞 Support

For issues or questions:
- **GitHub Issues:** https://github.com/prabeshx12/Topo-Brain/issues
- **Documentation:** See `docs/` folder
- **Examples:** See notebooks and `scripts/example_pipeline.py`

---

**Last Updated:** January 2, 2026  
**Project:** Topo-Brain MRI Super-Resolution  
**Status:** Active Development ✅
