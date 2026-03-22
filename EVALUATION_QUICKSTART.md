# Evaluation Quick Start Guide

**Fixed the path issue and added Dice + HD95 metrics!**

---

## Problem You Had

```
FileNotFoundError: No such file or no access: 'sub-01/ses-1/anat/...'
```

This happens because the script needs to know **where your data actually lives**.

---

## ✅ Solution: Provide `--data-root`

### Step 1: Find Your Data Root

Use the helper script:

```bash
python scripts/find_data_root.py --pairs_csv pairs_new.csv
```

This will:
- Search for your data files
- Suggest the correct `--data-root`
- Verify that all files exist

**Example output:**
```
✅ Suggested data-root:
   /eos/home-i04/p/ppokhrel/data/topobrain/

📝 Use in your command:
   --data-root '/eos/home-i04/p/ppokhrel/data/topobrain/'
```

### Step 2: Run Evaluation

Use the suggested data-root:

```bash
python scripts/evaluate_full_volume.py \
    --checkpoint output/checkpoint_latest.pt \
    --subject sub-01 \
    --pairs_csv pairs_new.csv \
    --data-root '/eos/home-i04/p/ppokhrel/data/topobrain/' \
    --output_dir results/sub-01/
```

---

## Command-Line Arguments Explained

### Required Arguments

```bash
--checkpoint        # Path to your trained model checkpoint
--subject          # Which subject to evaluate (e.g., sub-01)
--pairs_csv        # Path to pairs_new.csv
--data-root        # Base directory containing your data ⭐ IMPORTANT
```

### Optional Arguments

```bash
--masks-root       # Directory with segmentation masks (if different from data-root)
--output_dir       # Where to save results (default: results/full_volume_eval)
--overlap          # Patch overlap in voxels (default: 32)
```

---

## Full Examples

### Example 1: Basic Evaluation (Most Common)

```bash
python scripts/evaluate_full_volume.py \
    --checkpoint output/checkpoint_75000/checkpoint_75000.pt \
    --subject sub-01 \
    --pairs_csv pairs_new.csv \
    --data-root /eos/home-i04/p/ppokhrel/data/topobrain/ \
    --output_dir results/sub-01_step75k/
```

### Example 2: With Separate Masks Directory

```bash
python scripts/evaluate_full_volume.py \
    --checkpoint output/checkpoint_latest.pt \
    --subject sub-01 \
    --pairs_csv pairs_new.csv \
    --data-root /eos/home-i04/p/ppokhrel/data/topobrain/ \
    --masks-root /eos/home-i04/p/ppokhrel/masks/ \
    --output_dir results/sub-01/
```

### Example 3: Direct File Paths (No CSV)

```bash
python scripts/evaluate_full_volume.py \
    --checkpoint output/checkpoint_latest.pt \
    --input /full/path/to/3T_input.nii.gz \
    --target /full/path/to/7T_target.nii.gz \
    --output_dir results/custom_eval/
```

---

## Expected Output

### Console Output

```
📄 Resolved from CSV: pairs_new.csv
   Subject: sub-01
   Data root: /eos/home-i04/p/ppokhrel/data/topobrain

📂 File paths:
   Input:  /eos/home-i04/p/ppokhrel/data/topobrain/sub-01/ses-1/anat/sub-01_ses-1_desc-preproc_T1w_registered.nii.gz
   Target: /eos/home-i04/p/ppokhrel/data/topobrain/sub-01/ses-2/anat/sub-01_ses-2_desc-preproc_T1w.nii.gz

Device: cuda
Loading checkpoint: output/checkpoint_75000/checkpoint_75000.pt
Using EMA weights
Volume shape: (256, 256, 256)
Voxel spacing: (1.0, 1.0, 1.0) mm

Tiled inference: 100%|████████████████| 512/512 [05:23<00:00, 1.58patch/s]

Computing metrics for FreeSurfer...

================================================================================
METRICS SUMMARY
================================================================================
[FreeSurfer]
  SSIM:      0.9245
  PSNR:      34.56 dB
  Dice:      0.8934      ⬅️ NEW! Volumetric overlap
  HD95:      4.23 mm     ⬅️ NEW! Boundary accuracy

Saved: results/sub-01/predicted_7T.nii.gz
Saved: results/sub-01/predicted_seg.nii.gz
All outputs saved to: results/sub-01/
```

### Files Generated

```
results/sub-01/
├── predicted_7T.nii.gz           # Full-volume prediction
├── predicted_7T_raw.nii.gz       # Raw prediction (before masking)
├── predicted_seg.nii.gz          # Tissue segmentation
├── eval_axial_slice64.png        # Visualization (axial view)
├── eval_coronal_slice128.png     # Visualization (coronal view)
└── eval_sagittal_slice128.png    # Visualization (sagittal view)
```

---

## Troubleshooting

### Error: "No such file or no access"

**Cause:** Incorrect `--data-root`

**Fix:**
```bash
# Find the correct path
python scripts/find_data_root.py --pairs_csv pairs_new.csv

# Then use the suggested path
```

### Error: "CSV file not found"

**Cause:** Wrong path to pairs CSV

**Fix:**
```bash
# Provide absolute path
--pairs_csv /full/path/to/pairs_new.csv

# Or run from project root
cd /path/to/Topo-Brain/
python scripts/evaluate_full_volume.py ...
```

### Error: "subject 'sub-01' not found in CSV"

**Cause:** Subject ID doesn't match CSV

**Fix:**
```bash
# Check available subjects
head -20 pairs_new.csv

# Use exact subject ID from CSV
--subject sub-01  # (not Sub-01 or SUB-01)
```

---

## Batch Evaluation (All Subjects)

Create a simple bash loop:

```bash
#!/bin/bash
# evaluate_all.sh

DATA_ROOT="/eos/home-i04/p/ppokhrel/data/topobrain/"
CHECKPOINT="output/checkpoint_75000/checkpoint_75000.pt"

for subject in sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10; do
    echo "Evaluating $subject..."
    python scripts/evaluate_full_volume.py \
        --checkpoint $CHECKPOINT \
        --subject $subject \
        --pairs_csv pairs_new.csv \
        --data-root $DATA_ROOT \
        --output_dir results/${subject}_eval/
done
```

Run it:
```bash
chmod +x evaluate_all.sh
./evaluate_all.sh
```

---

## Understanding the New Metrics

### Dice Coefficient (0-1, higher is better)

- **>0.90**: Excellent volumetric overlap
- **>0.85**: Good (clinical acceptable)
- **>0.80**: Fair
- **<0.80**: Poor - model may be hallucinating tissue

**What it catches:** Volume errors that SSIM/PSNR miss

### HD95 (mm, lower is better)

- **<3mm**: Excellent boundary accuracy
- **<5mm**: Good (clinical acceptable)
- **<10mm**: Fair
- **>10mm**: Poor - boundaries are blurred

**What it catches:** Edge blur that Dice misses

### Why Both Matter

```
Scenario A: Dice=0.92, HD95=15mm  ⚠️  Volume OK but edges blurry
Scenario B: Dice=0.75, HD95=3mm   ⚠️  Sharp edges but wrong volume
Scenario C: Dice=0.92, HD95=3mm   ✅  Perfect - both volume and edges!
```

**See [docs/METRICS_GUIDE.md](docs/METRICS_GUIDE.md) for complete details**

---

## Quick Reference

```bash
# 1. Find your data location
python scripts/find_data_root.py

# 2. Run evaluation with suggested data-root
python scripts/evaluate_full_volume.py \
    --checkpoint output/checkpoint_latest.pt \
    --subject sub-01 \
    --pairs_csv pairs_new.csv \
    --data-root '<PATH_FROM_STEP_1>' \
    --output_dir results/sub-01/

# 3. Check results
ls results/sub-01/
```

---

## Need Help?

```bash
# Show all available options
python scripts/evaluate_full_volume.py --help

# Find your data location
python scripts/find_data_root.py --pairs_csv pairs_new.csv

# Verify a specific data-root
python scripts/find_data_root.py --pairs_csv pairs_new.csv --verify /path/to/test/
```

---

**The scripts now have better error messages and will guide you if something is wrong!** ✨
