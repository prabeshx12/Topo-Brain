# Topo-Brain: Road to Publication

**Paper framing:** Topology-Preserving Conditional Diffusion for Anatomically Faithful 3T-to-7T MRI Synthesis
**Target venues:** MICCAI Workshop (50-60%), MIDL 2027 (40-50%), IEEE JBHI (30-40%)

---

## Current Results (sub-06, 145k checkpoint, DDPM)

| Metric | Value | Target |
|--------|-------|--------|
| SSIM   | 0.895 | >0.90  |
| PSNR   | 19.84 | >25    |
| Dice   | 0.761 | >0.85  |
| HD95   | 3.31  | <5 mm  |

### Topology Results (sub-06, predicted_seg vs ground truth seg)

| Tissue | Dice  | Vol Diff | CC Predicted | CC Ground Truth |
|--------|-------|----------|-------------|----------------|
| WM     | 0.896 | +4.2%    | 1038        | 34             |
| GM     | 0.850 | +12.7%   | 1658        | 13             |
| CSF    | 0.762 | +36.5%   | 126         | 5              |
| Brain  | 0.944 | —        | —           | —              |
| HD95   | 2.05 mm | —     | —           | —              |

**Key finding:** Dice is high (0.85-0.89) but model over-fragments tissue —
predicts 1658 GM components vs 13 in ground truth. This fragmentation is the
core problem topology loss addresses. Ablation (without topo loss) should show
WORSE fragmentation — that is the paper's central proof.

---

## TIER 1 — Must Do (non-negotiable for any venue)

### 1. LOOCV (10-fold cross-validation)
- [ ] Set up 10 training runs, each leaving one subject out for testing
- [ ] Use 100k iterations per fold (sufficient based on convergence analysis)
- [ ] Report mean +/- std for SSIM, PSNR, Dice, HD95 across all 10 folds
- [ ] N=1 test subject is instant reject at any venue — this is priority #1
- **GPU time estimate:** 10 folds x ~24h each = ~10 days (parallelizable)

### 2. Ablation Study (with vs without topology loss)
- [ ] For each LOOCV fold, train a second model with `lambda_topo: 0.0`
- [ ] Compare metrics: full model vs ablated model
- [ ] Compare Betti numbers: full model vs ablated model
- [ ] This is the core evidence that topology loss works
- **Note:** Also serves as M4 baseline for AD classifier experiment later
- **GPU time estimate:** 10 more folds x ~24h = ~10 days (run alongside LOOCV)

### 3. Topology Verification (Betti numbers)
- [ ] Run `scripts/verify_topology.py` on all LOOCV test outputs
- [ ] Report beta0, beta1, beta2, Euler characteristic for synthetic vs real 7T
- [ ] Show that topology-aware model has closer Betti numbers to real 7T than ablated model
- [ ] Multi-threshold analysis across [-0.5, -0.25, 0.0, 0.25, 0.5]

### 4. FastSurfer Tissue Segmentation
- [ ] Get FreeSurfer license (free: https://surfer.nmr.mgh.harvard.edu/registration.html)
- [ ] Run FastSurfer on synthetic 7T and real 7T for all test subjects
- [ ] Compute tissue-class Dice (GM, WM, CSF separately) — replaces crude binary Dice
- [ ] Extract hippocampal volumes — compare synthetic vs real
- [ ] This is clinically meaningful, reviewers expect it

---

## TIER 2 — Strongly Recommended (for better venues)

### 5. Baseline Comparison
- [ ] Train a simple 3D U-Net regression (L1 loss, no diffusion, same encoder)
- [ ] Same LOOCV folds for fair comparison
- [ ] Shows that diffusion + topology adds value over plain regression
- **GPU time estimate:** 10 folds x ~12h = ~5 days

### 6. MRIQC Image Quality Metrics
- [ ] Run `scripts/compute_iqms.py` on all LOOCV outputs (3T, synthetic 7T, real 7T)
- [ ] Report CJV, CNR, SNR, FBER, EFC
- [ ] Show synthetic 7T quality is closer to real 7T than 3T input
- **No container needed — pure Python script**

### 7. Improve PSNR Investigation
- [ ] Check if evaluating only within brain mask (exclude background) changes PSNR
- [ ] Check normalization: verify data range is truly [-1, 1] end-to-end
- [ ] Try `--sampler ddim --ddim-steps 50` (already implemented, tested, marginal improvement)
- [ ] Consider: is 3T-to-7T registration quality limiting PSNR?

---

## TIER 3 — Nice to Have (strengthens paper)

### 8. Visual Quality Figures
- [ ] Multi-slice comparisons (axial/coronal/sagittal) at hippocampus level
- [ ] Difference maps (synthetic - real) to show where errors concentrate
- [ ] Comparison grid: 3T input | Synthetic 7T | Real 7T | Ablation output

### 9. Perceptual Metrics
- [ ] FID between synthetic and real 7T distributions
- [ ] LPIPS (learned perceptual similarity)

### 10. Blinded Reader Study
- [ ] Run `scripts/blinded_reader_prep.py` to generate panels
- [ ] Have 1-2 people score anatomical correctness without knowing which is synthetic
- [ ] Even informal results add credibility

---

## FUTURE WORK (separate paper or extension)

### 11. AD Classifiers on ADNI
- [ ] Download ADNI 3T T1w scans (AD vs CN, 100-200 subjects)
- [ ] Preprocess to match training format
- [ ] Run synthesis model on all ADNI subjects
- [ ] Train 4 matched classifiers (3T-only, synth-7T, fusion, ablation baseline)
- [ ] Compare AUC across conditions
- **This is a full paper's worth of additional work**

### 12. External Validation
- [ ] Test on data from a different scanner/site
- [ ] Shows generalizability beyond the training distribution

---

## Execution Timeline

```
Week 1-2:  LOOCV training (10 folds) + ablation training (10 folds)
           Run topology verification on current sub-06 results
           Get FreeSurfer license
Week 3:    Run FastSurfer on all LOOCV outputs
           Run compute_iqms.py on all outputs
           Train U-Net baseline (10 folds)
Week 4:    Aggregate all results into tables
           Generate visualization figures
           Run blinded reader prep
Week 5-6:  Write paper
           Target: MICCAI workshop or MIDL submission
```

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train_diffusion.py` | Training (use `--config` to set lambda_topo=0 for ablation) |
| `scripts/evaluate_full_volume.py` | Full-volume eval with SSIM/PSNR/Dice/HD95 |
| `scripts/verify_topology.py` | Betti numbers and Euler characteristic |
| `scripts/compute_iqms.py` | IQMs without MRIQC container |
| `scripts/blinded_reader_prep.py` | Generate comparison panels |
| `scripts/aggregate_metrics.py` | Aggregate results across subjects |

---

## Key Decisions Made

- **Test subject for single-fold:** sub-06 (val_fold=0, the only truly held-out subject with seed=42)
- **Sampler:** DDPM (DDIM tested, marginal difference — model is the bottleneck, not sampler)
- **Paper title drops "Alzheimer's"** unless AD classifier results are obtained
- **CERN paths:** All under `/eos/user/p/ppokhrel/Untitled Folder 1/`
- **Apptainer on CERN:** Must run from lxplus terminal, not SWAN
- **MRIQC cgroup fix:** `--bind /dev/null:/proc/1/cgroup` — but blocked on CERN, use compute_iqms.py instead
