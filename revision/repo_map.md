# Repo Map — TopoBrain (verified against code)

End-to-end pipeline, components, and where each manuscript claim lives in code. All entries
verified by reading the implementation (Phase 0 audit). Discrepancies are flagged → see
[discrepancies.md](discrepancies.md).

## 1. Data & split
- **Paired cohort:** 10 UNC subjects (chen2023), T1w (T2w available, unused), 3T=ses-1 / 7T=ses-2.
  7T FreeSurfer aparc+aseg → 4-class tissue GT (BG/CSF/GM/WM) via [scripts/preprocess_masks.py].
- **Loading:** [src/synthesis_dataset.py] — patch-based, **64³**, **32 patches/subject**,
  ≥10% brain/patch; manifest `pairs_new.csv` (subject, input_3t, target_7t, seg).
- **Normalization:** [-1,1] via 0.5/99.5-pct clip (diffusion convention), enforced at load.
- **Split:** seed=42 LOOCV shuffle → **sub-06 held out (n=1)**, other 9 train. The 10-fold
  splitter's `test_fold` is silently ignored ([synthesis_dataset.py:762]) → **D6**.

## 2. Preprocessing
[src/preprocessing.py], [src/preprocess_pipeline.py]: HD-BET skull-strip → N4 bias correction →
rigid 3T↔7T registration → [-1,1] normalization → 1 mm iso, RAS. (Never modify raw data dirs.)

## 3. Model — `AnatomyGuidedUNet` ([src/model.py:92])
3D U-Net, features (32,64,128,256), GroupNorm-8, SiLU, strided-conv down / transpose-conv up,
concat skips. **3T conditioning by channel-concat** (noisy-7T ⊕ 3T = 2 in-ch). Sinusoidal time
embedding into every ResBlock. Bottleneck multi-head **self-attention** (use_attention=True).
**Dual decoders:** noise head (1-ch) + tissue-seg head (4-ch). `forward` is a placeholder; real
path is `forward_full` ([model.py:215]) — aliased at import.
- **Params: 13.98 M** (computed, [scripts/count_params.py]), **not 22.4 M** → **D2**.

## 4. Diffusion ([src/diffusion.py])
`GaussianDiffusion`: cosine schedule, **T=200, last 15 excluded → 185 effective** ([diffusion.py:339]) → **D5**.
Objective pred_noise, L1. Sampling: DDPM `p_sample_loop` (185 steps) or `ddim_sample` (50 steps).

## 5. Losses ([src/diffusion.py:329-416], weights in [configs/train_diffusion.yaml])
Total = `loss_diff`(L1 noise, w=1.0) + `lambda_pixel`·`loss_pixel`(L1 on x̂₀, =1.0) +
`lambda_percep`·VGG(=0.25) + `lambda_topo`·topo(=0.2). **4 terms; paper Eq.1 shows 3 → D3.**
- **VGG perceptual** ([diffusion.py:49-102]): VGG16/ImageNet layers 0-15, every 8th slice,
  gray→3ch, ImageNet-norm, MSE on features, clamp 10. → R1.1.
- **Topology** ([src/topology_loss.py]): `MultiScaleTopologyLoss` (scales 1.0/0.5/0.25) of
  `EdgeAwareTopologyLoss` = edge-weighted CE (edge_weight 2.0) + 0.5·(Dice on Sobel edges).
  **Not** the gradient-CE + volumetric-Dice of paper Eq.2 → **D1**. No persistent homology.
- **Curriculum** ([scripts/train_diffusion.py:~369-412]): stage1 ≤3k (pixel only), stage2 ≤12k
  (full pixel), stage3 ≤30k (percep warmup 8k), stage4 ≤150k (topo warmup 25k).

## 6. Training ([scripts/train_diffusion.py], [configs/train_diffusion.yaml])
AdamW lr 1e-4 (→2e-5 @120k), wd 1e-4, batch 8, grad-clip 1.0, AMP, EMA from 2k decay 0.9999,
150k iters, save every 5k. **~1 s/step on RTX 4090 → ~40 h/run** ([TRAINING_REPORT.md:556]).
Selected checkpoint: **105k EMA**. ⚠️ checkpoint not on local disk — on Kaggle (needs restoring
before any inference experiment).

## 7. Evaluation ([src/metrics.py], [scripts/evaluate_full_volume.py])
SSIM, PSNR, Dice (fg + per-tissue), **HD95** ([metrics.py:125-192], surface distance), ASSD,
per-tissue CC + largest-CC fraction, tissue-volume error, per-region GM coverage.
- **True topology:** [scripts/compute_betti.py] computes β0/β1/β2 via gudhi — **present, not run
  on outputs** → R2.3/R2.8 win once checkpoint restored.

## 8. Downstream AD ([scripts/] — ADNI 3T, unpaired, no 7T GT)
ADNI 3T → synth seg → tissue-volume features → composite biomarker score.
- [build_adni_cohort.py]: 266 filtered (107 AD/159 CN) → 1:1 match 103+103; **only 30 (15/15)
  processed** ("smoke" cohort).
- [baseline_3t_vs_synth_features.py]: **C1 = unsupervised K-means(k=3) on 3T** vs
  **C2 = supervised synth seg** features. → 0.818 vs 0.658.
- [compare_c1_c2_risk_score.py] / [ad_risk_score.py]: NIA-AA composite z-score (CN reference),
  Mann-Whitney AUC. **In-sample, no CV** in original.
- [scripts/downstream_fair_recompute.py] (NEW): LOSO reference + bootstrap CI + DeLong →
  C1 0.809 [0.64,0.94], C2 0.653 [0.44,0.84], **DeLong p=0.073 (n.s.)**. → R1.5/R2.4.
- [train_ad_classifier.py]: supervised LR/RF, 5-fold CV (~0.60); [compare_classifiers.py]: DeLong.
- Surviving artifacts in [adni_smoke_analysis_30/]: feature CSVs + figures (volumes/checkpoint pruned).

## 9. Figures / manuscript
[scripts/thesis_figures.py], [paper_iet/generate_graphical_abstract.py]; manuscript in
[paper_iet/paper.tex]; refs [paper_iet/references.bib].

## New scripts added this revision
- [scripts/downstream_fair_recompute.py] — fair C1/C2 AUC (LOSO + CI + DeLong). ✅ run.
- [scripts/count_params.py] — settles the param count. ✅ run (13.98 M).
