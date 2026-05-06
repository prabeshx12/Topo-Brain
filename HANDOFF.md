# Topo-Brain — Session Handoff (2026-05-05)

This file captures active state for picking up in a new Claude Code chat.
**To resume:** open a new chat in this directory and say:
> "Read HANDOFF.md and continue. Next task is [X]."

The auto-memory in `C:\Users\Asus\.claude\projects\d--BCT-pcampus-...\memory\`
loads automatically — this file adds session-specific context on top.

---

## Where we are

- Single-fold sub-06 model trained at 145k iters (DDPM full loss, λ_topo=0.2).
- **Hippocampus preservation verified — AD downstream is viable.**
- ADNI access granted. DICOM download for AD/CN baseline cohort in progress.
- 8-week thesis plan settled.

---

## The headline empirical finding (sub-06 vs ses-2 aparc+aseg)

| Region | CC | Largest | Coverage | Vol Err |
|---|---|---|---|---|
| Hippo L | 6 | 99.8% | 85.2% | -14.8% |
| Hippo R | 2 | 100.0% | 92.2% | -7.8% |
| Amyg L | 1 | 100.0% | 81.8% | -18.2% |
| Amyg R | 1 | 100.0% | 95.4% | -4.6% |
| Thal L | 1 | 100.0% | 99.1% | -0.9% |
| Thal R | 1 | 100.0% | 98.7% | -1.3% |
| Caud L | 1 | 100.0% | 94.3% | -5.7% |
| Caud R | 1 | 100.0% | 95.4% | -4.6% |

**Reading:** subcortical structures are essentially single connected blobs.
The 1658-component global GM count is **cortical-ribbon fragmentation only**.
Systematic ~5-18% under-segmentation across all regions — uniform bias,
acceptable for AD as long as it stays uniform across AD/CN on ADNI.

ses-1 (3T-aseg) comparison was also run as sanity check; numbers are similar
but slightly noisier. **ses-2 is canonical reference everywhere.**

---

## Decisions locked in this session

1. **AD downstream stays in the project.** Hippocampus preservation justifies it.
2. **ses-2 aparc+aseg = canonical reference.** Higher-resolution FreeSurfer
   inputs → more accurate labels. The synthesis target is 7T, so reference is too.
   Verify all subjects' training masks were derived from ses-2, not ses-1.
3. **clDice replacement → future work.** Sobel "topology" loss is acceptable
   for the subcortical AD claim. Retraining all folds with clDice would burn
   1-2 weeks of GPU time we don't have. Acknowledged as limitation.
4. **Drop "super-resolution" framing.** 3T and 7T occupy the same voxel grid
   post-registration. Reframe as "contrast synthesis" / "cross-field-strength
   translation" everywhere in thesis.
5. **Compress LOOCV to 5 folds** if compute is tight: test on sub-02, 04, 06,
   08, 10. Full + ablation per fold.
6. **Drop "topology-preserving" from title** unless clDice replaces Sobel.
   Honest title: "Anatomy-Aware Conditional Diffusion for 3T-to-7T MRI
   Synthesis with Application to Alzheimer's Disease Classification."

---

## Active blocker — ADNI DICOM conversion

ADNI offers raw MPRAGE in DICOM only (no NIfTI option in their UI). Plan:

1. Finish DICOM download via 1-Click Download (DICOM, with metadata).
2. Also download from **DOWNLOAD → Study Data**:
   - `DXSUM_PDXCONV_ADNIALL.csv` (diagnosis labels — DXCURREN/DXCHANGE/DIAGNOSIS)
   - `PTDEMOG.csv` (age, sex, education)
   - `MRILIST.csv` or Mayo QC CSV (drop scans flagged as poor)
3. Batch convert with `dcm2niix -o out_dir -z y -f "${subj}_T1w" scan_dir`.
4. Cohort target: 100 AD + 100 CN baseline 3T MPRAGE, age/sex matched.

**Diagnosis encoding changes between ADNI phases:**
- ADNI1: `DXCURREN` (1=CN, 2=MCI, 3=AD)
- ADNI2/GO: `DXCHANGE`
- ADNI3: `DIAGNOSIS`
Check Data Dictionary at DOWNLOAD → Study Data → Study Info.

---

## Next concrete task

1. Finish DICOM download (running).
2. Run `dcm2niix` batch conversion on all subjects.
3. **Smoke test on 5 AD + 5 CN before committing to full 200:**
   - HD-BET → register → diffusion-normalize (existing pipeline)
   - Synthesis inference using current sub-06 checkpoint
   - Run hippocampus preservation script (see below) on outputs
   - **Decision rule:** if hippo largest > 95% and coverage > 80% on these 10,
     model generalizes to ADNI → invest in full 200. If numbers crash,
     diagnose domain shift before downloading more.

---

## Scripts to write (priority order)

1. **`scripts/check_region_preservation.py`** — formalize the hippocampus
   test from this session. Inputs: predicted_seg path, raw aparc+aseg path.
   Output: CSV row per subject of CC/largest/coverage/vol_err per region.
   Loops over LOOCV test subjects and ADNI subjects.

2. **`scripts/build_adni_cohort.py`** — join converted NIfTIs with DXSUM /
   PTDEMOG / MRILIST CSVs to produce clean `pairs_adni.csv` matching existing
   pipeline format. Filter for QC pass, baseline visit, age/sex matching.

3. **`scripts/preprocess_adni.py`** — HD-BET + register + diffusion-normalize
   ADNI batch. Reuses `src/preprocessing.py` infra.

4. **`scripts/launch_loocv.sh`** — train folds 0-4 of full + ablation in
   sequence (or parallel if multi-GPU). Reduce per-fold iters to 80k.

5. **`scripts/compute_betti.py`** — replace `scripts/verify_topology.py`'s
   `scipy.ndimage.label` (only counts β₀) with `gudhi.CubicalComplex` for
   real β₀, β₁, β₂. Use for honest "topology" reporting in thesis.

---

## 8-week thesis plan

| Week | Tasks |
|---|---|
| 1 (now) | DICOM → NIfTI; smoke-test 10 ADNI; brain-mask metrics; no-topo ablation training on sub-06 |
| 2-3 | 5-fold LOOCV (full + ablation); preprocess full 200 ADNI |
| 4-5 | Synthesis on all 200 ADNI; FastSurfer on synthetic 7T outputs |
| 6 | AD classifiers: C1=3T-only, C2=synth-7T, C3=fusion, C4=ablation. AUC comparison with paired tests |
| 7 | Hippocampal volumetry agreement on LOOCV (ICC, Bland-Altman, MAE%) — clinical safety net |
| 8 | Thesis writing |

---

## Limitations to acknowledge in thesis (be upfront)

- N=10 training cohort, 9 effective per fold
- PSNR 19.84 dB whole-volume → expected ~24-26 dB after brain-masking; still below SynDiff/CCA-GAN
- "Topology loss" is edge-aware CE+Dice (Sobel), not persistent homology. Cite Hu 2019, Clough 2020, Shit 2021 (clDice) and propose as future work.
- Cortical-ribbon fragmentation (1658 GM CC) — known limitation of edge-aware loss
- `verify_topology.py` only counts β₀ — replace with gudhi for real Betti before thesis submission
- Last 15/200 timesteps excluded from training (numerical instability — see [src/diffusion.py](src/diffusion.py) lines 336-340)
- 2D VGG perceptual on every-8th depth slice ([src/diffusion.py](src/diffusion.py) line 76) — likely should be removed; defer
- ADNI domain shift not yet validated — week 1 smoke test is the gate

---

## Files of interest with action notes

- [src/topology_loss.py](src/topology_loss.py) — current Sobel loss; do NOT call this "topology" in writing without clDice swap
- [src/diffusion.py:336-340](src/diffusion.py) — timestep exclusion workaround, document as known issue
- [src/diffusion.py:75-77](src/diffusion.py) — 2D VGG on subsampled slices
- [scripts/verify_topology.py:65-74](scripts/verify_topology.py) — replace with gudhi
- [src/metrics.py:28-34](src/metrics.py) — enforce brain-masking in PSNR
- [scripts/preprocess_masks.py:100-113](scripts/preprocess_masks.py) — verify aseg priority sort picks ses-2 not ses-1 for all subjects
- [src/model.py:209-212](src/model.py) — verify training loop calls `forward_full()` not `forward()` (`forward()` may return placeholder zeros for seg)
- [test_split.json](test_split.json) — current sub-06 split (only fold 0 held out)
- [TODO_PAPER.md](TODO_PAPER.md) — superseded by this handoff for execution priority

---

## Current branch

`feat/train-with-masks` — in-progress, NOT merged to main. Status:
- M `TODO_PAPER.md` (uncommitted edits)
- M `scripts/evaluate_full_volume.py` (uncommitted edits)
- ?? `configs/train_diffusion_ablation_notopo.yaml` (untracked, needed for ablation)

Recent commits: ad28e45 (verify topology), d76d55f (syntax fix), 67be6af (verify topology feat), c37ff17 (added args), aab1312 (DDIM eval).

---

## Open questions / things to verify in next session

1. Confirm training masks were generated from ses-2 aparc+aseg (not ses-1) for all 10 subjects.
2. Confirm `forward_full()` is the call site in the training loop, not `forward()`.
3. After ADNI smoke test: does the model generalize, or is fine-tuning needed?
4. After full LOOCV: does ablation (λ_topo=0) actually differ from full model in subcortical preservation? My prediction is **no significant difference** — but the data will say.
