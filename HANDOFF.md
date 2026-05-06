# Topo-Brain — Session Handoff (2026-05-07)

This file captures active state for picking up in a new Claude Code chat.
**To resume:** open a new chat in this directory and say:
> "Read HANDOFF.md and continue. Next task is [X]."

The auto-memory in `C:\Users\Asus\.claude\projects\d--BCT-pcampus-...\memory\`
loads automatically — this file adds session-specific context on top.

---

## Where we are

- Single-fold sub-06 model trained at 145k iters (DDPM full loss, λ_topo=0.2).
- **Hippocampus preservation verified — AD downstream is viable.**
- ADNI cohort built: 288 baseline 3T MPRAGE downloaded (AD/CN/MCI), 266 with
  baseline AD/CN diagnosis after DXSUM filter, 107 AD + 159 CN survive
  demographics filter. 1:1 age/sex matching → 103 AD ↔ 103 CN.
- **Kaggle smoke test in progress (10 subjects, 5 AD + 5 CN):** all 10
  preprocessed (HD-BET + N4 + diffusion-normalize); synthesis next.
  Bypassed `MRIPreprocessor` MONAI pipeline because Kaggle MONAI has a
  breaking `Orientationd(labels=None)` change — see commit 76a6b6e.
- ADNI cohort tooling (`build_adni_cohort.py`, `convert_adni_dicom.py`,
  `preprocess_adni.py`, `run_kaggle_smoke.py`) all written + smoke-tested.
- 8-week thesis plan settled.

---

## The headline empirical finding (sub-06 vs ses-2 aparc+aseg)

| Region | CC | Largest | Coverage | Vol Err |
| --- | --- | --- | --- | --- |
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

## Active task — Kaggle smoke-test synthesis

DICOM conversion is done (commit ea67c22). The 10-subject smoke pipeline is
the gate for committing to the full 200. Status as of this handoff:

| Stage | Status |
| --- | --- |
| ADNI 1-Click download (AD/CN/MCI) | ✅ done |
| Study CSVs (DXSUM, PTDEMOG, MRI3META, MRIQC, DATADIC) | ✅ in `adni_dataset/study_data/` (gitignored) |
| dcm2niix conversion | ✅ 10/10 ok via `convert_adni_dicom.py` |
| Cohort manifest | ✅ `pairs_adni_smoke.csv` from `build_adni_cohort.py` |
| HD-BET + N4 + diffusion-normalize | ✅ 10/10 ok on Kaggle T4 (~75 min CPU) |
| Synthesis (DDIM 50, sub-06 145k ckpt) | ⏳ running on Kaggle |
| Visual sanity check | ⏳ |
| FastSurfer on synth 7T (CERN) | ⏳ |
| `check_region_preservation.py` decision | ⏳ |

**Kaggle invocation (3 lines):**

```python
!git clone -b feat/train-with-masks https://github.com/prabeshx12/Topo-Brain.git /kaggle/working/Topo-Brain 2>/dev/null || (cd /kaggle/working/Topo-Brain && git pull)
!pip install -q HD-BET scikit-image
!python /kaggle/working/Topo-Brain/scripts/run_kaggle_smoke.py \
    --checkpoint /kaggle/input/<your-checkpoint>/checkpoint_145000.pt \
    --adni-dir   /kaggle/input/datasets/pratikadhikari9/adni-dataset
```

**Diagnosis encoding (current ADNI unified `DIAGNOSIS` column, used by `build_adni_cohort.py`):**
1=CN, 2=MCI, 3=AD/Dementia. Older `DXCURREN`/`DXCHANGE` columns are still
present but the unified DIAGNOSIS is the canonical field; we use only that.

---

## Next concrete task

1. ⏳ **Wait for Kaggle synthesis to finish** (~30-50 min on T4).
2. **Visual sanity check** on 1-2 outputs (notebook Cell 8) — does the synth
   7T look like a brain? If it's noise/garbage, debug domain shift before
   FastSurfer.
3. **Download `adni_smoke_results.zip`** from Kaggle's Output panel.
4. **On CERN (lxplus + apptainer):**
   - `scripts/run_fastsurfer.py --mode synthetic7t` on the 10 outputs (overnight).
   - `scripts/check_region_preservation.py` per subject:
     `predicted_seg.nii.gz` vs the FastSurfer-on-synth aparc+aseg as proxy GT.
5. **Decision rule:** if hippo `largest > 95%` and `coverage > 80%` on
   these 10, model generalizes → preprocess full 200 next. If numbers
   crash, diagnose domain shift before downloading more.

---

## Scripts — written and committed

| ✓ | Script | Commit | Notes |
| --- | --- | --- | --- |
| ✅ | `scripts/check_region_preservation.py` | (pre-existing 67be6af) | per-region CC/largest/coverage/vol_err. Requires GT aparc+aseg. |
| ✅ | `scripts/convert_adni_dicom.py` | ea67c22 | dcm2niix wrapper for AD/CN/MCI cohort folders |
| ✅ | `scripts/build_adni_cohort.py` | ea67c22 | DXSUM + PTDEMOG + MRI3META + MRIQC join → `pairs_adni.csv`. Optional 1:1 age/sex matching. |
| ✅ | `scripts/preprocess_adni.py` | 76a6b6e | HD-BET + N4 + diffusion-normalize. **Bypasses MRIPreprocessor / MONAI** because newer MONAI's `Orientationd(labels=None)` requires meta-tensor space info that doesn't survive temp-file roundtrip. |
| ✅ | `scripts/run_kaggle_smoke.py` | 3281c97 | end-to-end Kaggle runner. Stages → preprocess → zip resume bundle → synthesize → zip results. `--skip-preprocess` to resume after kernel restart. |
| ✅ | `scripts/compute_betti.py` | 1ffb4cb | real β₀/β₁/β₂ via `gudhi.CubicalComplex`. `--self-test` validates against shapes of known Betti — caught a real gudhi axis-order bug before it could pollute thesis numbers. |
| ⏳ | `scripts/launch_loocv.sh` | — | week 2-3 work; train folds 0-4 of full + ablation. Reduce per-fold iters to 80k. |

---

## 8-week thesis plan

| Week | Tasks |
| --- | --- |
| 1 (now) | DICOM → NIfTI; smoke-test 10 ADNI; brain-mask metrics; no-topo ablation training on sub-06 |
| 2-3 | 5-fold LOOCV (full + ablation); preprocess full 200 ADNI |
| 4-5 | Synthesis on all 200 ADNI; FastSurfer on synthetic 7T outputs |
| 6 | AD classifiers: C1=3T-only, C2=synth-7T, C3=fusion, C4=ablation. AUC comparison with paired tests |
| 7 | Hippocampal volumetry agreement on LOOCV (ICC, Bland-Altman, MAE%) — clinical safety net |
| 8 | Thesis writing |

---

## Limitations to acknowledge in thesis (be upfront)

- N=10 training cohort, 9 effective per fold
- PSNR brain-masking now correct in `src/metrics.py` (commit 8a3328f). Existing
  reported PSNR (19.84 dB whole-volume) was pre-fix; re-run eval to get the
  expected ~24-26 dB brain-masked number. Still below SynDiff/CCA-GAN at the
  state of the art.
- "Topology loss" is edge-aware CE+Dice (Sobel), not persistent homology. Cite
  Hu 2019, Clough 2020, Shit 2021 (clDice) and propose as future work.
- Cortical-ribbon fragmentation (1658 GM CC) — known limitation of edge-aware loss
- Real Betti numbers (β₀/β₁/β₂) now available via `scripts/compute_betti.py`
  (commit 1ffb4cb). `verify_topology.py` still uses `scipy.ndimage.label` for
  its connected-component count; either run `compute_betti.py` separately
  for the thesis-grade Betti report, or update `verify_topology.py` to call it.
- Last 15/200 timesteps excluded from training (numerical instability — see [src/diffusion.py](src/diffusion.py) lines 336-340)
- 2D VGG perceptual on every-8th depth slice ([src/diffusion.py](src/diffusion.py) line 76) — likely should be removed; defer
- ADNI domain shift not yet validated — week 1 smoke test is the gate

---

## Files of interest with action notes

- [src/topology_loss.py](src/topology_loss.py) — current Sobel loss; do NOT call this "topology" in writing without clDice swap
- [src/diffusion.py:336-340](src/diffusion.py) — timestep exclusion workaround, document as known issue
- [src/diffusion.py:75-77](src/diffusion.py) — 2D VGG on subsampled slices
- [scripts/verify_topology.py:65-74](scripts/verify_topology.py) — only counts β₀; for thesis, run `scripts/compute_betti.py` separately for full β₀/β₁/β₂
- [src/metrics.py:13-77](src/metrics.py) — `compute_ssim_psnr` now honors `mask=` (commit 8a3328f); existing eval scripts still have local copies, route through `src/metrics.py` when re-running eval
- [scripts/preprocess_masks.py:100-113](scripts/preprocess_masks.py) — verify aseg priority sort picks ses-2 not ses-1 for all subjects
- [src/model.py:209-212](src/model.py) — verify training loop calls `forward_full()` not `forward()` (`forward()` may return placeholder zeros for seg)
- [test_split.json](test_split.json) — current sub-06 split (only fold 0 held out)
- [TODO_PAPER.md](TODO_PAPER.md) — superseded by this handoff for execution priority
- [adni_dataset/](adni_dataset/) — gitignored. Contains DICOMs, study-data CSVs, and the cohort manifests. Not redistributable per ADNI DUA.

---

## Current branch

`feat/train-with-masks` — in-progress, NOT merged to main. Recent commits
(top of branch first):

- `1ffb4cb` feat: compute_betti.py — real β₀/β₁/β₂ via gudhi.CubicalComplex
- `8a3328f` fix: brain-mask support in compute_ssim_psnr
- `3281c97` feat: run_kaggle_smoke.py — single-command end-to-end smoke test
- `76a6b6e` fix: bypass MRIPreprocessor in preprocess_adni.py
- `7453e16` fix: pin Orientationd labels for MONAI compatibility
- `ea67c22` feat: ADNI cohort tooling and Kaggle smoke notebook
- `4ec5c2f` feat: support n_samples averaging in tiled_inference
- `ad28e45` fix: verify topology with true seg and predicted seg

`TODO_PAPER.md` has uncommitted edits (intentionally left out — superseded by this handoff).

---

## Open questions / things to verify in next session

1. Confirm training masks were generated from ses-2 aparc+aseg (not ses-1) for all 10 subjects.
2. Confirm `forward_full()` is the call site in the training loop, not `forward()`.
3. After Kaggle synthesis finishes: does the synthetic 7T look like a brain on
   visual inspection? If yes, proceed to FastSurfer + region check. If no,
   diagnose domain shift before any further compute spend.
4. After FastSurfer + `check_region_preservation.py` on the 10: does the model
   generalize to ADNI? Decision rule: hippo `largest > 95%` and `coverage > 80%`.
5. After full LOOCV: does ablation (λ_topo=0) actually differ from full model
   in subcortical preservation? My prediction is **no significant difference** —
   but the data will say.
6. Re-run evaluate_full_volume.py with the updated `src/metrics.py` to refresh
   the headline PSNR number (currently the trick-version is reported; new one
   computes brain-masked PSNR cleanly through the central API).
