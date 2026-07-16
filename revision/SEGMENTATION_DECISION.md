# Segmentation ground truth — audit, defects, and the decision

**Date: 16 July 2026.** Triggered by the right question from Pratik: *have we actually verified
the segmentation masks, or just the code that uses them?* Answer: we had only verified the code.
This documents the audit of the ground truth itself and the decision that follows.

The topology claim is the paper's contribution, and the segmentation is the **measuring
instrument** for it. It must be a named, standard, documented, QC'd tool, or the whole claim is
"trust us".

---

## 1. Provenance (established, not assumed)

The `*_seg.nii` files are produced by **`scripts/preprocess_masks.py`**:
- source: FreeSurfer / FastSurfer **`aseg`** (or `aparc+aseg`) — a real anatomical segmentation;
- resampled to the 7T target grid with **nearest-neighbour** (order=0) — correct for labels;
- mapped to 4 classes (0 BG / 1 CSF / 2 GM / 3 WM) by the `FS_MAPPING` lookup table.

So it is **not** the intensity-thresholding script `generate_tissue_masks.py` (that is abandoned
code). Proof: that script forces CSF = darkest 33% of brain (~1.2 M voxels); our actual CSF is
**41 k voxels (~1%)** — anatomical proportions, not equal thirds. Confirmed anatomical.

## 2. What the audit found (sub-06, 26-connectivity)

| class | β₀ (components) | real (>100 vox) | speckle (≤5 vox) | verdict |
|---|---|---|---|---|
| **GM** | **1** | 1 | 0 | **clean — the headline claim is safe** |
| CSF | 5 | 5 | 0 | real (ventricles are naturally separate) |
| **WM** | **15** | **1** | 7 (+7 small) | **93% of the components are noise** |

Brain mask (`seg>0`): 25,653 internal holes (0.71% of brain).

## 3. Defects in the label map (`FS_MAPPING`) — a reviewer *will* find these

WM is mapped from **only** labels `2, 41` (cerebral WM L/R). Consequently:

- **Corpus callosum (251–255) → background.** A large central WM structure is dropped, punching
  a hole through the middle of the white matter. Directly inflates WM topological error.
- **Entire cerebellum (7, 8, 46, 47) → background.** Cerebellar WM *and* cortex are dropped. The
  **brain mask is therefore cerebrum-only** — and this is stated nowhere in the manuscript.
- **Ventral DC (28, 60), choroid plexus (31, 63) → background.** Brainstem (16) → lumped into GM.

None of this is fraudulent — it is an incomplete LUT — but it has silently shaped every reported
number (WM topology, the brain mask, and hence brain-masked SSIM/PSNR), and it is exactly the
kind of thing that reads as sloppiness to a reviewer who opens the table.

## 4. The reframe that makes the claim defensible

The **true topology of a real brain is not a well-defined number** — it depends on resolution,
partial-volume voxels, and threshold. So *"we recover the true Betti number"* is indefensible.
The claim that survives review is **relative and consistent**:

> Measured against the same standard segmentation, our method produces substantially fewer
> spurious topological components than the baseline, and this holds across 10 held-out subjects.

This needs the GT only to be **consistent** between our method and the baselines — not perfect.

## 5. Decision

1. **Re-segment all 10 subjects with one standard, citable tool.** SynthSeg (Billot et al., MedIA
   2023) — robust, contrast-agnostic, GPU, one citation, no custom steps to attack. Alternative:
   a **complete, corrected** aseg LUT (add CC 251–255, cerebellum 7/8/46/47, ventral DC, etc.).
   Either way the mapping is explicit and version-controlled.
2. **QC it.** Cross-check the new segmentation against the existing one on ≥1 subject (Dice per
   class, topology agreement). Report the number — that is the provability evidence.
3. **Two different treatments of the residual speckle, on purpose:**
   - *Training target (loss):* supervise on **cleaned** topology (largest components per tissue),
     justified as "sub-voxel speckle is not anatomy." Otherwise the loss teaches the model to
     reproduce noise.
   - *Evaluation metric:* score against the **consistent** GT, report the claim as **relative**
     improvement, and show **raw and cleaned both** so nothing is hidden.
4. **Freeze the clean GT before the LOSO run** so all 10 folds train/evaluate against honest,
   consistent, complete targets.
5. **State the cerebrum-only / cerebellum question explicitly** in the manuscript, whichever way
   it is resolved.

## 6. Consequence

This **gates the LOSO run** (Phase 5) and resets the current sub-06/sub-07 numbers, which were
computed against the incomplete-LUT segmentation. It does **not** block CERN setup — the data
pull and environment work continue; a segmentation pass on the GPU is added before the recipe is
frozen.

## 7. What was NOT wrong

GM — the class the paper's headline rests on — is clean (β₀=1) under the current segmentation, so
the core finding (predicted GM β₀ ≫ 1) is real, not an artifact of the ground truth. The rebuild
here is about making the **instrument** unimpeachable, not about rescuing a false result.
