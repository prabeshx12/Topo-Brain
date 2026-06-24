# Paper ↔ Code Discrepancies

Code is ground truth. Each item lists what the manuscript says, what the code actually
does (with file:line), severity, and the fix. Verified items are marked ✅.

---

## D1 — Topology-loss equation does not match the implementation ✅ **High**
**Paper (§3.3, Eq. 2):**
`L_topo = L_CE(Ŝ,S) + λ_e·L_CE(∇Ŝ,∇S) + λ_d·L_Dice(Ŝ,S)`, with λ_e = λ_d = 0.5 —
i.e. a plain CE term + a **Sobel-gradient cross-entropy** term + a **volumetric Dice** term.

**Code ([src/topology_loss.py:103-181](../src/topology_loss.py#L103-L181)):**
- `EdgeAwareTopologyLoss` computes an **edge-WEIGHTED cross-entropy** (CE multiplied per-voxel
  by `1 + (edge_weight-1)·edge_map`, `edge_weight=2.0`) — a single CE term, *not* CE + a
  separate gradient-CE term.
- Plus `0.5 · L_boundary`, where `L_boundary` is a **Dice between soft Sobel edges of the
  predicted softmax and the binary target edge map** — i.e. Dice **on edge maps**, not a
  volumetric Dice on tissue classes.
- The whole thing is wrapped in `MultiScaleTopologyLoss` (3 scales [1.0, 0.5, 0.25],
  sqrt-weighted) — **multi-scale is undisclosed in the paper** ([topology_loss.py:184-248](../src/topology_loss.py#L184-L248), activated via `create_topology_loss(use_multiscale=True)` in [src/diffusion.py:164-183](../src/diffusion.py#L164-L183)).

**So:** no gradient-CE term; no volumetric Dice; an undocumented edge-weight (2.0) and
multi-scale wrapper. The single hard-coded internal coefficient is `0.5` on the boundary term.
**Fix:** WRITING — rewrite Eq. 2 to match code. Prerequisite for honestly answering R1.3/R2.8.

---

## D2 — Parameter count is 13.98 M, not 22.4 M ✅ **High**
**Paper:** "22.4-million-parameter 3D U-Net" (abstract, §1, §3.2). Reviewer R1.4 quotes it back.
**Code:** [scripts/count_params.py](../scripts/count_params.py) instantiates `AnatomyGuidedUNet`
with the exact training config (features (32,64,128,256), num_classes=4, use_attention=True)
→ **13,978,821 params (13.98 M)**. Matches [TRAINING_REPORT.md:283](../TRAINING_REPORT.md#L283)
("14-15 M"); contradicts the manuscript.
Per-module: downs 2.33M · mid_block1 3.55M · mid_attn 0.26M · mid_block2 3.55M · ups 2.14M ·
seg_ups 2.14M.
**Fix:** WRITING — change 22.4 M → 13.98 M everywhere. (Also mildly *helps* the overfitting
rebuttal: a 14 M model is smaller than claimed.) Reproducible via the script.

---

## D3 — Total-loss equation omits a term ✅ **Medium**
**Paper (§3.3, Eq. 1):** `L_total = 1.0·L_pixel + 0.2·L_topo + 0.25·L_percep` (3 terms);
text says L_pixel is "the standard L1 noise-prediction loss".
**Code ([src/diffusion.py:408](../src/diffusion.py#L408)):** **4 terms** —
`loss_diff` (L1 on noise, weight 1.0) **and** `loss_pixel` (L1 on the reconstructed image x̂₀,
`lambda_pixel=1.0`) **and** `lambda_percep·loss_vgg` **and** `lambda_topo·loss_topo`.
The paper collapses two distinct unit-weighted L1 terms (noise-space + image-space) into one.
**Fix:** WRITING — present all four terms (or justify merging) so the released code matches Eq. 1.

---

## D4 — Downstream synth AUC: 0.653 vs 0.658 ✅ **Medium**
**Paper (§4.6):** synthetic-derived composite AUC = **0.653**.
**Code/results:** [head_to_head_summary.txt:8](../adni_smoke_analysis_30/head_to_head_summary.txt#L8)
reports C2 (synth) = **0.658**; the fair LOSO recompute
([downstream_fair_recompute.txt](../adni_smoke_analysis_30/downstream_fair_recompute.txt))
gives **0.653** (LOSO) / 0.658 (in-sample). Two scripts, two numbers.
**Fix:** WRITING — pick the LOSO value (0.653) and report it consistently with its 95% CI
[0.444, 0.844] and the DeLong p (see [reviewer_response_matrix.md](reviewer_response_matrix.md), R1.5).

---

## D5 — Diffusion timesteps: 200 claimed, 185 effective ✅ **Low**
**Paper:** "T = 200 timesteps." **Code ([src/diffusion.py:339](../src/diffusion.py#L339)):**
training/inference sample `t ∈ [0, 184]`; last 15 excluded for cosine-schedule stability.
**Fix:** WRITING — one footnote ("200-step schedule, final 15 steps excluded for numerical
stability → 185 effective").

---

## D6 — 10-fold splitter exists but its `test_fold` is silently ignored ✅ **Low–Med**
**Code ([src/synthesis_dataset.py:762](../src/synthesis_dataset.py#L762)):** the dataloader calls
`get_split(..., test_fold=val_fold)`, overriding the config's `test_fold: 1`. So only fold 0
(sub-06) is ever held out (n=1), matching the paper's "sub-06 held out" — but a reviewer reading
the released code sees a LOOCV harness that is not actually used.
**Fix:** CODE — repair as part of the LOSO harness (R1.4/R2.7); note in response that the
single-holdout was intentional for the first version and LOSO is now run.

---

## Non-discrepancies (confirmed consistent)
- VGG perceptual-loss weight **0.25** matches code ([diffusion.py:374-375](../src/diffusion.py#L374), config `lambda_percep: 0.25`). R1.1 is a real *argument* to make, not a code mismatch.
- λ_topo = 0.2, λ_pixel = 1.0 match config.
- `scripts/compute_betti.py` implements **true Betti numbers via gudhi** — present but not yet run on outputs (R2.3/R2.8 cheap win once checkpoint is restored).
