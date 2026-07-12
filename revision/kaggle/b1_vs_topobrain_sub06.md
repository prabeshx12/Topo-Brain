# B1 regression baseline vs TopoBrain (110k) — sub-06

**This is the most consequential result of the revision. Reported exactly as produced.**

Answers reviewer **R1.2** (SOTA baseline) and **R2.6** (why diffusion over deterministic
regression). Produced by `topobrain_b1_eval.py`. Raw: `b1_eval_results.json`.

## Fair-comparison protocol
Identical backbone (`AnatomyGuidedUNet`, 13.98 M params), identical data, identical 9/1 split
(sub-06 held out), identical [-1,1] normalization, identical tiling/stitching and metrics.
**The only difference is the objective.**

- **TopoBrain**: conditional diffusion + seg head + edge-aware topology loss. 110k steps. DDIM-50
  inference (50 forwards/patch).
- **B1**: deterministic regression (no diffusion), L1 + plain weighted-CE seg head.
  **No topology loss.** Converged (L1 plateaued ~0.10 from step ~5k); trained to 17k.
  1 forward/patch (~50x cheaper inference).

## Results

### Image quality
| Metric | B1 | TopoBrain | winner |
|---|---|---|---|
| SSIM (SegMaskGT) | **0.9247** | 0.8998 | B1 |
| PSNR (SegMaskGT) | **22.36** | 20.85 | B1 |
| SSIM (Threshold)  | **0.9252** | 0.8709 | B1 |
| PSNR (Threshold)  | **22.61** | 19.38 | B1 |
| fg HD95 (Threshold) | 4.16 | **3.31** | TopoBrain |
| HD95 (SegMaskGT) | **2.60** | 2.68 | ~tie |

### Tissue Dice (predicted seg vs 7T-derived GT)
| Tissue | B1 | TopoBrain | winner |
|---|---|---|---|
| CSF | **0.8001** | 0.6811 | B1 (+0.12) |
| GM  | **0.8741** | 0.7823 | B1 (+0.09) |
| WM  | **0.9298** | 0.8821 | B1 (+0.05) |

### True topology (Betti; closer to GT is better)
| Tissue | | B1 | TopoBrain | GT |
|---|---|---|---|---|
| CSF | β0 | **8**  | 753  | 5 |
| GM  | β0 | **6**  | 5213 | 1 |
| WM  | β0 | **3**  | 1393 | 15 |
| GM  | β1 | **501** | 5991 | 939 |
| WM  | β1 | **40**  | 1585 | 25 |
| WM  | β2 | **6**   | 1411 | 101 |

## Interpretation — the thesis is refuted by its own baseline

1. **A plain regression U-Net beats TopoBrain on image quality, on every tissue Dice, and on
   topology** — with 1/6th the training steps and 1/50th the inference cost.
2. **B1 has NO topology loss, yet its topology is near ground truth** (β0 = 8/6/3 vs GT 5/1/15),
   while TopoBrain — *with* the topology loss — is shattered (753/5213/1393). **The loss designed
   to preserve topology yields far worse topology than not having it.**
3. **Mechanistic cause (testable):** TopoBrain's segmentation head is trained on *noisy diffusion
   latents* (x_t at random t in [0,184]), so it learns to segment noise-corrupted volumes and is
   never optimised for clean input. B1's seg head only ever sees clean input. Coupling the seg
   head to the diffusion process is what degrades it — and no boundary regulariser can compensate.

## Consequences
- The claim "anatomy-aware conditional diffusion preserves anatomy" **cannot be sustained** in its
  current form. R1.2/R2.6 are not merely unanswered — the honest answer currently favours the
  baseline.
- Combined with the topology finding (`cleaned_betti_sub06.md`), the "TopoBrain" framing is not
  supportable.

## Caveats (do not over-conclude)
- **n = 1 (sub-06).** It is the same held-out subject the paper's headline results use, so it is
  the fair comparison — but a conclusion this consequential must be confirmed across folds (LOSO).
- B1's advantage on SSIM/PSNR is partly expected (L1 optimises pixel fidelity; perception-
  distortion trade-off). Its advantage on **Dice and topology is NOT explained by that** and is the
  damaging part.

## Constructive paths (the diagnosis suggests fixes)
1. **Decouple the seg head** — run it on the clean 3T conditioning rather than the noisy latent.
2. **Timestep-restrict the seg supervision** — train the seg/topology loss only at low t.
3. Re-run this comparison after either fix; if diffusion still loses, the honest paper is the
   controlled negative result ("does conditional diffusion earn its cost for 3T->7T synthesis?").
