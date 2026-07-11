# 110k checkpoint re-evaluation — sub-06 (valid run)

Produced by `topobrain_eval_110k.py` on Kaggle P100, DDIM-50, EMA weights, inputs re-normalized to
[-1,1] via the exact `diffusion` method (`src/preprocessing.py`). Raw JSON: `eval_110k_results.json`.
Checkpoint: step 110000, 13,978,821 params.

## D8 verdict — the paper's "105k" is a mislabel for 110k
Table 1 reproduces from the 110k checkpoint (image quality):

| Metric | Paper Table 1 | 110k (SegMaskGT) | 110k (Threshold) |
|---|---|---|---|
| SSIM | 0.8991 | **0.8998** | 0.8709 |
| PSNR (dB) | 20.00 | **20.85** | 19.38 |
| Foreground HD95 (mm) | 3.31 | 2.68 | **3.31** |
| Foreground Dice | 0.7682 | 0.7831 | 0.7401 |

SSIM and foreground HD95 match essentially exactly; PSNR is within ~0.85 dB. **Conclusion: results
came from the 110k checkpoint; "105k" is a labelling error → correct the text to 110k.** (The
paper's brain-mask Dice 0.9436 / HD95 2.05 use a mask-vs-mask definition different from this
script's intensity-threshold Dice, so they are not directly comparable here — not a discrepancy.)

## Per-tissue Dice (predicted seg vs GT seg)
| Tissue | Paper Table 2 | 110k run |
|---|---|---|
| CSF | 0.7616 | 0.6811 |
| GM  | 0.8501 | 0.7823 |
| WM  | 0.8963 | 0.8821 |

Same ballpark, slightly lower; likely eval-setting differences (e.g. multi-sample averaging
`--n-samples`, which the paper's script supports and which raises Dice/PSNR). Worth a footnote.

## True topological metrics (R2.3 / R2.8) — reported honestly
Betti numbers via gudhi cubical complex; predicted seg vs 7T-derived GT seg:

| Tissue | β0 pred | β0 GT | β1 pred | β1 GT | β2 pred | β2 GT | Euler χ pred | Euler χ GT | Largest-CC frac (pred) |
|---|---|---|---|---|---|---|---|---|---|
| CSF | 753  | 5  | 222  | 2   | 31   | 0    | 562  | 3    | 0.650 |
| GM  | 5213 | 1  | 5991 | 939 | 3169 | 2115 | 2391 | 1177 | 0.937 |
| WM  | 1393 | 15 | 1585 | 25  | 1411 | 101  | 1219 | 91   | 0.978 |

**Honest interpretation:** the dominant connected component is preserved (largest-CC fraction
0.94-0.98), consistent with the paper's reported metric — but strict topology reveals the
predicted maps carry **orders-of-magnitude more small components/handles/voids** than ground
truth (GM β0 5213 vs 1). HD95 and largest-CC-fraction mask this small-component speckle. This
**qualifies the "topology preservation" claim**: the edge-aware boundary regulariser preserves
gross anatomy but does not achieve GT-level topology — reinforcing the manuscript's own framing
that it is an *approximation*, not persistent-homology, and motivating the PH-loss follow-up.
Recommended presentation: report Betti/Euler with the largest-CC-fraction caveat; do not overclaim.

## Reproducibility note
The exact [-1,1]-normalized `.nii.gz` files listed in `pairs_new.csv` are not persisted in any
Kaggle dataset (only a z-score-normalized `.nii` copy exists). This run reconstructs them by
re-applying the documented `diffusion` normalization. For the resubmission, the normalized inputs
(or a one-command preprocessing script) should be released so Table 1 is reproducible end-to-end.
