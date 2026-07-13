# Protocol sensitivity — how to manufacture a "SOTA-beating" result without changing the model

**Measured 14 July 2026** on the real held-out subject (sub-06), cascaded model, checkpoint
`cascaded_16677.pt`. One prediction. One model. Nothing changes between rows except **where the
metric is pointed**.

## The result

| Convention | PSNR | SSIM | vs honest | Source |
|---|---|---|---|---|
| **A brain-masked 3D** | **14.01** | — | — | **what we advocate** |
| B whole-volume 3D | 21.53 | — | +7.52 dB | field convention |
| C 2D per-slice, axis 0 (all slices) | 26.38 | 0.789 | +12.37 dB | Acs & Zhuang |
| C 2D per-slice, axis 1 (all slices) | 28.43 | 0.792 | +14.42 dB | Acs & Zhuang |
| **C 2D per-slice, axis 2 (all slices)** | **31.16** | **0.793** | **+17.15 dB** | **Acs & Zhuang** |
| D 101 central axial slices | 19.56 | 0.721 | +5.55 dB | FS-RWKV |
| E central 128×128 crop | 15.82 | 0.444 | +1.81 dB | LiteMamba Tab. 4 |
| F GT pasted outside brain | 22.23 | — | +8.22 dB | **the published bug** |

> **Spread: 17.15 dB. Same model. Same prediction.**

## The claim we could make, and will not

Acs & Zhuang report **23.25 dB / SSIM 0.737** — 2D per-slice, all slices, unmasked, and that is
their *best* plane (their coronal is 22.48, sagittal 22.05).

Scored **under exactly that protocol**, our model reports **31.16 dB / SSIM 0.793**.

We could therefore write:

> *"TopoBrain attains PSNR 31.16 and SSIM 0.793, outperforming the state of the art (23.25 /
> 0.737) by 7.9 dB."*

It would survive review. **It would also be worthless.** The same model, measured on the brain it
is supposed to synthesise, scores **14.01 dB**. Nothing was learned between those two numbers —
only the metric moved.

**We do not make this claim, and the paper says why.** This is the single most important
methodological point in the revision: it is a live demonstration, on our own model, of exactly
the failure mode that got the original manuscript rejected.

## Why the inflation is so large here

Our volumes are skull-stripped: **84.9 % of the volume is background** the model never has to
synthesise (3.6 M brain voxels of 24.0 M). Under a 2D per-slice mean, the near-empty slices at
the edges of the volume have vanishing MSE and therefore enormous per-slice PSNR, and they are
averaged in with equal weight. No published paper on this dataset states any slice-exclusion
rule.

Note also that the three orientations differ by **4.8 dB from each other** (26.38 / 28.43 /
31.16). Acs & Zhuang's three orientations differ by only 1.2 dB — consistent with their volumes
being full-head, where no slice is ever near-empty. **The orientation you report is a free
parameter, and they report their best one.**

## The direction of the bias is NOT fixed

From the phantom sweep (`scripts/test_metrics_protocols.py`), holding the brain error fixed at
σ = 0.09 and varying only the background difficulty:

| background error σ | brain-only | whole-volume | effect of unmasking |
|---|---|---|---|
| 0.02 | 22.67 | 30.32 | **+7.65 dB** (inflates) |
| 0.10 | 22.67 | 22.88 | +0.21 dB (**crossover**) |
| 0.30 | 22.67 | 15.25 | **−7.41 dB** (depresses) |

The sign flips exactly where background error ≈ brain error, as the arithmetic requires. So:

- an **easy** background (ours: stripped, constant, reproduced exactly) **inflates** the score;
- a background **harder than the brain** **depresses** it — which is how LiteMamba's central crop
  could *gain* 3.05 dB by *removing* background.

**Nobody may assert a single direction for "background inflates PSNR" — including us.** My own
earlier claims in *both* directions were wrong for precisely this reason. It depends on the data,
and none of the three competing papers release code or weights to check.

## Validation of the scorer

`src/metrics_protocols.py` was written independently of the evaluation harness, and it
reproduces both of our previously-reported numbers **to the decimal**:

| | previously reported | protocol scorer | |
|---|---|---|---|
| brain-masked 3D PSNR | 14.01 | **14.01** | ✅ |
| whole-volume 3D PSNR | 21.53 | **21.53** | ✅ |

Plus: PSNR matches its definition to 1e-9; SSIM matches `skimage` to 1e-9; the MSE = 0 degeneracy
(PSNR = ∞) is counted and excluded rather than allowed to saturate the mean.

**A normalisation bug was caught by this cross-check.** The first run compared the tanh-range
prediction against the *raw* ground truth (range [−1.512, 1.931]) and produced brain SSIM 0.079 /
PSNR 12.93. The pipeline's `dnorm` must be applied to the ground truth or the two are on
different scales. The fact that the corrected run lands exactly on 14.01 / 21.53 is what tells us
it is now right.

## What this becomes in the paper

Not a limitations footnote — a **contribution**:

1. **The benchmark on this dataset is unmoored.** Three papers, three protocols, none citing each
   other, none masking the brain, none releasing code, ±17 dB of available slack.
2. **We advocate the brain-masked 3D number** and report it as our headline, even though it is by
   far our *worst* number, because it is the only one that measures the thing the method claims
   to do.
3. **We report every other convention alongside it**, so a reader can compare us to anyone.
4. Fair comparison requires **running the baselines ourselves on our splits with our metrics** —
   which is what reviewers R1.2 / R2.1 demanded, and which is now cheap because Acs & Zhuang's
   FR-U-Net is essentially our own architecture.

## Reproduce

```bash
python scripts/test_metrics_protocols.py          # phantom: correctness + the sign-flip sweep
python scripts/protocol_sensitivity_real.py \
    --pred sub-06_cascaded_pred7T.nii.gz \
    --gt   sub-06_ses-2_desc-preproc_T1w.nii \
    --seg  sub-06_ses-2_desc-preproc_T1w_seg.nii
```
