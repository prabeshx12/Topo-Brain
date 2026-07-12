# Full code audit — TopoBrain

**Date: 12 July 2026.** Line-by-line audit of the whole pipeline, triggered by a controlled
experiment in which a deterministic regression U-Net (same backbone) beat the diffusion model on
image quality, tissue Dice, AND topology.

## THE HEADLINE CONCLUSION

> **The diffusion model was never given a fair chance.** Three diffusion-specific bugs — each
> independently sufficient to explain the loss — cripple it, and **the regression baseline (B1)
> accidentally avoids all three**. The B1-vs-TopoBrain comparison is therefore *not*
> "diffusion vs regression"; it is **"broken diffusion vs clean regression."**
>
> **We cannot publish "regression beats diffusion" from this experiment.** A reviewer would say
> the diffusion baseline is misimplemented, and they would be right. The bugs must be fixed and
> the comparison re-run before ANY conclusion is drawn.

---

## TIER 1 — Bugs that invalidate results (fix before anything else)

### A1. The model has only ever seen **288 unique patches** (0.33% of the brain)
`src/synthesis_dataset.py:465-476`
```python
center_idx = (patch_idx * 7919) % len(valid_centers)   # patch_idx = idx % 32
```
Patch choice is a **pure function of the index** — there is no random draw. Proven by execution:
```
valid centers available per volume : 9,699
distinct centers ever used         : 32   (0.33%)
epochs 0..4 identical               : True
EFFECTIVE TRAINING SET: 9 subjects x 32 = 288 unique patches, each shown ~4,167x
```
The ±5-voxel jitter (`:479-485`) gives 78% overlap — it does not create new patches.
**Fix:** draw `center = valid_centers[rng.integers(len(valid_centers))]` per `__getitem__`.
`patches_per_volume` should set epoch length only. **~34x more unique data.**
**Impact:** affects BOTH models equally (B1 used the same loader), so the head-to-head is
internally fair — but both are severely data-starved, and diffusion (far more data-hungry) is
hurt most.

### A2. The auxiliary pixel loss inverts the SNR weighting (algebraic identity, not a guess)
`src/diffusion.py:377-380`
Because `x_recon - x0 = -sqrt_recipm1(t)·(eps_pred - eps)` exactly:
```
loss_pixel(t)  ==  sqrt_recipm1(t) · L1(eps_pred, eps)
total          ==  [1 + lambda_pixel · sqrt_recipm1(t)] · L1(eps_pred, eps)
```
`loss_pixel` is **not an independent loss** — it is `loss_diff` scaled by a t-dependent factor.
Effective weight: **t=0 -> 1.02 ; t=184 -> 9.52**. 86% of the weight mass lands on the noisy half;
38% on the top 10% of timesteps. This is the **exact inverse** of Min-SNR / P2 weighting: the model
trains ~9x harder on the pure-noise regime than on the low-noise regime where PSNR/SSIM live.
**This is why B1 wins on image quality.** B1 has no such term.
**Fix:** drop `loss_pixel` (it is redundant) or normalize it by `sqrt_recipm1(t)`; or use
SNR-clipped weighting `min(SNR(t), gamma)`.

### A3. The segmentation head is trained at random t but only ever USED at t=0
`src/diffusion.py:353, :361`; `p_sample_loop:277-279`
`seg_pred` is computed on `x_noisy` at a **uniform random t in [0,185)**, but at inference the seg
is taken at **t=0**. Only **0.54%** of training steps are at t=0; only ~21% have alpha_bar > 0.9.
B1's seg head sees the test-time distribution on **100%** of steps.
=> TopoBrain's seg head gets a **~5x weaker training signal in the only regime it is evaluated in**,
and the topology loss inherits the same dilution. Plus exposure bias: training feeds
`q_sample(real 7T)`; inference feeds the model's own sample.
**This alone explains losing tissue Dice AND topology.** B1 has no such mismatch.
**Fix:** compute the seg/topo loss on a second forward at t=0, or bias timestep sampling toward
low t for those terms, or run the seg head off the clean conditioning path.

### A4. Checkpoint selection was done ON THE TEST SUBJECT — via a single patch
`src/synthesis_dataset.py:51, :761-763`; `scripts/evaluate_all_checkpoints.py:208-296`
```python
train, val, test = splitter.get_split(pairs, val_fold=val_fold, test_fold=val_fold)  # test := val
```
`test_fold: 1` in the config is **dead**. Proven: `val == test == ['sub-06']`, and train/val do NOT
overlap (no subject leakage — that part is fine). But `evaluate_all_checkpoints.py` then selects the
best checkpoint by **max SSIM on sub-06**, i.e. **model selection on the test subject**. Worse: it
builds that loader with `patches_per_volume=1`, so with A1 `center_idx = 0` —
**the 110k checkpoint was chosen by SSIM on ONE 64^3 patch, at one fixed location, of one subject.**
**Impact:** every reported test number is selection-biased upward.

### A5. The LOOCV harness is broken — a stateful RNG reshuffles on every call
`src/synthesis_dataset.py:79` (`self._rng.shuffle`) called from `get_split:111` and `get_cv_splits:147`
```
get_cv_splits() 10-fold LOOCV:
  NEVER validated : ['sub-01','sub-02','sub-04','sub-07']
  validated >1x   : sub-03 (x3), sub-08, sub-10
```
**Any LOSO/LOOCV run through this harness is invalid.** MUST be fixed before R1.4/R2.7 work.
**Fix:** compute folds once, or reseed `np.random.default_rng(seed)` at the top of `create_folds`.
(Training itself is reproducible — it makes a single `get_split` call on a fresh splitter.)

---

## TIER 2 — Bugs that degrade training/results

### B1. The 15-timestep exclusion rests on a FALSE premise, and causes an off-manifold sampler
`src/diffusion.py:228-229, :349-353`
The comment claims coefficients exceed 4000x for t > 185. Verified numerically:
```
max sqrt_recip   over t in [0,184] = 8.575   (clamp = 10.0) -> NEVER FIRES
max sqrt_recipm1 over t in [0,184] = 8.517   (clamp = 10.0) -> NEVER FIRES
4058x blowup occurs at t=199 ONLY;  t=198 is already fine (alpha_bar = 6.07e-05)
```
It is a **15x overcorrection for a 1-step problem**, and it *causes* the next bug:
`alphas_cumprod[184] = 0.0136` -> `sqrt = 0.117`, i.e. at t=184 the model always saw ~12% signal,
but both samplers start from **pure `torch.randn`** (0% signal). SNR mismatch ~1379x.
Partially self-healing for a *conditional* model, but a real defect.
**Fix:** `max_safe_timestep = len(betas) - 1` (drop only t=199) and raise/remove the dead clamps.

### B2. `p_sample` (DDPM) never clips x0-hat to the data range — and it is the DEFAULT sampler
`src/diffusion.py:235` (clamped to [-2,2], shared by training and sampling); `ddim_sample:327`
clamps to [-1,1] correctly. `scripts/evaluate_full_volume.py:538` **defaults to `--sampler ddpm`**.
Out-of-range mass accumulates over 185 steps -> lowers PSNR. (Our Table-1 rerun used DDIM, so it
escaped this; any DDPM-sampled result did not.)
**Fix:** add `clip_denoised=True` for sampling; keep [-2,2] for training only.

### B3. Zero-padding an image whose background is -1
`src/synthesis_dataset.py:439` (`np.zeros`) and `padding_mode="zeros"` at `:237`, `:246`
Background is **-1.0** after diffusion normalization, so padding with **0.0** injects a *bright,
brain-intensity* halo. The elastic/affine warps do the same at their borders.
**Fix:** `np.full(patch_size, -1.0)`; `padding_mode="border"` (or a -1 constant).

### B4. EMA freezes for 2000 steps after every resume; `ema_start` is dead config
`scripts/train_diffusion.py:53, :472, :519-526`
`ema_start` from the YAML is **never read**. `ema.step` is **not checkpointed**, so on every
`--resume` the warm-up counter restarts and the EMA receives **zero updates for 2000 steps** while
the online model keeps moving. This repo trains on Kaggle with repeated resumes.
**Fix:** pass `config['training']['ema_start']`, persist `ema.step`, and copy weights during warm-up.

### B5. Non-finite losses are silently zeroed -> stability logs are meaningless
`src/diffusion.py:406-418` (`_safe`) + `scripts/train_diffusion.py:425-440`
`_safe` replaces a non-finite loss with a fresh graph-less leaf (zero gradient for that term).
Because it sanitizes everything first, the training loop's `non_finite_keys` check **can never
trigger**, so the logged `skipped_steps`/`skip_ratio` structurally under-reports NaN-loss events as
**zero**. Any stability claim based on those logs is unfounded.

### B6. No `persistent_workers` -> caches are a no-op; severe GPU starvation
`src/synthesis_dataset.py:786-793`
An "epoch" is only 288/8 = 36 batches. Workers are re-forked every 36 steps, and each re-runs
`_get_valid_centers` (10,000 candidates x 64^3 mask means ~2.6e9 voxel ops per volume) from scratch,
in all 6 workers. Both the volume cache and the center cache never survive.
**Fix:** `persistent_workers=True`; precompute valid centers once (a summed-area table makes it <1s).

### B7. MONAI transforms are never reseeded per worker/epoch
`src/synthesis_dataset.py:221-248` with a plain `torch.utils.data.DataLoader`
All 6 workers inherit the **same** `RandomState` -> identical augmentations in lockstep; and because
workers re-fork each 36-batch epoch from a parent whose RNG never advances, **the augmentation
sequence restarts identically every epoch**.
**Fix:** use `monai.data.DataLoader`, or a `worker_init_fn` that reseeds per worker.

### B8. LR decay uses exact equality and sits after two `continue`s -> may never fire
`scripts/train_diffusion.py:461-468`: `if step == lr_decay_step`. If that exact step is skipped by a
NaN-grad guard, the LR never decays. **Fix:** `>=`, or a real `lr_scheduler`, placed before the guards.

---

## TIER 3 — Paper<->code mismatches a reviewer can find in 30 seconds
- **`dropout: 0.1`** in `configs/train_diffusion.yaml` is **silently ignored** — `AnatomyGuidedUNet`
  has no dropout parameter and the training script never passes it.
- **Dead `forward()`** (`src/model.py:167-212`) returns a hard-coded `torch.zeros(...)` segmentation
  placeholder; overwritten by `forward = forward_full` at `:251`. Harmless today, a landmine on any
  refactor, and confusing to any reviewer reading the repo.
- Parameter count (22.4M -> **13.98M**), the loss equations, and the "105k" checkpoint label were all
  wrong (fixed; see `discrepancies.md`).

---

## VERIFIED CORRECT (do not waste revision time here)
- **DDPM math**: cosine/linear schedules, `alphas_cumprod`, `posterior_variance`,
  `posterior_mean_coef1/2`, `q_sample`, `q_posterior`, `predict_start_from_noise` formula — all
  correct vs Ho et al. 2020 / Nichol & Dhariwal.
- **DDIM** (`:292-340`): correct vs Song et al.; `eta=0` is truly deterministic; timestep sequence has
  no duplicates; final step correct.
- **Segmentation extraction** happens at t=0 in BOTH samplers (the problem is A3, training dilution —
  not extraction).
- **`forward_full` skip-connection indexing** (`skips[-(i+2)]`) is correct; both decoders share the
  encoder features properly.
- **Augmentation is applied JOINTLY** to input/target/seg (MONAI dict transforms randomize once per
  call; seg uses `nearest`). **A spatial-transform desync was suspected and is NOT present.**
- **No train/val subject leakage** (train ∩ val = {}).
- **The live normalization path is correct** ([-1,1], background = -1).

---

## FIX ORDER (by expected recovery)
1. **A1** — random patch center per `__getitem__`. (~34x more data; biggest single change.)
2. **A3** — seg/topo loss on a t=0 (or low-t) forward. (Biggest win on Dice + topology.)
3. **A2** — drop or SNR-normalize `loss_pixel`. (Biggest win on PSNR/SSIM.)
4. **A4/A5** — real held-out test fold; stop selecting checkpoints on the test subject; fix the
   stateful splitter RNG **before** any LOSO.
5. **B1/B2** — restore a valid pure-noise init (drop only t=199) and clip x0-hat to [-1,1] when sampling.
6. **B3–B8** — padding value, EMA resume state, honest NaN accounting, `persistent_workers`, MONAI
   reseeding, LR decay.

## THEN, and only then
Retrain BOTH models on the fixed pipeline and re-run the head-to-head. Report whatever it says.
The 2026 literature (YODA, IEEE TMI; SynthRAD2023, MedIA) suggests regression may still win — but
that becomes a *defensible* claim only once the diffusion implementation is correct.
See `STRATEGY_2026.md`.
