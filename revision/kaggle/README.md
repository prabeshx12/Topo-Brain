# Kaggle GPU experiments (revision)

Reproducible harness for the checkpoint-gated revision experiments. Runs on Kaggle's free GPU
(~30 h/week), which is sufficient for all inference-only jobs.

## Why Kaggle
The trained checkpoint lives there as a Kaggle **Model** (`pratikadhikari9/mri-ckpt-110000`,
step 110,000, 224 MB), and the preprocessed volumes are Kaggle **Datasets**
(`preprocessed-mri-aligned`, `seg-masks`). No local GPU is required.

## Files
- `topobrain_eval_110k.py` — the kernel script. One tiled inference per sampler, then:
  - **Table 1 reproduction** (SSIM/PSNR/Dice/HD95) under three brain masks
    (`Threshold(-0.95)`, `SegMaskGT`, `BrainMaskNII`), for both `ddim` (50 steps) and
    `ddpm` (185 steps) — because the paper does not state which sampler produced Table 1.
  - **per-tissue Dice** (CSF/GM/WM) of predicted vs ground-truth segmentation.
  - **true topological metrics** — Betti numbers (β0, β1, β2) via `gudhi` cubical complex and
    the Euler characteristic χ = β0 − β1 + β2, plus connected-component counts and
    largest-CC fraction, for predicted **and** ground-truth segmentations (reviewer R2.3/R2.8).
  - Saves `sub-06_pred7T_<sampler>.nii.gz`, `sub-06_predseg_<sampler>.nii.gz`,
    and `eval_110k_results.json`.
- `kernel-metadata.json` — kernel config (GPU + internet on; attaches the model and both datasets).
- `make_code_bundle.py` — builds the POSIX-safe `topobrain_code.zip` uploaded as the
  `topobrain-code-rev` dataset. Ships an **empty `src/__init__.py`** so `src.diffusion` imports
  without pulling in `monai`.

## Run it
```bash
python revision/kaggle/make_code_bundle.py            # builds topobrain_code.zip
kaggle datasets create -p <bundle_dir>                # first time only (topobrain-code-rev)
kaggle datasets version -p <bundle_dir> -m "update"   # subsequent updates
kaggle kernels push -p revision/kaggle                # pushes AND runs
kaggle kernels status pratikadhikari9/topobrain-eval-110k
kaggle kernels output pratikadhikari9/topobrain-eval-110k -p results/kaggle_eval_110k
```

## What this decides
**D8** (see [../discrepancies.md](../discrepancies.md)): the paper attributes Table 1 to a
"105k EMA checkpoint", but only step **110,000** survives. If some sampler/mask combination
reproduces SSIM ≈ 0.8991 and brain HD95 ≈ 2.05, the "105k" label is a writing error and the text
is corrected. If nothing matches, the reported numbers came from a lost checkpoint and **all of
Table 1 must be regenerated at 110k**.

Results are reported exactly as produced. Nothing is fabricated.
