Run Instructions

This file explains how to run the preprocessing pipeline and where outputs are written.

Prerequisites
- Python 3.8+
- Install requirements: `pip install -r requirements.txt`
- Optional skull stripping tools:
  - HD-BET: `pip install HD-BET`
  - SynthStrip: install FreeSurfer 7.3+
  - FSL BET: install FSL

Quick Start (recommended pipeline)
1) Prepare a BIDS-like dataset root, e.g. `/path/to/BIDS`
2) Run preprocessing:

```bash
python -m scripts.preprocess_bids \
  --config configs/preprocess.yaml \
  --data-root /path/to/BIDS \
  --output-root /path/to/BIDS/derivatives/topobrain-preproc
```

Key flags
- `--skull-strip-method {hd-bet,synthstrip,fsl-bet,existing}`: Choose a skull stripping backend.
- `--skull-strip-device {cuda,cpu}`: Device for HD-BET.
- `--target-spacing X Y Z`: Enable resampling (set to `null` in config to disable).
- `--no-bias-correction`: Disable N4 bias correction.
- `--include-derivatives`: Include BIDS derivatives during discovery (useful if aligned data lives in `derivatives/`).
- `--overwrite`: Force reprocessing.
- `--no-resume`: Do not skip existing outputs.

Outputs
- Preprocessed volumes: `derivatives/topobrain-preproc/sub-*/ses-*/anat/*desc-preproc*.nii.gz`
- Brain masks: `derivatives/topobrain-preproc/sub-*/ses-*/anat/*desc-brainmask*.nii.gz`
- Manifest: `derivatives/topobrain-preproc/manifest.csv` and `manifest.jsonl`
- Pairing list: `derivatives/topobrain-preproc/pairs.csv` and `pairs.jsonl`
- Logs: `derivatives/topobrain-preproc/logs/preprocess.log`
- QC mask overlays (sampled): `derivatives/topobrain-preproc/qc/*.png`

Training and evaluation (using preprocessed data)

Train GAN:
```bash
python scripts/train_gan.py \
  --preprocessed-root /path/to/BIDS/derivatives/topobrain-preproc \
  --modality T1w \
  --device cuda
```

Evaluate GAN:
```bash
python scripts/eval_gan.py \
  --checkpoint /path/to/checkpoint.pth \
  --preprocessed-root /path/to/BIDS/derivatives/topobrain-preproc \
  --modality T1w \
  --device cuda
```

Legacy pipeline (not recommended)
The legacy MONAI pipeline can still be used, but it is not aligned with the research blueprint.

```bash
python scripts/example_pipeline.py --preprocess --legacy-preprocess
```
