# Kaggle smoke-test: 10-subject ADNI synthesis with the sub-06 145k checkpoint.
#
# This is jupytext-style: cells separated by `# %%`. Paste each cell into a
# Kaggle notebook (or `pip install jupytext && jupytext --to ipynb kaggle_adni_smoke.py`).
#
# REQUIRED INPUTS attached to the Kaggle notebook (Settings -> Input):
#   1. The 145k checkpoint Kaggle Dataset (the .pt file).
#   2. An ADNI dataset containing:
#        - adni_nifti/AD/<PTID>_T1w.nii.gz   (5 files)
#        - adni_nifti/CN/<PTID>_T1w.nii.gz   (5 files)
#        - pairs_adni_smoke.csv              (cohort manifest from build_adni_cohort.py)
#      Upload these from your local repo's adni_nifti/ + adni_dataset/pairs_adni_smoke.csv.
#
# Notebook accelerator: GPU T4 x1 (or P100). Internet: ON (HD-BET downloads weights).

# %% [markdown]
# ## Cell 1: Configure — EDIT THESE TWO PATHS

# %%
# Replace `<your-checkpoint-dataset>` with your Kaggle dataset slug.
# After attaching the dataset, find the actual file under /kaggle/input/.
CHECKPOINT_PATH = "/kaggle/input/<your-checkpoint-dataset>/checkpoint_145000.pt"
ADNI_DATASET_DIR = "/kaggle/input/<your-adni-dataset>"   # contains adni_nifti/ and pairs_adni_smoke.csv

REPO_BRANCH = "feat/train-with-masks"
SAMPLER = "ddim"        # ddim is ~10x faster than ddpm, similar PSNR
DDIM_STEPS = 50
N_SAMPLES = 1           # 1 for smoke test; 5 later for production

# %% [markdown]
# ## Cell 2: Clone repo and install missing deps

# %%
import os, subprocess, sys
WORK = "/kaggle/working"
REPO = f"{WORK}/Topo-Brain"
if not os.path.isdir(REPO):
    subprocess.check_call([
        "git", "clone", "-b", REPO_BRANCH,
        "https://github.com/prabeshx12/Topo-Brain.git", REPO,
    ])
os.chdir(REPO)
print("repo head:", subprocess.check_output(["git", "log", "--oneline", "-1"]).decode().strip())

# Kaggle base images already have torch, monai, nibabel, SimpleITK, pandas.
# Only HD-BET is consistently missing. nnUNet is HD-BET's dep.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "HD-BET", "scikit-image"])

# %% [markdown]
# ## Cell 3: Verify GPU + checkpoint visible

# %%
import torch
from pathlib import Path
assert torch.cuda.is_available(), "Enable GPU in Kaggle notebook settings."
print("GPU:", torch.cuda.get_device_name(0))

assert Path(CHECKPOINT_PATH).exists(), f"Checkpoint missing: {CHECKPOINT_PATH}"
assert Path(ADNI_DATASET_DIR).exists(), f"ADNI dataset missing: {ADNI_DATASET_DIR}"
print("checkpoint:", CHECKPOINT_PATH, "size:",
      f"{Path(CHECKPOINT_PATH).stat().st_size / 1e6:.1f} MB")

# %% [markdown]
# ## Cell 4: Stage ADNI inputs into /kaggle/working
#
# Kaggle inputs are read-only. We need to write brain masks alongside the
# NIfTIs, so we copy them to a writable location.

# %%
import shutil, pandas as pd
ADNI_LOCAL = f"{WORK}/adni_nifti"
shutil.copytree(f"{ADNI_DATASET_DIR}/adni_nifti", ADNI_LOCAL, dirs_exist_ok=True)

cohort_src = f"{ADNI_DATASET_DIR}/pairs_adni_smoke.csv"
cohort_local = f"{WORK}/pairs_adni_smoke.csv"
shutil.copy(cohort_src, cohort_local)

# The cohort CSV from build_adni_cohort.py has Windows-style absolute paths.
# Rewrite nifti_path to point at the staged Linux location on Kaggle.
df = pd.read_csv(cohort_local)
df["nifti_path"] = df.apply(
    lambda r: f"{ADNI_LOCAL}/{r['group']}/{r['ptid']}_T1w.nii.gz", axis=1)
df.to_csv(cohort_local, index=False)
print(df[["ptid", "group", "age", "sex", "nifti_path"]].to_string(index=False))

# %% [markdown]
# ## Cell 5: Preprocess (HD-BET + N4 + diffusion-normalize)
#
# Roughly 1-2 min/subject on T4. Bundles HD-BET first run weight download.

# %%
PRE_DIR = f"{WORK}/adni_preprocessed"
PRE_CSV = f"{WORK}/pairs_adni_smoke_preprocessed.csv"
subprocess.check_call([
    sys.executable, "scripts/preprocess_adni.py",
    "--cohort-csv", cohort_local,
    "--output-dir", PRE_DIR,
    "--output-csv", PRE_CSV,
    "--device", "cuda",
])
pd.read_csv(PRE_CSV)[["ptid", "group", "preprocess_status", "preprocessed_path"]]

# %% [markdown]
# ## Cell 6: Synthesize (DDIM, 50 steps, per-subject)
#
# Roughly 2-5 min/subject on T4. Outputs go to results/<PTID>/.

# %%
RESULTS = f"{WORK}/results_adni_smoke"
os.makedirs(RESULTS, exist_ok=True)

pre_df = pd.read_csv(PRE_CSV)
ok = pre_df[pre_df["preprocess_status"].isin(["ok", "skipped_exists"])]
print(f"Synthesizing {len(ok)} subjects.")

for _, sub in ok.iterrows():
    ptid = sub["ptid"]
    inp = sub["preprocessed_path"]
    out_subj = f"{RESULTS}/{ptid}"
    os.makedirs(out_subj, exist_ok=True)
    print(f"\n=== {ptid} ({sub['group']}, age {sub['age']}, {sub['sex']}) ===")
    subprocess.check_call([
        sys.executable, "scripts/evaluate_full_volume.py",
        "--checkpoint", CHECKPOINT_PATH,
        "--input", inp,
        "--output_dir", out_subj,
        "--sampler", SAMPLER,
        "--ddim-steps", str(DDIM_STEPS),
        "--n-samples", str(N_SAMPLES),
    ])

# %% [markdown]
# ## Cell 7: Visual sanity check — input vs synthetic on one subject
#
# Eyeballing midline sagittal/coronal/axial. If brains look like brains
# (no garbage / total noise), proceed to FastSurfer + region check on CERN.

# %%
import matplotlib.pyplot as plt, nibabel as nib, numpy as np
sample = ok.iloc[0]
ptid = sample["ptid"]
inp = nib.load(sample["preprocessed_path"]).get_fdata()
pred = nib.load(f"{RESULTS}/{ptid}/predicted_7T.nii.gz").get_fdata()

def show(vol, ax, title):
    z = vol.shape[2] // 2
    ax.imshow(np.rot90(vol[:, :, z]), cmap="gray")
    ax.set_title(title); ax.axis("off")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
show(inp, axes[0], f"{ptid} input 3T (preprocessed)")
show(pred, axes[1], f"{ptid} synthetic 7T")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Cell 8: Bundle outputs for download
#
# Kaggle saves /kaggle/working/ contents as the notebook's output. Zipping
# keeps the FastSurfer-ready NIfTIs together for transfer to CERN/local.

# %%
import shutil
shutil.make_archive(f"{WORK}/adni_smoke_results", "zip", RESULTS)
print("zipped:", os.path.getsize(f"{WORK}/adni_smoke_results.zip") / 1e6, "MB")
