# Kaggle smoke-test: 10-subject ADNI synthesis with the sub-06 145k checkpoint.
#
# This is jupytext-style: cells separated by `# %%`. Paste each cell into a
# Kaggle notebook (or `pip install jupytext && jupytext --to ipynb kaggle_adni_smoke.py`).
#
# REQUIRED INPUTS attached to the Kaggle notebook (Settings -> Input):
#   1. The 145k checkpoint Kaggle Dataset (the .pt file).
#   2. An ADNI dataset whose root contains adni_nifti/AD/<ptid>_T1w.nii(.gz)
#      and adni_nifti/CN/<ptid>_T1w.nii(.gz). Optional: pairs_adni_smoke.csv
#      from build_adni_cohort.py (we reconstruct it from filenames if missing).
#
# Notebook accelerator: GPU T4 x1 (or P100). Internet: ON (HD-BET downloads weights).
#
# Runtime expectations on T4:
#   Cell 5 preprocess     ~5-9 min/subject (N4 is CPU-only)        ~70-90 min total
#   Cell 7 synthesis      ~3-5 min/subject (DDIM 50 steps on GPU)  ~30-50 min total
#
# IMPORTANT: Kaggle wipes /kaggle/working/ when the kernel dies. Run Cell 6
# (zip preprocessed bundle) before Cell 7 so a tab reload doesn't cost you
# 90 minutes of N4. Then "Save Version" with output to persist as a Dataset.

# %% [markdown]
# ## Cell 1: Configure — EDIT THESE TWO PATHS

# %%
# Replace the placeholders with your actual Kaggle Dataset paths.
# Find them in /kaggle/input/ once the datasets are attached.
CHECKPOINT_PATH  = "/kaggle/input/<your-checkpoint-dataset>/checkpoint_145000.pt"
ADNI_DATASET_DIR = "/kaggle/input/<your-adni-dataset>"   # contains adni_nifti/

REPO_BRANCH = "feat/train-with-masks"
SAMPLER     = "ddim"     # ddim is ~10x faster than ddpm, similar PSNR
DDIM_STEPS  = 50
N_SAMPLES   = 1          # 1 for smoke test; 5+ for production averaging

# %% [markdown]
# ## Cell 2: Clone repo and install missing deps

# %%
import os, subprocess, sys
WORK = "/kaggle/working"
REPO = f"{WORK}/Topo-Brain"
if not os.path.isdir(REPO):
    subprocess.check_call(["git", "clone", "-b", REPO_BRANCH,
                           "https://github.com/prabeshx12/Topo-Brain.git", REPO])
else:
    subprocess.check_call(["git", "-C", REPO, "pull"])
os.chdir(REPO)
print("repo head:", subprocess.check_output(["git", "log", "--oneline", "-1"]).decode().strip())

# Kaggle base images already have torch, monai, nibabel, SimpleITK, pandas.
# Only HD-BET is consistently missing. nnUNet is HD-BET's transitive dep.
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
print(f"checkpoint: {CHECKPOINT_PATH}  "
      f"({Path(CHECKPOINT_PATH).stat().st_size/1e6:.1f} MB)")

# %% [markdown]
# ## Cell 4: Stage ADNI inputs + (re)build the cohort CSV
#
# Kaggle inputs are read-only — HD-BET writes brain masks alongside the
# NIfTIs, so we copy to a writable location. We also rebuild pairs CSV from
# filenames so this works whether or not the original CSV is in the dataset
# (and regardless of whether Kaggle decompressed `.nii.gz` -> `.nii`).

# %%
import shutil, pandas as pd

# Locate adni_nifti — accommodate both flat and nested dataset layouts.
candidates = [Path(ADNI_DATASET_DIR) / "adni_nifti",
              Path(ADNI_DATASET_DIR)]
ADNI_SRC = next((p for p in candidates
                 if any(p.glob("[AC][DN]/*_T1w.nii*"))), None)
assert ADNI_SRC, f"No adni_nifti structure found under {ADNI_DATASET_DIR}"
print(f"ADNI source: {ADNI_SRC}")

ADNI_LOCAL = Path(WORK) / "adni_nifti"
shutil.copytree(str(ADNI_SRC), str(ADNI_LOCAL), dirs_exist_ok=True)

rows = []
for grp in ("AD", "CN"):
    for f in sorted((ADNI_LOCAL / grp).glob("*_T1w.nii*")):
        rows.append({"ptid": f.name.split("_T1w.")[0],
                     "group": grp,
                     "nifti_path": str(f)})
COHORT_CSV = f"{WORK}/pairs_adni_smoke.csv"
df = pd.DataFrame(rows)
assert len(df) > 0, f"No NIfTIs found under {ADNI_LOCAL}"
df.to_csv(COHORT_CSV, index=False)
print(df.to_string(index=False))

# %% [markdown]
# ## Cell 5: Preprocess (HD-BET + N4 + diffusion-normalize)
#
# ~5-9 min/subject on Kaggle's CPU. About 75 min for 10 subjects.
# HD-BET masks are cached so re-runs only redo N4 + normalize.

# %%
PRE_DIR = f"{WORK}/adni_preprocessed"
PRE_CSV = f"{WORK}/pairs_adni_smoke_preprocessed.csv"
subprocess.check_call([sys.executable, "scripts/preprocess_adni.py",
    "--cohort-csv", COHORT_CSV,
    "--output-dir", PRE_DIR,
    "--output-csv", PRE_CSV,
    "--device", "cuda"], cwd=REPO)

print("\n=== preprocessing summary ===")
print(pd.read_csv(PRE_CSV)[["ptid", "group", "preprocess_status"]].to_string(index=False))

# %% [markdown]
# ## Cell 6: Save preprocessed bundle (resume checkpoint)
#
# Run this BEFORE synthesis so a kernel restart doesn't cost you 75 min of N4.
# Re-upload the zip as a private Kaggle Dataset to skip preprocessing next time.

# %%
import tempfile

PRE_BUNDLE = f"{WORK}/adni_preprocessed_bundle.zip"
stage = Path(tempfile.mkdtemp(prefix="prebundle_"))
shutil.copytree(PRE_DIR, stage / "adni_preprocessed")
shutil.copy(COHORT_CSV, stage / Path(COHORT_CSV).name)
shutil.copy(PRE_CSV,    stage / Path(PRE_CSV).name)

shutil.make_archive(PRE_BUNDLE.replace(".zip", ""), "zip", str(stage))
shutil.rmtree(stage)
print(f"preprocessed bundle: {PRE_BUNDLE}  "
      f"({os.path.getsize(PRE_BUNDLE)/1e6:.1f} MB)")
print("Download from Kaggle's Output panel before continuing.")

# %% [markdown]
# ## Cell 7: Synthesize (DDIM, 50 steps, per-subject)
#
# ~3-5 min/subject on T4 GPU. Outputs land in results_adni_smoke/<ptid>/.

# %%
RESULTS = f"{WORK}/results_adni_smoke"
os.makedirs(RESULTS, exist_ok=True)

pre_df = pd.read_csv(PRE_CSV)
ok = pre_df[pre_df["preprocess_status"].isin(["ok", "skipped_exists"])].reset_index(drop=True)
print(f"Synthesizing {len(ok)} subjects with {SAMPLER} ({DDIM_STEPS} steps).\n")

for i, sub in ok.iterrows():
    ptid = sub["ptid"]
    out_subj = f"{RESULTS}/{ptid}"
    os.makedirs(out_subj, exist_ok=True)
    print(f"\n=== [{i+1}/{len(ok)}] {ptid} ({sub['group']}) ===", flush=True)
    subprocess.check_call([sys.executable, "scripts/evaluate_full_volume.py",
        "--checkpoint", CHECKPOINT_PATH,
        "--input", sub["preprocessed_path"],
        "--output_dir", out_subj,
        "--sampler", SAMPLER,
        "--ddim-steps", str(DDIM_STEPS),
        "--n-samples", str(N_SAMPLES)], cwd=REPO)

# %% [markdown]
# ## Cell 8: Visual sanity check — input vs synthetic on one subject

# %%
import matplotlib.pyplot as plt, nibabel as nib, numpy as np
sample = ok.iloc[0]
inp  = nib.load(sample["preprocessed_path"]).get_fdata()
pred = nib.load(f"{RESULTS}/{sample['ptid']}/predicted_7T.nii.gz").get_fdata()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, vol, title in zip(axes, [inp, pred],
                          [f"{sample['ptid']} input 3T (preprocessed)",
                           f"{sample['ptid']} synthetic 7T"]):
    ax.imshow(np.rot90(vol[:, :, vol.shape[2]//2]), cmap="gray")
    ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Cell 9: Save synthesis bundle for FastSurfer transfer
#
# Zip predicted_7T + predicted_seg outputs for download. These go to CERN
# next for FastSurfer + region preservation.

# %%
RESULTS_BUNDLE = f"{WORK}/adni_smoke_results.zip"
shutil.make_archive(RESULTS_BUNDLE.replace(".zip", ""), "zip", RESULTS)
print(f"results bundle: {RESULTS_BUNDLE}  "
      f"({os.path.getsize(RESULTS_BUNDLE)/1e6:.1f} MB)")
print("Download from Kaggle's Output panel.")
