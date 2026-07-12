# Benchmark reality check — we are BEHIND the published methods

**Date: 12 July 2026.** Written to correct a claim I made prematurely: that our regression
baseline (whole-volume SSIM 0.884 / PSNR 20.24 dB) "beat published SOTA (0.726 / 0.711)".
**It does not.** Verified against the actual papers.

## 1. The decisive arithmetic

FS-RWKV reports **RMSE alongside PSNR**, which lets us put both on the same footing:

```
FS-RWKV:  20*log10(1/0.0898) = 20.93 dB  ~= their reported 21.00   (confirms data_range=1.0,
                                                                    whole image)
Ours:     whole-volume PSNR 20.24 dB @ data_range 2.0
          -> RMSE = 2/10^(20.24/20) = 0.1945 in [-1,1] units = 0.0973 in [0,1] units

    FS-RWKV    RMSE 0.0898
    LiteMamba  RMSE 0.0920
    OURS       RMSE 0.0973   <-- WORSE THAN BOTH
```

**We lose on PSNR even under our own most-favourable convention**, and that 20.24 dB is
measured on a voxel set where ~85% is trivially exact (pred = target = -1 in the background).
The easier population still loses. This conclusion is robust.

## 2. The SSIM "win" was a background artifact
SSIM said +0.16 ahead; PSNR said -0.76 dB behind. **That divergence is the signature of
background inflation** -- SSIM saturates to 1.0 in flat matching regions, PSNR does not.
A contradiction between the two metrics should have been read as a warning, not reported as
a win.

## 3. Their pixel population is not ours (three compounding differences)

**The UNC dataset is NOT skull-stripped.** Chen et al. (*Scientific Data* 2023) describe the
entire pipeline as PyDeface + FSL FLIRT linear registration. No brain extraction. So every
UNC competitor synthesises and scores **full-head** images (skull, scalp, eyes, marrow).

| Paper | Region scored | Range | 2D/3D | Skull-stripped | Split |
|---|---|---|---|---|---|
| FS-RWKV (21.00 / 0.726) | whole 256x256 slice, no mask | [0,1] | **2D**, 101 central axial slices | No | fixed 7 train / 3 test |
| LiteMamba-Synth (20.82 / 0.711) | whole slice, no mask | [0,1] min-max | **2D** axial | No | subject-level split |
| **Acs & Zhuang (23.25 / 0.737)** | whole volume, no mask | [0,1] | 3D 64^3 patches | **No (explicit)** | **10-fold LOOCV** |
| WATNet (28.27 / 0.878) | whole image | [0,1] | 2D slices | **Yes** | LOOCV, private data |
| ~~Cui et al. (25.60 / 0.914)~~ | -- | -- | 3D | -- | **DIFFERENT DATASET (UCSF TBI) -- DROP** |

Verified with word-boundary regex over the extracted full text: FS-RWKV, LiteMamba and Cui
contain **zero** occurrences of `mask`, `background`, `skull`, `foreground`, or
`brain extraction`. **Not one of the five masks the background out.**

Epistemic status: no paper *explicitly says* "we include background". The whole-image reading
is a strong inference from (a) the total absence of masking language, (b) pipelines that
produce full-head images, and (c) the RMSE/PSNR algebra above -- not a direct quote.

## 4. The real benchmark
**Acs & Zhuang, PLOS ONE 2025** (doi 10.1371/journal.pone.0333499) is the closest competitor:
*same dataset, same 64^3 patches, same 10-fold LOOCV*, and explicitly reports results
*"even without preprocessing steps like skull stripping"*. They achieve
**PSNR 23.25 / SSIM 0.737**. We are ~3 dB below.

Corroborating signal: WATNet is the only paper with a convention close to ours
(skull-stripped, [0,1], unmasked whole image) and reports SSIM 0.878 / **PSNR 28.27** --
essentially our SSIM with 8 dB more PSNR. (Private dataset, so not a strict comparison, but
it indicates what an unmasked skull-stripped SSIM near 0.88 *should* come with.)

## 5. Consequences for the rebuild

1. **No borrowed comparisons.** We cannot put their numbers in a table against ours. The only
   defensible route is to **run the baselines ourselves, on our pipeline, with our metrics** --
   which is what reviewer R1.2 demanded anyway.
2. **The model must actually get better**, not merely be measured honestly. Grounds for
   optimism: B1 reached 20.24 dB while trained on **288 unique patches for 17k steps**. The
   Phase-3 run has **147x more unique patches** plus a cascaded seg head whose gradient
   actually reaches the generator.
3. **Report both conventions**, transparently: (a) whole-volume, no pasting -- the field's
   convention; (b) brain-region only -- the honest one. State the protocol difference
   explicitly rather than letting a reviewer discover it.
4. If we ever want a literal head-to-head with FS-RWKV/LiteMamba, we must reproduce their
   protocol exactly: released defaced+FLIRT volumes **without skull-stripping**, min-max to
   [0,1], 101 central axial slices, resized 256x256, SSIM/PSNR per slice with data_range=1.0,
   fixed 7/3 split, slice-level averaging (NOT per-subject LOOCV means).

## Lesson
A claim of superiority was made from an assumed metric convention. A reviewer holding
FS-RWKV's RMSE column could have refuted it in one line. Verify the convention before
comparing -- always.
