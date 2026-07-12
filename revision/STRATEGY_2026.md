# Strategy after the B1 result — evidence from the 2025/2026 literature

**Date: 12 July 2026.** Written after the controlled experiment in which a deterministic
regression U-Net (same backbone, no diffusion, no topology loss, no perceptual loss) **beat**
TopoBrain on image quality, tissue Dice, and topology (see `kaggle/b1_vs_topobrain_sub06.md`).

## 1. The result is not an anomaly — it reproduces a published finding

- **Rassmann, Kügler, Ewert, Reuter — "Regression is all you need for medical image translation"
  (YODA).** arXiv:2505.02048; reported as IEEE TMI 2026 (VERIFY venue before citing).
  Code: github.com/Deep-MI/YODA
  - Regression matches/exceeds full diffusion sampling across 5 datasets, beating 8 SOTA
    baselines (ResViT, I2I-Mamba, Ea-GAN; SynDiff, SelfRDB, MADM, ALDM, Choo et al.).
  - Mechanism (verbatim): *"Superior perceptual realism of DMs is largely driven by their
    replication of acquisition noise, rather than improvements in medically-relevant accuracy."*
  - Downstream **segmentation Dice was better with regression** — matches our CSF/GM/WM result.
- **Huijben et al., SynthRAD2023 challenge report,** *Medical Image Analysis* 2024
  (arXiv:2403.08447): Transformer 0.88 > CNN/U-Net 0.85 > GAN 0.83 > **Diffusion 0.82** (last).
- **Moschetto et al.,** arXiv:2507.14575 (2025): for paired T1w→T2w, **pix2pix beats diffusion
  AND flow matching**; flow models overfit on small datasets.

**Conclusion: diffusion is the wrong tool for PAIRED translation. We did not do it wrong.**
Perception-distortion (Blau & Michaeli, CVPR 2018) explains the SSIM/PSNR gap — but it does
**NOT** explain a 100–1000x Betti blowup. That is residual-noise fragmentation, and we must
prove it (see §4).

## 2. Three hard truths

1. **Our "edge-aware topology loss" is neither novel nor topological.** FS-RWKV (Lei et al.,
   arXiv:2510.08951) and LiteMamba-Synth (Zhang et al., *Front. Neuroanat.* 2026,
   doi 10.3389/fnana.2026.1886476) both use **L1 + SSIM + Sobel edge** on this exact task and do
   not call it topology. Naming it "topology" is likely a direct cause of rejection.
2. **Our setup is already published.** **Acs & Zhuang, PLOS ONE 2025**
   (doi 10.1371/journal.pone.0333499): *same* 10-subject Chen dataset, *same* 64³ patches,
   *same* 9-train/1-test, with **10-fold LOOCV**. Must cite and beat. A reviewer will find it.
3. **Diffusion is absent from the 3T→7T literature.** FS-RWKV compares 8 baselines, LiteMamba 5 —
   none are diffusion. SOTA is efficient deterministic regression (transformer / RWKV / Mamba).
   Also relevant: **Cui et al., MICCAI 2024** (arXiv:2403.08979) — V-Net-**SSeg** is a
   segmentation-guided regression net, i.e. our dual-decoder idea, already published.

## 3. The genuine, unclaimed novelty
Persistent-homology losses exist for **segmentation** (Hu 2019; Clough 2020; Byrne TMI 2022;
Stucki MICCAI 2024) and for **GANs** (TopoGAN, ECCV 2020) — but **no paper applies a PH-based
topology loss to paired MRI *synthesis*.** That gap is ours.

Our **dual decoder** (image + 4-class seg), which currently earns nothing, is exactly what a PH
loss needs: it emits the discrete label map PH operates on. It finally has a purpose.

## 4. Recommended paper

> ~~"TopoBrain: conditional diffusion + edge-aware topology loss"~~
> **"Diffusion is not needed for 3T→7T synthesis: a deterministic regression U-Net with a true
> topological loss."**

**Architecture:** keep the 13.98 M dual-decoder U-Net (32/64/128/256, attention bottleneck).
**Delete the diffusion process.** Train deterministically (converges in ~17k steps, not 110k).

**Loss:**
```
L = L1(image) + λ_ssim·(1 − SSIM) + λ_seg·(CE + Dice)(seg) + λ_topo·L_PH_multiclass(seg)
```
- **Add:** multi-class persistent-homology loss — **Byrne, Clough, Valverde, Montana, King,
  IEEE TMI 2022**, doi 10.1109/TMI.2022.3203309 — explicitly built for **multi-class 3D
  volumetric** anatomy. Best fit.
  - Cheaper fallback: **fast Euler-characteristic loss**, Li, Ma, Ouyang, Paetzold, Rueckert,
    Kainz, IEEE TMI 2025 (arXiv:2507.23763). (Public code NOT confirmed — check.)
- **Drop VGG perceptual** (keep one ablation row proving it doesn't help Dice/Betti — itself a
  finding, consistent with YODA).
- **Drop/rename the Sobel loss** — cite FS-RWKV/LiteMamba as prior art.
- **Staged training is essential:** ~10–15k steps of L1+CE first, *then* switch on the topology
  loss at low weight. PH losses are unstable on early garbage predictions.

**clDice (Shit et al., CVPR 2021) is the WRONG tool — do not use.** Its guarantee requires
foreground/background homotopy-equivalent to *graphs* (tubular: vessels, roads, neurons). CSF/GM/WM
are blobs/sheets. Using it is an easy reviewer kill.

## 5. Five must-fixes (non-negotiable)
1. **n=1 → full 10-fold LOOCV.** Both direct competitors do it. Subject-level statistics only
   (n=10) — never patch-level (pseudo-replication).
2. **Explain the diffusion Betti blowup (753 vs GT 5)** as residual-noise fragmentation and
   **prove it** (denoise the DDPM output → Betti collapses while SSIM barely moves). Otherwise a
   reviewer concludes "your diffusion baseline is broken" and rejects.
3. **Report voxel connectivity (6/18/26)** for every Betti number, and report **Betti *matching***
   error, not raw counts — **Berger, Lux, Weers, Menten, Rueckert, Paetzold, "Pitfalls of
   topology-aware image segmentation," arXiv:2412.14619.** Also: GT labels themselves carry
   topological artifacts (up to 43% of measured error).
4. **Cite and compare against Acs & Zhuang (PLOS ONE 2025)** — same dataset, same protocol.
5. **Fix the evaluation** — see `AUDIT.md`: SSIM/PSNR are inflated by pasting GT into ~81% of the
   volume; "Dice"/HD95 are computed on intensity thresholds, not segmentations.

**Armour for a "simple method wins" paper:** Isensee et al., nnU-Net, *Nature Methods* 2021
(*"pipeline configuration is more important than architectural variations"*); Varoquaux &
Cheplygina, *npj Digital Medicine* 2022 (weak baselines / inflated claims); Maier-Hein et al.,
"Metrics reloaded," *Nature Methods* 2024.

## 6. Ranked options (publishability × feasibility, ~100 GPU-h on a Kaggle P100)
| # | Option | Verdict |
|---|---|---|
| 1 | Regression U-Net + multi-class PH loss (Byrne TMI 2022) + full 10-fold LOOCV | ✅ **DO THIS** |
| 2 | Same, with Euler-characteristic loss instead of PH | ✅ fallback if PH is too costly |
| 3 | Regression U-Net, no topo loss; rigorous LOOCV + strong baselines + honest diffusion ablation | ✅ **safety floor — guaranteed publishable** (MELBA/MIDL/Sci Rep/MICCAI workshop) |
| 4 | Swap backbone to Mamba/RWKV/transformer | ⚠️ incremental; cite as baselines, don't lead |
| 5 | Keep/fix diffusion | ❌ fights YODA + SynthRAD; 110k × 10 folds unaffordable. Baseline row only. |

## Citations to VERIFY before use (flagged honestly by the research pass)
- YODA venue (IEEE TMI 2026) — arXiv:2505.02048 is certain; confirm the journal reference.
- 7T-Restormer, *Medical Physics* 2026 (doi 10.1002/mp.70496) — arXiv:2507.08655 confirmed.
- Topograph (arXiv:2411.03228) — no peer-reviewed venue confirmed; treat as preprint.
- Euler-characteristic loss (arXiv:2507.23763) — public code NOT confirmed.
- BrainIAC as a perceptual backbone — model is real/public, but this *use* is untested.
