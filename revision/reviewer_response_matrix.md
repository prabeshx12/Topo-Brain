# Reviewer → Code Response Matrix

One row per comment. **Validity** = VALID / PARTIAL / MISUNDERSTANDING. **Fix class** =
WRITING-ONLY · CODE+RERUN · NEW-EXPERIMENT · REFRAME. **Risk** = does it threaten the core
contribution? Effort/compute estimates assume one RTX 4090 (~1 s/step; full run ~40 h,
50k-step run ~13 h).

Legend for status: ✅ done (no-compute) · 🟡 queued (needs GPU/checkpoint/data) · 📝 writing pending.

---

## Reviewer 1

### R1.1 — LPIPS rejected for eval but VGG16 perceptual loss used in training
- **Validity:** VALID (genuine inconsistency in our writing, not a code bug).
- **Root cause in code:** [diffusion.py:49-102](../src/diffusion.py#L49-L102) uses VGG16/ImageNet
  features (layers 0-15) on 2D slices, weight 0.25 — confirmed. §3.4 rejects VGG/ImageNet for LPIPS.
- **Fix class:** WRITING-ONLY (core) + optional CODE+RERUN (robustness).
- **Plan:** Argue the real distinction — VGG-on-slices is a *weak, late-warmed-in training
  regularizer* (an uncalibrated prior is acceptable when it only nudges gradients), whereas LPIPS
  would be a *reported perceptual metric* presented as ground-truth quality (needs calibration).
  Offer the medical-pretrained swap (S6) as a queued robustness check.
- **Risk:** Low. **Status:** 📝

### R1.2 — No SOTA baselines, only self-ablation
- **Validity:** VALID. **Root cause:** repo has no baseline implementations.
- **Fix class:** NEW-EXPERIMENT. **Compute:** regression U-Net ~13-40 h; pix2pix/CycleGAN-3D ~40 h;
  WATNet/VCT only if public code runs on our data (feasibility research first).
- **Plan (S5):** Train (a) deterministic 3D regression U-Net (reuse our backbone, drop diffusion),
  (b) a 3D GAN baseline, on the same split; compare SSIM/PSNR/Dice/HD95/Betti. Cite WATNet/VCT and
  attempt at least one if code is available.
- **Risk:** **HIGH — threatens core.** Can't claim method quality without comparison. **Status:** 🟡

### R1.3 — Ablation too coarse (topology on/off only); isolate Sobel vs Dice
- **Validity:** VALID, and tied to D1 (the loss isn't literally "Sobel + Dice" — it's
  edge-weighted-CE + edge-Dice + multiscale).
- **Fix class:** CODE+RERUN. **Compute:** 3 short runs (~40 h total).
- **Plan (S2):** Add config flags to isolate {edge-weighted-CE only, boundary-Dice only,
  single-scale vs multi-scale}; retrain short variants; report Dice/HD95/CC + Betti per variant.
  First correct the loss description (D1) so the ablation maps to real terms.
- **Risk:** Medium. **Status:** 🟡

### R1.4 — 9 train / 1 test, 22.4 M params → overfitting
- **Validity:** VALID (n=1 quantitative test is the headline weakness). Note: params are 13.98 M
  not 22.4 M (D2) — slightly weakens the overfitting magnitude but not the n=1 concern.
- **Fix class:** NEW-EXPERIMENT + DATA + REFRAME. **Compute:** LOSO 3-5 short folds ~40-65 h.
- **Plan:** (S4) fix the splitter bug (D6), run LOSO on the 10 UNC subjects, report mean±std;
  (S7) add the free Figshare+ **20-subject** paired 3T/7T set as an *expanded external test set*
  (n=1 → n≥20); pursue the 279-pair cohort. Reframe generalization claims to the evidence.
- **Risk:** **HIGH — threatens core.** **Status:** 🟡 (LOSO/data) · ✅ (param-count correction)

### R1.5 — Weak downstream: synth AUC 0.653 < native-3T K-means 0.818
- **Validity:** VALID but **commonly misread**. The honest position (now quantified):
  - Fair LOSO recompute ([downstream_fair_recompute.txt](../adni_smoke_analysis_30/downstream_fair_recompute.txt)):
    C1 = 0.809 [0.640, 0.938], C2 = 0.653 [0.444, 0.844]; **DeLong p = 0.073 → difference NOT
    significant**; CIs overlap heavily. In-sample optimism was negligible (+0.009/+0.004), so the
    baseline is *not* unfair — the gap is simply **within noise at n=30**.
  - Both methods are **coarse 4-class** tissue features, where AD atrophy is already largely
    separable; neither tests the synthesis hypothesis (finer parcellation).
- **Fix class:** REFRAME (core) ✅ + NEW-EXPERIMENT (finer parcellation, S8) 🟡.
- **Plan:** Report 0.653 with its CI and the DeLong p; drop any implicit "synthesis is worse";
  state plainly that 4-class saturates and the mechanistic test is FastSurfer hippocampal-subfield
  features (S8). **Risk:** **HIGH — threatens the AD-application half.** **Status:** ✅ reframe / 🟡 S8

---

## Reviewer 2

### R2.1 / R2.2 — Missing recent SOTA comparison and recent citations
- **Validity:** VALID. **Fix class:** WRITING (citations) + NEW-EXPERIMENT (overlaps R1.2).
- **Plan:** Add 2023-2025 refs (WATNet, VCT/eidex2024, gicquel2025, bhattacharya2025, yoon2024
  already cited — extend); tie to the baseline experiments. **Risk:** Med. **Status:** 📝 / 🟡

### R2.3 — Show topology preservation via TRUE topological metrics, not HD95
- **Validity:** VALID. **Root cause:** [scripts/compute_betti.py](../scripts/compute_betti.py)
  computes real Betti numbers (gudhi) but is **not run** on outputs; paper reports HD95/CC only.
- **Fix class:** CODE+RERUN (cheap once checkpoint restored). **Compute:** CPU.
- **Plan (S3):** Add Euler characteristic; run β0/β1/β2 on synth vs GT seg and across the ablation;
  add a results table. **Risk:** Low (a clear win). **Status:** 🟡 (needs synth volumes)

### R2.4 — Does synth 7T actually improve AD discrimination vs 3T? (overlaps R1.5)
- See R1.5. Current honest answer: **not demonstrated at 4-class granularity** (difference not
  significant). Mechanistic test = finer parcellation (S8). **Status:** ✅ reframe / 🟡 S8

### R2.5 — Generalization beyond the single scanner/protocol
- **Validity:** PARTIAL. ADNI deployment *is* multi-scanner OOD, but has no paired 7T GT, so only
  topology/biomarker-direction evidence exists.
- **Fix class:** WRITING + DATA. **Plan:** Frame ADNI as OOD evidence; add the 20-subject (and if
  obtained, 279-pair) external paired test set (S7) for quantitative cross-cohort SSIM/Dice/HD95.
- **Risk:** Med. **Status:** 📝 / 🟡

### R2.6 — Justify conditional diffusion over GAN/latent/transformer/regression
- **Validity:** VALID. **Fix class:** WRITING-ONLY (seed material in [TRAINING_REPORT.md §6.2](../TRAINING_REPORT.md#L315)).
  The deterministic-regression and GAN baselines (S5) make this empirical, not just rhetorical.
- **Risk:** Low. **Status:** 📝 (+ strengthened by 🟡 S5)

### R2.7 — Single hold-out vs k-fold/LOSO; stable across folds? (overlaps R1.4)
- See R1.4. **Fix:** LOSO harness (S4), report fold stability. **Status:** 🟡

### R2.8 — Compare loss to PH / topo-Dice / clDice / Euler-characteristic regularization
- **Validity:** VALID. Paper already cites these as future work; reviewer wants comparison.
- **Fix class:** WRITING (positioning) + optional NEW-EXPERIMENT (implement clDice/PH as a loss
  and compare — expensive, multiple retrains).
- **Plan:** Minimum = honest positioning + report Betti/Euler as *evaluation* metrics (S3) even
  where we don't adopt them as losses; stretch = one clDice retrain if budget allows.
- **Risk:** Med. **Status:** 📝 / 🟡

---

## Summary: what threatens the core, and the chosen option
| Cluster | Option | Why |
|---|---|---|
| R1.2/R2.1 baselines | **FIX** (≥1 regression + 1 GAN baseline) | Non-negotiable for a second round |
| R1.4/R2.7 cohort | **FIX** (LOSO) + **DATA** (20-subj external test) | n=1 is indefensible; data is obtainable |
| R1.5/R2.4 downstream | **REFRAME** now (CI + DeLong) + **FIX** later (subfields, S8) | Gap is within noise; stop overclaiming |

Everything else is WRITING + cheap reruns.
