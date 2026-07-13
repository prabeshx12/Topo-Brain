# Benchmark reality — the published numbers on this dataset are NOT comparable to each other

**Rewritten 13 July 2026, after reading all three competing papers in full.**

This file previously asserted *"we are BEHIND published SOTA — this conclusion is robust."*
**That claim was wrong and is retracted.** So was the claim before it (*"we beat SOTA"*). Both
were made by comparing numbers whose measurement conventions I had not checked. The correct
answer is narrower, and it is now sourced:

> **Nobody knows who is ahead on this dataset, because no two papers measure the same quantity.**

---

## 1. The smoking gun — from a competitor's own ablation

LiteMamba-Synth (Front. Neuroanat. 2026) evaluate **the same model, the same weights, the same
data**, changing only *where they point the metric*:

| Evaluation region | PSNR | SSIM | RMSE |
|---|---|---|---|
| Full 256×256 image | 20.820 | 0.711 | 0.092 |
| Central 128×128 crop | **23.870** | 0.719 | 0.064 |

> *"Compared with the full-image evaluation, the central-region evaluation yields higher
> reconstruction accuracy, with PSNR increasing from 20.820 dB to 23.870 dB."* — LiteMamba §Table 4

**A 3.05 dB swing from the evaluation region alone.** Acs & Zhuang's headline 23.254 dB falls
*inside* that spread. This is a published, same-dataset demonstration that the headline PSNR on
this benchmark is worth ±3 dB depending purely on where you measure it.

**Corroborating signal:** SSIM — which is far less sensitive to the air/background region than
PSNR — is *consistent* across all three papers (0.737 / 0.726 / 0.711, a ±0.02 spread), while
PSNR diverges by 2.3 dB. When one metric agrees and the other does not, the disagreement lives
in the **measurement**, not in the models.

---

## 2. Why my own "decisive arithmetic" was not decisive

The old §1 converted PSNR to RMSE and concluded we lose. Recomputed on the **cascaded** model
(the old version used B1), on the identical footing:

| System | PSNR | RMSE (in [0,1] units) |
|---|---|---|
| B1 regression (whole-vol 3D, skull-stripped) | 20.24 | 0.0973 — *worse* than FS-RWKV |
| **Cascaded (whole-vol 3D, skull-stripped)** | 21.53 | **0.0838 — *better* than FS-RWKV** |
| Cascaded, BRAIN-ONLY (the honest number) | 14.01 | 0.1993 — far worse than everyone |
| *FS-RWKV (2D, 101 central slices, full-head)* | *21.00* | *0.0898* |
| *LiteMamba (2D, full image, full-head)* | *20.82* | *0.0920* |

**The conclusion reverses depending on which checkpoint you pick.** A conclusion that flips like
that was never robust. And the flip must NOT be read as "we are now ahead": our 0.0838 is
measured on a **skull-stripped** volume where ~85 % of voxels are background that we set to a
constant and reproduce exactly. FS-RWKV's 0.0898 is measured on **full-head central slices**,
where the skull, scalp and 7T MP2RAGE background noise are genuinely hard.

**Different pixel populations. The comparison is invalid in BOTH directions.**

---

## 3. What each paper actually does (all sourced from full text)

| | **Acs & Zhuang** (PLOS ONE 2025) | **FS-RWKV** (BIBM 2025) | **LiteMamba** (Front. Neuroanat. 2026) |
|---|---|---|---|
| Headline | 23.25 / 0.737 | 21.00 / 0.726 | 20.82 / 0.711 |
| …which is actually | **transverse plane ONLY** (coronal 22.48, sagittal 22.05) | 2D, 101 central slices | 2D, full image |
| Architecture | **3D U-Net, 64³ patches, stride 32, 32/64/128/256 ch** | RWKV, 45.4 M | ConvMamba, 2.15 M |
| Loss | MSE + 0.7·SSIM | smooth-L1 + 0.4·SSIM + 0.3·Sobel | smooth-L1 + SSIM + Sobel |
| Skull-stripped? | **No — stated 4×, it is their selling point** | Not stated (→ full-head) | Not stated (→ full-head) |
| Brain mask on metric? | **No** — "mask" never appears in their metrics section | No | No |
| Slices scored | **ALL 308**, incl. air-only end slices | 101 **central** only | central only |
| Dimensionality | 3D volume → sliced → **2D per-slice PSNR** | 2D | 2D |
| Validation | **10-fold LOOCV** | **fixed 7/3 split** | fixed split |
| Extra harmonisation | none | **+ histogram matching** | **+ N4** |
| Topology / Betti | **NONE** | **NONE** | **NONE** |
| Code / weights | **None** | None | None |

**Nobody brain-masks. Nobody reports 3D SSIM. Nobody reports a topological metric. Nobody
releases code. And FS-RWKV and LiteMamba do not even cite Acs & Zhuang** — the 21-vs-23.25 gap
has never been adjudicated by anyone in the field.

---

## 4. Their architecture is essentially ours

Acs & Zhuang is **not** an exotic backbone. It is a 3D patch-based U-Net: **64³ patches, stride
32, encoder 32/64/128/256** — channel for channel, our `UNetGenerator`. No GAN, no diffusion, no
transformer (they explicitly declined to compare against diffusion or transformers).

**There is therefore nothing to "adopt" from their design to beat them.** Any gap is training
completeness, data convention, or evaluation convention — not architecture.

---

## 5. Weaknesses in the "SOTA" we were measuring ourselves against

Stated without gloating, because they bear directly on how hard the bar really is:

1. **Acs & Zhuang early-stop on the test subject.** 10 subjects, 9 train + 1 held out, no third
   split; "validation" and "testing" are used interchangeably (*"Subject 8 was used for
   validation, while the remaining 9 subjects were used for training"*; 567 × 9 = 5103 training
   patches leaves no room for a val subject). **This is the identical leak we found and fixed in
   our own pipeline (bug A4).** They do not list it as a limitation.
2. **Their own ablation shows PSNR getting WORSE** as they add their contributions (transverse
   24.98 → 24.76; sagittal 23.40 → 22.91), while the text claims *"each architectural
   modification progressively improves ... performance metrics."*
3. **Their headline semi-supervised contribution is statistically null** — no significant
   difference vs their own supervised baseline on any metric or orientation, **all p > 0.28**.
   They state this themselves.
4. Their ablation is reported on a **single favourable fold** (subject 8 → 24.76 dB, well above
   their own 10-fold mean of 23.25).
5. **Siam et al. reach SSIM > 80 % *with* skull-stripping**, which beats Acs & Zhuang's 0.737 —
   a fact Acs & Zhuang acknowledge but frame as a practicality argument rather than a fairness one.

---

## 6. Consequences for the rebuild

1. **No borrowed comparisons. Ever.** Their numbers cannot go in a table against ours. The only
   defensible route is to **run the baselines ourselves, on our pipeline, with our metrics** —
   which is exactly what reviewer R1.2/R2.1 demanded anyway.
2. **Acs & Zhuang is now a CHEAP baseline to reproduce**, because their FR-U-Net is ~90 % our
   existing code (same patch size, same channels, same stride). Reimplementing it and training it
   on our splits gives a genuine head-to-head under identical data and identical evaluation.
   No code is released, so reimplementation from the text is the only route — and the text omits
   the parameter count, the consistency-loss weights α and λ, the patch-reassembly blending rule,
   the SSIM window/constants, and the PSNR data_range. **Those omissions must be stated as
   reproduction caveats.**
3. **Report multiple conventions, transparently** — brain-masked 3D (honest), whole-volume 3D
   (nearest to the field), and 2D-per-slice (theirs). Let the reader see the spread instead of
   discovering it.
4. **The contribution is not "our PSNR is bigger."** The field has no consistent protocol here,
   ±3 dB of slack, no masking, no topology, no code. Ours is: **a reproducible evaluation
   protocol + genuine topological metrics + a verified topology loss.** That ground is
   unoccupied — confirmed across all three papers.

---

## Lesson (unchanged, and now demonstrated twice)

I claimed superiority from an assumed convention, then claimed inferiority from another assumed
convention. **Both were wrong for the same reason.** Verify the measurement before comparing —
and when the measurement cannot be verified, the honest output is "not comparable", not a
direction.
