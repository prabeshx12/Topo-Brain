# Research plan — verified against prior art + a hostile reviewer (16 July 2026)

This plan was stress-tested by three independent checks before being written: a broad prior-art
search, a recency scan (to mid-2026), and a hostile Q1-reviewer pass. It is deliberately more
modest than the first sketch, which oversold a thesis that turned out to be already published.

---

## 1. The corrected thesis (what we can honestly claim)

**NOT** "the field measures synthesis wrong; we measure honestly." That is taken -- Gicquel et al.
(arXiv:2507.13782, Jul 2025) already show a PSNR/SSIM-vs-task rank inversion for 3T->7T; Acs &
Zhuang (PLOS ONE 2025) already replace PSNR/SSIM with segmentation-consistency; Deo et al.
(arXiv:2505.07175) already argue "beyond PSNR/SSIM." We cannot headline that.

**What we CAN claim (confirmed unclaimed as of mid-2026):**

> The first **topology-preserving loss for 3T->7T MRI translation**, together with a **topological-
> invariant evaluation** (Betti numbers / connected-component fragmentation of segmented tissue)
> showing that current 3T->7T methods produce topologically broken anatomy that segmentation-Dice
> and PSNR/SSIM do not capture.

Novelty scope: the **topological** noun on the **3T->7T** task. Narrow, but real and defensible.

## 2. Positioning (cite precisely, or a reviewer reads us as scooped)

| Prior work | What they did | How we differ -- state explicitly |
|---|---|---|
| Gicquel 2025; Acs & Zhuang 2025 | "structure matters," seg-**Dice**/Hausdorff eval for 3T->7T | we use **topological invariants** (fragmentation/holes) that Dice cannot see |
| Deo 2025 "Metrics that Matter" | argue PSNR/SSIM inadequate, propose downstream eval | they propose **no topological metric**; we fill the exact gap they flag |
| TopoGAN 2020, TPOT 2024, TopoGen 2025 | topology loss for **shape/mask/natural-image** generation | first application to **intensity medical MRI translation** |
| Li et al., IEEE TMI Dec 2025 (fast Euler chi) | fast differentiable chi for **segmentation** + correction net | we **adapt fast chi into a differentiable loss for synthesis** (they did segmentation) |
| LiteMamba-Synth 2026 | abstract *says* "preserving topological features" | it has **no topology loss and no topological metric**; we actually measure/optimize it |

## 3. The reviewer's three killers -- and how the plan defuses each

1. **Circularity.** We train on Euler chi and report Betti numbers -- "they optimized topology and
   reported topology improved." **Defuse:** the headline result must be a win on a metric we did
   **NOT** train against (see the Q1 gate, section 5).
2. **Topology vs blur.** Reducing Betti-0 is trivially achievable by smoothing, which erases the
   detail 7T is for. **Defuse:** a **matched-blur control** -- a Gaussian-blurred fidelity baseline
   must NOT reach our Betti improvement, or the loss contributes nothing.
3. **Wrong target.** Real 7T is not topologically clean (noise/partial volume give it nonzero
   Betti). Optimizing toward an idealized beta0=1 moves *away* from the real 7T. **Defuse:** target
   and score against the **paired real 7T's** Betti numbers (identical segmentation, fixed
   threshold, + a threshold-robustness sweep). Our current "GM beta0 -> 1" framing is WRONG and is
   corrected here.

## 4. Lead with the AD failure, do not hide it

Synthetic-7T AD AUC 0.653 < native-3T 0.818 is positive evidence that translation *loses*
information. Reported as a **cautionary finding** ("hidden structural damage has a downstream
cost") it supports the thesis. Buried, it reads as spin and a reviewer who finds it distrusts the
whole paper. It is a scoping footnote, never a contribution. The paper does NOT claim clinical or
diagnostic utility -- our own data refute it.

## 5. The Q1 gate: ONE causal, non-circular win

The paper is IET/Q2 without it and reaches for JBHI/CMIG with it. The win:

> Show topology-aware synthesis **reduces the error of a real anatomical measurement, validated
> against the paired real 7T, that the best fidelity baseline cannot match** -- e.g., cortical
> surface/thickness reconstruction, or tissue-boundary HD95 on thin/branching structure. The chain
> "topology loss -> lower real-7T Betti error -> lower measurement error, and ONLY our method gets
> there" converts the thesis from assertion to result.

If, after honest effort, no such win exists -> pivot to the **benchmark paper**: "SOTA 3T->7T
synthesis is topologically broken, standard metrics hide it, here is the topological protocol, do
not use synthetic 7T downstream." Legitimate, true, Q2.

## 6. Experiment matrix (what the CERN GPU buys)

| Phase | Run | Answers | Gate |
|---|---|---|---|
| P3 | cascaded, topo OFF, 40k, val06/test07 | does finishing training lift fidelity? | RUNNING |
| P4 | topo ON, resume P3 | does the HARD monitor fall (real topology, not confidence)? | **make-or-break** |
| P4b | matched-blur control | is the topo gain more than smoothing? | reviewer killer #2 |
| L0 | LOSO baseline (topo off) x10 | honest fidelity + topology, n=10 | needs P3 healthy |
| L1 | LOSO topo x10 | the method vs baseline (ablation) | needs P4 positive |
| B | FR-U-Net + pix2pix + regression + **a diffusion baseline** x LOSO | fair comparison (reviewer: no diffusion baseline = auto "incomplete") | GPU-parallel |
| W | **the causal win** (section 5) | Q1 gate | the decider |
| X | Beijing external cohort inference | cross-site generalisation | CPU-preproc + fast infer |

## 7. Honest venue targets

- **IET / Q2 -- the reliable floor.** Topological loss + topological benchmark + reproduced
  baselines, honest metrics. Legitimate even if fidelity is unremarkable and the causal win is
  modest.
- **JBHI / CMIG (specialised Q1/Q2) -- the stretch.** Requires the section-5 causal win AND the
  diffusion baseline. Reachable, not guaranteed.
- **TMI / MedIA / NeuroImage -- OFF the table** with this data (n=10+20, healthy-only, no
  pathology). Not negotiable without a larger/clinical cohort.

## 8. Sequencing under a ~1-month GPU window (floor first, then climb)

- **Week 1:** P3 done -> eval_cern -> P4 (+P4b blur control). DECISION: does topology work?
- **Week 2:** baseline LOSO + baselines (incl. diffusion). **-> a complete IET paper is now secured.**
- **Weeks 3-4:** the causal win (section 5) + Beijing external (CPU-bound, doesn't compete for GPU)
  + writing. This is the Q1 push, attempted only after the floor is banked.
- **Discipline:** never gamble the finished IET floor for the uncertain Q1 upgrade. A finished
  modest paper beats an unfinished ambitious one.

## 9. What NOT to claim (so we never get caught)

- NOT "first to note PSNR/SSIM miss structure" (Gourdeau 2022; Gicquel/Acs&Zhuang 2025).
- NOT "first structure/downstream evaluation of synthesis" (standard; both 2025 3T->7T papers do it).
- NOT "topology loss for synthesis" as a category first (TopoGAN 2020; TPOT 2024; TopoGen 2025).
- NOT any diagnostic/clinical-utility claim (our own AD result refutes it).
- NOT "we recover true 7T detail" (physics: can't invent absent information).

## 10. Open risk to verify early

The brain-only SSIM ~0.5 floor may be partly **registration/contrast mismatch**, not the
perception-distortion wall. If the 3T/7T pairs are not truly voxel-aligned, the whole fidelity
narrative is confounded. **Check pairing/registration quality on 2-3 subjects before leaning on
the "metrics are the problem" story.**

---

**One-line summary for the supervisor:** a first topology-preserving loss + topological evaluation
for 3T->7T MRI synthesis; honest, IET/Q2-solid; reaches for specialised Q1 only if one causal,
non-circular, real-7T-validated win can be secured; top-tier is out of reach with this cohort.
