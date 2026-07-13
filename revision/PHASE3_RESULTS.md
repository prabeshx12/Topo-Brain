# Phase 3 results — rebuilt model, first training session

**Checkpoint:** `cascaded_16677.pt` (step 16,677 of a planned 40,000 — the session hit the
10.3 h Kaggle cap because of the dataloader perf bug, since fixed).
**Evaluated on:** sub-06 (the VAL fold). sub-07 (the TEST fold) is deliberately untouched —
the old pipeline's fatal flaw was selecting checkpoints on the test subject.
**Metrics:** `src/metrics_honest.py` — brain-only SSIM/PSNR, segmentation Dice,
surface-to-surface HD95, true Betti numbers (26-connectivity).

## Head-to-head, same subject, same yardstick

| Metric | B1 (old pipeline) | **Cascaded (rebuilt)** | Δ |
|---|---|---|---|
| Whole-volume PSNR | 20.24 dB | **21.53 dB** | **+1.29 dB** |
| Whole-volume SSIM | 0.8840 | **0.8917** | +0.008 |
| Brain-only SSIM | 0.4436 | 0.4499 | +0.006 (flat) |
| Brain-only PSNR | 14.12 dB | 14.01 dB | −0.11 (flat) |
| CSF Dice | 0.8001 | **0.8318** | **+0.032** |
| GM Dice | 0.8741 | 0.8750 | +0.001 |
| WM Dice | 0.9298 | 0.9273 | −0.003 |
| CSF HD95 | 2.34 mm | **1.59 mm** | **−32 %** |
| GM HD95 | 1.45 mm | **1.30 mm** | −10 % |
| WM HD95 | 0.92 mm | 0.92 mm | — |
| Brain Dice | 0.9556 | **0.9578** | +0.002 |
| Brain HD95 | 4.11 mm | **3.31 mm** | **−19 %** |

## Progress toward the real benchmark
```
Acs & Zhuang (PLOS ONE 2025, same dataset, same LOOCV):  23.25 dB
B1                                                        20.24 dB   -> 3.01 dB behind
Cascaded (16.7k/40k steps, LR never annealed)             21.53 dB   -> 1.72 dB behind
                                                                        43% of the gap closed
```

## Honest reading

**What improved (real):**
- **+1.29 dB whole-volume PSNR** on an unfinished model.
- **Anatomy across the board**: CSF Dice +0.032, CSF HD95 −32 %, brain HD95 −19 %.
  These are the metrics the paper is actually about.

**What did NOT improve (the open problem):**
- **Brain-only SSIM is flat** (0.444 → 0.450) and brain-only PSNR is flat (14.12 → 14.01).
  The gains came from the background and from anatomy, NOT from brain image fidelity.
  Two candidate explanations, not yet distinguished:
    1. **Undertrained** — 16.7k of 40k steps; the cosine LR never annealed (still ~1.3e-4).
    2. **Perception-distortion wall** — 7T detail absent from the 3T input cannot be
       invented; the conditional mean is inherently blurry, capping brain SSIM. This is
       why no paper reports brain-only SSIM: it is an unforgiving number.
  The resume session (with the perf fix) discriminates between these.

**A regression to fix:**
- **GM beta0 worsened: 6 -> 65** (GT = 1). HD95 and Dice improved, so this is small-component
  speckle rather than structural failure — but it is the wrong direction, and it is precisely
  what Phase 4 (a real topology loss, now that the gradient reaches the generator) exists to fix.

## Topology detail (26-connectivity)
| Tissue | beta0 pred | beta0 GT | beta1 pred | beta1 GT | beta2 pred | beta2 GT |
|---|---|---|---|---|---|---|
| CSF | 6 | 5 | 0 | 2 | 0 | 0 |
| GM | 65 | 1 | 524 | 939 | 450 | 2115 |
| WM | 30 | 15 | 76 | 25 | 11 | 101 |

CSF topology is now essentially correct (beta0 6 vs GT 5). GM/WM still carry excess
components and handles.

## Compute status
**Kaggle's 30 h/week GPU quota is EXHAUSTED.** Blocked until it resets:
- Phase 3 session 2 (resume to 40k with the perf fix — 2-3x faster per step)
- Phase 4 (real topology loss)
- Phase 5 (10-fold LOOCV — ~100 GPU-h on its own)

Kaggle **CPU** is free and unmetered; all evaluation/metric/topology work runs there.

**The supervisor GPU request is now the critical path**, and the case is concrete: the harness
is built and proven, seven pipeline bugs are fixed, the architecture is rebuilt, the metrics
are validated, and 43 % of the gap to SOTA is closed on an unfinished model. Finishing needs
roughly 100-150 GPU-hours.
