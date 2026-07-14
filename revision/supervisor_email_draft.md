# Draft — email to supervisor (GPU request)

**Read the notes at the bottom before sending.** Adjust the tone to your relationship; the
substance should stay.

---

## SHORT VERSION — send this one

**Subject:** TopoBrain resubmission — progress, and a request for GPU access

Dear Professor [NAME],

A progress update on the IET revision, and a request for compute.

Before changing anything, I audited our code against the manuscript. It surfaced problems I think
the referees were sensing but could not name:

- **Our reported metrics were produced by the evaluation harness, not the model.** Ground truth
  was being pasted into ~80% of the volume before SSIM/PSNR were computed. Scored honestly on
  brain tissue, the submitted checkpoint gives **SSIM 0.07 / PSNR 8.3 dB**, not the 0.90 / 20.0 dB
  we reported.
- **The topology loss sent exactly zero gradient to the image decoder.** It sat on a parallel
  branch, so it could never influence the synthesised image. Our central claim was not just
  unsupported — it was structurally impossible.
- Dice/HD95 were computed on intensity thresholds rather than segmentations, and the ablation was
  circular.

I have since rebuilt the pipeline: honest brain-masked metrics with true Betti numbers, a
cascaded architecture where the topology gradient does reach the generator (0 → 4.6×10¹), a
genuine Euler-characteristic topology loss verified against `gudhi`, and a reimplementation of the
current state of the art (Acs & Zhuang, PLOS ONE 2025) at matched capacity for a fair baseline.
Every claim is reproducible from a test script in the repo.

One finding I think stands on its own: **the benchmark on this dataset is not comparable across
papers.** On a single fixed model of ours, changing only the evaluation convention moves PSNR from
14.0 dB (brain-masked) to 31.2 dB (the convention Acs & Zhuang publish, under which they report
23.25). For comparison, every architecture ever tried on this dataset — CNN, GAN, transformer,
Mamba, 0.29 M to 123 M parameters — spans just 1.55 dB. The protocol matters ~11× more than the
architecture. I could claim we beat the state of the art; I don't think we should, and the paper
will say why.

**The request:** all of the above was done without a GPU, but finishing cannot be. I need roughly
**120–150 GPU-hours** to complete training, run the topology ablation and the baseline, and do
leave-one-subject-out cross-validation (which fixes the n=1 test set both referees objected to).
Kaggle's free quota is exhausted.

Could I get access to a departmental or lab GPU (≥16 GB)? Failing that, a small cloud budget would
do — roughly **$50–75** at current rates. With compute I estimate **4–6 weeks** to a resubmittable
manuscript; I will keep working on the write-up and reviewer response meanwhile.

Happy to walk you through any of the audit findings in detail.

Thank you,
Pratik Adhikari

---

## LONG VERSION — for reference, or if they ask for detail

**Subject:** TopoBrain resubmission — audit findings, rebuilt pipeline, and a request for GPU access

Dear Professor [NAME],

I want to update you on the TopoBrain revision, and to ask for help with compute.

**Short version:** before rebuilding anything, I ran a full audit of our own code against the
manuscript. It turned up several serious problems that I believe the referees were circling
without being able to name. I have fixed them, rebuilt the model, and the revised work is now
more defensible than the version we submitted. To finish it I need roughly **120–150 GPU-hours**,
which I do not have.

---

### 1. What the audit found

I treated the code as ground truth and checked every claim in the paper against it. The
substantive findings:

- **Our reported metrics were inflated by the evaluation harness, not earned by the model.**
  Ground-truth voxels were pasted into ~80% of the volume before SSIM/PSNR were computed. Scored
  honestly on brain tissue only, the published checkpoint gives **SSIM 0.071 / PSNR 8.27 dB**,
  not the 0.899 / 20.00 dB we reported.
- **The reported Dice and HD95 were computed on intensity thresholds, not on segmentations.** The
  HD95 implementation measured distance-to-object rather than surface-to-surface, which means it
  *rewarded* fragmentation. That is how HD95 could read 2.05 mm while the Betti number β₀ was
  5,213 (it should be 1).
- **The "topology loss" sent exactly zero gradient to the image decoder.** It hung off a parallel
  segmentation branch, so it was structurally incapable of influencing the synthesised image. The
  central claim of the paper — anatomy-aware synthesis — was not merely unsupported; it was
  impossible in that architecture.
- **The no-topology ablation was circular**, and several pipeline bugs (patch sampler, fold
  assignment, checkpoint selection on the test subject) were inflating results further.

None of this was visible from the manuscript alone, which is why it survived to submission. I am
not comfortable resubmitting anything built on it, and I don't think it would survive a second
round.

### 2. What I have rebuilt (no compute required)

- **Honest evaluation** — brain-masked SSIM/PSNR, Dice on actual segmentations, true
  surface-to-surface HD95, and real Betti numbers (β₀/β₁/β₂) validated against an independent
  persistent-homology library.
- **A cascaded architecture** in which the segmentation is computed *from* the synthesised image,
  so a topology loss must backpropagate through the generator. Measured: gradient into the image
  decoder goes from **0.00 → 4.6×10¹**. The old design is retained as a switch, so we can now
  *ablate* this choice rather than assert it.
- **A genuine topological loss** — a differentiable Euler characteristic (χ = V − E + F − C on the
  cubical complex), verified exact against `gudhi` on shapes with known topology.
- **A reimplementation of the current state of the art** (Acs & Zhuang, PLOS ONE 2025) at
  near-identical capacity (11.67 M vs our 11.90 M), so we can run a fair head-to-head on our own
  splits — which is precisely what Reviewers 1.2 and 2.1 demanded.

### 3. An unexpected result that I think is publishable in its own right

While checking how the competing papers measure their numbers, I found that **the benchmark on
this dataset is not comparable across papers at all.** Measuring one of our own models, changing
*nothing but the evaluation convention*:

| Evaluation convention | PSNR |
|---|---|
| Brain-masked 3D (honest) | **14.01 dB** |
| Whole-volume 3D | 21.53 dB |
| 2D per-slice, as published by Acs & Zhuang | **31.16 dB** |

**17.15 dB of the score is the convention, not the model.** Under the protocol Acs & Zhuang
publish, our model would report 31.16 dB against their 23.25 — a claim I could make, and will
not, because the same model scores 14.01 dB on the brain it is supposed to synthesise.

For context: across every architecture ever tried on this dataset (CNN, GAN, transformer, RWKV,
Mamba — 0.29 M to 123 M parameters), the total spread is **1.55 dB**. **The evaluation protocol
matters roughly eleven times more than the architecture.** I believe a reproducible evaluation
protocol, plus genuine topological metrics — which none of the three competing papers report — is
a stronger and more honest contribution than another fractional PSNR win.

### 4. What I need

Everything above was done without a GPU. The remaining work cannot be:

| Task | Est. GPU-hours |
|---|---|
| Finish the main training run (it is 42% complete; the LR never annealed) | ~10 |
| Train with the topology loss + a non-circular ablation | ~25 |
| The FR-U-Net baseline on our splits | ~12 |
| Leave-one-subject-out cross-validation (fixes the n=1 test set the referees objected to) | ~70–100 |
| **Total** | **~120–150** |

I have exhausted Kaggle's free quota (30 h/week), and its GPUs are in any case an older
architecture that current PyTorch barely supports.

**My request, in order of preference:**

1. **Access to a departmental / lab GPU** (anything with ≥16 GB — an RTX 3090, 4090, A5000 or
   better would finish this comfortably).
2. **Failing that, a small cloud-compute budget.** At current rates a rented RTX 4090 is roughly
   $0.30–0.50/hour, so the full remaining programme is about **$50–75**.

If neither is possible, I can still submit a reduced version, but it would have to drop the
cross-validation and the baselines — the two things the referees asked for most insistently — and
I don't rate its chances.

### 5. Timeline

With compute, I estimate **4–6 weeks** to a complete, resubmittable manuscript. I will keep
working on the write-up, the protocol analysis, and the response-to-reviewers in the meantime, so
no time is lost while this is arranged.

I am happy to walk you through any of the audit findings in detail — I have kept a full record,
and every claim above is reproducible from a test script in the repository.

Thank you,

Pratik Adhikari

---

## Notes before you send this

1. **The hardest paragraph is §1, and it is the one you must not soften.** Your supervisor is
   likely a co-author on the submitted version. Finding these errors yourself, before a referee
   or a reader did, is the single most creditable thing in this whole project — but only if you
   say it plainly. If you bury it and it surfaces later, it becomes a much worse conversation.
2. **Do not send the numbers in §3 as a boast.** The point of that section is *"here is a claim I
   could make and am refusing to make."* If it reads as "we beat SOTA," it will be read as
   exactly the thing you just finished apologising for.
3. **Check my GPU-hour estimates against your own judgement** before you send them. They assume
   ~1 s/step at batch 8 on 64³ patches, which is what we measured, but on unfamiliar hardware
   they could be off by a factor of two. It is better to ask for a range than to come back a
   second time.
4. **Have an answer ready for "so is the paper salvageable?"** Mine would be: *the paper we
   submitted is not; the paper we can now write is, and it is a better one.* Be ready to say
   which venue — IET (realistic) versus a Q1 journal (a stretch that needs an external cohort).
5. **Rotate your Kaggle API token before you do anything else** — it was exposed in plaintext and
   should be treated as compromised.
