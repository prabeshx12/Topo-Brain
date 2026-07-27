# The reframed paper — blueprint (circular evaluation in MRI synthesis)

Written 17 Jul 2026, after the topology-preservation thesis was refuted by our own validity
checks. This is the paper the surviving, verified evidence supports. `paper_iet/paper.tex` is the
OLD paper and will be rewritten to follow this blueprint — do not edit the old one piecemeal.

**One-line thesis.** Learned structural-fidelity evaluation of MRI synthesis is *circular* when
the segmenter is coupled to the generator: a co-trained segmenter certifies synthetic grey matter
as anatomically preserved, while an independent segmenter shows the same images are ~10x
topologically simpler than real 7T. The verdict reverses in 9 of 10 subjects.

---

## Title (working)

**"Self-Confirming Fidelity: Coupled Segmenters Mask Over-Smoothing in MRI Synthesis Evaluation."**

Alt: *"Circular Evaluation in Learned Structural Assessment of 3T->7T MRI Synthesis."*

## Abstract (draft — honest, ~210 words)

> Structural-fidelity claims for synthesized MRI are increasingly supported by segmentation-based
> evaluation: a segmentation network scores whether synthetic anatomy is preserved. We show this
> practice is *circular* when the segmenter is coupled to the generator, and that the circularity
> is large enough to invert the conclusion. Using paired 3T/7T brain MRI (10 subjects) and a
> deterministic 3T->7T translation network with an auxiliary segmentation head, we evaluate the
> topology (connected-component count, beta-0) of synthetic grey matter under two judges applied to
> the identical images: the model's own co-trained head, and an independent segmenter that never
> saw the generator. The co-trained head rates synthetic grey matter as structurally comparable to
> real 7T (median synthetic/real beta-0 ratio 0.98), whereas the independent segmenter reveals it
> is roughly ten times topologically simpler -- severe over-smoothing (median ratio 0.10). The
> verdict reverses in 9 of 10 subjects, including both held-out subjects, so it is a property of
> the metric, not of overfitting. Bias scales with coupling (joint > detached > independent). We
> further show standard whole-volume PSNR/SSIM overstate fidelity by ~8 dB / 0.45 SSIM versus
> brain-masked metrics, reproducing a published paper's headline numbers. We conclude that only an
> independent, real-data-trained segmenter is an admissible structural judge for synthesis, and
> release the protocol.

## Contributions (precise — do not overclaim)

1. **The finding (headline).** First empirical demonstration that a generator-coupled segmenter
   yields a *self-confirming* structural-fidelity verdict for medical image synthesis, and that an
   independent segmenter *reverses* it — with a consistent, quantified effect (GM verdict reverses
   9/10; independent synth/real beta-0 ratio 0.09-0.12 in every subject).
2. **Bias scales with coupling.** A dose-response along joint -> detached -> independent shows the
   inflation grows with how tightly the segmenter is tied to the generator. *(Detached point: the
   `--detach-seg` run, in progress.)*
3. **A protocol + corroboration.** (a) An independent, real-data-trained segmenter is the only
   admissible structural judge; (b) standard whole-volume IQMs overstate fidelity (~8 dB / 0.45
   SSIM) and reproduce a published inflated headline — a second, well-known-but-ignored pitfall in
   the same lineage.

## Positioning (cite precisely; anchor in the Metrics-Reloaded lineage)

| Prior work | What they did | How we differ |
|---|---|---|
| Metrics Reloaded (Maier-Hein et al., Nat. Methods 2023/24) | framework for *selecting* validation metrics per problem | we identify a *specific circularity failure mode* in synthesis eval and quantify a verdict reversal |
| Deo et al. 2025 ("Metrics that Matter", arXiv:2505.07175) | IQMs miss localized morphological errors; use downstream task | they use pre-trained segmenters, never contrast co-trained vs independent, no reversal, no topology |
| Dohmen et al. 2025 (Sci Rep; arXiv:2405.08431) | similarity/quality metrics for MR i2i; downstream seg | no circularity framing, no coupling axis, no reversal |
| Acs & Zhuang 2025 (PLOS ONE) | 3T->7T with segmentation-consistency assessment (VolBrain, *independent*) | uses an independent segmenter but never *names/measures* the bias it thereby avoids; we do |
| Shape-/cycle-consistency losses (Zhang 2018 +) | bake a segmenter INTO generator training | exactly the coupled setup we show is circular to *evaluate* with |
| BrainMRDiff 2025 | topology-*guided* generation (persistence loss) | generation-side prior, not an evaluation-circularity finding |

**Correction for refs:** the paper previously miscited "Onofrey" — the correct authors are
**Dohmen et al.** Fix before submission.

## Section outline (what goes where, with the real numbers)

- **1. Introduction.** Segmentation-based fidelity evaluation is now common in medical synthesis;
  when the segmenter is coupled to the generator it can confirm itself. State the reversal result
  up front. Frame as a validation-methodology contribution, not a synthesis method.
- **2. Related work.** The Metrics-Reloaded lineage; downstream-task evaluation (Deo, Dohmen);
  segmentation-consistency in 3T->7T (Acs & Zhuang); coupled segmenters in generator training
  (shape/cycle consistency); topology in generative MRI (BrainMRDiff) — as *context*, not competitor.
- **3. Data & models.** 10 paired 3T/7T subjects, 0.65 mm; deterministic cascaded translation net
  (generator + auxiliary seg head) as the *demonstration vehicle*; the independent judge (GMM tissue
  probe; + FastSurfer/SynthSeg if installable). Honest: the net is a vehicle, not a contribution.
- **4. The circularity experiment.**
  - Two judges on identical synthetic images: co-trained head vs independent segmenter.
  - **Core table (n=10, GM):** median co-trained synth/real beta-0 = 0.98 ("preserved"),
    independent = 0.10 ("~10x over-smoothed"), reversal 9/10. Independent ratio 0.09-0.12 every
    subject. Held-out 06/07 identical to in-sample -> not overfitting.
  - **Dose-response:** joint -> detached -> independent (bias vs coupling).
  - **Tissue specificity:** reversal is GM-only; CSF/WM show over-smoothing under *both* judges
    (report honestly — the head can only mask the defect where structure is richest).
- **5. The metric-convention pitfall (secondary).** Whole-volume vs brain-masked (+~8 dB / +0.45
  SSIM); reproduce a published headline and show it collapses under masking.
- **6. Discussion / protocol.** Only an independent, real-data-trained segmenter is admissible;
  segmentation-consistency results using co-trained segmenters should be re-examined.
- **7. Limitations.** (see below) **8. Conclusion.**

## Key results table (fill final numbers from circularity_summary.json)

```
GM, synth/real beta-0 ratio, per subject:
  judge = co-trained head (A/B):  median 0.98   (>=1 => "looks preserved")
  judge = independent probe (C/D): median 0.10  (<<1 => "~10x over-smoothed")
  verdict reversal: 9 / 10 subjects   (only sub-08 borderline: co-trained 0.40)
  held-out subjects 06, 07: reversal YES, ratio 0.10 / 0.09 (== in-sample behaviour)
```

## Limitations (state all of these — they are what make it credible)

- **n=10, healthy, single training cohort.** Frame the pitfalls as the deliverable; scope
  generalizability honestly. Beijing 20-subject cohort held for revision if reviewers push.
- **8/10 subjects are in-sample** for the trained model. Acceptable because the claim is about the
  *metric*; the 2 held-out subjects behave identically. State plainly.
- **Reversal is GM-specific.** Report it; it is mechanistically sensible (co-trained head can only
  hide the defect where structure is most complex).
- **The independent judge is an unsupervised GMM probe** (no topology prior, image-only) — a
  strength for the argument, but add FastSurfer/SynthSeg if feasible so a reviewer trusts a
  standard tool. beta-0 depends on segmenter/threshold; compare only *within* a judge.
- **No new synthesis method / no clinical claim.** The AD downstream is a one-paragraph caveat
  (synthetic-7T AUC 0.66 < native-3T 0.82, n=30, underpowered) — not a contribution.

## Experiment status

- [x] Core circularity, n=10 (co-trained vs independent), GM reversal 9/10 — DONE.
- [ ] Dose-response middle point (detached-head run) — TRAINING (condor).
- [ ] Cross-method rank-flip: FR-U-Net + pix2pix — pending GPU.
- [ ] FastSurfer/SynthSeg as a second independent judge — try to install.
- [ ] Metric-inflation reproduction subsection — mostly have the numbers.

## Venue

MELBA (free, best scope fit) primary; Scientific Reports / IEEE Access as reputable/fast
fallbacks; IET resubmission as the warm-lead backup. MICCAI/SASHIMI workshops closed until 2027.
