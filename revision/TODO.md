# Revision TODO — IET Image Processing Round 2

Branch: `revision/iet-round2`. Living tracker. ✅ done · 🟡 needs GPU/checkpoint/data · 📝 writing.

## A. No-compute (in progress now)
- [x] Phase 0 orientation + audits → [repo_map.md](repo_map.md)
- [x] Discrepancy log → [discrepancies.md](discrepancies.md)
- [x] Reviewer response matrix → [reviewer_response_matrix.md](reviewer_response_matrix.md)
- [x] **Fair downstream recompute** (LOSO + bootstrap CI + DeLong) — [scripts/downstream_fair_recompute.py]
      → C1 0.809 [0.640,0.938], C2 0.653 [0.444,0.844], **DeLong p=0.073 (n.s.)**
- [x] **Param count** — [scripts/count_params.py] → **13.98 M** (not 22.4 M)
- [ ] 📝 Rewrite §3.3 loss section to match code (D1 topology eq, D3 4-term total)
- [ ] 📝 Correct "22.4 M" → "13.98 M" everywhere (abstract, §1, §3.2) (D2)
- [ ] 📝 R1.1 VGG justification (training regularizer vs reported metric)
- [ ] 📝 R2.6 design-justification subsection (diffusion vs GAN/latent/transformer/regression)
- [ ] 📝 Reframe §4.6 downstream with CI + DeLong; drop implicit "synthesis worse" (R1.5/R2.4)
- [ ] 📝 R2.1/R2.2 add recent citations
- [ ] 📝 T=200→185 footnote (D5); reconcile 0.653/0.658 (D4)
- [ ] 📝 Draft `response_to_reviewers.md` (point-by-point; mark pending experiments)

## B. Needs the 105k checkpoint (restore from Kaggle first) — blocker S1
- [ ] 🟡 **S1** Restore 105k EMA checkpoint locally / to GPU env
- [ ] 🟡 **S3** Run Betti (β0/β1/β2) + Euler on synth vs GT seg + across ablation (R2.3/R2.8) — CPU after inference
- [ ] 🟡 **S7** Inference on 20-subject Figshare+ paired set → external SSIM/Dice/HD95/Betti (R1.4/R2.5)
- [ ] 🟡 **S8** FastSurfer hippocampal-subfield features on synth vs 3T (ADNI) → DeLong (R1.5/R2.4)

## C. Needs GPU training (supervisor compute) — ~100 GPU-h total
- [ ] 🟡 **S2** Finer ablation flags: edge-CE-only / boundary-Dice-only / single-vs-multiscale (R1.3) — 3 short runs
- [ ] 🟡 **S4** Fix splitter (D6) + LOSO harness; run 3-5 folds, report mean±std (R1.4/R2.7)
- [ ] 🟡 **S5** Baselines: deterministic 3D regression U-Net + 3D GAN; attempt WATNet/VCT (R1.2/R2.1)
- [ ] 🟡 **S6** Medical-pretrained perceptual-loss option (MedicalNet/RadImageNet) — 1 retrain (R1.1 robustness)

## D. Data acquisition (parallel, no compute) — VERIFIED
- [ ] **Download the 20-subject paired 3T/7T set** ✅ PUBLIC (Scientific Data descriptor,
      Nature s41597-025-04586-9; Figshare 25634115 + IEEE DataPort). Gives 3T T1/T2, 7T T1/T2 +
      **manual hippocampal-subfield labels**. Use as external test cohort (R2.5/R1.4) AND
      subfield-level downstream ground truth (R1.5/R2.4). HIGH VALUE, free.
- [ ] Email Chu et al. re: **279/351-pair** 3T/7T T1w (coc61@pitt.edu) — ⚠️ likely PRIVATE
      (study cohort, no repository; HBM 10.1002/hbm.70195). Ask, but do NOT plan on it.
      This is the gate for Q1-at-scale.
- [ ] Research public code for WATNet / VCT (feasibility for S5)

## Strategy: SEQUENCED (IET now -> NeuroImage later)
- **IET resubmission (now):** current flaw fixes + 1-2 baselines + LOSO + true-topo metrics +
  the free 20-subject external/subfield validation. Q2, invited, bankable.
- **NeuroImage follow-up (later, contingent):** add the Chu cohort IF it lands + full
  topology-loss benchmark (edge vs clDice vs PH vs Euler) + downstream as the "so what".
  Q1 is data-gated; do not abandon the IET invitation for it.

## E. Logistics
- [x] Draft supervisor email (lab GPU + ~$100 cloud credit ask)
- [ ] Manuscript: revised sections compile under paper_iet/ LaTeX
- [ ] Each NEW-EXPERIMENT script ships a tiny self-test before any GPU spend; pending = "run me", never faked
