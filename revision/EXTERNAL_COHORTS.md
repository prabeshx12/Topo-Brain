# External cohorts for validation (verified 16 July 2026)

The path from "IET-achievable" to "Q1-plausible" is external validation: test our UNC-trained
model on an INDEPENDENT cohort. Two candidates, verified against the actual papers (not memory --
I had this muddled earlier in the project, so these are checked).

---

## 1. Beijing 20-subject paired 3T/7T -- FREE, downloadable, genuinely independent

- Paper: Chu, Ma, Dong, He, Che, Li, Zeng, Zhang. *Scientific Data* 2025.
  https://www.nature.com/articles/s41597-025-04586-9  (PMID 39948093, PMC11825668)
- Data (free, Figshare+):
  https://plus.figshare.com/articles/dataset/A_paired_dataset_of_multi-modal_MRI_at_3_Tesla_and_7_Tesla_with_manual_hippocampal_subfield_segmentations_on_7T_T2-weighted_images/26075713
- Site: **Beijing MRI Center for Brain Research (BMCBR), China.** Siemens Prisma 3T + MAGNETOM 7T.
- 20 healthy young adults (18-25, 10M/10F). T1w + T2w + rfMRI (+DWI at 3T). Manual hippocampal
  subfield labels on 7T T2w.

**Independence: CONFIRMED.** Different country/site/scanner/subjects/group from our UNC training
data (Chen/Qu/Xie/Ahmad/Yap, University of North Carolina). A UNC->Beijing test is genuine
cross-site external validation.

Caveats:
- Narrow population (healthy 18-25) -- fixes cohort-generalisation, NOT the AD downstream.
- Likely the same broad Beijing/Li group whose UNPAIRED 7T data Acs & Zhuang used for their
  semi-supervised augmentation. Still fully independent of OUR training data; cite the lineage
  honestly.
- Their labelled emphasis is 7T T2w + hippocampal subfields; ours is T1w. Preprocessing to match
  our pipeline is real work (days).

## 2. Chu et al. 279-pair (351-subject) 3T/7T -- larger, but access-gated

- Chu, Santini, Marsland, Gianaros, Ibrahim. *Alzheimer's & Dementia* 2024/2025.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11716720/  (DOI 10.1002/alz.093423)
- 351 healthy participants (~47 yo); 279 pairs passed QA. Pitt (Ibrahim lab).
- **CONFERENCE ABSTRACT, not an open data release.** No public repo found; access would require
  emailing the corresponding author. Slower, uncertain.

---

## Where this fits

- **NOT needed for the IET resubmission.** IET stands on UNC + LOSO + baselines (in progress).
- **The lever toward NeuroImage/TMI/MedIA.** External inference on cohort #1 turns "works on 10 UNC
  subjects" into "generalises to an independent site" -- the thing a Q1 reviewer demands.
- Sequence: land IET first (current work), then treat cohort #1 as the next-project rung.

## Venue tiers, for calibration (reputational, not exact JCR)

Direct competitors on THIS benchmark are mid-tier: Acs & Zhuang (PLOS ONE, non-selective),
FS-RWKV (IEEE BIBM conf), LiteMamba (Frontiers), Siam (IEEE Access). => IET (Q2) is a realistic
target. The Q1 references we build on (YODA/Byrne in IEEE TMI, SynthRAD in MedIA, Blau/clDice in
CVPR) are foundational, not same-task competitors -- reaching that tier needs the external cohort.
