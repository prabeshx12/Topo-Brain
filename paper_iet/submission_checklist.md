# IET Image Processing — Pre-Submission Checklist

Tick each item before submitting via the journal portal at
<https://mc.manuscriptcentral.com/iet-ipr> (or the current Wiley
portal — confirm on the journal's author-guidelines page).

## Manuscript

- [ ] **Title** is informative, $<$ 200 characters, no abbreviations
      that aren't expanded on first use.
      *Current:* "TopoBrain: Anatomy-Aware Conditional Diffusion for
      3T-to-7T MRI Synthesis with Application to Alzheimer's Disease
      Biomarkers" (149 chars).

- [ ] **Authors** in the order to appear; each author has an ORCID
      registered. *(IET strongly encourages ORCIDs; Wiley submission
      portal will request them.)*

- [ ] **Affiliations** are correct, with explicit corresponding
      author and email.

- [ ] **Abstract** is structured (problem, methods, results,
      conclusion) and 200--250 words.
      *Current word count:* ~245 words. **OK.**

- [ ] **Keywords**: 4–6 keywords. *Current:* 7 keywords — trim to 6.

- [ ] **Section structure** matches IET house style:
      - Introduction
      - Related work / Background
      - Materials and Methods
      - Results
      - Discussion
      - Conclusion
      - Acknowledgements
      - Conflict of interest
      - Data availability statement
      - Author contributions
      - References

- [ ] **References** in IET house numeric style (Vancouver-like).
      Verify with `IEEEtran.bst` — output should look like
      "[1] J. Ho et al., ..." and be ordered by first citation.

- [ ] **Figures** are at $\geq$\,300 dpi for production quality.
      Caption is below figure, beginning "Fig.\ X.".

- [ ] **Tables** have caption above the table; rule lines per IET
      house style (`booktabs` package: `\toprule`, `\midrule`,
      `\bottomrule`).

- [ ] **Equations** are numbered only when referenced.

- [ ] **Page count** for two-column research article: typically
      8--14 pages. Verify final compiled PDF.

- [ ] **Math symbols** are consistent: italic for variables, upright
      for operators (sin, log).

## Mandatory declarations

- [ ] **Conflict of interest** statement (already present in
      `paper.tex`).

- [ ] **Data availability statement** (already present;
      ADNI under DUA, code on GitHub, paired 3T/7T cohort referenced
      to original distributors).

- [ ] **Author contributions** statement (CRediT-style; already
      present in `paper.tex`).

- [ ] **Funding** statement (add if any; we used unfunded GPU
      access from E.K. Solutions — note in acknowledgements only).

- [ ] **Ethics statement**: ADNI is a public dataset collected with
      its own IRB approvals; the manuscript cites it. UNC paired 3T/7T
      cohort: confirm originating institution's ethics statement is
      cited if it appears in their distribution.

## Files to upload (separate uploads, not bundled)

- [ ] `paper.pdf` — compiled manuscript (DOUBLE-CHECK figures render).
- [ ] `paper.tex` — source LaTeX file.
- [ ] `references.bib` — bibliography source.
- [ ] All `figures/*.png` — separate file uploads, NOT zipped.
      **NOTE:** the `paper_iet/figures/` folder is gitignored on the public
      Topo-Brain repository (the global `.gitignore` excludes `*.png`
      because the figures are derived from ADNI imaging covered by the
      ADNI Data Use Agreement, which prohibits public redistribution).
      The figures live ONLY on your local machine — verify before
      submission that you have the following files locally before you
      open the journal portal:
      ```text
      paper_iet/figures/graphical_abstract.png
      paper_iet/figures/cohort_overview.png
      paper_iet/figures/AD_compare.png            (renamed from 006_S_4153_compare.png)
      paper_iet/figures/CN_compare.png            (renamed from 002_S_4213_compare.png)
      paper_iet/figures/ad_risk_distribution.png
      paper_iet/figures/roc_with_bootstrap_ci.png    (supplementary)
      paper_iet/figures/feature_boxplots.png         (supplementary)
      paper_iet/figures/cohen_d_bar.png              (supplementary)
      paper_iet/figures/ranking_compare.png          (supplementary)
      ```
- [ ] `cover_letter.pdf` (compiled from `cover_letter.tex`).
- [ ] `supplementary.pdf` (compiled from `supplementary.tex`).
- [ ] **Highlights** (paste content of `highlights.txt` into the portal
      form; do NOT upload the file).
- [ ] **Graphical abstract** (`figures/graphical_abstract.png` —
      uploaded to its own portal slot, NOT inline in the manuscript).

## Suggested reviewers (optional but recommended)

The journal portal lets you suggest 3 reviewers. Candidates with
relevant expertise:

- [ ] Reviewer 1: medical-image super-resolution (CycleGAN/diffusion).
- [ ] Reviewer 2: topology-aware deep segmentation (clDice / persistent
      homology).
- [ ] Reviewer 3: Alzheimer's neuroimaging biomarkers / FreeSurfer
      pipelines.

Do NOT suggest direct collaborators or those at your own institution.

## Wiley-specific items

- [ ] Confirm copyright transfer / license type (CC-BY for OA, or
      Wiley's Standard License).
- [ ] APC (article-processing charge) decision: subscription model
      (no fee at acceptance) vs. Open Access (ca. £2500 at acceptance).
      The decision is made at submission, not at acceptance.
- [ ] If Open Access is chosen, confirm a funding source for APC or
      that the journal participates in a transformative agreement
      with your institution (Tribhuvan University does NOT typically
      have such an agreement — likely subscription-only, no APC).

## Final pre-submission sanity check

- [ ] Compile `paper.tex` from a fresh terminal: `pdflatex; bibtex; pdflatex; pdflatex`. No errors.
- [ ] Spell-check the abstract (Wiley's portal does NOT auto-correct).
- [ ] Verify every cited reference appears in the bibliography
      (BibTeX warning "citation undefined" is a stop-the-press
      issue — fix before submitting).
- [ ] Verify every bibliography entry is cited (otherwise
      `IEEEtran.bst` will silently drop them, but the bibliography
      file should be cleaned up regardless).
- [ ] Check every section, table, and figure cross-reference renders
      correctly in the PDF (no `??` placeholders).
- [ ] Verify `https://github.com/prabeshx12/Topo-Brain` is publicly
      accessible.
- [ ] Verify ADNI Data Use Agreement compliance: no ADNI subject
      identifiers (PTIDs) appear in the manuscript text. Subject
      labels in figures must be aliased (e.g., "AD-1", "CN-1") if
      shown — currently figures use raw PTIDs in filenames; consider
      regenerating with anonymised labels before final submission.

## After submission

- [ ] Note submission date and manuscript ID in your project journal.
- [ ] Expected first decision: 6–10 weeks (IET typical).
- [ ] If "major revision": prepare point-by-point response, plan
      revision turnaround within journal-stipulated window
      (typically 60 days).
