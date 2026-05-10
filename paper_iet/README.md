# IET Image Processing — Submission Package

Folder for the journal manuscript submitted to **IET Image Processing**
(ISSN 1751-9667, Wiley/IET, Q2). Compiled from the existing thesis
report (`../report/latex__3_/`) and the analysis pipeline at
`https://github.com/prabeshx12/Topo-Brain`.

## File map

```text
paper_iet/
├── paper.tex                       ← main manuscript (two-column research article)
├── supplementary.tex               ← supplementary materials (extended tables)
├── references.bib                  ← Vancouver numeric bibliography (34 entries)
├── highlights.txt                  ← 5 bullets ≤85 chars + 85-word plain summary
├── cover_letter.tex                ← submission cover letter
├── generate_graphical_abstract.py  ← reproducible graphical-abstract builder
├── README.md                       ← this file
├── submission_checklist.md         ← pre-submission checklist + Wiley specifics
└── figures/
    ├── graphical_abstract.png      ← 4-panel infographic (300 dpi, 1.8M)
    ├── cohort_overview.png         ← all 30 ADNI synth outputs
    ├── 006_S_4153_compare.png      ← AD subject input vs synth, 3 planes
    ├── 002_S_4213_compare.png      ← CN subject input vs synth, 3 planes
    ├── ad_risk_distribution.png    ← composite AD-risk score scatter
    ├── roc_with_bootstrap_ci.png   ← ROC + 95% bootstrap CI (supplementary)
    ├── feature_boxplots.png        ← per-feature AD vs CN (supplementary)
    ├── cohen_d_bar.png             ← effect-size magnitudes (supplementary)
    └── ranking_compare.png         ← C1 vs C2 ranking visualisation (supplementary)
```

## How to compile

`paper.tex` uses the standard `article` class with `twocolumn` so it
compiles offline (e.g., on the Overleaf "free" tier):

```bash
pdflatex paper
bibtex   paper
pdflatex paper
pdflatex paper
```

`supplementary.tex` and `cover_letter.tex` are independent:

```bash
pdflatex supplementary
bibtex   supplementary
pdflatex supplementary
pdflatex supplementary
pdflatex cover_letter
```

To regenerate the graphical abstract from the analysis NIfTIs:

```bash
python generate_graphical_abstract.py
```

## How to convert to the actual Wiley/IET template

IET Image Processing requires the publisher template
(`wileyNJD-v2.cls`) for final submission, which is restricted-download
from `https://authorservices.wiley.com`. The conversion is a
near-mechanical drop-in:

1. Download Wiley's IET research-article template after creating the
   submission record on the journal portal.
2. Open Wiley's `main.tex` skeleton.
3. Copy our `\title`, `\author[]{...}`, and `\affil[]{...}` blocks
   into the equivalent Wiley `\title{...}`, `\author{...}`,
   `\address{...}` macros (the field names differ slightly).
4. Copy the body of our `paper.tex` (everything between
   `\begin{document}` and `\end{document}`) into Wiley's body, between
   their declared abstract macro and `\bibliography{}`.
5. Keep our `references.bib` as-is — Wiley uses a numeric Vancouver
   style consistent with `IEEEtran.bst`.
6. Update the `\bibliographystyle{IEEEtran}` line if Wiley specifies
   a different `.bst`.

Total conversion time: ~15 minutes once the template is downloaded.

## Manuscript headline numbers (sanity-check table)

These numbers must match between the thesis report (`../report/`),
the manuscript (`paper.tex`), and the analysis outputs (`../adni_smoke_analysis/`):

| Result | Value | Source |
| --- | --- | --- |
| SSIM (held-out paired) | 0.8991 | report 06-results.tex; paper §4.1 |
| PSNR (held-out paired, dB) | 20.00 | report 06-results.tex; paper §4.1 |
| Brain-mask Dice | 0.9436 | report 06-results.tex; paper §4.1 |
| Brain-mask HD95 (mm) | 2.05 | report 06-results.tex; paper §4.1 |
| Topology-loss HD95 reduction | 13.34 → 2.05 (6.5×) | report 06-results.tex; paper §4.3 |
| Hippocampus L largest-CC | 0.998 | report 06-results.tex; paper §4.4 |
| ADNI cohort (smoke) | 15 AD + 15 CN | report 07-adniextension.tex; paper §3.1 |
| CSF/GM ratio Cohen's d | +0.681 | report 07-adniextension.tex; paper §4.5 |
| Composite AD-risk AUC | 0.653 (synth) vs 0.818 (3T) | report 07-adniextension.tex; paper §4.6 |
| Inference time (A100) | 3.1 min/subject | report 06-results.tex; paper §4.1 |

If any of these need updating after expanded ADNI runs, update both
`paper.tex` and the source thesis report so the two stay consistent.

## Submission checklist

See `submission_checklist.md`. Ensure all items marked **must** are
satisfied before clicking submit.

## Versioning

This is version 1.0 of the manuscript prepared from the thesis
material on 2026-05-08. Subsequent revisions should carry an updated
date and a brief change log here.
