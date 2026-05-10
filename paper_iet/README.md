# IET Image Processing — Submission Package

Folder for the journal manuscript submitted to **IET Image Processing**
(ISSN 1751-9667, Wiley/IET, Q2). Compiled from the existing thesis
report (`../report/latex__3_/`) and the analysis pipeline at
`https://github.com/prabeshx12/Topo-Brain`.

## File map

```
paper_iet/
├── paper.tex            ← main manuscript (10–12 page two-column research article)
├── references.bib       ← Vancouver-style numeric bibliography (32 entries)
├── cover_letter.tex     ← submission cover letter (compiles standalone)
├── README.md            ← this file
├── submission_checklist.md
└── figures/             ← all 8 figures referenced in paper.tex
    ├── cohort_overview.png
    ├── 006_S_4153_compare.png
    ├── 002_S_4213_compare.png
    ├── ad_risk_distribution.png
    ├── roc_with_bootstrap_ci.png
    ├── feature_boxplots.png
    ├── cohen_d_bar.png
    └── ranking_compare.png
```

## How to compile

`paper.tex` uses the standard `article` class with `twocolumn` so it
compiles offline (e.g., on the Overleaf "free" tier) for review:

```bash
pdflatex paper
bibtex   paper
pdflatex paper
pdflatex paper
```

The `cover_letter.tex` is independent:

```bash
pdflatex cover_letter
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
|---|---|---|
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
