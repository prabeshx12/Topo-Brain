# True topological metrics, raw vs cleaned — sub-06 (110k checkpoint)

Answers reviewer **R2.3** ("can topology preservation be demonstrated using true topological
metrics rather than HD95?") and informs **R2.8**. Produced by `topobrain_cleanbetti.py` (CPU,
gudhi cubical complex). Raw JSON: `cleaned_betti.json`.

"pred>=Nv" = predicted segmentation after removing connected components smaller than N voxels
(standard morphological cleanup). GT = FreeSurfer 7T-derived labels.

## β0 — connected components
| Tissue | GT | pred raw | pred ≥20v | pred ≥50v | pred ≥100v |
|---|---|---|---|---|---|
| CSF | 5  | 753  | 78  | 54  | 36  |
| GM  | 1  | 5213 | 297 | 225 | 175 |
| WM  | 15 | 1393 | 34  | 21  | 13  |

## β1 — handles/tunnels
| Tissue | GT | pred raw | pred ≥50v |
|---|---|---|---|
| CSF | 2   | 222  | 183  |
| GM  | 939 | 5991 | 4679 |
| WM  | 25  | 1585 | 1130 |

## β2 — voids/cavities
| Tissue | GT | pred raw | pred ≥50v |
|---|---|---|---|
| CSF | 0    | 31   | 31   |
| GM  | 2115 | 3169 | 3152 |
| WM  | 101  | 1411 | 1406 |

## Largest-CC fraction (the metric the paper reports)
CSF 0.650 → 0.690 (≥50v) · GM 0.937 → 0.946 · WM 0.978 → 0.983 (GT: 0.844 / 1.000 / 0.9999)

## Honest interpretation — this QUALIFIES the central claim

1. **β0: cleanup rescues WM, not GM.** WM drops 1393 → 21 vs GT 15 (essentially matched — that
   fragmentation *was* single-voxel speckle). But **GM only falls to 225 vs GT 1**: those are 225
   genuine components of ≥50 voxels, *not* noise. CSF stays ~10× GT.
2. **β1/β2 cannot be cleaned away.** Handles and voids live *inside* the large components, so
   small-island removal does not touch them. WM is a single connected blob (β0 ✅) that contains
   **~1130 spurious tunnels (GT: 25) and ~1406 cavities (GT: 101)**. No post-processing fixes this.
3. **Therefore: by strict topological metrics the edge-aware loss does NOT preserve topology.**
   The `largest-CC fraction ≥ 0.99` reported in the paper is a **weak metric that hides this** — a
   single component riddled with holes still scores ~0.99.

### Context (fair, not an excuse)
- The GT is **FreeSurfer, which applies explicit topological correction**. No raw CNN segmentation
  matches that without an explicit topological mechanism, so this is a harsh but legitimate and
  reviewer-requested comparison.
- The loss *does* help on what it actually optimises: brain-mask HD95 6.5× better and reduced
  fragmentation vs the no-topology ablation. Those results stand.

### Required manuscript consequences
- **Do not claim "topology preservation."** Reframe as an **edge-aware boundary regulariser** that
  improves boundary fidelity and suppresses fragmentation, and report these β0/β1/β2 numbers
  honestly as a limitation (consistent with the §3.3/§5 "approximation, not persistent-homology"
  framing already adopted).
- Report largest-CC fraction *alongside* Betti numbers and state plainly that the former is
  insensitive to interior handles/voids.
- This is the strongest possible motivation for the persistent-homology / Euler-characteristic
  loss as the follow-up (R2.8, and the Q1 paper).

**Reported exactly as produced. Nothing fabricated.**
