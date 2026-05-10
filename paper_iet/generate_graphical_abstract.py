"""
Generate the graphical abstract for the IET Image Processing submission.

Wiley/IET requirement: a single figure (typically 1500x800 px or 600x300 mm)
that summarises the paper visually. This script produces a clean
infographic-style three-panel layout:

  [Panel A: input 3T] --[diffusion model]--> [Panel B: synth 7T + seg]
  [Panel C: downstream biomarker — AD-risk score histogram]

Uses real volumes from the ADNI smoke cohort (one AD subject + one CN subject)
and the AD-risk-score distribution from the same cohort.

Usage:
    python paper_iet/generate_graphical_abstract.py

Reads from:
  ../adni_preprocessed/adni_preprocessed/AD/<ad>_T1w_preprocessed.nii.gz
  ../adni_smoke_results/results_adni_smoke/<ad>/predicted_7T.nii.gz
  ../adni_smoke_results/results_adni_smoke/<ad>/predicted_seg.nii.gz
  ../adni_smoke_analysis/ad_risk_score_per_subject.csv

Writes:
  paper_iet/figures/graphical_abstract.png  (300 dpi, ~1800x900)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
import pandas as pd


SUBJECT_AD = "006_S_4153"
SUBJECT_CN = "002_S_4213"

REPO_ROOT = Path(__file__).resolve().parent.parent
PRE_DIR  = REPO_ROOT / "adni_preprocessed" / "adni_preprocessed"
RES_DIR  = REPO_ROOT / "adni_smoke_results" / "results_adni_smoke"
RISK_CSV = REPO_ROOT / "adni_smoke_analysis" / "ad_risk_score_per_subject.csv"
OUT_PNG  = REPO_ROOT / "paper_iet" / "figures" / "graphical_abstract.png"


def midcoronal(vol):
    h = vol.shape[1]
    return np.rot90(vol[:, h // 2, :])


def main():
    # Load AD subject's input + synth
    inp_ad   = nib.load(str(PRE_DIR / "AD" / f"{SUBJECT_AD}_T1w_preprocessed.nii.gz")).get_fdata()
    synth_ad = nib.load(str(RES_DIR / SUBJECT_AD / "predicted_7T.nii.gz")).get_fdata()
    seg_ad   = nib.load(str(RES_DIR / SUBJECT_AD / "predicted_seg.nii.gz")).get_fdata()

    # AD-risk score data
    risk = pd.read_csv(RISK_CSV)
    ad_scores = risk[risk["group"] == "AD"]["composite"].to_numpy()
    cn_scores = risk[risk["group"] == "CN"]["composite"].to_numpy()

    # ===== Build the figure =====
    fig = plt.figure(figsize=(15, 5.2), dpi=300, facecolor="white")
    gs = fig.add_gridspec(1, 7, width_ratios=[2.4, 0.5, 2.4, 0.6, 2.4, 0.5, 3.0],
                          left=0.02, right=0.98, top=0.92, bottom=0.10,
                          wspace=0.05)

    # ---- Panel A: input 3T ----
    axA = fig.add_subplot(gs[0])
    axA.imshow(midcoronal(inp_ad), cmap="gray", aspect="equal")
    axA.set_title("Input 3T MRI", fontsize=13, fontweight="bold", pad=8)
    axA.text(0.5, -0.08, "1 mm$^3$ isotropic, T1-weighted",
             transform=axA.transAxes, ha="center", fontsize=9, color="dimgrey")
    axA.axis("off")

    # ---- Arrow A→B with model description ----
    axArr1 = fig.add_subplot(gs[1])
    axArr1.axis("off")
    axArr1.annotate("", xy=(1, 0.5), xytext=(0, 0.5),
                    arrowprops=dict(arrowstyle="-|>", lw=2.6, color="#222"))
    axArr1.text(0.5, 0.78, "TopoBrain", ha="center", va="center",
                fontsize=11, fontweight="bold", color="#222",
                transform=axArr1.transAxes)
    axArr1.text(0.5, 0.66, "22.4M params", ha="center", va="center",
                fontsize=8, color="#444", transform=axArr1.transAxes)
    axArr1.text(0.5, 0.30, "DDPM + seg", ha="center", va="center",
                fontsize=8, color="#444", transform=axArr1.transAxes)
    axArr1.text(0.5, 0.18, "+ topology loss", ha="center", va="center",
                fontsize=8, color="#444", transform=axArr1.transAxes)

    # ---- Panel B: synthesised 7T ----
    axB = fig.add_subplot(gs[2])
    axB.imshow(midcoronal(synth_ad), cmap="gray", aspect="equal")
    axB.set_title("Synthesised 7T", fontsize=13, fontweight="bold", pad=8)
    axB.text(0.5, -0.08, "SSIM 0.899  •  PSNR 20.00 dB  •  HD95 2.05 mm",
             transform=axB.transAxes, ha="center", fontsize=9, color="dimgrey")
    axB.axis("off")

    # ---- Arrow B→C ----
    axArr2 = fig.add_subplot(gs[3])
    axArr2.axis("off")
    axArr2.annotate("", xy=(1, 0.5), xytext=(0, 0.5),
                    arrowprops=dict(arrowstyle="-|>", lw=2.6, color="#222"))
    axArr2.text(0.5, 0.78, "tissue volumes", ha="center", va="center",
                fontsize=10, fontweight="bold", color="#222",
                transform=axArr2.transAxes)
    axArr2.text(0.5, 0.30, "GM  WM  CSF", ha="center", va="center",
                fontsize=9, color="#444", transform=axArr2.transAxes)
    axArr2.text(0.5, 0.18, "atrophy index", ha="center", va="center",
                fontsize=9, color="#444", transform=axArr2.transAxes)

    # ---- Panel C: predicted seg colour overlay ----
    axC = fig.add_subplot(gs[4])
    rgb = np.zeros(midcoronal(synth_ad).shape + (3,))
    base = midcoronal(synth_ad)
    base = (base - base.min()) / max(base.max() - base.min(), 1e-9)
    seg_slice = midcoronal(seg_ad).astype(int)
    # GM = green, WM = blue, CSF = red, BG = gray
    for c in range(3):
        rgb[..., c] = base * 0.65
    rgb[seg_slice == 1, 0] = 0.95   # CSF -> red
    rgb[seg_slice == 1, 1] = 0.30
    rgb[seg_slice == 1, 2] = 0.30
    rgb[seg_slice == 2, 0] = 0.30   # GM -> green
    rgb[seg_slice == 2, 1] = 0.85
    rgb[seg_slice == 2, 2] = 0.30
    rgb[seg_slice == 3, 0] = 0.20   # WM -> blue
    rgb[seg_slice == 3, 1] = 0.55
    rgb[seg_slice == 3, 2] = 0.95
    axC.imshow(rgb, aspect="equal")
    axC.set_title("Tissue segmentation", fontsize=13, fontweight="bold", pad=8)
    handles = [mpatches.Patch(color=(0.30, 0.85, 0.30), label="GM"),
               mpatches.Patch(color=(0.20, 0.55, 0.95), label="WM"),
               mpatches.Patch(color=(0.95, 0.30, 0.30), label="CSF")]
    axC.legend(handles=handles, loc="lower right", fontsize=8,
               frameon=True, framealpha=0.85)
    axC.axis("off")

    # ---- Arrow C→D ----
    axArr3 = fig.add_subplot(gs[5])
    axArr3.axis("off")
    axArr3.annotate("", xy=(1, 0.5), xytext=(0, 0.5),
                    arrowprops=dict(arrowstyle="-|>", lw=2.6, color="#222"))
    axArr3.text(0.5, 0.78, "AD-risk score", ha="center", va="center",
                fontsize=10, fontweight="bold", color="#222",
                transform=axArr3.transAxes)

    # ---- Panel D: AD-risk score scatter ----
    axD = fig.add_subplot(gs[6])
    rng = np.random.default_rng(0)
    for x_pos, vals, color, label in [
        (0, cn_scores, "#4070b8", f"CN (n={len(cn_scores)})"),
        (1, ad_scores, "#c33b3b", f"AD (n={len(ad_scores)})"),
    ]:
        xs = x_pos + rng.uniform(-0.13, 0.13, size=len(vals))
        axD.scatter(xs, vals, s=70, color=color, alpha=0.85,
                    edgecolor="black", linewidth=0.5, label=label)
        if len(vals):
            axD.hlines(np.mean(vals), x_pos - 0.18, x_pos + 0.18,
                       colors=color, lw=2.5)

    axD.set_xticks([0, 1])
    axD.set_xticklabels(["CN", "AD"], fontsize=11)
    axD.set_ylabel("AD-risk z-score", fontsize=10)
    axD.set_title("ADNI external cohort\n($d = +0.68$, AUC $=$ 0.65)",
                  fontsize=12, fontweight="bold", pad=4)
    axD.grid(alpha=0.3)
    axD.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.85)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT_PNG} ({OUT_PNG.stat().st_size / 1e3:.0f} KB)")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
