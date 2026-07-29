"""
Paper figures for the circular-evaluation result. No GPU; reads the summary JSONs.

Fig 1 (fig_reversal.png): per-subject grey-matter synthetic/real beta0 ratio under the co-trained
head vs an independent segmenter. Each subject is a vertical segment from its co-trained ratio
(~1, "preserved") down to its independent ratio (~0.1, "over-smoothed"); the length of the segment
IS the verdict reversal. Log y-axis; reference line at 1.0.

Fig 2 (fig_dose_response.png): grey-matter ratio vs segmenter-generator coupling, all three judges
applied to the SAME images (joint head -> detached head -> independent GMM). Faint per-subject
lines + bold median. Monotonic decrease = the dose-response.

Usage (on CERN, after the runs):
  python scripts/plot_circularity.py \
      --reversal-json    /eos/user/p/ppokhrel/topobrain/runs/circularity_summary/circularity_summary.json \
      --confound-json    /eos/user/p/ppokhrel/topobrain/runs/confound_free/confound_free_dose.json \
      --out /eos/user/p/ppokhrel/topobrain/runs/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # no display on the batch/login node
import matplotlib.pyplot as plt
import numpy as np

# colorblind-safe (Okabe-Ito)
C_COUPLED = "#0072B2"   # blue
C_INDEP = "#D55E00"     # vermillion
C_MID = "#009E73"       # green


def gm_ratios_reversal(path):
    """From aggregate_circularity output: per-subject (co-trained ratio, independent ratio) for GM."""
    rows = json.load(open(path))
    out = {}
    for r in rows:
        if r.get("tissue") != "GM":
            continue
        out[r["subject"]] = (r.get("cot_ratio"), r.get("ind_ratio"))
    return {k: v for k, v in out.items() if v[0] is not None and v[1] is not None}


def gm_ratios_dose(path):
    """From confound_free_dose output: per-subject GM ratio for each of the 3 judges (same images)."""
    rows = json.load(open(path))
    judges = ["joint_head", "detached_head", "gmm_probe"]
    out = {}
    for r in rows:
        s = r["_subject"]
        vals = {}
        for j in judges:
            if j in r:
                sn = r[j]["synth"]["GM"]["n_cc"]
                rn = r[j]["real"]["GM"]["n_cc"]
                vals[j] = (sn / rn) if rn else None
        if all(vals.get(j) is not None for j in judges):
            out[s] = [vals[j] for j in judges]
    return out


def fig_reversal(data, out):
    subs = sorted(data.keys())
    cot = [data[s][0] for s in subs]
    ind = [data[s][1] for s in subs]
    x = np.arange(len(subs))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for xi, c, i in zip(x, cot, ind):
        ax.plot([xi, xi], [c, i], color="0.6", lw=1.2, zorder=1)
    ax.scatter(x, cot, s=44, color=C_COUPLED, zorder=3, label="co-trained head (coupled)")
    ax.scatter(x, ind, s=44, marker="s", color=C_INDEP, zorder=3, label="independent segmenter")
    ax.axhline(1.0, color="0.4", ls="--", lw=1, zorder=0)
    ax.text(len(subs) - 0.5, 1.05, "preserved (ratio = 1)", ha="right", va="bottom", fontsize=8, color="0.4")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("sub-", "") for s in subs])
    ax.set_xlabel("subject")
    ax.set_ylabel(r"GM synthetic/real $\beta_0$ ratio")
    ax.set_title("Same images, opposite verdicts: the reversal is the segment length")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "fig_reversal.png", dpi=300)
    fig.savefig(out / "fig_reversal.pdf")
    plt.close(fig)


def fig_dose(data, out):
    labels = ["joint head\n(gradient-coupled)", "detached head\n(output-coupled)", "independent GMM\n(uncoupled)"]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    arr = np.array([data[s] for s in data])                # (n_subj, 3)
    for row in arr:
        ax.plot(x, row, color="0.75", lw=1, marker="o", ms=3, zorder=1)
    med = np.median(arr, axis=0)
    ax.plot(x, med, color=C_MID, lw=2.6, marker="o", ms=8, zorder=3, label="median")
    for xi, m in zip(x, med):
        ax.annotate(f"{m:.2f}", (xi, m), textcoords="offset points", xytext=(8, 6),
                    fontsize=10, fontweight="bold", color=C_MID)
    ax.axhline(1.0, color="0.4", ls="--", lw=1, zorder=0)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"GM synthetic/real $\beta_0$ ratio")
    ax.set_title("Bias scales with coupling (all judges, identical images)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_dose_response.png", dpi=300)
    fig.savefig(out / "fig_dose_response.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reversal-json", required=True)
    ap.add_argument("--confound-json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rev = gm_ratios_reversal(a.reversal_json)
    print(f"reversal: {len(rev)} subjects")
    fig_reversal(rev, out)
    print(f"  wrote {out/'fig_reversal.png'}")

    dose = gm_ratios_dose(a.confound_json)
    print(f"dose-response: {len(dose)} subjects")
    fig_dose(dose, out)
    print(f"  wrote {out/'fig_dose_response.png'}")


if __name__ == "__main__":
    main()
