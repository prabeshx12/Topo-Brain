"""
Aggregate the circular-evaluation result across subjects -> the paper's core table.

THE FINDING, stated as this script measures it. For each subject and tissue we compare the
SYNTH-vs-REAL structural verdict under two judges:

    co-trained head    : beta0(synth) [arm A]  vs  beta0(real) [arm B]   ->  ratio  A/B
    independent probe  : beta0(synth) [arm C]  vs  beta0(real) [arm D]   ->  ratio  C/D

The co-trained head reports A/B ~ 1 (synthetic looks as structurally rich as the real 7T, i.e.
"fidelity preserved"), while the INDEPENDENT segmenter reports C/D << 1 (synthetic is many times
topologically SIMPLER than real, i.e. "severely over-smoothed"). Same images, opposite verdicts:
the co-trained judge is self-confirming; the independent judge reveals the over-smoothing it hid.
That verdict reversal, shown to be CONSISTENT across subjects, is the contribution.

beta0 = number of 26-connected components. We read it from the `n_cc` field, which is present
and identical whether the source run used full gudhi or --b0-only, so old and new runs aggregate
the same way.

Usage:
  python scripts/aggregate_circularity.py \
      --runs-dir /eos/user/p/ppokhrel/topobrain/runs \
      --glob 'circularity_*/external_seg_topology.json' \
      --out /eos/user/p/ppokhrel/topobrain/runs/circularity_summary
"""
import argparse
import glob
import json
import os
from pathlib import Path

TISSUES = ("CSF", "GM", "WM")
# arm keys; support both the GMM probe and a real external segmenter for C/D
A_KEY = "A_ourhead_on_synth"
B_KEY = "B_ourhead_on_real"
C_KEYS = ("C_probe_on_synth", "C_external_on_synth")
D_KEYS = ("D_probe_on_real", "D_external_on_real")


def b0(entry, tissue):
    """beta0 for a tissue from an arm dict; prefer n_cc (always present), fall back to betti[0]."""
    if entry is None or tissue not in entry:
        return None
    t = entry[tissue]
    if t.get("n_cc") is not None:
        return int(t["n_cc"])
    bt = t.get("betti")
    return int(bt[0]) if bt and bt[0] is not None else None


def first_present(res, keys):
    for k in keys:
        if k in res:
            return res[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--glob", default="circularity_*/external_seg_topology.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio-thresh", type=float, default=0.5,
                    help="reversal = co-trained ratio >= thresh (says 'not over-smoothed') AND "
                         "independent ratio <= thresh (says 'over-smoothed')")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.runs_dir, a.glob)))
    if not paths:
        raise SystemExit(f"no JSONs matched {os.path.join(a.runs_dir, a.glob)}")

    rows = []
    for p in paths:
        res = json.load(open(p))
        subj = res.get("_subject", Path(p).parent.name)
        A, B = res.get(A_KEY), res.get(B_KEY)
        C, D = first_present(res, C_KEYS), first_present(res, D_KEYS)
        for t in TISSUES:
            a_, b_, c_, d_ = b0(A, t), b0(B, t), b0(C, t), b0(D, t)
            cot = (a_ / b_) if (a_ is not None and b_) else None            # co-trained synth/real
            ind = (c_ / d_) if (c_ is not None and d_) else None            # independent synth/real
            rows.append({"subject": subj, "tissue": t,
                         "cot_synth": a_, "cot_real": b_, "cot_ratio": cot,
                         "ind_synth": c_, "ind_real": d_, "ind_ratio": ind})

    def fmt(x, nd=2):
        return "  n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    print("=" * 92)
    print("CIRCULAR EVALUATION: synth-vs-real topology (beta0) under two judges, per subject")
    print("=" * 92)
    print(f"  {'subject':9} {'tis':3} | {'co-trained head (A/B)':>26} | {'independent probe (C/D)':>28} | flip")
    print(f"  {'':9} {'':3} | {'synth':>7} {'real':>7} {'ratio':>8} | {'synth':>7} {'real':>8} {'ratio':>9} |")
    print("  " + "-" * 88)
    flips = {t: [] for t in TISSUES}
    for r in rows:
        cot, ind = r["cot_ratio"], r["ind_ratio"]
        is_flip = (cot is not None and ind is not None
                   and cot >= a.ratio_thresh and ind <= a.ratio_thresh)
        if cot is not None and ind is not None:
            flips[r["tissue"]].append(is_flip)
        print(f"  {r['subject']:9} {r['tissue']:3} | {fmt(r['cot_synth'],0):>7} {fmt(r['cot_real'],0):>7} "
              f"{fmt(cot):>8} | {fmt(r['ind_synth'],0):>7} {fmt(r['ind_real'],0):>8} {fmt(ind):>9} | "
              f"{'YES' if is_flip else '-'}")

    print("\n" + "=" * 92)
    print("VERDICT REVERSAL CONSISTENCY  (co-trained says 'preserved', independent says 'over-smoothed')")
    print("=" * 92)
    import statistics as st
    for t in TISSUES:
        fl = flips[t]
        n = len(fl)
        if not n:
            continue
        cots = [r["cot_ratio"] for r in rows if r["tissue"] == t and r["cot_ratio"] is not None]
        inds = [r["ind_ratio"] for r in rows if r["tissue"] == t and r["ind_ratio"] is not None]
        med_cot = st.median(cots) if cots else float("nan")
        med_ind = st.median(inds) if inds else float("nan")
        print(f"  {t:3}: reversal in {sum(fl)}/{n} subjects | "
              f"median co-trained ratio {med_cot:.2f} (>=1 => 'looks preserved'), "
              f"median independent ratio {med_ind:.2f} (<<1 => 'over-smoothed')")

    gm = flips.get("GM", [])
    print("\n  HEADLINE (GM): the co-trained head rates synthetic grey matter as structurally")
    print(f"  comparable to real (median A/B ~ {st.median([r['cot_ratio'] for r in rows if r['tissue']=='GM' and r['cot_ratio'] is not None]):.2f}),")
    print(f"  while an independent segmenter shows it is ~{1/st.median([r['ind_ratio'] for r in rows if r['tissue']=='GM' and r['ind_ratio'] is not None]):.0f}x")
    print(f"  topologically simpler than real. The verdict reverses in {sum(gm)}/{len(gm)} subjects.")

    outp = Path(a.out); outp.mkdir(parents=True, exist_ok=True)
    (outp / "circularity_summary.json").write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nwrote {outp / 'circularity_summary.json'}  ({len(paths)} subjects)")


if __name__ == "__main__":
    main()
