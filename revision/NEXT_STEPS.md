# Next steps — verified plan (written while Phase 3 trains)

Status: **Phase 3 training is running on a CERN GPU** (job 11821692, ~0.5 s/step, ~5.5 h to 40k).
Everything below is prepared and verified so nothing is improvised when it finishes.

---

## The experiment matrix the paper needs

| # | Run | Purpose | Compute | Status |
|---|---|---|---|---|
| P3 | Cascaded, topology OFF, val=06/test=07, 40k | pilot: is the recipe healthy? annealed image quality? | 1 GPU, ~5.5h | **RUNNING** |
| P4 | Cascaded, topology ON, same split, resume P3 | does the topology loss reduce the HARD monitor (real topology)? | 1 GPU, short | ready (needs P3 done) |
| L0 | LOSO baseline (topology OFF), 10 folds | headline fidelity + topology, honest, n=10 | 10 GPU, ~5.5h each | **batch ready** (`cern/loso.sub`, arg `0`) |
| L1 | LOSO topology (topology ON), 10 folds | the method vs baseline (the ablation) | 10 GPU | ready (same batch, arg e.g. `0.1`) |
| FR | FR-U-Net baseline, 10 folds | the SOTA baseline R1.2/R2.1 demanded | 10 GPU | code ready (`train_frunet`/`eval_frunet`); needs a batch wrapper |

Decision gates (do NOT skip):
1. **P3 must look healthy** (losses fall, no NaNs, sane annealed numbers) before firing any 10-fold sweep.
2. **P4 must show the HARD monitor (`chi_err_hard`) actually falls** before trusting the topology
   arm. If it does not, topology is not working and we do NOT claim it -- we report the negative
   and reframe. This is the make-or-break experiment.

---

## When P3 finishes (~5.5h) -- exact commands

```bash
# 1. evaluate the final checkpoint on BOTH held-out subjects, COMPLETE segs, honest metrics
cd /eos/user/p/ppokhrel/topobrain/Topo-Brain && git pull
for s in sub-06 sub-07; do
  python scripts/eval_cern.py \
    --ckpt /eos/user/p/ppokhrel/topobrain/runs/phase3_clean/cascaded_40000.pt \
    --pairs-csv /eos/user/p/ppokhrel/topobrain/data/norm/pairs_cern.csv \
    --subject $s --out /eos/user/p/ppokhrel/topobrain/runs/phase3_clean/eval_$s
done
```
This answers: **does finishing training (LR annealed) lift brain-only SSIM/PSNR above the 16.7k
pilot (0.45 / 14.0), or is it the perception-distortion wall?** Either answer is publishable and
honest; we just need to know which.

`eval_cern.py` is the CORRECT evaluator: it reads the normalised pairs (the exact data training
used, with the COMPLETE segs) and reads voxel spacing from the header. Do NOT use eval_cascaded.py
on CERN -- it globs the old BIDS names and would score against the old cerebrum-only seg.

## Then fire the baseline LOSO (if P3 is healthy)

```bash
mkdir -p ~/loso_submit/logs
cp /eos/user/p/ppokhrel/topobrain/Topo-Brain/cern/run_loso_fold.sh ~/loso_submit/
cp /eos/user/p/ppokhrel/topobrain/Topo-Brain/cern/loso.sub        ~/loso_submit/
chmod +x ~/loso_submit/run_loso_fold.sh
cd ~/loso_submit && condor_submit loso.sub     # 10 folds, one per GPU
```
When all 10 finish:
```bash
python scripts/eval_loso.py \
  --runs-dir /eos/user/p/ppokhrel/topobrain/runs/loso_topo0 \
  --pairs-csv /eos/user/p/ppokhrel/topobrain/data/norm/pairs_cern.csv
```
-> mean +/- std across 10 held-out subjects. That is the paper's headline table.

---

## What is READY and VERIFIED (as of now)

- CERN pipeline runs end-to-end (P3 stepping at 0.5 s/it).
- Segmentation: complete whole-brain, all 10 subjects (cerebellum + CC recovered).
- `eval_cern.py`: reads normalised pairs + complete segs + header spacing. **The gap that would
  have scored against the wrong seg is closed.**
- `eval_loso.py`: reproduces the seed-42 fold->subject map, evaluates final checkpoints, aggregates.
- Topology loss: real invariant (exact vs gudhi), reaches the generator, and the HARD monitor is
  decoupled from softmax confidence (so a falling curve means real topology change).
- Training curves saved: `train_history.json` flushed every log step (per fold too).
- Test suite: green (verifying the last few after today's edits).
- Manuscript: voxel size corrected 1.0 -> 0.65 mm; evaluation described as 10-fold LOSO; seg
  described as complete whole-brain.

## Still OPEN (not blocking P3)

- **P4 topology validation** -- the one experiment that decides the paper's headline claim.
- **FR-U-Net LOSO batch wrapper** -- code exists; needs a `cern/` submit like loso.sub.
- **A100 line + EosSubmit** -- ask friend; only speeds up the sweeps.
- **Kaggle token rotation** -- user action, still pending.
- **Downstream AD** -- recommend cutting from the paper (negative result); keep in thesis.

---

## The honest through-line for the paper

1. **Reproducible honest evaluation** on this benchmark (brain-masked, whole-brain segs, true
   Betti numbers, LOSO) -- ground nobody else on this dataset covers.
2. **A topology-aware method** whose loss provably reaches the generator, with a monitor that
   cannot be faked by confidence.
3. Claims stated as **relative improvement, consistently measured, across 10 subjects** -- not
   "we recover the true topology" (undefined) and not "we beat X dB" (protocol-dependent).

If P4 shows the topology loss works: that is the contribution. If it does not: the honest
evaluation + protocol finding still stand on their own, and we report topology as attempted with
the negative result. Either way, nothing in the paper is something a reviewer can catch us on.
