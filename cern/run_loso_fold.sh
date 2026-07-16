#!/bin/bash
# ============================================================================
# run_loso_fold.sh -- ONE leave-one-subject-out fold on a CERN GPU node.
# HTCondor runs 10 of these (one per GPU) via cern/loso.sub, $(ProcId) = fold.
#
# Each fold: train on 9 subjects, hold ONE out entirely, with a FROZEN recipe
# (same steps/LR every fold, decided from the Phase-3 pilot BEFORE any fold ran)
# and take the FINAL weights -- NO checkpoint selection, so the held-out subject
# is never seen during training. That is what makes 9-train honest, unlike the
# papers that early-stop on the test subject.
#
# args:  FOLD(0-9)   N_ITERS   LAM_TOPO
#   LAM_TOPO=0 -> baseline (no topology). >0 -> the topology method.
# ============================================================================
set -e
FOLD=${1:?fold index 0-9}
NITERS=${2:-40000}
LAMTOPO=${3:-0}
echo "=== LOSO fold $FOLD | $NITERS steps | lam_topo=$LAMTOPO | $(hostname) @ $(date) ==="

source /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh
export PATH=$HOME/.local/bin:$PATH
nvidia-smi || { echo "NO GPU -- aborting"; exit 1; }

BASE=/eos/user/p/ppokhrel/topobrain
CODE=$BASE/Topo-Brain
PAIRS=$BASE/data/norm/pairs_cern.csv
OUT=$BASE/runs/loso_topo${LAMTOPO}/fold_${FOLD}
mkdir -p "$OUT"
cd "$CODE"

# val-fold == test-fold == FOLD  ->  the dataloader excludes ONLY this one subject,
# leaving 9 to train. We NEVER select on it; we take the final checkpoint below.
python scripts/train_cascaded.py \
    --pairs-csv "$PAIRS" \
    --config configs/train_diffusion.yaml \
    --n-iters "$NITERS" \
    --batch-size 8 \
    --lam-topo "$LAMTOPO" \
    --val-fold "$FOLD" --test-fold "$FOLD" \
    --save-freq 5000 \
    --max-hours 20 \
    --out-dir "$OUT"

echo "=== LOSO fold $FOLD done @ $(date); final checkpoint in $OUT ==="
