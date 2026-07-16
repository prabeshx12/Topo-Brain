#!/bin/bash
# ============================================================================
# run_loso_fold.sh -- ONE fold of leave-one-subject-out cross-validation.
#
# Called once per fold (0..9). HTCondor runs 10 of these in parallel, one per
# GPU, via cern/loso.sub. Each fold:
#   * trains on 9 subjects, holds ONE out entirely
#   * uses a FROZEN recipe (same steps/LR for every fold, decided from the
#     pilot BEFORE any fold ran -- so nothing is tuned on a test subject)
#   * takes the FINAL EMA weights -- NO checkpoint selection, so the held-out
#     subject is never looked at during training. This is the whole point: it
#     is what makes 9-train honest, unlike the papers that early-stop on test.
#
# Usage:  run_loso_fold.sh <FOLD 0-9> <N_ITERS> <OUTDIR> <PAIRS_CSV> <CONFIG>
# ============================================================================
set -e
FOLD=${1:?fold index 0-9}
N_ITERS=${2:?frozen step count from the pilot}
OUT=${3:?output dir}
PAIRS=${4:?pairs csv}
CONFIG=${5:-configs/train_diffusion.yaml}

mkdir -p "$OUT/fold_$FOLD"
echo "=== LOSO fold $FOLD : train on 9, hold out fold $FOLD, $N_ITERS steps ==="

# val-fold == test-fold == FOLD  ->  the dataloader excludes ONLY this one
# subject, leaving 9 to train (src: create_synthesis_dataloaders excludes both
# val and test fold; setting them equal excludes exactly one subject).
# We NEVER select on it -- run_loso_fold takes the final weights below.
python scripts/train_cascaded.py \
    --pairs-csv "$PAIRS" \
    --config "$CONFIG" \
    --val-fold "$FOLD" \
    --test-fold "$FOLD" \
    --n-iters "$N_ITERS" \
    --batch-size 8 \
    --max-hours 20 \
    --out-dir "$OUT/fold_$FOLD"

# evaluate the FINAL checkpoint on the held-out subject -- no selection.
CK=$(ls -1 "$OUT/fold_$FOLD"/cascaded_*.pt | sort -t_ -k2 -n | tail -1)
echo "=== fold $FOLD : evaluating held-out subject with FINAL ckpt $CK ==="
python scripts/eval_cascaded.py \
    --ckpt "$CK" \
    --config "$CONFIG" \
    --pairs-csv "$PAIRS" \
    --fold "$FOLD" \
    --out "$OUT/fold_$FOLD"

echo "=== fold $FOLD done ==="
