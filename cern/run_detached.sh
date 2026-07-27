#!/bin/bash
# ============================================================================
# run_detached.sh -- the DOSE-RESPONSE middle point for the circular-evaluation paper.
#
# IDENTICAL to run_job.sh (Phase 3) in every respect -- same split (val=06/test=07), same
# 40k steps, same batch 8, lam-topo 0 -- EXCEPT it adds --detach-seg, which severs the cascade
# so the segmentation head's gradient no longer reaches the generator. That is the ONLY variable.
#
# WHY: it fills the coupling axis  joint (P3)  ->  detached (this)  ->  independent (GMM probe),
# letting the paper show the circular-evaluation bias SCALES with how tightly the segmenter is
# coupled to the generator. Compute-matched to P3 so the comparison is fair. Output goes to a
# SEPARATE dir so it can never collide with P3's checkpoints.
#
# Uses the same proven LCG_110_cuda view -- no container.
# ============================================================================
set -e
echo "=== detached job on $(hostname) @ $(date) ==="

source /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh
export PATH=$HOME/.local/bin:$PATH

nvidia-smi || { echo "NO GPU -- aborting"; exit 1; }
python3 -c "import torch; assert torch.cuda.is_available(); print('CUDA', torch.cuda.get_device_name(0))"

BASE=/eos/user/p/ppokhrel/topobrain
CODE=$BASE/Topo-Brain
PAIRS=$BASE/data/norm/pairs_cern.csv
OUT=$BASE/runs/phase3_detached
mkdir -p "$OUT"
cd "$CODE"

python scripts/train_cascaded.py \
    --pairs-csv "$PAIRS" \
    --config configs/train_diffusion.yaml \
    --n-iters 40000 \
    --batch-size 8 \
    --lam-topo 0 \
    --detach-seg \
    --val-fold 0 --test-fold 1 \
    --save-freq 5000 \
    --max-hours 20 \
    --out-dir "$OUT"

echo "=== done @ $(date); checkpoints in $OUT ==="
