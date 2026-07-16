#!/bin/bash
# ============================================================================
# run_job.sh -- the PAYLOAD HTCondor runs on the GPU node.
# Phase 3 (clean segs): train the cascaded model to 40k steps, topology OFF.
# This establishes the image-quality baseline on the COMPLETE segmentation and
# confirms training is healthy at batch 8 on a dedicated GPU. Topology (Phase 4)
# is a separate job after the entropy fix.
#
# Uses the LCG_110_cuda view we proved works (torch 2.11 + CUDA) -- NO container.
# ============================================================================
set -e
echo "=== job on $(hostname) @ $(date) ==="

# 1. environment: the proven CUDA software view + user pip packages (monai/nibabel/gudhi)
source /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh
export PATH=$HOME/.local/bin:$PATH

# 2. prove we actually got a GPU (fail loudly in the log if not)
nvidia-smi || { echo "NO GPU -- aborting"; exit 1; }
python3 -c "import torch; assert torch.cuda.is_available(); print('CUDA', torch.cuda.get_device_name(0))"

# 3. paths (all on EOS)
BASE=/eos/user/p/ppokhrel/topobrain
CODE=$BASE/Topo-Brain
PAIRS=$BASE/data/norm/pairs_cern.csv
OUT=$BASE/runs/phase3_clean
mkdir -p "$OUT"
cd "$CODE"

# 4. train. val=sub-06 / test=sub-07 (same split as the pilot, so it is directly comparable
#    to the old 14.01/21.53 numbers -- now on the COMPLETE segs). Fresh, topology off.
python scripts/train_cascaded.py \
    --pairs-csv "$PAIRS" \
    --config configs/train_diffusion.yaml \
    --n-iters 40000 \
    --batch-size 8 \
    --lam-topo 0 \
    --val-fold 0 --test-fold 1 \
    --save-freq 5000 \
    --max-hours 20 \
    --out-dir "$OUT"

echo "=== done @ $(date); checkpoints in $OUT ==="
