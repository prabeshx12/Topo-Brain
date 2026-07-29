#!/bin/bash
# ============================================================================
# run_frunet.sh -- FR-U-Net baseline (Acs & Zhuang) for the cross-method result.
#
# A second, independent synthesis architecture on the SAME split (val=06/test=07) and SAME seed
# as our model, so its synthetic outputs can be scored by the same two judges (co-trained head vs
# independent segmenter). Purpose: show the circular-evaluation pitfall is not specific to one
# architecture. FR-U-Net's own hyperparameters (lr 2e-5, lambda_SSIM 0.7) are the paper's defaults
# in train_frunet.py and are left untouched. Output to runs/frunet_baseline (no collision).
# ============================================================================
set -e
echo "=== frunet baseline on $(hostname) @ $(date) ==="

source /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh
export PATH=$HOME/.local/bin:$PATH

nvidia-smi || { echo "NO GPU -- aborting"; exit 1; }
python3 -c "import torch; assert torch.cuda.is_available(); print('CUDA', torch.cuda.get_device_name(0))"

BASE=/eos/user/p/ppokhrel/topobrain
CODE=$BASE/Topo-Brain
PAIRS=$BASE/data/norm/pairs_cern.csv
OUT=$BASE/runs/frunet_baseline
mkdir -p "$OUT"
cd "$CODE"

python scripts/train_frunet.py \
    --pairs-csv "$PAIRS" \
    --config configs/train_diffusion.yaml \
    --n-iters 40000 \
    --batch-size 8 \
    --val-fold 0 --test-fold 1 \
    --save-freq 5000 \
    --max-hours 20 \
    --out-dir "$OUT"

echo "=== done @ $(date); checkpoints in $OUT ==="
