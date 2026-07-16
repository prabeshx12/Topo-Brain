#!/bin/bash
# ============================================================================
# run_job.sh -- the PAYLOAD that HTCondor runs ON the GPU node.
#
# HTCondor does NOT run your commands directly. It runs THIS script on whatever
# GPU machine it allocates. So everything the job needs to do goes in here:
#   1. show which GPU we got (so the log proves it worked)
#   2. enter a container that already has torch/monai/gudhi
#   3. resume training from the checkpoint and run to 40k steps
#
# Anything printed here lands in the .out log HTCondor writes back.
# ============================================================================
set -e                      # stop immediately if any command fails
echo "=== job started on $(hostname) at $(date) ==="

# ---- 1. what GPU did we actually get? -------------------------------------
nvidia-smi || { echo "NO GPU VISIBLE -- aborting"; exit 1; }
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ---- 2. paths (EDIT THESE ONCE YOU KNOW YOUR CERN LAYOUT) ------------------
# AFS home is small (~10 GB) and slow. Put code+data+checkpoints on EOS or a
# work area. These are placeholders -- we fill them in after you tell me where
# your data landed.
WORK=/eos/user/p/ppokhrel/topobrain      # <-- TBD: your EOS work dir
CODE=$WORK/Topo-Brain                    # the git repo
DATA=$WORK/data                          # the normalised nii files
CKPT=$WORK/checkpoints/cascaded_16677.pt # the checkpoint to resume from
OUT=$WORK/out_$(date +%Y%m%d_%H%M%S)     # this run's outputs
mkdir -p "$OUT"

# ---- 3. run inside a container that has the Python env ---------------------
# CERN provides GPU-ready container images. apptainer mounts them read-only and
# runs your command inside. The exact image path is something your friend/CERN
# docs will give us -- placeholder for now.
IMAGE=/cvmfs/unpacked.cern.ch/registry.hub.docker.com/pytorch/pytorch:latest  # <-- TBD

apptainer exec --nv "$IMAGE" bash -c "
    set -e
    cd $CODE
    pip install --user -q monai gudhi nibabel pyyaml    # anything the image lacks
    python scripts/train_cascaded.py \
        --pairs-csv $DATA/pairs_cern.csv \
        --resume $CKPT \
        --n-iters 40000 \
        --batch-size 8 \
        --max-hours 20 \
        --out-dir $OUT
"
echo "=== job finished at $(date); outputs in $OUT ==="
