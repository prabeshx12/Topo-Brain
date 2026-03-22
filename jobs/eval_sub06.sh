#!/bin/bash
# Evaluation job for sub-06 with checkpoint_110000
# Runs on CERN HTCondor GPU node

set -e

PROJECT=/eos/home-i04/p/ppokhrel/topobrain/Topo-Brain
DATA_ROOT=/eos/home-i04/p/ppokhrel/data/topobrain
CHECKPOINT=$PROJECT/output/checkpoint_110000/checkpoint_110000.pt

cd $PROJECT

# Activate conda environment
source ~/.bashrc
conda activate topobrain

echo "=========================================="
echo "Starting evaluation: sub-06"
echo "Checkpoint: $CHECKPOINT"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

# Step 1: Full-volume evaluation
python scripts/evaluate_full_volume.py \
    --checkpoint $CHECKPOINT \
    --subject sub-06 \
    --pairs_csv pairs_new.csv \
    --data-root $DATA_ROOT \
    --output_dir results/sub-06/

echo ""
echo "Evaluation complete. Results in: results/sub-06/"

# Step 2: Aggregate metrics
python scripts/aggregate_metrics.py \
    --results-dir results/ \
    --subjects sub-06 \
    --output results/test_metrics.csv

echo "Done."
