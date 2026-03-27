#!/bin/bash
# Emergency fix script for CERNbox
# Run this to restart training with the correct loss weights

echo "=========================================="
echo "Topo-Brain: Emergency Training Restart"
echo "=========================================="
echo ""

# Step 1: Stop current training
echo "Step 1: Stopping current training..."
pkill -f train_diffusion.py
sleep 2
echo "✓ Training stopped"
echo ""

# Step 2: Pull latest changes
echo "Step 2: Pulling latest code from GitHub..."
cd /eos/home-i04/p/ppokhrel/Untitled\ Folder\ 1/Topo-Brain
git fetch origin
git reset --hard origin/feat/branch-new-pipeline
echo "✓ Code updated"
echo ""

# Step 3: Verify config
echo "Step 3: Verifying loss weights in config..."
if grep -q "lambda_topo: 0.3" configs/train_diffusion.yaml; then
    echo "✓ Config has new loss weights (lambda_topo: 0.3)"
else
    echo "❌ Config still has old weights!"
    echo "   Please check configs/train_diffusion.yaml manually"
    exit 1
fi
echo ""

# Step 4: Resume from 100k checkpoint
echo "Step 4: Resuming training from 100k checkpoint..."
echo "   Using new loss weights:"
echo "   - lambda_pixel: 0.05 (reduced from 0.1)"
echo "   - lambda_percep: 0.3 (reduced from 0.5)"
echo "   - lambda_topo: 0.3 (increased from 0.1)"
echo ""

python scripts/train_diffusion.py \
    --resume /eos/home-i04/p/ppokhrel/Untitled\ Folder\ 1/results/checkpoints/checkpoint_100000.pt \
    --config configs/train_diffusion.yaml \
    2>&1 | tee restart_training.log

echo ""
echo "=========================================="
echo "Training restarted with correct weights!"
echo "=========================================="
