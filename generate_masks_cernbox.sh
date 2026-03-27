#!/bin/bash
# Quick tissue mask generation on CERNbox
# Run this on CERNbox to generate synthetic tissue masks

cd /eos/home-i04/p/ppokhrel/Untitled\ Folder\ 1/Topo-Brain

echo "=========================================="
echo "Generating Synthetic Tissue Masks"
echo "=========================================="
echo ""

# Generate tissue masks from existing data
python scripts/generate_tissue_masks.py \
    --manifest pairs.csv \
    --output_dir /eos/home-i04/p/ppokhrel/Untitled\ Folder\ 1/tissue_masks

echo ""
echo "=========================================="
echo "✅ Tissue masks generated!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Update dataset.pairs_csv in configs/train_diffusion.yaml"
echo "2. Resume training from 100k checkpoint"
echo "3. Watch loss_topo decrease!"
