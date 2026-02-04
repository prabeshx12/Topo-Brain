#!/bin/bash
# Install and run FastSurfer on CERNbox

echo "=========================================="
echo "FastSurfer Installation for CERNbox"
echo "=========================================="
echo ""

# Step 1: Install FastSurfer
echo "Step 1: Installing FastSurfer..."
cd ~
git clone https://github.com/Deep-MI/FastSurfer.git
cd FastSurfer

# Create conda environment (if not exists)
if ! conda env list | grep -q fastsurfer; then
    conda create -n fastsurfer python=3.10 -y
fi

# Activate and install dependencies
source activate fastsurfer
pip install -r requirements.txt

echo "✓ FastSurfer installed"
echo ""

# Step 2: Download pre-trained models
echo "Step 2: Downloading pre-trained models..."
python FastSurferCNN/download_checkpoints.py

echo "✓ Models downloaded"
echo ""

# Step 3: Run on your data
echo "Step 3: Processing 7T volumes..."
cd /eos/home-i04/p/ppokhrel/Untitled\ Folder\ 1/Topo-Brain

OUTPUT_DIR="/eos/home-i04/p/ppokhrel/Untitled Folder 1/fastsurfer_output"
mkdir -p "$OUTPUT_DIR"

# Process each 7T volume
while IFS=, read -r input_3t target_7t mask subject modality
do
    # Skip header
    if [[ "$target_7t" == "target_7t_path" ]]; then
        continue
    fi
    
    echo "Processing: $subject ($modality)"
    
    # Run FastSurfer (segmentation only, no surface reconstruction)
    ~/FastSurfer/run_fastsurfer.sh \
        --t1 "$target_7t" \
        --sd "$OUTPUT_DIR" \
        --sid "${subject}_${modality}" \
        --seg_only \
        --device cuda \
        --batch 1
    
    echo "✓ $subject complete"
    
done < pairs.csv

echo ""
echo "=========================================="
echo "✅ FastSurfer processing complete!"
echo "=========================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Next: Convert FastSurfer output to 4-class tissue masks"
