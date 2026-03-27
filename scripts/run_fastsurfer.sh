#!/bin/bash
# Generate professional tissue segmentations using FastSurfer
# Requires: FastSurfer installed (https://github.com/Deep-MI/FastSurfer)

MANIFEST="pairs.csv"
OUTPUT_DIR="fastsurfer_segmentations"

mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "FastSurfer Tissue Segmentation Pipeline"
echo "=========================================="
echo ""

# Read 7T volumes from manifest
while IFS=, read -r input_3t target_7t mask subject modality
do
    # Skip header
    if [[ "$target_7t" == "target_7t_path" ]]; then
        continue
    fi
    
    echo "Processing: $subject"
    
    # Run FastSurfer
    fastsurfer \
        --t1 "$target_7t" \
        --sd "$OUTPUT_DIR" \
        --sid "$subject" \
        --seg_only \
        --device cuda
    
    # Convert aparc+aseg to 4-class tissue mask
    python scripts/convert_fastsurfer_to_tissue.py \
        --input "$OUTPUT_DIR/$subject/mri/aparc.DKTatlas+aseg.deep.mgz" \
        --output "$OUTPUT_DIR/${subject}_tissue_mask.nii.gz"
    
    echo "✓ $subject complete"
    echo ""
    
done < "$MANIFEST"

echo "=========================================="
echo "✅ All segmentations complete!"
echo "=========================================="
