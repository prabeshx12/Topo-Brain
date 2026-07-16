#!/bin/bash
# ============================================================================
# rebuild_segs.sh -- replace the incomplete 4-class segs with COMPLETE ones
# re-mapped from the original FreeSurfer aparc+aseg (recovers cerebellum + CC).
#
# Pulls the ses-2 aparc+aseg for all 10 subjects from prabeshbashyal/aligned-mri-dataset,
# re-maps each with the complete LUT (scripts/remap_aseg_complete.py), and OVERWRITES the
# seg files that pairs_cern.csv already points at -- so training/eval pick them up with no
# further change.
#
# Run on lxplus (kaggle token in ~/.kaggle, LCG view sourced):
#   bash cern/rebuild_segs.sh
# ============================================================================
set -e
DATA=/eos/user/p/ppokhrel/topobrain/data
NORM=$DATA/norm
ASEG=$DATA/aseg_full
DS=prabeshbashyal/aligned-mri-dataset
CODE=/eos/user/p/ppokhrel/topobrain/Topo-Brain

mkdir -p "$ASEG"
export PATH=$HOME/.local/bin:$PATH

for i in 01 02 03 04 05 06 07 08 09 10; do
    s=sub-$i
    f="Aligned/$s/ses-2/anat/aparc+aseg.nii"
    dst="$ASEG/${s}_aparc+aseg.nii"
    if [ ! -f "$dst" ]; then
        echo "=== pulling aparc+aseg for $s ==="
        kaggle datasets download "$DS" -f "$f" -p "$ASEG" --force
        # kaggle may deliver as .nii or zipped; normalise the name
        [ -f "$ASEG/aparc+aseg.nii" ] && mv "$ASEG/aparc+aseg.nii" "$dst"
        [ -f "$ASEG/aparc+aseg.nii.zip" ] && (cd "$ASEG" && unzip -o "aparc+aseg.nii.zip" && mv "aparc+aseg.nii" "$dst" && rm -f "aparc+aseg.nii.zip")
    fi
    echo "=== re-mapping $s (complete LUT) -> overwriting $NORM/${s}_seg.nii.gz ==="
    python "$CODE/scripts/remap_aseg_complete.py" \
        --aseg "$dst" \
        --ref  "$NORM/${s}_7t.nii.gz" \
        --out  "$NORM/${s}_seg.nii.gz"
done

echo ""
echo "=== DONE. pairs_cern.csv already points at $NORM/*_seg.nii.gz (now COMPLETE). ==="
echo "    cerebellum + corpus callosum recovered for all 10 subjects."
