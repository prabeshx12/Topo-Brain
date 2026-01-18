
import argparse
import logging
import csv
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from pathlib import Path
from tqdm import tqdm
import json

# FreeSurfer Lookup Table (simplified for TopoBrain)
# Mapping: FreeSurfer ID -> TopoBrain Class (0:BG, 1:CSF, 2:GM, 3:WM)
FS_MAPPING = {
    # 0: Background
    0: 0, 
    # CSF-like structures (Ventricles, etc.) -> 1
    4: 1, 14: 1, 15: 1, 43: 1, 44: 1, 72: 1, 213: 1, 221: 1, 
    # White Matter (Left/Right) -> 3
    2: 3, 41: 3, 
    # Grey Matter (Cortex + Subcortical GM) -> 2
    # Cortex
    3: 2, 42: 2, 
    # Subcortical GM (Thalamus, Caudate, Putamen, Pallidum, Hippocampus, Amygdala)
    10: 2, 11: 2, 12: 2, 13: 2, 17: 2, 18: 2, 26: 2,
    49: 2, 50: 2, 51: 2, 52: 2, 53: 2, 54: 2, 58: 2,
    # Brainstem
    16: 2 
}

def map_labels(data, mapping):
    """Maps varied source labels to a standardized set."""
    out = np.zeros_like(data)
    for src, dst in mapping.items():
        out[data == src] = dst
    
    # Anything else stays 0 (Background)
    
    # For cortical ribbon (often labeled 3/42 in simple aseg, but sometimes 1000+ in aparc+aseg)
    # If using aparc+aseg, cortical labels are >1000 and >2000. 
    # We map them to GM (2).
    out[(data >= 1000) & (data < 3000)] = 2
    
    return out

def main():
    parser = argparse.ArgumentParser(description="Preprocess segmentation masks")
    parser.add_argument("--pairs-csv", default="derivatives/topobrain-preproc/pairs.csv")
    parser.add_argument("--data-root", required=True, help="Root of raw BIDS data for finding asegs")
    parser.add_argument("--output-col", default="seg", help="Column name to add to pairs.csv")
    args = parser.parse_args()
    
    pairs_path = Path(args.pairs_csv)
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")
        
    with open(pairs_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
        
    if args.output_col not in fieldnames:
        fieldnames.append(args.output_col)
        
    root = Path(args.data_root)
    
    updated_rows = []
    
    for row in tqdm(rows, desc="Processing Masks"):
        subj = row['subject']
        target_path = Path(row['target_7t']) # Already processed 7T
        
        # 1. Find Raw Aseg
        # Strategy: Look in BIDS structure derivatives/freesurfer or similar?
        # Or look for `aseg.nii.gz` inside the subject folder structure provided
        # Current pattern: {data_root}/{subject}/...
        # We'll search recursively for *aseg*.nii*
        
        search_dir = root / subj
        candidates = list(search_dir.rglob("*aseg*.nii*"))
        
        # Filter out "aparc" if we just want simple aseg, or keep it.
        # Prefer "aseg.nii.gz" or "aseg.mgz" (freesurfer output)
        aseg_path = None
        
        # Priority sort: aseg.nii.gz > aseg.mgz > *aseg*
        candidates = sorted(candidates, key=lambda p: (
            p.name != 'aseg.nii.gz', 
            p.name != 'aseg.mgz', 
            len(str(p))
        ))
        
        if candidates:
            aseg_path = candidates[0]
        
        if not aseg_path or not aseg_path.exists():
            print(f"Skipping {subj}: No aseg found.")
            updated_rows.append(row)
            continue
            
        # 2. Resample to match Target 7T
        # We need the geometry of the PROCESSED 7T file
        if not target_path.exists():
            # If path is relative to some root not here, this might fail.
            # Assume running from project root where pairs.csv is.
            # Try prepending 'derivatives/topobrain-preproc' if needed?
            # Actually, user usually passes data-root. But for this script, let's assume
            # we can read the file as listed in csv.
            pass
            
        try:
            target_img = nib.load(target_path)
            aseg_img = nib.load(aseg_path)
            
            # Resample (Nearest Neighbor for masks!)
            resampled_aseg = resample_from_to(aseg_img, target_img, order=0)
            
            # 3. Map Labels
            data = resampled_aseg.get_fdata().astype(np.int32)
            mapped_data = map_labels(data, FS_MAPPING)
            
            # 4. Save
            out_name = target_path.name.replace(".nii.gz", "_seg.nii.gz")
            out_path = target_path.parent / out_name
            
            new_img = nib.Nifti1Image(mapped_data.astype(np.uint8), target_img.affine)
            nib.save(new_img, out_path)
            
            # 5. Update Row
            # Store relative path if original was relative, or absolute?
            # Usually pairs.csv has relative paths.
            row[args.output_col] = str(out_path).replace("\\", "/") # Ensure forward slashes
            
        except Exception as e:
            print(f"Failed {subj}: {e}")
            
        updated_rows.append(row)
        
    # Write back pairs.csv
    with open(pairs_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
        
    print(f"Updated {pairs_path} with segmentation paths.")

if __name__ == "__main__":
    main()
