
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
    parser.add_argument("--pairs-csv", required=True, help="Path to pairs.csv from preprocessing")
    parser.add_argument("--data-root", required=True, help="Root of RAW data (for finding input asegs)")
    parser.add_argument("--preproc-root", required=True, help="Root of PREPROCESSED data (for finding target 7T reference)")
    parser.add_argument("--output-csv", default=None, help="Path to save updated pairs.csv (defaults to overtime input)")
    parser.add_argument("--mask-output-dir", default=None, help="Directory to save generated masks (optional, for read-only source)")
    parser.add_argument("--output-col", default="seg", help="Column name to add to pairs.csv")
    args = parser.parse_args()
    
    pairs_path = Path(args.pairs_csv)
    preproc_root = Path(args.preproc_root)
    output_csv_path = Path(args.output_csv) if args.output_csv else pairs_path
    
    mask_output_root = Path(args.mask_output_dir) if args.mask_output_dir else None
    if mask_output_root:
        mask_output_root.mkdir(parents=True, exist_ok=True)
    
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
        
        # Resolve target 7T path (Reference Geometry)
        target_rel_path = row['target_7t']
        target_path = preproc_root / target_rel_path
        
        if not target_path.exists():
             # Try absolute path fallback if CSV has absolute paths
             if Path(target_rel_path).exists():
                 target_path = Path(target_rel_path)
             # Try replacing extension .nii.gz <-> .nii
             elif target_path.with_name(target_path.name.replace(".nii.gz", ".nii")).exists():
                 target_path = target_path.with_name(target_path.name.replace(".nii.gz", ".nii"))
             elif target_path.with_suffix(".gz").exists():
                 target_path = target_path.with_suffix(".gz")
             else:
                 print(f"Skipping {subj}: Target 7T not found at {target_path}")
                 updated_rows.append(row)
                 continue
        # added search dir       
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
            # Should have been caught above, but safety check
             print(f"Skipping {subj}: Target 7T disappeared?")
             updated_rows.append(row)
             continue
            
        try:
            target_img = nib.load(target_path)
            aseg_img = nib.load(aseg_path)
            
            # Resample (Nearest Neighbor for masks!)
            resampled_aseg = resample_from_to(aseg_img, target_img, order=0)
            
            # 3. Map Labels
            data = resampled_aseg.get_fdata().astype(np.int32)
            mapped_data = map_labels(data, FS_MAPPING)
            
            # 4. Save
            if target_path.name.endswith(".nii.gz"):
                out_name = target_path.name.replace(".nii.gz", "_seg.nii.gz")
            elif target_path.name.endswith(".nii"):
                out_name = target_path.name.replace(".nii", "_seg.nii.gz") # Always save masks as .nii.gz
            else:
                out_name = target_path.name + "_seg.nii.gz"
                
            if mask_output_root:
                try:
                    rel_structure = target_path.parent.relative_to(preproc_root)
                except ValueError:
                    rel_structure = Path(subj) / "anat"
                
                out_dir = mask_output_root / rel_structure
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / out_name
            else:
                out_path = target_path.parent / out_name
            
            new_img = nib.Nifti1Image(mapped_data.astype(np.uint8), target_img.affine)
            nib.save(new_img, out_path)
            
            # 5. Update Row
            # Store absolute path in CSV for maximum safety if separate dir used
            # Or relative if inside preproc root?
            if mask_output_root:
                 # Absolute path is safest for separate output dir
                 row[args.output_col] = str(out_path.absolute()).replace("\\", "/") 
            else:
                # Store relative path if under preproc root
                try:
                    rel_out_path = out_path.relative_to(preproc_root)
                    row[args.output_col] = str(rel_out_path).replace("\\", "/") 
                except ValueError:
                    row[args.output_col] = str(out_path).replace("\\", "/") 
            
        except Exception as e:
            print(f"Failed {subj}: {e}")
            
        updated_rows.append(row)
        
    # Write back pairs.csv
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
        
    print(f"Updated {output_csv_path} with segmentation paths.")

if __name__ == "__main__":
    main()
