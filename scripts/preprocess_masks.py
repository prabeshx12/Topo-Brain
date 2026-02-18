
import argparse
import logging
import csv
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from pathlib import Path
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("preprocess_masks")

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

def find_aseg(folder: Path):
    """
    Find the best FreeSurfer segmentation file in a folder.
    Priority: aparc+aseg.nii.gz > aparc+aseg.nii > aseg.nii.gz > aseg.nii
    
    aparc+aseg is preferred because it includes cortical parcellation
    labels (1000-2999) which get mapped to GM, giving full cortical coverage.
    """
    # Search in the given folder AND recursively below it
    search_dirs = [folder]
    
    candidates = []
    for d in search_dirs:
        for pattern in ["aparc+aseg.nii.gz", "aparc+aseg.nii",
                        "aseg.nii.gz", "aseg.nii",
                        "*aparc+aseg*.nii*", "*aseg*.nii*"]:
            candidates.extend(d.glob(pattern))
    
    if not candidates:
        # Try recursive search as fallback
        candidates = list(folder.rglob("*aseg*.nii*"))
    
    if not candidates:
        return None
    
    # Priority sort
    def priority(p):
        name = p.name.lower()
        if name == "aparc+aseg.nii.gz":
            return 0
        elif name == "aparc+aseg.nii":
            return 1
        elif name == "aseg.nii.gz":
            return 2
        elif name == "aseg.nii":
            return 3
        elif "aparc" in name:
            return 4
        else:
            return 5
    
    candidates.sort(key=priority)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(
        description="Create 4-class tissue masks from FreeSurfer aseg files.\n\n"
                    "Finds aseg/aparc+aseg in the --aligned-root directory tree,\n"
                    "maps FreeSurfer labels to 4 classes (BG/CSF/GM/WM),\n"
                    "resamples to match target_7t geometry, and updates pairs CSV.\n\n"
                    "Example:\n"
                    "  python preprocess_masks.py \\\n"
                    "    --pairs-csv /eos/.../pairs.csv \\\n"
                    "    --aligned-root /eos/.../Aligned \\\n"
                    "    --data-root /eos/.../tissue_masks \\\n"
                    "    --output-dir /eos/.../tissue_masks/masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pairs-csv", required=True,
        help="Path to pairs CSV (will be updated with tissue_mask_path column)",
    )
    parser.add_argument(
        "--aligned-root", required=True,
        help="Root of the Aligned folder containing aseg files. "
             "e.g. '/eos/user/p/ppokhrel/Untitled Folder 1/Aligned'",
    )
    parser.add_argument(
        "--data-root", default=None,
        help="Root directory that target_7t paths in CSV are relative to. "
             "If target_7t paths are absolute, leave this empty.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to save mapped masks. Default: same folder as target_7t",
    )
    parser.add_argument(
        "--output-csv", default=None,
        help="Save updated CSV to a new file instead of overwriting. "
             "Default: overwrites --pairs-csv",
    )
    parser.add_argument(
        "--mask-col", default="tissue_mask_path",
        help="Column name for mask path in CSV (default: tissue_mask_path)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Just find and report aseg files without processing",
    )
    args = parser.parse_args()
    
    aligned_root = Path(args.aligned_root)
    if not aligned_root.exists():
        raise FileNotFoundError(f"Aligned root not found: {aligned_root}")
    
    data_root = Path(args.data_root) if args.data_root else None
    
    pairs_path = Path(args.pairs_csv)
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")
        
    # Read CSV
    with open(pairs_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
        
    if args.mask_col not in fieldnames:
        fieldnames.append(args.mask_col)
        
    logger.info(f"Processing {len(rows)} subjects from {pairs_path}")
    
    updated_rows = []
    success_count = 0
    
    for row in tqdm(rows, desc="Processing masks"):
        subj = row['subject']
        target_rel = row['target_7t']
        
        # Resolve target_7t to absolute path
        target_path = Path(target_rel)
        if not target_path.is_absolute() and data_root:
            target_path = data_root / target_rel
        
        # Derive aseg search folder from aligned root
        # CSV has: sub-01/ses-2/anat/sub-01_ses-2_desc-preproc_T1w.nii.gz
        # Aligned has: {aligned_root}/sub-01/ses-2/anat/aparc+aseg.nii
        # So we take the DIRECTORY part of target_7t relative path
        target_rel_dir = Path(target_rel).parent  # e.g. sub-01/ses-2/anat
        aseg_search_dir = aligned_root / target_rel_dir
        
        logger.info(f"  {subj}: Searching for aseg in {aseg_search_dir}")
        
        # Find aseg
        aseg_path = find_aseg(aseg_search_dir)
        
        if aseg_path is None:
            # Fallback: try subject root in aligned (e.g. {aligned_root}/sub-01/)
            subj_dir = aligned_root / subj
            if subj_dir.exists():
                aseg_path = find_aseg(subj_dir)
        
        if aseg_path is None:
            logger.warning(f"  {subj}: No aseg found in {aseg_search_dir} or {aligned_root / subj}. Skipping.")
            updated_rows.append(row)
            continue
        
        logger.info(f"  {subj}: Found {aseg_path.name} at {aseg_path}")
        
        if args.dry_run:
            success_count += 1
            updated_rows.append(row)
            continue
        
        try:
            # Load target (for geometry/affine reference)
            target_img = nib.load(str(target_path))
            aseg_img = nib.load(str(aseg_path))
            
            logger.info(f"    aseg shape: {aseg_img.shape}, target shape: {target_img.shape}")
            
            # Check if resampling is needed (different geometry)
            needs_resample = (aseg_img.shape != target_img.shape or
                              not np.allclose(aseg_img.affine, target_img.affine, atol=1e-3))
            
            if needs_resample:
                logger.info(f"    Resampling aseg to match target geometry ...")
                resampled = resample_from_to(aseg_img, target_img, order=0)  # Nearest neighbor
                data = resampled.get_fdata().astype(np.int32)
            else:
                logger.info(f"    Geometry matches, no resampling needed.")
                data = aseg_img.get_fdata().astype(np.int32)
            
            # Map FreeSurfer labels → 4 classes
            mapped = map_labels(data, FS_MAPPING)
            
            # Report class distribution
            unique, counts = np.unique(mapped, return_counts=True)
            total = mapped.size
            logger.info(f"    Class distribution:")
            class_names = {0: "BG", 1: "CSF", 2: "GM", 3: "WM"}
            for u, c in zip(unique, counts):
                pct = 100 * c / total
                logger.info(f"      {class_names.get(u, u)}: {c:>10,} voxels ({pct:.1f}%)")
            
            # Save mapped mask
            if args.output_dir:
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_name = f"{subj}_tissue_mask.nii.gz"
                out_path = out_dir / out_name
            else:
                # Save alongside target_7t (use resolved absolute path)
                target_folder = target_path.parent
                target_folder.mkdir(parents=True, exist_ok=True)
                out_name = target_path.name.replace(".nii.gz", "_tissue_seg.nii.gz").replace(".nii", "_tissue_seg.nii")
                out_path = target_folder / out_name
            
            mask_img = nib.Nifti1Image(mapped.astype(np.uint8), target_img.affine, target_img.header)
            nib.save(mask_img, str(out_path))
            
            # Update row
            row[args.mask_col] = str(out_path).replace("\\", "/")
            success_count += 1
            logger.info(f"    Saved: {out_path}")
            
        except Exception as e:
            logger.error(f"  {subj}: Failed — {e}")
            import traceback
            traceback.print_exc()
            
        updated_rows.append(row)
    
    if args.dry_run:
        logger.info(f"\nDry run complete. Found aseg for {success_count}/{len(rows)} subjects.")
        return
    
    # Write updated CSV
    out_csv = Path(args.output_csv) if args.output_csv else pairs_path
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Done! Processed {success_count}/{len(rows)} subjects.")
    logger.info(f"Updated CSV: {out_csv}")
    logger.info(f"Mask column: '{args.mask_col}'")
    logger.info(f"{'=' * 60}")
    logger.info(f"\nTo use these masks for training, ensure your training config")
    logger.info(f"pairs_csv points to: {out_csv}")
    logger.info(f"The dataset will automatically load the masks via the")
    logger.info(f"'{args.mask_col}' column for topology loss supervision.")

if __name__ == "__main__":
    main()
