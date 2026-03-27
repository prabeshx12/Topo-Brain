"""
Script to regenerate pairs.csv from an existing manifest.csv.
fixes the issue where 'skipped' items (already preprocessed) were excluded from pairs.
"""
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def regenerate_pairs(manifest_path: Path, output_path: Path, make_relative: bool = False):
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return

    # Read manifest
    entries = []
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        entries = list(reader)

    logger.info(f"Loaded {len(entries)} entries from manifest.")

    # Configuration (matching default preprocess config)
    # Assuming standard session names if not provided
    session_3t = "ses-1" 
    session_7t = "ses-2"
    modalities = ["T1w", "T2w"]

    by_subject: Dict[str, Dict[str, Dict[str, str]]] = {}
    
    for entry in entries:
        # ACCEPT both 'ok' and 'skipped'
        if entry.get("status") not in ["ok", "skipped"]:
            continue
            
        subject = entry["subject"]
        session = entry["session"]
        modality = entry["modality"]
        field_strength = entry.get("field_strength")
        output_path_str = entry.get("output_path")

        if not output_path_str:
            continue
            
        # Make relative if requested
        if make_relative:
            # Try to find 'sub-XX' and slice from there
            try:
                # Normalize slashes
                p = output_path_str.replace("\\", "/")
                idx = p.find(f"/{subject}/")
                if idx != -1:
                    # Keep everything starting from subject
                    output_path_str = p[idx+1:]
                elif p.startswith(subject):
                     pass
                else:
                    # Fallback: try to split by 'preprocessed' or 'derivatives'
                    parts = p.split("/")
                    if subject in parts:
                        si = parts.index(subject)
                        output_path_str = "/".join(parts[si:])
            except Exception as e:
                logger.warning(f"Failed to make relative path for {output_path_str}: {e}")

        label = None
        # Determine 3T vs 7T
        if field_strength == "3T" or session == session_3t:
            label = "3T"
        elif field_strength == "7T" or session == session_7t:
            label = "7T"
        
        if label is None:
            continue
            
        # Store entry
        if subject not in by_subject:
            by_subject[subject] = {}
        
        key = f"{label}_{modality}"
        by_subject[subject][key] = output_path_str

    # Generate pairs
    pairs = []
    for subject, items in by_subject.items():
        # Check for primary T1w pair
        path_3t = items.get("3T_T1w")
        path_7t = items.get("7T_T1w")
        
        if path_3t and path_7t:
            pair = {
                "subject": subject,
                "modality": "T1w",
                "input_3t": path_3t,
                "target_7t": path_7t,
            }
            
            # Optional T2
            path_3t_t2 = items.get("3T_T2w")
            if path_3t_t2:
                pair["input_3t_t2"] = path_3t_t2
                
            pairs.append(pair)
        else:
            logger.warning(f"Subject {subject} incomplete: 3T={bool(path_3t)}, 7T={bool(path_7t)}")

    if not pairs:
        logger.warning("No valid pairs found.")
        return

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
        writer.writeheader()
        writer.writerows(pairs)
        
    logger.info(f"Successfully wrote {len(pairs)} pairs to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to manifest.csv")
    parser.add_argument("--output", required=True, help="Path to output pairs.csv")
    parser.add_argument("--make-relative", action="store_true", help="Strip absolute paths, keeping only from 'sub-XX' onwards")
    args = parser.parse_args()
    
    # Custom logic to modify how paths are stored before writing
    # We essentially intercept the regenerate_pairs function logic by modifying it or handling it here
    # For simplicity, let's update regenerate_pairs to take this flag
    regenerate_pairs(Path(args.manifest), Path(args.output), make_relative=args.make_relative)

if __name__ == "__main__":
    main()
