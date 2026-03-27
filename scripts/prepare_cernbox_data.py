"""
Script to prepare and rebase manifest paths for CERNbox/EOS training.
Converts the registration manifest.csv into a synthesis pairs.csv.
"""
import pandas as pd
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert registration manifest to synthesis pairs")
    parser.add_argument("--manifest", type=str, default="/eos/home-i04/p/ppokhrel/Untitled Folder 1/preprocessed/manifest.csv")
    parser.add_argument("--output", type=str, default="pairs.csv")
    parser.add_argument("--target-root", type=str, default="/eos/home-i04/p/ppokhrel/Untitled Folder 1/preprocessed/")
    args = parser.parse_args()

    print(f"Loading manifest: {args.manifest}")
    df = pd.read_csv(args.manifest)

    # 1. Update paths to current environment (CERNbox)
    # We replace the old /kaggle/ path with the new /eos/ path
    path_cols = ['output_path', 'mask_output_path']
    for col in path_cols:
        if col in df.columns:
            # Rebase based on the subject/session/filename structure
            # Example: .../preprocessed/sub-01/ses-1/...
            def rebase_path(old_path):
                if pd.isna(old_path): return old_path
                p = Path(old_path)
                try:
                    # Find the 'sub-' part and append to target_root
                    parts = p.parts
                    idx = next(i for i, part in enumerate(parts) if part.startswith("sub-"))
                    return str(Path(args.target_root) / Path(*parts[idx:]))
                except StopIteration:
                    return old_path
            
            df[col] = df[col].apply(rebase_path)

    # 2. Pair 3T (ses-1) and 7T (ses-2)
    # Filter for T1w mostly, or whatever modality you want to synthesize
    t1_3t = df[(df['field_strength'] == '3T') & (df['modality'] == 'T1w')]
    t1_7t = df[(df['field_strength'] == '7T') & (df['modality'] == 'T1w')]

    pairs = []
    for _, row_3t in t1_3t.iterrows():
        sub = row_3t['subject']
        # Find matching 7T for this subject
        match_7t = t1_7t[t1_7t['subject'] == sub]
        
        if not match_7t.empty:
            row_7t = match_7t.iloc[0]
            pairs.append({
                "subject": sub,
                "input_3t": row_3t['output_path'],
                "target_7t": row_7t['output_path'],
                "mask": row_7t['mask_output_path'] # Use 7T mask (or 3T, they should be aligned)
            })

    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv(args.output, index=False)
    print(f"Successfully created {len(pairs_df)} pairs in {args.output}")
    print(f"Sample paths:\n{pairs_df['input_3t'].iloc[0]}")

if __name__ == "__main__":
    main()
