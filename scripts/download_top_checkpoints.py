"""
Download (zip) the top-N checkpoints based on evaluation results.

Reads a CSV produced by evaluate_all_checkpoints.py or find_best_checkpoint.py,
ranks checkpoints by SSIM (or PSNR), copies the top-N .pt files into a zip archive.

Usage:
    python scripts/download_top_checkpoints.py \
        --results_csv checkpoint_evaluation_results.csv \
        --checkpoint_dir "/eos/.../checkpoints/" \
        --top_n 10 \
        --output top10_checkpoints.zip

    # Or use find_best_checkpoint.py results:
    python scripts/download_top_checkpoints.py \
        --results_csv results/checkpoint_search/checkpoint_results.csv \
        --checkpoint_dir "/eos/.../checkpoints/" \
        --top_n 10
"""
import argparse
import zipfile
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Zip the top-N checkpoints by evaluation metrics"
    )
    parser.add_argument(
        "--results_csv", type=str, required=True,
        help="CSV from evaluate_all_checkpoints.py or find_best_checkpoint.py",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Directory containing checkpoint_*.pt files",
    )
    parser.add_argument(
        "--top_n", type=int, default=10,
        help="Number of top checkpoints to include (default: 10)",
    )
    parser.add_argument(
        "--metric", type=str, default="ssim_mean",
        choices=["ssim_mean", "psnr_mean"],
        help="Metric to rank by (default: ssim_mean)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output zip path (default: top<N>_checkpoints.zip)",
    )
    args = parser.parse_args()

    # ---- Load results ----
    df = pd.read_csv(args.results_csv)
    print(f"Loaded {len(df)} checkpoint results from {args.results_csv}")

    if args.metric not in df.columns:
        # Try without _mean suffix
        alt = args.metric.replace("_mean", "")
        if alt in df.columns:
            args.metric = alt
        else:
            print(f"ERROR: Column '{args.metric}' not found in CSV.")
            print(f"Available columns: {list(df.columns)}")
            return

    # ---- Rank and select top N ----
    df_sorted = df.sort_values(args.metric, ascending=False)
    top = df_sorted.head(args.top_n)

    print(f"\nTop {args.top_n} checkpoints by {args.metric}:")
    print("-" * 60)
    for i, (_, row) in enumerate(top.iterrows(), 1):
        iteration = int(row.get("iteration", 0))
        metric_val = row[args.metric]
        name = row.get("checkpoint", f"checkpoint_{iteration}.pt")
        print(f"  {i:>2}. {name:<30s}  {args.metric} = {metric_val:.4f}")
    print("-" * 60)

    # ---- Resolve checkpoint file paths ----
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {ckpt_dir}")
        return

    files_to_zip = []
    missing = []
    for _, row in top.iterrows():
        iteration = int(row.get("iteration", 0))
        name = row.get("checkpoint", f"checkpoint_{iteration}.pt")
        path = ckpt_dir / name
        if not path.exists():
            # Try alternative naming
            alt_path = ckpt_dir / f"checkpoint_{iteration}.pt"
            if alt_path.exists():
                path = alt_path
            else:
                missing.append(name)
                continue
        files_to_zip.append((path, name))

    if missing:
        print(f"\nWARNING: {len(missing)} checkpoint files not found:")
        for m in missing:
            print(f"  - {m}")

    if not files_to_zip:
        print("ERROR: No checkpoint files found to zip.")
        return

    # ---- Create zip ----
    output_path = args.output or f"top{args.top_n}_checkpoints.zip"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nZipping {len(files_to_zip)} checkpoints to {output_path} ...")
    total_size = sum(f.stat().st_size for f, _ in files_to_zip)
    print(f"Total size (uncompressed): {total_size / (1024**3):.2f} GB")

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_STORED) as zf:
        # Add a summary CSV inside the zip
        summary_csv = top.to_csv(index=False)
        zf.writestr("top_checkpoints_summary.csv", summary_csv)

        for filepath, arcname in tqdm(files_to_zip, desc="Zipping", unit="file"):
            zf.write(filepath, arcname=arcname)

    final_size = output_path.stat().st_size
    print(f"\nDone! Saved: {output_path}")
    print(f"Zip size: {final_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()
