"""
Download (zip) a specific checkpoint by iteration number.

Usage:
    python scripts/download_checkpoint.py \
        --checkpoint-dir "/eos/.../checkpoints/" \
        --iteration 100000

    # Custom output path:
    python scripts/download_checkpoint.py \
        --checkpoint-dir "/eos/.../checkpoints/" \
        --iteration 100000 \
        --output checkpoint_100k.zip

    # Multiple checkpoints at once:
    python scripts/download_checkpoint.py \
        --checkpoint-dir "/eos/.../checkpoints/" \
        --iteration 100000 141000
"""
import argparse
import zipfile
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Zip specific checkpoint(s) by iteration number"
    )
    parser.add_argument(
        "--checkpoint-dir", required=True,
        help="Directory containing checkpoint_*.pt files",
    )
    parser.add_argument(
        "--iteration", type=int, nargs="+", required=True,
        help="Iteration number(s) to download, e.g. 100000 or 100000 141000",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output zip path. Default: checkpoint_<iter>.zip",
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {ckpt_dir}")
        return

    # Find checkpoint files
    files_to_zip = []
    for iteration in args.iteration:
        # Try common naming patterns
        candidates = [
            f"checkpoint_{iteration}.pt",
            f"ckpt_{iteration}.pt",
            f"checkpoint_{iteration:06d}.pt",
        ]
        found = False
        for name in candidates:
            path = ckpt_dir / name
            if path.exists():
                files_to_zip.append((path, name))
                size_mb = path.stat().st_size / (1024 ** 2)
                print(f"  Found: {name} ({size_mb:.0f} MB)")
                found = True
                break
        if not found:
            print(f"  WARNING: No checkpoint found for iteration {iteration}")
            print(f"    Tried: {', '.join(candidates)}")
            # List available checkpoints near the target
            all_ckpts = sorted(ckpt_dir.glob("checkpoint_*.pt"))
            if all_ckpts:
                iters = []
                for c in all_ckpts:
                    try:
                        i = int(c.stem.split("_")[1])
                        iters.append(i)
                    except (ValueError, IndexError):
                        pass
                nearest = sorted(iters, key=lambda x: abs(x - iteration))[:5]
                print(f"    Nearest available: {nearest}")

    if not files_to_zip:
        print("\nERROR: No checkpoint files found.")
        return

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    elif len(args.iteration) == 1:
        output_path = Path(f"checkpoint_{args.iteration[0]}.zip")
    else:
        iters_str = "_".join(str(i) for i in args.iteration)
        output_path = Path(f"checkpoints_{iters_str}.zip")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_size = sum(f.stat().st_size for f, _ in files_to_zip)
    print(f"\nZipping {len(files_to_zip)} checkpoint(s) to {output_path}")
    print(f"Total size (uncompressed): {total_size / (1024**3):.2f} GB")

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_STORED) as zf:
        for filepath, arcname in tqdm(files_to_zip, desc="Zipping", unit="file"):
            zf.write(filepath, arcname=arcname)

    final_size = output_path.stat().st_size
    print(f"\nDone! Saved: {output_path} ({final_size / (1024**3):.2f} GB)")


if __name__ == "__main__":
    main()
