#!/usr/bin/env python3
"""
Helper script to find the correct --data-root path for your evaluation.

Searches for data files from your pairs CSV and suggests the correct data-root.
"""
import csv
import sys
from pathlib import Path
import argparse


def find_data_root(pairs_csv: Path, search_dirs: list = None):
    """
    Find the data root by searching for files listed in the pairs CSV.

    Args:
        pairs_csv: Path to pairs CSV file
        search_dirs: Optional list of directories to search (defaults to common locations)
    """
    print(f"🔍 Analyzing pairs CSV: {pairs_csv}\n")

    if not pairs_csv.exists():
        print(f"❌ ERROR: CSV file not found: {pairs_csv}")
        return None

    # Read first entry from CSV
    with open(pairs_csv) as f:
        reader = csv.DictReader(f)
        first_row = next(reader, None)

    if not first_row:
        print("❌ ERROR: CSV file is empty")
        return None

    subject = first_row.get('subject', '?')
    input_3t = first_row.get('input_3t', '')

    if not input_3t:
        print("❌ ERROR: CSV missing 'input_3t' column")
        return None

    print(f"Sample entry from CSV:")
    print(f"  Subject: {subject}")
    print(f"  Relative path: {input_3t}")
    print()

    # Default search directories
    if search_dirs is None:
        search_dirs = [
            Path.home(),
            Path('/eos/user'),
            Path('/eos/home-i04'),
            Path('/data'),
            Path.cwd().parent,
            Path.cwd(),
        ]
        # Filter to only existing directories
        search_dirs = [d for d in search_dirs if d.exists()]

    print(f"🔎 Searching in {len(search_dirs)} directories...")
    print("This may take a moment...\n")

    # Extract the filename to search for
    filename = Path(input_3t).name

    found_paths = []
    for search_dir in search_dirs:
        try:
            # Use rglob to search recursively
            for found in search_dir.rglob(filename):
                if found.is_file():
                    found_paths.append(found)
                    print(f"✓ Found: {found}")

                    # Only check first few to avoid slowness
                    if len(found_paths) >= 3:
                        break
        except PermissionError:
            continue

        if len(found_paths) >= 3:
            break

    if not found_paths:
        print("❌ No matching files found.")
        print("\n💡 Manual search:")
        print(f"   find ~ -name '{filename}' 2>/dev/null")
        return None

    # Find the common root
    print(f"\n📊 Analysis:")
    for found_path in found_paths:
        # Try to find where the CSV path starts in the full path
        found_str = str(found_path)
        input_parts = Path(input_3t).parts

        # Try to match the CSV path structure
        for i, part in enumerate(input_parts):
            if part in found_str:
                # Find position in full path
                try:
                    idx = found_str.index(part)
                    # Data root is everything before this part
                    data_root = found_str[:idx].rstrip('/')
                    if data_root:
                        print(f"\n✅ Suggested data-root:")
                        print(f"   {data_root}")
                        print(f"\n📝 Use in your command:")
                        print(f"   --data-root '{data_root}'")
                        return Path(data_root)
                except ValueError:
                    continue

    # Fallback: suggest parent directories
    if found_paths:
        suggested = found_paths[0].parent
        for _ in range(len(Path(input_3t).parts) - 1):
            suggested = suggested.parent

        print(f"\n✅ Suggested data-root (inferred):")
        print(f"   {suggested}")
        return suggested

    return None


def verify_data_root(pairs_csv: Path, data_root: Path, max_check: int = 5):
    """
    Verify that the data-root works for files in the CSV.

    Args:
        pairs_csv: Path to pairs CSV
        data_root: Proposed data root
        max_check: Maximum number of files to check
    """
    print(f"\n🧪 Verifying data-root: {data_root}\n")

    with open(pairs_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    checked = 0
    found = 0
    missing = []

    for row in rows[:max_check]:
        checked += 1
        subject = row.get('subject', '?')
        input_path = data_root / row.get('input_3t', '')
        target_path = data_root / row.get('target_7t', '')

        inp_exists = input_path.exists()
        tgt_exists = target_path.exists()

        status = '✓' if inp_exists and tgt_exists else '✗'

        print(f"{status} {subject}:")
        print(f"   Input:  {input_path} {'✓' if inp_exists else '✗ MISSING'}")
        print(f"   Target: {target_path} {'✓' if tgt_exists else '✗ MISSING'}")

        if inp_exists and tgt_exists:
            found += 1
        else:
            missing.append(subject)

    print(f"\n📊 Verification:")
    print(f"   Checked: {checked} subjects")
    print(f"   Found:   {found} subjects")
    print(f"   Missing: {checked - found} subjects")

    if found == checked:
        print(f"\n✅ SUCCESS! All files found with data-root: {data_root}")
        return True
    else:
        print(f"\n⚠️  Some files missing: {missing}")
        print(f"   Check that your data-root is correct")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the correct --data-root for your evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--pairs_csv', type=str, default='pairs_new.csv',
                        help='Path to pairs CSV file (default: pairs_new.csv)')
    parser.add_argument('--search', type=str, nargs='+',
                        help='Additional directories to search')
    parser.add_argument('--verify', type=str,
                        help='Verify a specific data-root path')
    args = parser.parse_args()

    pairs_csv = Path(args.pairs_csv)

    if args.verify:
        # Verify mode
        data_root = Path(args.verify)
        if not data_root.exists():
            print(f"❌ ERROR: Directory does not exist: {data_root}")
            sys.exit(1)

        verify_data_root(pairs_csv, data_root)
    else:
        # Search mode
        search_dirs = [Path(d) for d in args.search] if args.search else None
        data_root = find_data_root(pairs_csv, search_dirs)

        if data_root and data_root.exists():
            # Automatically verify
            print("\n" + "="*60)
            verify_data_root(pairs_csv, data_root)

            print("\n" + "="*60)
            print("💡 Complete evaluation command:")
            print(f"""
python scripts/evaluate_full_volume.py \\
    --checkpoint output/checkpoint_latest.pt \\
    --subject sub-01 \\
    --pairs_csv {pairs_csv} \\
    --data-root '{data_root}' \\
    --output_dir results/sub-01/
            """)
