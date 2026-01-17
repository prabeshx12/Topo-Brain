
import argparse
import logging
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import numpy as np

# Add src to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from src.bids import discover_bids_files, BIDSFile

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load NIfTI file and return data and affine."""
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    return data, img.affine


def normalize_for_display(data: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> np.ndarray:
    """Normalize data to [0, 1] based on robust percentiles."""
    non_zeros = data[data > 0]
    if len(non_zeros) == 0:
        return data
    vmin, vmax = np.percentile(non_zeros, [lower_percentile, upper_percentile])
    data_norm = np.clip(data, vmin, vmax)
    data_norm = (data_norm - vmin) / (vmax - vmin)
    return data_norm


def get_middle_slices(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract middle slices for sagittal, coronal, and axial views."""
    x, y, z = data.shape
    sagittal = np.rot90(data[x // 2, :, :])
    coronal = np.rot90(data[:, y // 2, :])
    axial = np.rot90(data[:, :, z // 2])
    return sagittal, coronal, axial


def plot_ortho_slices(
    ax_row: List[plt.Axes],
    data: np.ndarray,
    title_prefix: str = "",
    cmap: str = "gray",
    overlay: Optional[np.ndarray] = None
):
    """Plot orthogonal slices on a given row of axes."""
    sag, cor, ax = get_middle_slices(data)
    
    slices = [sag, cor, ax]
    names = ["Sagittal", "Coronal", "Axial"]
    
    if overlay is not None:
        ov_sag, ov_cor, ov_ax = get_middle_slices(overlay)
        overlays = [ov_sag, ov_cor, ov_ax]
    
    for i, (slice_data, name) in enumerate(zip(slices, names)):
        ax_cell = ax_row[i]
        ax_cell.imshow(slice_data, cmap=cmap, origin="upper", aspect="equal")
        
        if overlay is not None:
            # Mask overlay (e.g., brain mask in red)
            # Create RGBA for overlay
            masked_overlay = np.ma.masked_where(overlays[i] == 0, overlays[i])
            ax_cell.imshow(masked_overlay, cmap="autumn", alpha=0.3, origin="upper", aspect="equal")
            
        ax_cell.set_title(f"{title_prefix} {name}", fontsize=10)
        ax_cell.axis("off")


def plot_histogram(ax: plt.Axes, data: np.ndarray, label: str, color: str):
    """Plot intensity histogram excluding zeros."""
    # For normalized data (e.g. z-score), values can be negative.
    # We only want to exclude the background which is typically exactly 0.
    values = data[data != 0].flatten()
    if len(values) == 0:
        return
    ax.hist(values, bins=100, density=True, alpha=0.6, color=color, label=label)
    ax.set_ylabel("Density")
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)


def find_corresponding_preproc(raw_entry: BIDSFile, preproc_root: Path, suffix: str = "desc-preproc") -> Optional[Path]:
    """Attempt to find the preprocessed file corresponding to a raw BIDS entry."""
    # Expected path: output_root/sub-X/ses-Y/anat/sub-X_ses-Y_..._desc-preproc_T1w.nii.gz
    # Construct expected filename stem
    
    # Logic similar to pipeline but simplified for discovery
    # We look for files in the expected preproc folder that contain the subject, session, modality
    preproc_dir = preproc_root / raw_entry.subject / raw_entry.session / "anat"
    if not preproc_dir.exists():
        return None
        
    # Robust search: Look for files containing suffix AND modality, regardless of order
    # Supports both .nii and .nii.gz
    for cand in preproc_dir.glob("*"):
        name = cand.name
        if suffix in name and raw_entry.modality in name and (name.endswith(".nii") or name.endswith(".nii.gz")):
            return cand
    return None


def find_corresponding_mask(preproc_path: Path) -> Optional[Path]:
    """Find the brain mask corresponding to a preprocessed file."""
    # Typically _desc-brainmask.nii.gz
    name = preproc_path.name
    if "desc-preproc" in name:
        mask_name = name.replace("desc-preproc", "desc-brainmask")
        mask_path = preproc_path.with_name(mask_name)
        if mask_path.exists():
            return mask_path
    return None


def visualize_subject(
    subject: str,
    raw_files: List[BIDSFile],
    preproc_root: Path,
    output_dir: Optional[Path] = None,
    show_mask: bool = True,
    show_plot: bool = False
) -> Optional[plt.Figure]:
    """Generate comprehensive visualization for a single subject."""
    
    # Group by session/modality
    # We want to show: Raw 3T vs Preproc 3T, Raw 7T vs Preproc 7T
    
    logger.info(f"Generating visualization for subject: {subject}")
    
    # 1. Gather Data pairs
    pairs = []
    for entry in raw_files:
        if entry.subject != subject:
            continue
            
        preproc_path = find_corresponding_preproc(entry, preproc_root)
        if preproc_path:
            pairs.append((entry, preproc_path))
            
    if not pairs:
        logger.warning(f"No preprocessed pairs found for {subject}")
        return None

    # Create figure
    # Layout: One big row per pair (Raw Slices | Preproc Slices | Histograms)
    n_pairs = len(pairs)
    fig = plt.figure(figsize=(20, 6 * n_pairs))
    
    outer_grid = gridspec.GridSpec(n_pairs, 1, figure=fig, hspace=0.3)
    
    for i, (raw_entry, preproc_path) in enumerate(pairs):
        inner_grid = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=outer_grid[i], width_ratios=[1, 1, 1, 1.2], wspace=0.1
        )
        
        # Load Data
        raw_data, _ = load_nifti(raw_entry.path)
        prep_data, _ = load_nifti(preproc_path)
        
        # Normalize raw for display
        raw_disp = normalize_for_display(raw_data)
        prep_disp = normalize_for_display(prep_data)
        
        mask_data = None
        if show_mask:
            mask_path = find_corresponding_mask(preproc_path)
            if mask_path:
                mask_data, _ = load_nifti(mask_path)
        
        gs_left = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=inner_grid[0], wspace=0.05)
        gs_mid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=inner_grid[1], wspace=0.05)
        
        # Plot Histograms (Split into 2 vertical subplots to handle different scales)
        gs_hist = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=inner_grid[3], hspace=0.4)
        ax_hist_raw = fig.add_subplot(gs_hist[0])
        ax_hist_prep = fig.add_subplot(gs_hist[1])

        # Overlay/Diff
        gs_diff = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=inner_grid[2], wspace=0.05)

        # Plot Raw
        ax_raw = [fig.add_subplot(gs_left[j]) for j in range(3)]
        plot_ortho_slices(ax_raw, raw_disp, title_prefix="Raw", overlay=None) 
        
        # Plot Preproc
        ax_prep = [fig.add_subplot(gs_mid[j]) for j in range(3)]
        plot_ortho_slices(ax_prep, prep_disp, title_prefix="Prep", overlay=mask_data if show_mask else None)

        # Plot Diff/Mask Check on Raw
        ax_diff = [fig.add_subplot(gs_diff[j]) for j in range(3)]
        
        if raw_data.shape == prep_data.shape:
             plot_ortho_slices(ax_diff, raw_disp, title_prefix="Mask Check", overlay=mask_data)
        else:
             plot_ortho_slices(ax_diff, prep_disp, title_prefix="Clean", cmap="magma")

        # Plot Histograms
        plot_histogram(ax_hist_raw, raw_data, label="Raw", color="gray")
        ax_hist_raw.set_title(f"Raw Intensity", fontsize=10)
        
        plot_histogram(ax_hist_prep, prep_data, label="Preproc", color="green")
        ax_hist_prep.set_title(f"Preproc Intensity", fontsize=10)
        
    
    # Title
    fig.suptitle(f"Preprocessing QC: Subject {subject}", fontsize=16, fontweight='bold', y=0.95)
    
    if output_dir:
        output_path = Path(output_dir) / f"viz_{subject}.png"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved visualization to {output_path}")

    if show_plot:
        plt.show()
    elif output_dir is None:
        pass
    else:
        plt.close()
    
    return fig


def visualize_subject_by_id(
    subject_id: str,
    data_root: str = "data",
    preproc_root: str = "derivatives/topobrain-preproc",
    output_dir: Optional[str] = None,
    show_plot: bool = True
) -> Optional[plt.Figure]:
    """
    Convenience wrapper for Notebooks/Kaggle.
    """
    data_path = Path(data_root)
    preproc_path = Path(preproc_root)
    out_path = Path(output_dir) if output_dir else None
    
    # Quick discovery
    raw_files = discover_bids_files(data_path, modalities=["T1w", "T2w"])
    
    return visualize_subject(subject_id, raw_files, preproc_path, out_path, show_plot=show_plot)


def main():
    parser = argparse.ArgumentParser(description="Generate visual report cards for dataset preprocessing.")
    parser.add_argument("--data-root", default="data", type=Path, help="BIDS data root")
    parser.add_argument("--output-root", default="derivatives/topobrain-preproc", type=Path, help="Preprocessing output root")
    parser.add_argument("--viz-dir", default="docs/visualizations", type=Path, help="Where to save images")
    parser.add_argument("--subject", type=str, help="Specific subject to visualize (e.g., sub-1). If not set, picks random.")
    parser.add_argument("--count", type=int, default=1, help="Number of subjects to visualize if picking randomly")
    
    args = parser.parse_args()
    
    args.viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Discover Raw Files
    logger.info("Discovering raw BIDS files...")
    # Basic discovery - assuming standard config or just probing T1w/T2w
    raw_files = discover_bids_files(args.data_root, modalities=["T1w", "T2w"])
    
    if not raw_files:
        logger.error("No raw files found!")
        return
        
    all_subjects = sorted(list(set(f.subject for f in raw_files)))
    logger.info(f"Found {len(all_subjects)} subjects.")
    
    # 2. Select Subjects
    selected_subjects = []
    if args.subject:
        if args.subject in all_subjects:
            selected_subjects = [args.subject]
        else:
            logger.error(f"Subject {args.subject} not found in raw data.")
            return
    else:
        # Pick random
        count = min(args.count, len(all_subjects))
        selected_subjects = random.sample(all_subjects, count)
        
    # 3. Process
    for subj in selected_subjects:
        visualize_subject(subj, raw_files, args.output_root, args.viz_dir)
        
    logger.info("Done!")


if __name__ == "__main__":
    main()
