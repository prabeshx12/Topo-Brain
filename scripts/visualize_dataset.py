
import argparse
import logging
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

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
    # Handle signed data (like Z-scores) by only excluding exact zeros (background)
    non_zeros = data[data != 0]
    if len(non_zeros) == 0:
        return data
    vmin, vmax = np.percentile(non_zeros, [lower_percentile, upper_percentile])
    data_norm = np.clip(data, vmin, vmax)
    data_norm = (data_norm - vmin) / (vmax - vmin)
    return data_norm

# ... (rest of code)

def plot_histogram(ax: plt.Axes, data: np.ndarray, label: str, color: str):
    """Plot intensity histogram excluding zeros."""
    values = data[data != 0].flatten()
    if len(values) == 0:
        return
    ax.hist(values, bins=100, density=True, alpha=0.6, color=color, label=label)
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.set_ylabel("Density", fontsize=8)


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
    names = ["Sag", "Cor", "Ax"]
    
    if overlay is not None:
        ov_sag, ov_cor, ov_ax = get_middle_slices(overlay)
        overlays = [ov_sag, ov_cor, ov_ax]
    
    for i, (slice_data, name) in enumerate(zip(slices, names)):
        ax_cell = ax_row[i]
        ax_cell.imshow(slice_data, cmap=cmap, origin="upper", aspect="equal")
        
        if overlay is not None:
            # Mask overlay (e.g., brain mask in red)
            masked_overlay = np.ma.masked_where(overlays[i] == 0, overlays[i])
            ax_cell.imshow(masked_overlay, cmap="autumn", alpha=0.3, origin="upper", aspect="equal")
            
        ax_cell.set_title(f"{title_prefix} {name}", fontsize=9)
        ax_cell.axis("off")


def plot_histogram(ax: plt.Axes, data: np.ndarray, label: str, color: str):
    """Plot intensity histogram excluding zeros."""
    values = data[data != 0].flatten()
    if len(values) == 0:
        return
    ax.hist(values, bins=100, density=True, alpha=0.6, color=color, label=label)
    ax.tick_params(axis='both', which='major', labelsize=7)
    # y-ticks restored



def find_corresponding_preproc(raw_entry: BIDSFile, preproc_root: Path, suffix: str = "desc-preproc") -> Optional[Path]:
    """Attempt to find the preprocessed file corresponding to a raw BIDS entry."""
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
    # Try multiple common patterns
    name = preproc_path.name
    # 1. Replace desc-preproc with desc-brainmask
    if "desc-preproc" in name:
        patterns = ["desc-brainmask", "desc-mask", "_mask"]
        for pat in patterns:
            mask_name = name.replace("desc-preproc", pat)
            mask_path = preproc_path.with_name(mask_name)
            if mask_path.exists():
                return mask_path
    
    # 2. General glob in the same folder if strict replacement failed
    # Look for same modality + "mask"
    parent = preproc_path.parent
    # Identify modality from filename logic or passed args? We don't have args here.
    # Simple heuristic: split by underscore
    parts = name.split('_')
    for part in parts:
        if part in ["T1w", "T2w"]:
            # Try to find *T1w*mask*
            cands = list(parent.glob(f"*{part}*mask*"))
            # Filter for same siblings (rough check)
            if cands:
                return cands[0]
                
    return None


def visualize_subject(
    subject: str,
    raw_files: List[BIDSFile],
    preproc_root: Path,
    output_dir: Optional[Path] = None,
    show_mask: bool = True,
    show_plot: bool = False
) -> List[plt.Figure]:
    """
    Generate comprehensive visualization for a single subject.
    Creates a separate 'QC Card' image for each scan pair.
    Returns a list of generated figures.
    """
    
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
        return []

    # Prepare output directory
    subject_out_dir = None
    if output_dir:
        subject_out_dir = output_dir / subject
        subject_out_dir.mkdir(parents=True, exist_ok=True)

    generated_figs = []

    # 2. Generate QC Card for each pair
    for i, (raw_entry, preproc_path) in enumerate(pairs):
        # Create a dedicated figure for this scan
        # Layout: [Raw] | [Preproc] | [Mask] | [Hist]
        fig = plt.figure(figsize=(20, 5))
        
        # Grid: 4 columns
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.6], wspace=0.1, figure=fig)
        
        # Load Data
        raw_data, _ = load_nifti(raw_entry.path)
        prep_data, _ = load_nifti(preproc_path)
        
        raw_disp = normalize_for_display(raw_data)
        prep_disp = normalize_for_display(prep_data)
        
        mask_data = None
        mask_source = "None"
        
        if show_mask:
            mask_path = find_corresponding_mask(preproc_path)
            if mask_path:
                mask_data, _ = load_nifti(mask_path)
                mask_source = "File"
            else:
                # Fallback: Infer mask from non-zero preprocessed data
                # This shows the "Effective Mask"
                mask_data = (prep_data != 0).astype(np.float32)
                mask_source = "Inferred (>0)"
        
        # --- Col 1: Raw ---
        gs_raw = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.05)
        ax_raw = [fig.add_subplot(gs_raw[j]) for j in range(3)]
        plot_ortho_slices(ax_raw, raw_disp, title_prefix="Raw")
        
        # --- Col 2: Preproc ---
        gs_prep = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], wspace=0.05)
        ax_prep = [fig.add_subplot(gs_prep[j]) for j in range(3)]
        plot_ortho_slices(ax_prep, prep_disp, title_prefix="Prep")

        # --- Col 3: Mask on Raw ---
        gs_mask = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2], wspace=0.05)
        ax_mask = [fig.add_subplot(gs_mask[j]) for j in range(3)]
        if mask_data is not None:
             plot_ortho_slices(ax_mask, raw_disp, title_prefix=f"Mask ({mask_source})", overlay=mask_data)
        else:
             # Just show blank or text? Should not happen with inference fallback
             for ax in ax_mask: ax.axis('off')

        # --- Col 4: Histograms ---
        gs_hist = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[3], hspace=0.3)
        ax_hist_raw = fig.add_subplot(gs_hist[0])
        ax_hist_prep = fig.add_subplot(gs_hist[1])
        
        plot_histogram(ax_hist_raw, raw_data, label="Raw", color="silver")
        ax_hist_raw.set_title("Raw Intensity", fontsize=9, loc='left')
        
        plot_histogram(ax_hist_prep, prep_data, label="Prep", color="mediumseagreen")
        ax_hist_prep.set_title("Preproc Intensity", fontsize=9, loc='left')
        
        # Header / Title
        scan_info = f"{raw_entry.session} | {raw_entry.modality}"
        if raw_entry.field_strength:
            scan_info += f" | {raw_entry.field_strength}"
            
        fig.suptitle(f"Subject: {subject}   Scan: {scan_info}", fontsize=14, fontweight='bold', y=0.98)
        
        # Save
        if subject_out_dir:
            fname = f"{subject}_{raw_entry.session}_{raw_entry.modality}_QC.png"
            save_path = subject_out_dir / fname
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved QC card: {save_path}")
            
        generated_figs.append(fig)
        
        if not show_plot and output_dir:
            plt.close(fig)

    if show_plot:
        plt.show()
        
    return generated_figs


def visualize_subject_by_id(
    subject_id: str,
    data_root: str = "data",
    preproc_root: str = "derivatives/topobrain-preproc",
    output_dir: Optional[str] = None,
    show_plot: bool = True
) -> List[plt.Figure]:
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
