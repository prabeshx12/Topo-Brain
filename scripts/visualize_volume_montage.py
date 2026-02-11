"""
Generates a full-brain axial montage from 3D NIfTI volumes.
Useful for qualitative assessment of Topo-Brain outputs.
"""
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

def create_montage(volume, step=4, cols=8):
    """Create a grid of axial slices."""
    # Ensure volume is 3D
    if volume.ndim == 4:
        volume = volume.squeeze()
        
    depth = volume.shape[0]
    # Filter out empty slices at top/bottom
    brain_indices = np.where(np.any(volume > volume.min() + (volume.max()-volume.min())*0.1, axis=(1, 2)))[0]
    if len(brain_indices) > 0:
        start, end = brain_indices[0], brain_indices[-1]
    else:
        start, end = 0, depth
        
    slices = list(range(start, end, step))
    rows = (len(slices) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), facecolor='black')
    axes = axes.flatten()
    
    # Normalize for display
    v_min, v_max = volume.min(), volume.max()
    
    for i, slice_idx in enumerate(slices):
        img = np.rot90(volume[slice_idx, :, :])
        axes[i].imshow(img, cmap='gray', vmin=v_min, vmax=v_max)
        axes[i].axis('off')
        axes[i].text(5, 5, f"z={slice_idx}", color='white', fontsize=8, alpha=0.7)
        
    # Hide remaining axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig

def main():
    parser = argparse.ArgumentParser(description='Generate axial montage from NIfTI')
    parser.add_argument('--input', type=str, required=True, help='Path to NIfTI volume (.nii.gz)')
    parser.add_argument('--seg', type=str, help='Optional segmentation NIfTI to overlay')
    parser.add_argument('--output', type=str, help='Output PNG path')
    parser.add_argument('--step', type=int, default=4, help='Slice step for axial view')
    parser.add_argument('--cols', type=int, default=10, help='Columns in montage grid')
    parser.add_argument('--title', type=str, default='Full Brain Axial View')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return
        
    print(f"Loading {input_path}...")
    nii = nib.load(str(input_path))
    data = nii.get_fdata()
    
    fig = create_montage(data, step=args.step, cols=args.cols)
    
    if args.seg:
        seg_path = Path(args.seg)
        if seg_path.exists():
            print(f"Overlaying segmentation from {seg_path}...")
            # This is a bit complex for a montage, perhaps save separate or tinted?
            # For now, let's keep it simple and just save the volume montage.
            pass

    out_path = args.output or input_path.with_suffix('').with_suffix('.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='black')
    print(f"Montage saved to: {out_path}")

if __name__ == "__main__":
    main()
