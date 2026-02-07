"""
Find the best checkpoint using DETERMINISTIC evaluation.

Unlike evaluate_all_checkpoints.py (which uses random patches), this script:
  1) Loads full volumes for 1-2 subjects
  2) Extracts FIXED patches at predetermined brain locations
  3) Evaluates every checkpoint on the EXACT SAME patches
  4) Results are reproducible & comparable across checkpoints

Usage:
    python scripts/find_best_checkpoint.py \
        --checkpoint_dir "/eos/.../checkpoints/" \
        --output_dir results/checkpoint_search
"""
import os
import sys
import csv
import torch
import yaml
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.diffusion import GaussianDiffusion
from src.model import AnatomyGuidedUNet


# ---------------------------------------------------------------------------
# Metrics  (identical to sample_diffusion.py & evaluate_full_volume.py)
# ---------------------------------------------------------------------------

def compute_metrics(pred_np, target_np):
    """Both arrays in [-1, 1]. Rescale to [0, 1], data_range=1.0."""
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr

    pred_np = np.clip(pred_np, -1.0, 1.0)
    target_np = np.clip(target_np, -1.0, 1.0)
    pred_01 = (pred_np + 1.0) / 2.0
    tgt_01  = (target_np + 1.0) / 2.0

    return ssim(tgt_01, pred_01, data_range=1.0), psnr(tgt_01, pred_01, data_range=1.0)


# ---------------------------------------------------------------------------
# Fixed patch extraction
# ---------------------------------------------------------------------------

def extract_fixed_patches(volume, patch_size=64):
    """
    Extract deterministic patches from a 3-D volume at anatomically
    meaningful locations.  Returns list of (patch_array, label_str).

    Locations:
      - Center of volume
      - 6 offset positions (±quarter along each axis from center)
      - Brain centroid (center-of-mass of non-background voxels)
    """
    D, H, W = volume.shape
    r = patch_size // 2                       # 32
    cd, ch, cw = D // 2, H // 2, W // 2      # volume center

    def crop(d, h, w, label):
        """Safe crop centred at (d, h, w)."""
        d = max(r, min(d, D - r))
        h = max(r, min(h, H - r))
        w = max(r, min(w, W - r))
        return volume[d-r:d+r, h-r:h+r, w-r:w+r].copy(), label

    patches = [
        crop(cd, ch, cw,               "center"),
        crop(cd - D//4, ch, cw,         "sup"),       # superior
        crop(cd + D//4, ch, cw,         "inf"),       # inferior
        crop(cd, ch - H//4, cw,         "anterior"),
        crop(cd, ch + H//4, cw,         "posterior"),
        crop(cd, ch, cw - W//4,         "left"),
        crop(cd, ch, cw + W//4,         "right"),
    ]

    # Brain centroid patch
    brain = volume > -0.95
    if brain.any():
        coords = np.argwhere(brain)
        centroid = coords.mean(axis=0).astype(int)
        patches.append(crop(centroid[0], centroid[1], centroid[2], "centroid"))

    return patches


# ---------------------------------------------------------------------------
# Load subject volumes from pairs CSV
# ---------------------------------------------------------------------------

def load_subject_from_csv(pairs_csv, subject_id):
    """Return (input_vol, target_vol) numpy float32 arrays for one subject."""
    with open(pairs_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('subject', '') == subject_id:
                inp_path = row.get('input_3t', row.get('input'))
                tgt_path = row.get('target_7t', row.get('target'))
                inp = nib.load(inp_path).get_fdata().astype(np.float32)
                tgt = nib.load(tgt_path).get_fdata().astype(np.float32)
                return inp, tgt
    raise ValueError(f"Subject '{subject_id}' not found in {pairs_csv}")


# ---------------------------------------------------------------------------
# Evaluate one checkpoint on all fixed patches
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_checkpoint(model, diffusion, patch_pairs, device):
    """
    Args:
        patch_pairs: list of (input_patch, target_patch, label)

    Returns:
        dict with ssim_mean, psnr_mean, per-patch breakdown
    """
    model.eval()
    ssims, psnrs = [], []

    for inp_patch, tgt_patch, label in patch_pairs:
        inp_t = torch.from_numpy(inp_patch).float().unsqueeze(0).unsqueeze(0).to(device)
        shape = inp_t.shape

        pred_t, _ = diffusion.p_sample_loop(
            conditioning=inp_t, shape=shape, return_all=True,
        )

        pred_np = pred_t.cpu().numpy().squeeze()
        s, p = compute_metrics(pred_np, tgt_patch)
        ssims.append(s)
        psnrs.append(p)

    return {
        'ssim_mean': float(np.mean(ssims)),
        'psnr_mean': float(np.mean(psnrs)),
        'ssim_std':  float(np.std(ssims)),
        'psnr_std':  float(np.std(psnrs)),
        'ssim_per_patch': ssims,
        'psnr_per_patch': psnrs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Find best checkpoint (deterministic)')
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/train_diffusion.yaml')
    parser.add_argument('--output_dir', type=str, default='results/checkpoint_search')
    parser.add_argument('--subjects', nargs='+', default=['sub-01'],
                        help='Subject IDs to evaluate on (default: sub-01)')
    parser.add_argument('--pairs_csv', type=str, default=None,
                        help='Override pairs CSV (defaults to config)')
    parser.add_argument('--every_n', type=int, default=1,
                        help='Evaluate every N-th checkpoint (1=all, 5=every 5th)')
    args = parser.parse_args()

    # ---- config ----
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    csv_path = args.pairs_csv or config['dataset']['pairs_csv']
    patch_size = config['dataset']['patch_size'][0]

    # ---- extract fixed patches ONCE ----
    print(f"\nExtracting fixed {patch_size}³ patches ...")
    patch_pairs = []   # list of (input_patch, target_patch, label_str)

    for subj in args.subjects:
        print(f"  Loading {subj} ...")
        inp_vol, tgt_vol = load_subject_from_csv(csv_path, subj)
        print(f"    Volume shape: {inp_vol.shape}")

        inp_patches = extract_fixed_patches(inp_vol, patch_size)
        tgt_patches = extract_fixed_patches(tgt_vol, patch_size)

        for (ip, lbl), (tp, _) in zip(inp_patches, tgt_patches):
            patch_pairs.append((ip, tp, f"{subj}_{lbl}"))

    print(f"  Total fixed patches: {len(patch_pairs)}")

    # ---- model / diffusion ----
    model = AnatomyGuidedUNet(
        in_channels=config['model']['in_channels'],
        cond_channels=config['model'].get('cond_channels', 1),
        out_channels=config['model']['out_channels'],
        num_classes=config['model']['num_classes'],
        features=config['model']['features'],
        use_attention=config['model']['use_attention'],
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        timesteps=config['diffusion']['timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
    ).to(device)

    # ---- find checkpoints ----
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_files = sorted(ckpt_dir.glob('checkpoint_*.pt'))

    # Filter to numbered checkpoints & parse iteration
    ckpt_list = []
    for p in ckpt_files:
        parts = p.stem.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            ckpt_list.append((int(parts[1]), p))
    ckpt_list.sort(key=lambda x: x[0])

    # Subsample if requested
    if args.every_n > 1:
        ckpt_list = ckpt_list[::args.every_n]

    print(f"\nEvaluating {len(ckpt_list)} checkpoints on {len(patch_pairs)} fixed patches ...\n")

    # ---- output ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- evaluate ----
    results = []
    for iteration, ckpt_path in tqdm(ckpt_list, desc="Checkpoints"):
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'ema' in ckpt:
            model.load_state_dict(ckpt['ema'])
        elif 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)

        metrics = evaluate_checkpoint(model, diffusion, patch_pairs, device)
        metrics['iteration'] = iteration
        metrics['checkpoint'] = ckpt_path.name
        results.append(metrics)

        tqdm.write(
            f"  Iter {iteration:>6d}:  "
            f"SSIM = {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.3f}   "
            f"PSNR = {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.1f} dB"
        )

    if not results:
        print("No checkpoints evaluated.")
        return

    # ---- results dataframe ----
    df = pd.DataFrame([{
        'iteration': r['iteration'],
        'checkpoint': r['checkpoint'],
        'ssim_mean': r['ssim_mean'],
        'ssim_std': r['ssim_std'],
        'psnr_mean': r['psnr_mean'],
        'psnr_std': r['psnr_std'],
    } for r in results]).sort_values('iteration')

    csv_out = out_dir / 'checkpoint_results.csv'
    df.to_csv(csv_out, index=False)

    # ---- learning curve plot ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(df['iteration'], df['ssim_mean'], 'b-', linewidth=1)
    ax1.fill_between(df['iteration'],
                     df['ssim_mean'] - df['ssim_std'],
                     df['ssim_mean'] + df['ssim_std'],
                     alpha=0.2, color='b')
    best_ssim_row = df.loc[df['ssim_mean'].idxmax()]
    ax1.axvline(best_ssim_row['iteration'], color='r', linestyle='--', alpha=0.7)
    ax1.annotate(f"Best: {best_ssim_row['ssim_mean']:.4f}\n@ {int(best_ssim_row['iteration'])}k",
                 xy=(best_ssim_row['iteration'], best_ssim_row['ssim_mean']),
                 fontsize=10, color='r',
                 xytext=(10, -20), textcoords='offset points')
    ax1.set_ylabel('SSIM', fontsize=12)
    ax1.set_title('Checkpoint Evaluation (Fixed Patches – Deterministic)', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(df['iteration'], df['psnr_mean'], 'g-', linewidth=1)
    ax2.fill_between(df['iteration'],
                     df['psnr_mean'] - df['psnr_std'],
                     df['psnr_mean'] + df['psnr_std'],
                     alpha=0.2, color='g')
    best_psnr_row = df.loc[df['psnr_mean'].idxmax()]
    ax2.axvline(best_psnr_row['iteration'], color='r', linestyle='--', alpha=0.7)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'learning_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- report ----
    sep = '=' * 70
    print(f"\n{sep}")
    print("RESULTS  (deterministic fixed-patch evaluation)")
    print(sep)
    print(f"CSV saved to:      {csv_out}")
    print(f"Learning curve:    {out_dir / 'learning_curve.png'}")

    best = df.loc[df['ssim_mean'].idxmax()]
    print(f"\n★  BEST SSIM:  {best['ssim_mean']:.4f}  at iteration {int(best['iteration'])}")
    print(f"   PSNR:       {best['psnr_mean']:.2f} dB")

    best_p = df.loc[df['psnr_mean'].idxmax()]
    print(f"\n★  BEST PSNR:  {best_p['psnr_mean']:.2f} dB  at iteration {int(best_p['iteration'])}")
    print(f"   SSIM:       {best_p['ssim_mean']:.4f}")

    print(f"\n{sep}")
    print("TOP 10 BY SSIM:")
    print(sep)
    for _, row in df.nlargest(10, 'ssim_mean').iterrows():
        print(f"  Iter {int(row['iteration']):>6d}:  "
              f"SSIM = {row['ssim_mean']:.4f} ± {row['ssim_std']:.3f}   "
              f"PSNR = {row['psnr_mean']:.2f} dB")
    print(sep)


if __name__ == '__main__':
    main()
