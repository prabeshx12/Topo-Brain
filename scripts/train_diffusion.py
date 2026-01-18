"""
Training Script for Anatomy-Guided Diffusion Model.
Implements Blueprint Section 7 & Q1-Level Requirements.
"""
import argparse
import logging
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import time
import sys

# Import Topo-Brain modules
sys.path.append(os.getcwd())

from src.model import AnatomyGuidedUNet
from src.diffusion import GaussianDiffusion
from src.synthesis_dataset import create_synthesis_dataloaders, load_pairs_manifest
from src.utils import setup_logging

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA:
    """Exponential Moving Average for model weights."""
    def __init__(self, beta):
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_diffusion.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Run a single batch for verification")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--data-root", type=str, default=None, help="Root directory for data (prepended to CSV paths)")
    args = parser.parse_args()

    # Load Config
    if not os.path.exists(args.config):
        config = {
            "dataset": {"pairs_csv": "pairs.csv", "batch_size": 4},
            "model": {"features": [32, 64, 128, 256], "num_classes": 3},
            "diffusion": {"timesteps": 1000},
            "training": {"lr": 1e-4, "n_iters": 100000, "save_freq": 1000}
        }
    else:
        config = load_config(args.config)

    # Setup Logging
    log_dir = Path("logs") / time.strftime("%Y%m%d-%H%M%S")
    
    class LoggingConfig:
        def __init__(self, log_dir):
            self.log_dir = log_dir
            self.log_level = "INFO"
            
    setup_logging(LoggingConfig(log_dir))
    logger = logging.getLogger(__name__)

    # Data Loading
    pairs_path = Path(config["dataset"]["pairs_csv"])
    has_data = False
    
    if args.dry_run and not pairs_path.exists():
        logger.info("Dry run: Skipping missing dataset.")
    else:
        try:
            if pairs_path.exists():
                pairs = load_pairs_manifest(pairs_path)
                
                # Prepend data root if provided
                if args.data_root:
                    root = Path(args.data_root)
                    logger.info(f"Prepending data root: {root}")
                    for p in pairs:
                        # Assuming keys like 'input_3t', 'target_7t', 'mask'
                        if 'input_3t' in p: p['input_3t'] = str(root / p['input_3t'])
                        if 'target_7t' in p: p['target_7t'] = str(root / p['target_7t'])
                        if 'mask' in p and p['mask']: p['mask'] = str(root / p['mask']) # Mask might be None
                
                train_loader, _, _ = create_synthesis_dataloaders(pairs, batch_size=config["dataset"].get("batch_size", 4))
                train_iter = cycle(train_loader)
                has_data = True
            else:
                 logger.warning(f"Pairs file {pairs_path} missing.")
        except Exception as e:
            logger.warning(f"Failed to load dataset: {e}")
            if not args.dry_run:
                raise e

    # Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AnatomyGuidedUNet(
        in_channels=1,
        cond_channels=1,
        out_channels=1,
        num_classes=config["model"].get("num_classes", 3),
        features=tuple(config["model"].get("features", (32, 64, 128, 256)))
    ).to(device)

    diffusion = GaussianDiffusion(model, timesteps=config["diffusion"].get("timesteps", 1000)).to(device)

    ema = EMA(0.995)
    ema_model = AnatomyGuidedUNet(
        in_channels=1, cond_channels=1, out_channels=1,
        num_classes=config["model"].get("num_classes", 3),
        features=tuple(config["model"].get("features", (32, 64, 128, 256)))
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.requires_grad_(False)

    optimizer = optim.Adam(model.parameters(), lr=float(config["training"]["lr"]))
    start_step = 0

    # Training Loop
    n_iters = 10 if args.dry_run else config["training"]["n_iters"]
    
    for step in tqdm(range(start_step, n_iters)):
        optimizer.zero_grad()
        
        # Batch Data
        if has_data:
            batch = next(train_iter)
            x_start = batch['target'].to(device)
            cond = batch['input'].to(device)
            # Fallback for segmentation target
            seg_target = batch.get('seg', torch.zeros_like(x_start, dtype=torch.long)) # Dummy
            if len(seg_target.shape) == 5: seg_target = seg_target.squeeze(1) # [B, D, H, W]
        else:
            x_start = torch.randn(2, 1, 64, 64, 64).to(device)
            cond = torch.randn(2, 1, 64, 64, 64).to(device)
            seg_target = torch.randint(0, 3, (2, 64, 64, 64)).to(device)

        # Curriculum Stages
        epoch_approx = step // 1000 
        lambda_pixel = 1.0 
        
        if epoch_approx < 50:
            lambda_topo, lambda_percep = 0.0, 0.0
        elif epoch_approx < 150:
            lambda_topo, lambda_percep = 0.1, 0.1
        else:
            lambda_topo, lambda_percep = 0.1, 0.1
            if epoch_approx == 150 and step % 1000 == 0:
                for pg in optimizer.param_groups: pg['lr'] *= 0.1

        loss_dict = diffusion(x_start, cond, seg_target, lambda_pixel=lambda_pixel, lambda_percep=lambda_percep, lambda_topo=lambda_topo)
        
        loss_dict["loss"].backward()
        optimizer.step()
        ema.step_ema(ema_model, model)
        
        if step % 10 == 0:
            # Concise log
            tqdm.write(f"Step {step}: L={loss_dict['loss'].item():.4f} D={loss_dict['loss_diff'].item():.4f} S={loss_dict['loss_topo'].item():.4f}")

    logger.info("Training Complete.")

if __name__ == "__main__":
    main()
