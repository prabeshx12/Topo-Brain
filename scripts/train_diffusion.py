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
from src.utils import setup_logging, TensorBoardLogger

# ... (inside main) ...

    # Setup Logging
    log_dir = Path("logs") / time.strftime("%Y%m%d-%H%M%S")
    
    class LoggingConfig:
        def __init__(self, log_dir):
            self.log_dir = log_dir
            self.log_level = "INFO"
            
    setup_logging(LoggingConfig(log_dir))
    logger = logging.getLogger(__name__)
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="topobrain", help="W&B Project Name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B Entity (Team/User)")
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

    # W&B Setup
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project, 
                entity=args.wandb_entity,
                config=config, 
                name=f"run_{time.strftime('%Y%m%d-%H%M%S')}"
            )
            logger.info(f"Weights & Biases initialized: {args.wandb_entity}/{args.wandb_project}")
        except ImportError:
            logger.warning("wandb not installed. Skipping W&B initialization.")
            args.use_wandb = False

    # TensorBoard
    tb_logger = TensorBoardLogger(log_dir)

    # ... (skipping to loop) ...

        loss_dict = diffusion(x_start, cond, seg_target, lambda_pixel=lambda_pixel, lambda_percep=lambda_percep, lambda_topo=lambda_topo)
        
        loss_dict["loss"].backward()
        optimizer.step()
        ema.step_ema(ema_model, model)
        
        # Log to TensorBoard
        tb_logger.log_scalar("Loss/Total", loss_dict["loss"].item(), step)
        tb_logger.log_scalar("Loss/Diff", loss_dict["loss_diff"].item(), step)
        tb_logger.log_scalar("Loss/Topo", loss_dict["loss_topo"].item(), step)
        
        # Log to W&B
        if args.use_wandb:
            wandb.log({
                "loss_total": loss_dict["loss"].item(),
                "loss_diff": loss_dict["loss_diff"].item(),
                "loss_topo": loss_dict["loss_topo"].item(),
                "step": step
            })
        
        if step % 10 == 0:
            # Concise log to console
            tqdm.write(f"Step {step}: L={loss_dict['loss'].item():.4f} D={loss_dict['loss_diff'].item():.4f} S={loss_dict['loss_topo'].item():.4f}")

        # Saving
        if step > 0 and step % config["training"]["save_freq"] == 0:
            save_path = log_dir / f"checkpoint_{step}.pt"
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'ema': ema_model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, save_path)
            logger.info(f"Saved checkpoint to {save_path}")

    logger.info("Training Complete.")

if __name__ == "__main__":
    main()
