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
from torch.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import time
import sys

# Import Topo-Brain modules
sys.path.append(os.getcwd())

from src.model import AnatomyGuidedUNet
from src.diffusion import GaussianDiffusion
from src.synthesis_dataset import create_synthesis_dataloaders, load_pairs_manifest, PatchConfig, SplitConfig
from src.utils import setup_logging, TensorBoardLogger

# ... imports ...

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
    parser.add_argument("--masks-root", type=str, default=None, help="Root directory for masks (prepended to seg paths)")
    parser.add_argument("--output", type=str, default=None, help="Output directory for checkpoints and logs")
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
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = output_dir / "logs" / time.strftime("%Y%m%d-%H%M%S")
    else:
        log_dir = Path("logs") / time.strftime("%Y%m%d-%H%M%S")
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    class LoggingConfig:
        def __init__(self, log_dir):
            self.log_dir = log_dir
            self.log_level = "INFO"
            
    setup_logging(LoggingConfig(log_dir))
    logger = logging.getLogger(__name__)
    logger.info(f"Output directory: {log_dir.parent.parent if args.output else log_dir.parent}")

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
                        if 'input_3t' in p: p['input_3t'] = str(root / p['input_3t'])
                        if 'target_7t' in p: p['target_7t'] = str(root / p['target_7t'])
                        # Fallback for old masks/t2 if no mask root
                        if not args.masks_root:
                             if 'mask' in p and p['mask']: p['mask'] = str(root / p['mask'])
                             if 'seg' in p and p['seg']: p['seg'] = str(root / p['seg'])
                
                # Prepend masks root if provided (overrides data root for masks)
                if args.masks_root:
                    mask_root = Path(args.masks_root)
                    logger.info(f"Prepending masks root: {mask_root}")
                    for p in pairs:
                        if 'seg' in p and p['seg']: p['seg'] = str(mask_root / p['seg'])
                        elif 'mask' in p and p['mask']: p['mask'] = str(mask_root / p['mask'])
                
                # Validate seg mask paths at startup
                logger.info("=" * 60)
                logger.info("Seg Mask Path Validation:")
                _mf, _mm = 0, 0
                for p in pairs:
                    _subj = p.get("subject", "?")
                    _seg = p.get("seg", "")
                    if _seg:
                        _sp = Path(_seg)
                        if _sp.exists():
                            logger.info("  FOUND  %s: %s", _subj, _sp)
                            _mf += 1
                        else:
                            _alt = None
                            if _sp.suffix == ".gz":
                                _alt = _sp.with_suffix("").with_suffix(".nii")
                            elif _sp.suffix == ".nii":
                                _alt = _sp.with_suffix(".nii.gz")
                            if _alt and _alt.exists():
                                logger.info("  FOUND  %s: %s (alt ext)", _subj, _alt)
                                _mf += 1
                            else:
                                logger.warning("  MISS   %s: %s", _subj, _sp)
                                _mm += 1
                    else:
                        logger.warning("  MISS   %s: no seg in CSV", _subj)
                        _mm += 1
                logger.info("Masks: %d found, %d missing (fallback=binary)", _mf, _mm)
                logger.info("=" * 60)

                dataset_cfg = config.get("dataset", {})
                patch_cfg = PatchConfig(
                    patch_size=tuple(dataset_cfg.get("patch_size", (64, 64, 64))),
                    patches_per_volume=int(dataset_cfg.get("patches_per_volume", 32)),
                    min_brain_fraction=float(dataset_cfg.get("min_brain_fraction", 0.1)),
                    use_t2=bool(dataset_cfg.get("use_t2", False)),
                    seed=int(dataset_cfg.get("seed", 42)),
                )
                split_cfg = SplitConfig(
                    n_folds=int(dataset_cfg.get("n_folds", 10)),
                    val_fold=int(dataset_cfg.get("val_fold", 0)),
                    test_fold=int(dataset_cfg.get("test_fold", 1)),
                    use_loocv=bool(dataset_cfg.get("use_loocv", True)),
                    seed=int(dataset_cfg.get("seed", 42)),
                )
                train_loader, _, _ = create_synthesis_dataloaders(
                    pairs,
                    config=patch_cfg,
                    split_config=split_cfg,
                    batch_size=int(dataset_cfg.get("batch_size", 4)),
                    num_workers=int(dataset_cfg.get("num_workers", 4)),
                    val_fold=int(dataset_cfg.get("val_fold", 0)),
                )
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
    
    # Dynamically set channels based on multi-modal config (Section 6.3)
    use_t2 = config["dataset"].get("use_t2", False)
    in_channels = 2 if use_t2 else 1
    logger.info(f"Input channels: {in_channels} (use_t2={use_t2})")

    model = AnatomyGuidedUNet(
        in_channels=in_channels,
        cond_channels=1,
        out_channels=1,
        num_classes=config["model"].get("num_classes", 3),
        features=tuple(config["model"].get("features", (32, 64, 128, 256))),
        use_attention=config["model"].get("use_attention", False)
    ).to(device)

    # Use beta_schedule from config (cosine recommended)
    beta_schedule = config["diffusion"].get("beta_schedule", "cosine")
    diffusion = GaussianDiffusion(
        model, 
        timesteps=config["diffusion"].get("timesteps", 1000),
        beta_schedule=beta_schedule
    ).to(device)
    logger.info(f"Using {beta_schedule} beta schedule")

    # EMA with configurable decay
    ema_decay = config["training"].get("ema_decay", 0.9999)
    ema = EMA(ema_decay)
    ema_model = AnatomyGuidedUNet(
        in_channels=in_channels, cond_channels=1, out_channels=1,
        num_classes=config["model"].get("num_classes", 3),
        features=tuple(config["model"].get("features", (32, 64, 128, 256))),
        use_attention=config["model"].get("use_attention", False)
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.requires_grad_(False)

    # AdamW optimizer with weight decay
    lr = float(config["training"]["lr"])
    weight_decay = float(config["training"].get("weight_decay", 1e-4))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-5)
    logger.info(f"Optimizer: AdamW, lr={lr}, weight_decay={weight_decay}")
    start_step = 0

    # Gradient clipping value
    grad_clip = config["training"].get("grad_clip", 1.0)

    # Mixed precision
    use_amp = bool(config["training"].get("use_amp", True))
    scaler = GradScaler(enabled=use_amp, init_scale=2**12)  # 4096 — safe for 3D medical volumes

    # Skip statistics for stability monitoring
    skipped_steps = 0
    attempted_steps = 0

    # Resume from checkpoint
    if args.resume:
        if os.path.exists(args.resume):
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Robust state dict loading for multi-task heads
            model_dict = model.state_dict()
            pretrained_dict = checkpoint['model']
            
            # Track if we need to skip optimizer due to head changes
            head_changed = False
            
            # Filter out the segmentation head if classes changed
            for k in ['seg_outc.weight', 'seg_outc.bias']:
                if k in pretrained_dict and pretrained_dict[k].shape != model_dict[k].shape:
                    logger.warning(f"Shape mismatch in {k}, re-initializing segmentation head.")
                    pretrained_dict.pop(k)
                    head_changed = True
            
            model.load_state_dict(pretrained_dict, strict=False)
            
            if 'ema' in checkpoint:
                ema_pretrained_dict = checkpoint['ema']
                for k in ['seg_outc.weight', 'seg_outc.bias']:
                    if k in ema_pretrained_dict and ema_pretrained_dict[k].shape != ema_model.state_dict()[k].shape:
                        ema_pretrained_dict.pop(k)
                ema_model.load_state_dict(ema_pretrained_dict, strict=False)
                
            # Only load optimizer if head hasn't changed
            if 'optimizer' in checkpoint:
                if head_changed:
                    logger.warning("Segmentation head changed (num_classes). Skipping optimizer state to avoid momentum shape mismatch. UNet weights are preserved, optimizer will re-initialize.")
                else:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer'])
                    except Exception as e:
                        logger.warning(f"Optimizer state could not be loaded: {e}. Starting with fresh optimizer state.")
            if 'scaler' in checkpoint and use_amp:
                try:
                    scaler.load_state_dict(checkpoint['scaler'])
                    logger.info("Loaded AMP GradScaler state from checkpoint.")
                except Exception as e:
                    logger.warning(f"GradScaler state could not be loaded: {e}. Starting with fresh scaler state.")
            
            if 'step' in checkpoint:
                start_step = checkpoint['step'] + 1
                
            logger.info(f"Resumed at step {start_step}")
        else:
            logger.error(f"Checkpoint not found at {args.resume}")
            raise FileNotFoundError(f"Checkpoint not found at {args.resume}")

    # Training Loop
    n_iters = 10 if args.dry_run else config["training"]["n_iters"]
    
    for step in tqdm(range(start_step, n_iters), initial=start_step, total=n_iters):
        attempted_steps += 1
        optimizer.zero_grad()
        
        # Batch Data
        if has_data:
            batch = next(train_iter)
            x_start = batch['target'].to(device)
            cond = batch['input'].to(device)
            # Ensure segmentation target is on the correct device
            # synthesis_dataset.py now guarantees 'seg' key exists (even if zeros)
            seg_target = batch['seg'].to(device)
            if len(seg_target.shape) == 5:
                seg_target = seg_target.squeeze(1)
        else:
            x_start = torch.randn(2, 1, 64, 64, 64).to(device)
            cond = torch.randn(2, 1, 64, 64, 64).to(device)
            seg_target = torch.randint(0, 3, (2, 64, 64, 64)).to(device)

        # Guard segmentation labels against out-of-range values from masks.
        # Topology CE expects class ids in [0, num_classes - 1].
        num_classes = int(config["model"].get("num_classes", 4))
        invalid_mask = (seg_target < 0) | (seg_target >= num_classes)
        if invalid_mask.any():
            invalid_count = int(invalid_mask.sum().item())
            max_label = int(seg_target.max().item())
            min_label = int(seg_target.min().item())
            logger.warning(
                f"Step {step}: seg_target has {invalid_count} invalid voxels "
                f"(label range [{min_label}, {max_label}], num_classes={num_classes}). "
                "Remapping invalid labels to background."
            )
            seg_target = seg_target.clone()
            seg_target[invalid_mask] = 0

        # Curriculum Stages based on config
        stages = config.get("stages", {})
        stage1_end = stages.get("stage1_end", 10000)
        stage2_end = stages.get("stage2_end", 50000)
        stage3_end = stages.get("stage3_end", 100000)
        
        # Get loss weights from config (with fallback defaults)
        loss_config = config.get("loss_weights", {})
        final_lambda_pixel = loss_config.get("lambda_pixel", 0.05)
        final_lambda_percep = loss_config.get("lambda_percep", 0.3)
        final_lambda_topo = loss_config.get("lambda_topo", 0.3)
        topo_warmup_steps = loss_config.get("topo_warmup_steps", 25000)
        percep_warmup_steps = loss_config.get("percep_warmup_steps", 20000) 
        
        # Progressive loss weighting (Blueprint-aligned curriculum)
        if step < stage1_end:
            # Stage 1: Diffusion + Pixel loss (blueprint requires pixel from start)
            lambda_pixel, lambda_percep, lambda_topo = final_lambda_pixel * 0.5, 0.0, 0.0
        elif step < stage2_end:
            # Stage 2: Full pixel loss
            lambda_pixel, lambda_percep, lambda_topo = final_lambda_pixel, 0.0, 0.0
        elif step < stage3_end:
            # Stage 3: Add perceptual detail loss with its own warm-up
            steps_into_stage3 = step - stage2_end
            if steps_into_stage3 < percep_warmup_steps:
                lambda_percep = final_lambda_percep * (steps_into_stage3 / percep_warmup_steps)
            else:
                lambda_percep = final_lambda_percep
                
            lambda_pixel, lambda_topo = final_lambda_pixel, 0.0
        else:
            # Stage 4: Full curriculum with gradual topology warm-up
            # Topology loss warm-up starts at stage3_end (config-driven)
            steps_into_stage4 = step - stage3_end
            
            if steps_into_stage4 < topo_warmup_steps:
                # Gradual ramp from 0.0 to final_lambda_topo
                lambda_topo = final_lambda_topo * (steps_into_stage4 / topo_warmup_steps)
            else:
                lambda_topo = final_lambda_topo
                
            lambda_pixel, lambda_percep = final_lambda_pixel, final_lambda_percep
        
        # Log loss weights on first iteration of Stage 4 for verification
        if step == stage3_end + 1:
            logger.info(f"Stage 4 started - Loss weights from config:")
            logger.info(f"  lambda_pixel: {final_lambda_pixel}")
            logger.info(f"  lambda_percep: {final_lambda_percep}")
            logger.info(
                f"  lambda_topo: {final_lambda_topo} "
                f"(warm-up over {topo_warmup_steps} steps starting at step {stage3_end})"
            )

        with autocast(device_type=device.type, enabled=use_amp):
            loss_dict = diffusion(
                x_start,
                cond,
                seg_target,
                lambda_pixel=lambda_pixel,
                lambda_percep=lambda_percep,
                lambda_topo=lambda_topo,
            )

        # Non-finite loss guard before backward
        non_finite_keys = [
            k for k, v in loss_dict.items()
            if isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
        ]
        if non_finite_keys:
            skipped_steps += 1
            logger.warning(
                f"Step {step}: non-finite losses in {non_finite_keys} "
                f"(diff={loss_dict['loss_diff'].item()}, "
                f"pixel={loss_dict['loss_pixel'].item()}, "
                f"percep={loss_dict['loss_vgg'].item()}, "
                f"topo={loss_dict['loss_topo'].item()}). Skipping step."
            )
            optimizer.zero_grad()
            scaler.update()
            continue
        
        scaler.scale(loss_dict["loss"]).backward()
        
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        
        # NaN Safety: Skip optimizer step if any gradient is NaN/Inf
        has_nan_grad = any(
            torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
            for p in model.parameters() if p.grad is not None
        )
        if has_nan_grad:
            skipped_steps += 1
            logger.warning(f"Step {step}: NaN/Inf gradient detected - skipping optimizer step")
            optimizer.zero_grad()
            scaler.update()
            continue
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Learning Rate Decay (Manual Scheduler)
        lr_decay_step = config["training"].get("lr_decay_step", 0)
        if lr_decay_step > 0 and step == lr_decay_step:
            lr_decay_factor = config["training"].get("lr_decay_factor", 0.5)
            new_lr = config["training"]["lr"] * lr_decay_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            logger.info(f"Step {step}: Learning rate decayed from {config['training']['lr']} to {new_lr}")

        scaler.step(optimizer)
        scaler.update()
        ema.step_ema(ema_model, model)
        
        # Log to TensorBoard (all loss components)
        tb_logger.log_scalar("Loss/Total", loss_dict["loss"].item(), step)
        tb_logger.log_scalar("Loss/Diff", loss_dict["loss_diff"].item(), step)
        tb_logger.log_scalar("Loss/Pixel", loss_dict["loss_pixel"].item(), step)
        tb_logger.log_scalar("Loss/Perceptual", loss_dict["loss_vgg"].item(), step)
        tb_logger.log_scalar("Loss/Topo", loss_dict["loss_topo"].item(), step)
        tb_logger.log_scalar("LossWeights/Pixel", lambda_pixel, step)
        tb_logger.log_scalar("LossWeights/Percep", lambda_percep, step)
        tb_logger.log_scalar("LossWeights/Topo", lambda_topo, step)
        
        # Log to W&B
        if args.use_wandb:
            wandb.log({
                "loss_total": loss_dict["loss"].item(),
                "loss_diff": loss_dict["loss_diff"].item(),
                "loss_pixel": loss_dict["loss_pixel"].item(),
                "loss_perceptual": loss_dict["loss_vgg"].item(),
                "loss_topo": loss_dict["loss_topo"].item(),
                "lambda_pixel": lambda_pixel,
                "lambda_percep": lambda_percep,
                "lambda_topo": lambda_topo,
                "skipped_steps": skipped_steps,
                "skip_ratio": (skipped_steps / attempted_steps) if attempted_steps > 0 else 0.0,
                "step": step
            })
        
        if step % 50 == 0:
            # Comprehensive loss logging to console
            skip_ratio = (skipped_steps / attempted_steps) if attempted_steps > 0 else 0.0
            tqdm.write(
                f"Step {step}: Total={loss_dict['loss'].item():.4f} | "
                f"Diff={loss_dict['loss_diff'].item():.4f} Pixel={loss_dict['loss_pixel'].item():.4f} "
                f"Percep={loss_dict['loss_vgg'].item():.4f} Topo={loss_dict['loss_topo'].item():.4f} | "
                f"lambda=({lambda_pixel:.2f},{lambda_percep:.2f},{lambda_topo:.2f}) | "
                f"skip={skipped_steps}/{attempted_steps} ({100.0 * skip_ratio:.2f}%)"
            )

        # Saving
        if step > 0 and step % config["training"]["save_freq"] == 0:
            output_root = Path(args.output) if args.output else Path(".")
            
            # Save numbered checkpoint in its own subdirectory
            step_dir = output_root / f"checkpoint_{step}"
            step_dir.mkdir(parents=True, exist_ok=True)
            
            ckpt_data = {
                'step': step,
                'model': model.state_dict(),
                'ema': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'config': config,
            }
            
            save_path = step_dir / f"checkpoint_{step}.pt"
            torch.save(ckpt_data, save_path)
            logger.info(f"Saved checkpoint to {save_path}")
            
            # Also save as latest (in output root for easy discovery)
            output_root.mkdir(parents=True, exist_ok=True)
            latest_path = output_root / "checkpoint_latest.pt"
            torch.save(ckpt_data, latest_path)
            logger.info(f"Saved latest checkpoint to {latest_path}")

    final_skip_ratio = (skipped_steps / attempted_steps) if attempted_steps > 0 else 0.0
    logger.info(
        f"Training Complete. skipped_steps={skipped_steps}, "
        f"attempted_steps={attempted_steps}, skip_ratio={100.0 * final_skip_ratio:.2f}%"
    )

if __name__ == "__main__":
    main()
