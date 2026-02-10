"""
Sanity check for diffusion training configuration.
Validates ranges, schedule ordering, and reports derived milestones.
"""
import argparse
from pathlib import Path
import yaml


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def require(cond: bool, message: str, errors: list):
    if not cond:
        errors.append(message)


def warn(cond: bool, message: str, warnings: list):
    if not cond:
        warnings.append(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check diffusion training config")
    parser.add_argument("--config", type=str, default="configs/train_diffusion.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)

    dataset = cfg.get("dataset", {})
    training = cfg.get("training", {})
    diffusion = cfg.get("diffusion", {})
    stages = cfg.get("stages", {})
    loss = cfg.get("loss_weights", {})

    errors = []
    warnings = []

    # Dataset checks
    batch_size = int(dataset.get("batch_size", 0))
    patch_size = dataset.get("patch_size", [64, 64, 64])
    patches_per_volume = int(dataset.get("patches_per_volume", 0))
    min_brain_fraction = float(dataset.get("min_brain_fraction", 0.0))

    require(batch_size > 0, "dataset.batch_size must be > 0", errors)
    require(len(patch_size) == 3, "dataset.patch_size must be length 3", errors)
    require(all(int(p) > 0 for p in patch_size), "dataset.patch_size must be positive", errors)
    require(patches_per_volume > 0, "dataset.patches_per_volume must be > 0", errors)
    require(0.0 <= min_brain_fraction <= 1.0, "dataset.min_brain_fraction must be in [0,1]", errors)

    # Diffusion checks
    timesteps = int(diffusion.get("timesteps", 0))
    require(timesteps > 0, "diffusion.timesteps must be > 0", errors)

    # Training checks
    n_iters = int(training.get("n_iters", 0))
    lr = float(training.get("lr", 0.0))
    lr_decay_step = int(training.get("lr_decay_step", 0))
    lr_decay_factor = float(training.get("lr_decay_factor", 1.0))
    grad_clip = float(training.get("grad_clip", 0.0))

    require(n_iters > 0, "training.n_iters must be > 0", errors)
    require(lr > 0, "training.lr must be > 0", errors)
    require(0 < lr_decay_factor <= 1.0, "training.lr_decay_factor must be in (0,1]", errors)
    warn(lr_decay_step == 0 or lr_decay_step < n_iters, "training.lr_decay_step should be < n_iters", warnings)
    warn(grad_clip >= 0, "training.grad_clip should be >= 0", warnings)

    # Stage schedule checks
    stage1_end = int(stages.get("stage1_end", 0))
    stage2_end = int(stages.get("stage2_end", 0))
    stage3_end = int(stages.get("stage3_end", 0))
    require(stage1_end > 0, "stages.stage1_end must be > 0", errors)
    require(stage1_end < stage2_end, "stages.stage1_end must be < stage2_end", errors)
    require(stage2_end < stage3_end, "stages.stage2_end must be < stage3_end", errors)
    warn(stage3_end < n_iters, "stages.stage3_end should be < training.n_iters", warnings)

    # Loss warmups
    topo_warmup_steps = int(loss.get("topo_warmup_steps", 0))
    percep_warmup_steps = int(loss.get("percep_warmup_steps", 0))
    warn(topo_warmup_steps > 0, "loss_weights.topo_warmup_steps should be > 0", warnings)
    warn(percep_warmup_steps > 0, "loss_weights.percep_warmup_steps should be > 0", warnings)

    # Derived schedule report
    stage1_range = (0, stage1_end)
    stage2_range = (stage1_end, stage2_end)
    stage3_range = (stage2_end, stage3_end)
    stage4_range = (stage3_end, n_iters)

    percep_full_step = stage2_end + percep_warmup_steps
    topo_full_step = stage3_end + topo_warmup_steps

    warn(percep_full_step <= stage3_end, "Perceptual warmup finishes after stage3 ends; check percep_warmup_steps", warnings)
    warn(topo_full_step <= n_iters, "Topology warmup finishes after training ends; check topo_warmup_steps", warnings)

    # Print report
    print("Sanity Check Report")
    print("===================")
    print(f"Config: {config_path}")
    print("")
    print("Dataset")
    print(f"  batch_size: {batch_size}")
    print(f"  patch_size: {patch_size}")
    print(f"  patches_per_volume: {patches_per_volume}")
    print(f"  min_brain_fraction: {min_brain_fraction}")
    print("")
    print("Diffusion")
    print(f"  timesteps: {timesteps}")
    print("")
    print("Training")
    print(f"  n_iters: {n_iters}")
    print(f"  lr: {lr}")
    print(f"  lr_decay_step: {lr_decay_step}")
    print(f"  lr_decay_factor: {lr_decay_factor}")
    print(f"  grad_clip: {grad_clip}")
    print("")
    print("Stages")
    print(f"  stage1: {stage1_range[0]} -> {stage1_range[1]}")
    print(f"  stage2: {stage2_range[0]} -> {stage2_range[1]}")
    print(f"  stage3: {stage3_range[0]} -> {stage3_range[1]}")
    print(f"  stage4: {stage4_range[0]} -> {stage4_range[1]}")
    print("")
    print("Warmups")
    print(f"  perceptual warmup ends at step: {percep_full_step}")
    print(f"  topology warmup ends at step:   {topo_full_step}")
    print("")

    if errors:
        print("ERRORS")
        for e in errors:
            print(f"  - {e}")
        print("")

    if warnings:
        print("WARNINGS")
        for w in warnings:
            print(f"  - {w}")
        print("")

    if not errors and not warnings:
        print("All checks passed.")


if __name__ == "__main__":
    main()
