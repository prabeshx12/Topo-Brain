"""
Settle the parameter-count claim (reviewer-checkable, manuscript says 22.4 M,
internal TRAINING_REPORT.md says "14-15 M").

Instantiates AnatomyGuidedUNet with the EXACT config used for training
(configs/train_diffusion.yaml) and reports total + per-top-level-module
parameter counts. No GPU needed.

Usage:
    python scripts/count_params.py
"""
import sys
from pathlib import Path

import importlib.util

import torch  # noqa: F401  (import validates the env)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load src/model.py directly: src/__init__.py imports monai (not needed here),
# so we bypass the package and load the self-contained model module by path.
_spec = importlib.util.spec_from_file_location("tb_model", ROOT / "src" / "model.py")
_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_model)
AnatomyGuidedUNet = _model.AnatomyGuidedUNet

# Args mirror configs/train_diffusion.yaml -> model:
CFG = dict(
    in_channels=1,
    cond_channels=1,
    out_channels=1,
    num_classes=4,
    features=(32, 64, 128, 256),
    use_attention=True,
)


def count(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def main():
    model = AnatomyGuidedUNet(**CFG)
    total, trainable = count(model)

    print("AnatomyGuidedUNet parameter count")
    print("=" * 50)
    print(f"  config: {CFG}")
    print("-" * 50)
    for name, child in model.named_children():
        t, _ = count(child)
        print(f"  {name:<16} {t/1e6:>8.3f} M")
    print("-" * 50)
    print(f"  {'TOTAL':<16} {total/1e6:>8.3f} M  ({total:,} params)")
    print(f"  {'trainable':<16} {trainable/1e6:>8.3f} M")
    print("=" * 50)
    print(f"\nManuscript claims 22.4 M. Computed: {total/1e6:.2f} M.")
    if abs(total/1e6 - 22.4) > 0.5:
        print(">>> MISMATCH: update the manuscript to the computed value.")
    else:
        print(">>> Consistent with the 22.4 M claim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
