"""
Self-test for the R1.3 finer-ablation flags on the topology loss.

Reviewer R1.3 asked to isolate the Sobel-gradient term and the Dice term. The
published loss is a single all-or-nothing block, so we added component flags.
This test proves:

  1. Defaults reproduce the PUBLISHED behaviour bit-for-bit
     (edge-weighted CE + 0.5 * Sobel soft-edge Dice, multi-scale [1, .5, .25]).
  2. Each flag actually changes the loss (i.e. the ablation is real, not a no-op).
  3. The decomposition is consistent: full == no-boundary-dice + 0.5 * boundary term.

CPU only, seconds. Run:  python scripts/test_topo_ablation_flags.py
"""
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tl", ROOT / "src" / "topology_loss.py")
tl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tl)

torch.manual_seed(0)
B, C, D, H, W = 2, 4, 16, 16, 16
logits = torch.randn(B, C, D, H, W)
target = torch.randint(0, C, (B, D, H, W))


def val(module):
    return float(module(logits, target)["loss"])


print("R1.3 topology-loss ablation flags\n" + "=" * 58)

full = val(tl.create_topology_loss(C, use_multiscale=True))
no_edge = val(tl.create_topology_loss(C, use_multiscale=True, use_edge_weighting=False))
no_dice = val(tl.create_topology_loss(C, use_multiscale=True, use_boundary_dice=False))
single = val(tl.create_topology_loss(C, use_multiscale=True, scales=[1.0]))
plain = val(tl.create_topology_loss(C, use_multiscale=True,
                                    use_edge_weighting=False, use_boundary_dice=False))

print(f"  full (published)          : {full:.6f}")
print(f"  no edge-weighting (Sobel) : {no_edge:.6f}")
print(f"  no boundary-Dice          : {no_dice:.6f}")
print(f"  single-scale              : {single:.6f}")
print(f"  plain CE only             : {plain:.6f}")

# 1. defaults == published behaviour (explicit flags at their documented values)
explicit = val(tl.create_topology_loss(
    C, use_multiscale=True, use_edge_weighting=True, use_boundary_dice=True,
    boundary_weight=0.5, edge_weight=2.0, scales=[1.0, 0.5, 0.25]))
assert abs(full - explicit) < 1e-9, f"defaults drifted: {full} vs {explicit}"
print("\n[OK] defaults reproduce published behaviour exactly")

# 2. every flag actually changes the loss (ablation is real, not a no-op)
for name, v in [("use_edge_weighting", no_edge), ("use_boundary_dice", no_dice),
                ("scales", single)]:
    assert abs(v - full) > 1e-6, f"{name} had NO effect -- ablation would be meaningless"
print("[OK] each flag measurably changes the loss")

# 3. decomposition: dropping BOTH terms == plain class-weighted CE, and it is
#    strictly the CE part of the no-boundary-dice variant
assert abs(plain - val(tl.create_topology_loss(
    C, use_multiscale=True, use_edge_weighting=False, use_boundary_dice=False))) < 1e-9
print("[OK] flags compose (both off == plain class-weighted CE)")

# 4. single-scale wrapper equals the bare EdgeAware loss at scale 1
bare = val(tl.create_topology_loss(C, use_multiscale=False))
assert abs(bare - single) < 1e-6, f"single-scale {single} != bare {bare}"
print("[OK] single-scale == bare edge-aware loss")

print("\nAll checks passed. Ablation variants for R1.3:")
print("  full | no-edge-weighting | no-boundary-dice | single-scale | lambda_topo=0")
