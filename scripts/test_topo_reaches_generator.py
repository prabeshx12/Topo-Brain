"""
THE decisive test. A topology loss that cannot change the synthesised image is decoration.

The old system failed exactly here, and nobody checked:
  * the old "topology loss" hung off a PARALLEL segmentation decoder, so
    d(L_topo)/d(image decoder) was EXACTLY 0.00e+00 -- grad was None for 42/42 params.
    The paper's central claim ("anatomy-aware synthesis") was structurally impossible.
  * it was not topological anyway: edge-weighted CE + Sobel-edge Dice. No Betti number,
    no connectivity, no persistence.

This test asserts BOTH halves of the fix, on the real network, end to end:
  1. src/topology_euler.py computes a REAL invariant (proved exact vs gudhi in
     test_topology_euler.py) and
  2. its gradient actually reaches the IMAGE decoder through the cascade.

and it re-asserts that the OLD wiring (detach_seg_input=True) still measures 0.00e+00, so the
paper can ABLATE this design choice rather than merely assert it.

CPU, seconds.  Run:  python scripts/test_topo_reaches_generator.py
"""
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# synthesise the `src` package so model_cascaded's relative imports resolve, WITHOUT executing
# src/__init__.py (which pulls in monai).
pkg = importlib.util.module_from_spec(importlib.util.spec_from_loader("src", loader=None))
pkg.__path__ = [str(ROOT / "src")]
sys.modules["src"] = pkg
for _name in ("model", "model_cascaded", "topology_euler"):
    _spec = importlib.util.spec_from_file_location(f"src.{_name}", ROOT / "src" / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[f"src.{_name}"] = _mod
    _spec.loader.exec_module(_mod)

te = sys.modules["src.topology_euler"]
mc = sys.modules["src.model_cascaded"]

torch.manual_seed(0)

B, S, C = 2, 32, 4
x3t = torch.randn(B, 1, S, S, S)
target = torch.randint(0, C, (B, S, S, S))

topo = te.EulerTopologyLoss(num_classes=C, include_background=False)


def grad_into_generator(detach: bool):
    """Backprop ONLY the topology loss; measure what reaches the image decoder."""
    torch.manual_seed(0)
    net = mc.CascadedSynthesisNet(num_classes=C, detach_seg_input=detach)
    net.zero_grad(set_to_none=True)

    out = net(x3t)
    loss = topo(out["seg"], target)["loss"]
    loss.backward()

    # the image decoder + the head that actually emits the synthesised 7T volume
    dec = [p for n, p in net.named_parameters()
           if n.startswith("generator.ups") or n.startswith("generator.outc")]
    total = sum(float(p.grad.abs().sum()) for p in dec if p.grad is not None)
    n_none = sum(1 for p in dec if p.grad is None)
    return float(loss.detach()), total, n_none, len(dec)


print("=" * 78)
print("DOES THE TOPOLOGY LOSS REACH THE IMAGE GENERATOR?")
print("=" * 78)
print("  Backpropagating ONLY L_topo (no L1, no SSIM, no CE) and measuring the gradient")
print("  that arrives at the image decoder -- the part that synthesises the 7T volume.\n")

print(f"  {'wiring':40} {'L_topo':>8} {'sum|grad| into image decoder':>30}")
print("  " + "-" * 76)

l_old, g_old, none_old, n_old = grad_into_generator(detach=True)
print(f"  {'OLD (parallel / detached seg input)':40} {l_old:>8.4f} {g_old:>30.4e}")
print(f"  {'':40} {'':>8} {f'grad is None for {none_old}/{n_old} params':>30}")

l_new, g_new, none_new, n_new = grad_into_generator(detach=False)
print(f"  {'NEW (cascade: seg = SegHead(synth))':40} {l_new:>8.4f} {g_new:>30.4e}")
print(f"  {'':40} {'':>8} {f'grad is None for {none_new}/{n_new} params':>30}")

assert g_old == 0.0 and none_old == n_old, \
    "the OLD wiring must send exactly zero gradient -- that is the bug being ablated"
assert none_new == 0, "every image-decoder param must receive a gradient in the cascade"
assert g_new > 0, "the topology loss MUST be able to change the synthesised image"
assert torch.isfinite(torch.tensor(g_new)), "gradient must be finite"

print(f"\n  [OK] OLD: {g_old:.4e}  ->  NEW: {g_new:.4e}")
print("       The topology loss can now actually change the synthesised image.")
print("       Because seg = SegHead(synth_image), ANY loss on the segmentation must")
print("       backpropagate THROUGH the generator. The network is forced to synthesise an")
print("       image that is segmentable with the correct anatomy -- which is what")
print("       'topology-aware synthesis' was always supposed to mean, and what the old")
print("       architecture made impossible.\n")

print("=" * 78)
print("ABLATION IS NOW HONEST")
print("=" * 78)
print("  The old no-topology ablation was CIRCULAR: setting lambda_topo = 0 left the seg head")
print("  at RANDOM INIT, so the ablation compared a trained head against an untrained one --")
print("  it could not have come out any other way.")
print("  Here `detach_seg_input` isolates the ONE design choice (cascade vs parallel) with the")
print("  seg head trained identically in both arms. That is a real ablation.")
print("=" * 78)
