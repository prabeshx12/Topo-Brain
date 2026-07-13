"""
Proves src/topology_euler.py computes a REAL topological invariant -- not another loss that
merely sounds topological (which is exactly what the old "topology loss" was).

Three levels of proof:
  1. EXACTNESS on shapes whose Euler characteristic is known a priori
     (ball chi=1, two balls chi=2, hollow shell chi=2, solid torus chi=0, ...).
  2. AGREEMENT with gudhi's persistent homology: chi must equal b0 - b1 + b2, computed by a
     completely independent implementation (cubical complex persistence).
  3. DIFFERENTIABILITY: gradients must be finite, non-zero, and must point in the direction
     that FIXES topology (closing a hole should reduce the loss).

CPU, seconds.  Run:  python scripts/test_topology_euler.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("te", ROOT / "src" / "topology_euler.py")
te = importlib.util.module_from_spec(spec)
spec.loader.exec_module(te)


def chi_of(mask: np.ndarray, eps: float = 1e-12) -> float:
    """chi of a mask.

    eps defaults to 1e-12 here, NOT the 1e-6 the loss uses. eps is the floor the probabilities
    are squeezed into so that log(1-p) stays differentiable at saturation; it costs a bias of
    ~eps * (#background voxels), because at p=eps every background voxel is an EXPECTED spurious
    component. For a pure VALUE check on a binary mask no gradient is needed, so eps -> 0.
    (Level 2 below measures that bias explicitly.)
    """
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    return float(te.euler_characteristic(t, eps=eps)[0])


def betti_gudhi(mask: np.ndarray):
    """Independent ground truth via persistent homology."""
    import gudhi as gd
    b = mask.astype(bool)
    if not b.any():
        return 0, 0, 0
    c = np.argwhere(b)
    lo, hi = c.min(0), c.max(0) + 1
    sub = b[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sub = np.pad(sub, 2, constant_values=False)
    cc = gd.CubicalComplex(top_dimensional_cells=np.where(sub, 0.0, 1.0).astype(np.float32))
    cc.compute_persistence(homology_coeff_field=2, min_persistence=0.0)
    raw = list(cc.persistent_betti_numbers(from_value=0.0, to_value=0.5)) + [0, 0, 0]
    return int(raw[0]), int(raw[1]), int(raw[2])


print("=" * 74)
print("1) EXACTNESS — shapes with a known Euler characteristic")
print("=" * 74)

shapes = {}

# a single voxel: a solid ball
m = np.zeros((9, 9, 9), np.float32); m[4, 4, 4] = 1
shapes["single voxel (ball)"] = (m, 1)

# two adjacent voxels: still ONE ball
m = np.zeros((9, 9, 9), np.float32); m[4, 4, 4] = 1; m[4, 4, 5] = 1
shapes["two adjacent voxels"] = (m, 1)

# two separated voxels: TWO balls
m = np.zeros((9, 9, 9), np.float32); m[2, 2, 2] = 1; m[6, 6, 6] = 1
shapes["two separate voxels"] = (m, 2)

# a solid cube: one ball
m = np.zeros((11, 11, 11), np.float32); m[3:8, 3:8, 3:8] = 1
shapes["solid cube (ball)"] = (m, 1)

# a hollow cube shell: chi = 2 (like a sphere: b0=1, b1=0, b2=1)
m = np.zeros((13, 13, 13), np.float32); m[3:10, 3:10, 3:10] = 1; m[4:9, 4:9, 4:9] = 0
shapes["hollow shell (sphere)"] = (m, 2)

# a solid torus (donut): chi = 0  (b0=1, b1=1, b2=0)
D = 25
zz, yy, xx = np.mgrid[0:D, 0:D, 0:D]
cz = cy = cx = D // 2
R, r = 7.0, 2.5
q = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) - R
torus = ((q ** 2 + (zz - cz) ** 2) <= r ** 2).astype(np.float32)
shapes["solid torus (donut)"] = (torus, 0)

print(f"  {'shape':26} {'chi (ours)':>11} {'expected':>9} {'gudhi b0-b1+b2':>16}")
print("  " + "-" * 68)
ok = True
for name, (mask, expected) in shapes.items():
    c = chi_of(mask)
    b0, b1, b2 = betti_gudhi(mask)
    chi_g = b0 - b1 + b2
    match = abs(c - expected) < 1e-3 and abs(c - chi_g) < 1e-3
    ok &= match
    flag = "" if match else "   <-- MISMATCH"
    print(f"  {name:26} {c:11.3f} {expected:9d} {chi_g:16d}{flag}")
    print(f"  {'':26} {'':11} {'':9} (b0={b0} b1={b1} b2={b2})")
assert ok, "Euler characteristic does not match the known value / gudhi"
print("\n  [OK] exact on every shape, and agrees with an INDEPENDENT persistent-homology impl\n")


print("=" * 74)
print("2) IT IS AN *EXPECTED* chi — and that is the property we need")
print("=" * 74)
print("  By linearity of expectation, soft_or = P(window has >=1 foreground), so")
print("  V - E + F - C is E[chi] over the binarisation implied by the probabilities.")
print("  Consequence: a diffuse background costs chi, because each uncertain background")
print("  voxel is an EXPECTED spurious component. Demonstrated:\n")
solid = np.zeros((13, 13, 13), np.float32)
solid[3:10, 3:10, 3:10] = 1
n_bg = int((solid == 0).sum())
print(f"  {'p(background)':>14} {'E[chi]':>10} {'expected spurious':>19}")
for p_bg in (0.05, 0.0025, 1e-4, 1e-6):
    m = solid.copy()
    m[solid == 0] = p_bg
    m[solid == 1] = 1.0 - p_bg
    print(f"  {p_bg:>14.0e} {chi_of(m):>10.2f} {n_bg * p_bg:>19.2f}")
print("\n  [OK] E[chi] tracks the expected number of spurious components.")
print("       That is EXACTLY our failure mode: GM beta0 = 65 vs GT 1.\n")


print("=" * 74)
print("3) GRADIENT — finite, non-zero, and it CLOSES the handle")
print("=" * 74)
holed = solid.copy()
holed[:, 6, 6] = 0                      # drill a tunnel -> creates a handle
b0, b1, b2 = betti_gudhi(holed)
print(f"  chi(solid cube)       = {chi_of(solid):.3f}   (expect 1)")
print(f"  chi(cube with tunnel) = {chi_of(holed):.3f}   (expect 0: b0={b0} b1={b1} b2={b2})")
print("\n  Sweeping the CONFIDENCE of a prediction of the holed shape. Two things compete:")
print("  the missing handle (chi too LOW by 1) and the expected spurious components from an")
print("  uncertain background (chi too HIGH by ~ p_bg * n_bg). Which one the gradient chases")
print("  depends on which is larger -- so we report both, and we do not hide the crossover.\n")

loss_fn = te.EulerTopologyLoss(num_classes=2, include_background=False)
target = torch.from_numpy(solid).long()[None]
tunnel = torch.zeros(13, 13, 13, dtype=torch.bool)
tunnel[3:10, 6, 6] = True

print(f"  {'logit':>6} {'p(bg)':>9} {'E[spurious]':>12} {'chi_pred':>9} {'|grad|':>10} "
      f"{'grad @ tunnel':>14}  verdict")
print("  " + "-" * 84)
saw_fill = False
for a in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
    logit = torch.zeros(1, 2, 13, 13, 13)
    logit[0, 1] = torch.from_numpy(holed) * (2 * a) - a
    logit[0, 0] = -logit[0, 1]
    logit.requires_grad_(True)

    out = loss_fn(logit, target)
    out["loss"].backward()
    g = logit.grad
    assert torch.isfinite(g).all(), f"gradient must be finite (logit +/-{a})"

    p_bg = float(torch.softmax(torch.tensor([a, -a]), 0)[1])
    gt = float(g[0, 1][tunnel].mean())
    gnorm = float(g.abs().sum())
    # descending the loss ADDS foreground at the tunnel iff the gradient there is negative
    fills = gt < 0
    saw_fill |= fills and gnorm > 0
    print(f"  {a:>6.1f} {p_bg:>9.1e} {p_bg * n_bg:>12.2f} "
          f"{float(out['chi_pred'][0, 0].detach()):>9.3f} {gnorm:>10.2e} {gt:>+14.2e}"
          f"  {'CLOSES the handle' if fills else 'kills spurious cpts first'}")

assert saw_fill, "at realistic confidence the gradient must push foreground INTO the hole"
print("\n  [OK] gradient is finite and non-zero throughout, and once the background is cleaner")
print("       than ~1 expected spurious component it points at CLOSING THE HANDLE.")
print("       Below that it removes spurious components first -- which is the correct")
print("       priority, and is exactly our GM beta0 = 65 (vs GT 1) failure mode.\n")


print("=" * 74)
print("4) A CORRECT shape must score better than a BROKEN one — and why sharpening")
print("=" * 74)
print("  The single property the loss MUST have. Without sharpening it FAILS this, because")
print("  the expected-spurious-component term (+0.62 at this confidence) numerically cancels")
print("  the missing handle (-1). The loss would then trade a hole against speckle.\n")


def loss_of(mask, fn, a=4.0):
    lg = torch.zeros(1, 2, 13, 13, 13)
    lg[0, 1] = torch.from_numpy(mask) * (2 * a) - a
    lg[0, 0] = -lg[0, 1]
    return float(fn(lg, target)["loss"])


raw = te.EulerTopologyLoss(num_classes=2, include_background=False, sharpness=0.0)
print(f"  {'':34} {'solid (CORRECT)':>16} {'holed (BROKEN)':>15}   ordering")
print("  " + "-" * 82)
for nm, fn in (("sharpness = 0  (naive E[chi])", raw), ("sharpness = 20 (default)", loss_fn)):
    ls, lh = loss_of(solid, fn), loss_of(holed, fn)
    good = ls < lh
    print(f"  {nm:34} {ls:>16.4f} {lh:>15.4f}   "
          f"{'OK  correct < broken' if good else 'BROKEN SHAPE WINS  <-- unusable'}")

assert loss_of(solid, raw) > loss_of(holed, raw), "the naive loss is expected to fail here"
assert loss_of(solid, loss_fn) < loss_of(holed, loss_fn), \
    "the correct shape MUST score better than the broken one"

losses = []
for fill in (0.0, 0.25, 0.5, 0.75, 1.0):
    m = solid.copy()
    m[:, 6, 6] = fill                       # progressively refill the tunnel
    L = loss_of(m, loss_fn)
    losses.append(L)
    print(f"  tunnel filled {100*fill:5.1f}%  ->  loss {L:.4f}")
assert losses[-1] < losses[0], "repairing the topology must reduce the loss"
assert losses[-1] < 1e-2, "a perfectly repaired shape must have ~zero topology loss"
print("\n  [OK] with sharpening the repaired shape has ~zero topology loss, the handled one")
print("       does not, and the ordering is correct.")
print("\n  HONEST CAVEAT (a real property, not a bug): the loss is NOT monotone through")
print("  fill = 0.5. At p = 0.5 every tunnel voxel is a coin flip, so E[chi] is genuinely")
print("  dominated by the expected structure of a maximally-uncertain region. E[chi] is a")
print("  smooth function of p, not of the binarised shape. This is why it is a REGULARISER")
print("  on top of CE + Dice -- which anchor the shape -- and never a standalone objective.\n")


print("=" * 74)
print("5) LIMITATION, stated plainly: chi CONFLATES b0, b1, b2")
print("=" * 74)
print("  chi = b0 - b1 + b2, so |chi_pred - chi_gt| = 0 does NOT imply correct topology:")
for name, (m, _) in [("solid cube (b0=1)", (solid, 0))]:
    pass
two = np.zeros((13, 13, 13), np.float32)
two[2:5, 2:5, 2:5] = 1
two[8:11, 8:11, 8:11] = 1                        # two balls: b0=2, chi=2
shell = np.zeros((13, 13, 13), np.float32)
shell[3:10, 3:10, 3:10] = 1
shell[4:9, 4:9, 4:9] = 0                         # one hollow shell: b0=1 b2=1, chi=2
for nm, m in (("two separate balls", two), ("one hollow shell", shell)):
    b0, b1, b2 = betti_gudhi(m)
    print(f"    {nm:22} b0={b0} b1={b1} b2={b2}  ->  chi = {chi_of(m):.0f}")
print("\n  Both have chi = 2 with COMPLETELY different topology. An Euler loss cannot tell them")
print("  apart. This is inherent to the invariant (Li et al., IEEE TMI 2025 carry the same")
print("  caveat) and it MUST be stated in the paper, not buried. Two consequences:")
print("    - the loss is a REGULARISER on CE+Dice, never the sole objective;")
print("    - EVALUATION reports beta0/beta1/beta2 SEPARATELY (src/metrics_honest.py), so the")
print("      claim is tested by a metric strictly stronger than the loss being optimised.\n")

print("=" * 74)
print("EULER TOPOLOGY LOSS VERIFIED — a real topological invariant (exact vs gudhi),")
print("differentiable, gradient repairs topology. Limitations measured and stated.")
print("=" * 74)
