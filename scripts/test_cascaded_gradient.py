"""
THE decisive test for Phase 2.

The old model ran two PARALLEL decoders off the shared bottleneck, so a loss on the
segmentation could not reach the image decoder. Measured on the real model
(revision/AUDIT.md, bug 6), after loss_topo.backward():

    IMAGE decoder .ups   sum|grad| = 0.00e+00   (grad is None for 42/42 params)
    IMAGE head    .outc  sum|grad| = 0.00e+00   (grad is None for  2/2 params)

i.e. "topology-preserving synthesis" was structurally impossible -- the topology loss
could never shape the synthesised image.

This test asserts that in the CASCADED model (seg computed FROM the synth image) a
segmentation/topology loss produces a NON-ZERO gradient throughout the generator,
including the final image head.

It also verifies the ablation switch (`detach_seg_input=True`) reproduces the OLD broken
behaviour exactly -- so the paper can show, empirically, that the cascade is what matters.

CPU, seconds.  Run:  python scripts/test_cascaded_gradient.py
"""
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# load src.model (needed by model_cascaded) without importing src/__init__ (which pulls monai)
pkg = importlib.util.module_from_spec(importlib.util.spec_from_loader("src", loader=None))
pkg.__path__ = [str(ROOT / "src")]
sys.modules["src"] = pkg
for name in ("model", "model_cascaded"):
    spec = importlib.util.spec_from_file_location(f"src.{name}", ROOT / "src" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"src.{name}"] = mod
    spec.loader.exec_module(mod)

from src.model import AnatomyGuidedUNet          # noqa: E402
from src.model_cascaded import CascadedSynthesisNet  # noqa: E402

torch.manual_seed(0)
B, D = 1, 32
x3t = torch.randn(B, 1, D, D, D)
x7t = torch.randn(B, 1, D, D, D)
seg_gt = torch.randint(0, 4, (B, D, D, D))


def grad_report(module, tag):
    tot, n_none, n_tot = 0.0, 0, 0
    for p in module.parameters():
        n_tot += 1
        if p.grad is None:
            n_none += 1
        else:
            tot += float(p.grad.abs().sum())
    return tot, n_none, n_tot


print("=" * 74)
print("OLD (parallel decoders): can a segmentation loss reach the image decoder?")
print("=" * 74)
old = AnatomyGuidedUNet(in_channels=1, cond_channels=1, out_channels=1,
                        num_classes=4, features=(32, 64, 128, 256), use_attention=True)
old.zero_grad(set_to_none=True)
t = torch.zeros(B, dtype=torch.long)
out = old(torch.zeros_like(x3t), t, x3t)
loss_seg = F.cross_entropy(out["segmentation"].float(), seg_gt)   # a SEG-only loss
loss_seg.backward()

for tag, mod in [("encoder  (shared)", old.downs), ("bottleneck (shared)", old.mid_block1),
                 ("IMAGE decoder .ups", old.ups), ("IMAGE head    .outc", old.outc),
                 ("SEG decoder .seg_ups", old.seg_ups), ("SEG head  .seg_outc", old.seg_outc)]:
    g, nn_, nt = grad_report(mod, tag)
    flag = "  <-- ZERO" if g == 0.0 else ""
    print(f"  {tag:22} sum|grad| = {g:10.4e}   (grad is None for {nn_}/{nt} params){flag}")

g_img_old, none_img_old, _ = grad_report(old.ups, "")
g_head_old, _, _ = grad_report(old.outc, "")
assert g_img_old == 0.0 and g_head_old == 0.0, \
    "expected the old parallel design to give ZERO image-decoder gradient"
print("\n  [CONFIRMED] the topology loss could NEVER shape the synthesised image.\n")


print("=" * 74)
print("NEW (cascaded): seg is computed FROM the synth image")
print("=" * 74)
net = CascadedSynthesisNet(num_classes=4, features=(32, 64, 128, 256), use_attention=True)
print(f"  params: {net.n_params()}")
net.zero_grad(set_to_none=True)
out = net(x3t)
loss_seg = F.cross_entropy(out["seg"].float(), seg_gt)            # a SEG-only loss again
loss_seg.backward()

parts = [("generator encoder", net.generator.downs),
         ("generator bottleneck", net.generator.mid_block1),
         ("generator IMAGE decoder", net.generator.ups),
         ("generator IMAGE head", net.generator.outc),
         ("seg head", net.seg_head)]
for tag, mod in parts:
    g, nn_, nt = grad_report(mod, tag)
    print(f"  {tag:26} sum|grad| = {g:10.4e}   (grad is None for {nn_}/{nt})")

g_img_new, none_img_new, ntot_img = grad_report(net.generator.ups, "")
g_head_new, none_head_new, _ = grad_report(net.generator.outc, "")
assert g_img_new > 0.0, "image decoder MUST now receive segmentation/topology gradient"
assert g_head_new > 0.0, "image head MUST now receive segmentation/topology gradient"
assert none_img_new == 0, "every image-decoder param must get a gradient"
print("\n  [OK] the segmentation loss now flows THROUGH the image decoder.")
print("       The generator is finally forced to synthesise an image that is")
print("       segmentable with the correct anatomy -- which is what 'topology-aware")
print("       synthesis' was always supposed to mean.\n")


print("=" * 74)
print("ABLATION SWITCH: detach_seg_input=True must reproduce the OLD broken behaviour")
print("=" * 74)
det = CascadedSynthesisNet(num_classes=4, detach_seg_input=True)
det.zero_grad(set_to_none=True)
out = det(x3t)
F.cross_entropy(out["seg"].float(), seg_gt).backward()
g_img_det, none_det, ntot_det = grad_report(det.generator.ups, "")
print(f"  detached: generator IMAGE decoder sum|grad| = {g_img_det:.4e} "
      f"(grad is None for {none_det}/{ntot_det})")
assert g_img_det == 0.0, "the detached variant must sever the gradient (it is the ablation)"
print("  [OK] the paper can now ablate EXACTLY this design choice and show the")
print("       parallel/detached variant is what fails.\n")

print("=" * 74)
print("PHASE 2 VERIFIED: topology gradient now reaches the generator")
print("=" * 74)
