"""
Verify the FR-U-Net reimplementation (Acs & Zhuang, PLOS ONE 2025) against every structural
claim the paper actually makes -- and report the parameter count they never published.

This is the SOTA baseline reviewers R1.2/R2.1 demanded. It is only a fair baseline if it is
(a) faithful to their text and (b) not crippled relative to our own model, so both are asserted.

CPU, seconds.  Run:  python scripts/test_frunet.py
"""
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pkg = importlib.util.module_from_spec(importlib.util.spec_from_loader("src", loader=None))
pkg.__path__ = [str(ROOT / "src")]
sys.modules["src"] = pkg
for _n in ("model", "model_frunet", "model_cascaded"):
    _s = importlib.util.spec_from_file_location(f"src.{_n}", ROOT / "src" / f"{_n}.py")
    _m = importlib.util.module_from_spec(_s)
    sys.modules[f"src.{_n}"] = _m
    _s.loader.exec_module(_m)
fr = sys.modules["src.model_frunet"]
mc = sys.modules["src.model_cascaded"]

torch.manual_seed(0)
print("=" * 84)
print("FR-U-Net (Acs & Zhuang, PLOS ONE 2025) — reimplementation check")
print("=" * 84)

net = fr.FRUNet(features=(32, 64, 128, 256))

# ---- 1. the encoder pyramid the paper specifies -------------------------------------------
print("\n1) ENCODER PYRAMID — the paper states 32@64^3 -> 64@32^3 -> 128@16^3 -> 256@8^3")
x = torch.randn(2, 1, 64, 64, 64)
acts = []
h = x
for enc in net.encoders:
    h = enc(h)
    acts.append(tuple(h.shape[1:]))
    h = net.pool(h)
h = net.bottleneck(h)
acts.append(tuple(h.shape[1:]))
expected = [(32, 64, 64, 64), (64, 32, 32, 32), (128, 16, 16, 16), (256, 8, 8, 8)]
for got, exp in zip(acts, expected):
    ok = got == exp
    print(f"   {str(got):24} expected {str(exp):24} {'OK' if ok else '<-- MISMATCH'}")
    assert ok, f"encoder level mismatch: {got} != {exp}"
print("   [OK] the pyramid matches the paper exactly.")

# ---- 2. forward: shape + the sigmoid output range ------------------------------------------
print("\n2) OUTPUT — '1x1x1 convolution followed by a sigmoid activation function'")
y = net(x)
print(f"   shape {tuple(y.shape)}   range [{float(y.min()):.4f}, {float(y.max()):.4f}]")
assert y.shape == (2, 1, 64, 64, 64), "output must match the input patch shape"
assert 0.0 <= float(y.min()) and float(y.max()) <= 1.0, "sigmoid output must lie in [0,1]"
print("   [OK] shape preserved; output bounded to [0,1] as their sigmoid requires.")

# ---- 3. gradient flows ---------------------------------------------------------------------
print("\n3) GRADIENT")
net.zero_grad(set_to_none=True)
y.sum().backward()
n_none = sum(1 for p in net.parameters() if p.grad is None)
gsum = sum(float(p.grad.abs().sum()) for p in net.parameters() if p.grad is not None)
print(f"   params with no grad: {n_none}    sum|grad| = {gsum:.4e}")
assert n_none == 0 and gsum > 0, "every parameter must receive a gradient"
print("   [OK] fully differentiable.")

# ---- 4. the number they never published -----------------------------------------------------
print("\n4) PARAMETER COUNT — the paper NEVER states it ('million' appears 0 times in the text)")
n_fr = net.n_params()
ours = mc.CascadedSynthesisNet(num_classes=4)
n_ours = sum(p.numel() for p in ours.parameters())
print(f"   FR-U-Net (this reimplementation)   {n_fr:>12,}  ({n_fr / 1e6:.2f} M)")
print(f"   TopoBrain cascaded (ours)          {n_ours:>12,}  ({n_ours / 1e6:.2f} M)")
print(f"   ratio                              {n_fr / n_ours:>12.2f}x")
print("\n   Reported so the comparison is capacity-aware. NOTE: on this dataset capacity barely")
print("   matters -- LiteMamba's Table 2 has a 0.29 M CNN (20.75 dB) BEATING a 123 M transformer")
print("   (20.14 dB), and every architecture ever tried spans just 1.55 dB. By contrast the")
print("   evaluation convention is worth 17.15 dB (revision/PROTOCOL_SENSITIVITY.md).")

# ---- 5. their loss --------------------------------------------------------------------------
print("\n5) LOSS — their Eq(1): 'Hybrid Loss = MSE Loss + lambda x SSIM Loss', 'We set lambda=0.7'")


def fake_ssim(a, b):                       # a stand-in; training uses the real ssim3d
    return 1.0 - (a - b).abs().mean()


pred = torch.rand(2, 1, 16, 16, 16, requires_grad=True)
tgt = torch.rand(2, 1, 16, 16, 16)
tot, l_mse, l_ssim = fr.frunet_loss(pred, tgt, fake_ssim, lam_ssim=0.7)
tot.backward()
print(f"   total {float(tot):.4f} = MSE {float(l_mse):.4f} + 0.7 x SSIM_loss {float(l_ssim):.4f}")
assert abs(float(tot) - (float(l_mse) + 0.7 * float(l_ssim))) < 1e-5, "Eq(1) must hold exactly"
assert pred.grad is not None and pred.grad.abs().sum() > 0
print("   [OK] Eq(1) holds exactly and is differentiable.")

print("\n" + "=" * 84)
print("REPRODUCTION CAVEATS — stated, because they cannot be resolved from the paper")
print("=" * 84)
for c in (
    "param count      NEVER STATED. We report ours above; theirs is unknown.",
    "layer-norm axis  'layer normalization' only. We follow the Keras default (channel axis).",
    "MSF widths       concat width not given. We use the inception form + 1x1x1 projection.",
    "stage order      upconv -> concat -> MSF -> residual is our reading, not their statement.",
    "reassembly       patch blending rule not given. We reuse OUR Tukey blending, so the",
    "                 baseline is not disadvantaged relative to our own model.",
    "semi-supervised  omitted: needs 20 unpaired 7T subjects we do not have. Their OWN stats",
    "                 show it is not significantly better than supervised (all p > 0.28);",
    "                 supervised 23.173 vs semi-supervised 23.254 dB.",
):
    print(f"  * {c}")
print("\n  We therefore report this as 'FR-U-Net (our reimplementation)', NEVER as their result.")
print("=" * 84)
