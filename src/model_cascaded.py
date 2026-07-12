"""
Cascaded synthesis model: 3T -> synthetic 7T -> tissue segmentation.

WHY THIS EXISTS
---------------
The original `AnatomyGuidedUNet` runs TWO PARALLEL decoders off the shared bottleneck:

    bottleneck --+--> image decoder (ups)     --> synth 7T
                 +--> seg   decoder (seg_ups) --> seg  --> topology loss

A topology loss on `seg` therefore CANNOT reach the image decoder. Measured on the real
model (revision/AUDIT.md, bug 6), after loss_topo.backward():

    encoder      (shared)   sum|grad| = 1.88e+01
    bottleneck   (shared)   sum|grad| = 2.08e+01
    IMAGE decoder .ups      sum|grad| = 0.00e+00   <-- grad is None for 42/42 params
    IMAGE head    .outc     sum|grad| = 0.00e+00   <-- grad is None for  2/2 params

So the layers that actually synthesise the 7T image received ZERO topology gradient. The
"topology-preserving synthesis" claim was structurally impossible: the loss could only ever
regularise the shared encoder. That is the single deepest reason a plain regression baseline
with no topology loss matched (and beat) it on topology.

THE FIX: CASCADE, DON'T PARALLELISE
-----------------------------------
    3T --> U-Net --> synth 7T --> SegHead --> seg --> topology / Dice / CE loss
                        ^                              |
                        +---------- gradient ----------+

The segmentation is now computed FROM THE SYNTHESISED IMAGE. Any loss on the segmentation
backpropagates THROUGH the image decoder, so the network is forced to synthesise an image
that is *segmentable with the correct anatomy and topology*. That is what
"topology-aware synthesis" was always supposed to mean.

The 3T input is also given to the SegHead (concatenated) because tissue boundaries are
partly determined by the input anatomy; this does not break the gradient path to the image.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .model import ResnetBlock, SelfAttention3D, SinusoidalPosEmb  # reuse the proven blocks


class UNetGenerator(nn.Module):
    """Deterministic 3D U-Net: 3T -> 7T. Same capacity/blocks as the published backbone
    (features 32/64/128/256, GroupNorm-8, SiLU, attention bottleneck), with the diffusion
    machinery removed.

    A timestep embedding is retained but fixed at t=0 so the ResnetBlocks can be reused
    unchanged; it is a constant bias and costs ~8k params.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Tuple[int, ...] = (32, 64, 128, 256),
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention
        dim = features[0]
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.inc = nn.Conv3d(in_channels, features[0], 3, padding=1)

        self.downs = nn.ModuleList()
        ch = features[0]
        for f in features[1:]:
            self.downs.append(nn.ModuleList([
                ResnetBlock(ch, ch, time_emb_dim=dim),
                nn.Conv3d(ch, f, 3, stride=2, padding=1),
            ]))
            ch = f

        mid = features[-1]
        self.mid_block1 = ResnetBlock(mid, mid, time_emb_dim=dim)
        self.mid_attn = SelfAttention3D(mid) if use_attention else nn.Identity()
        self.mid_block2 = ResnetBlock(mid, mid, time_emb_dim=dim)

        self.ups = nn.ModuleList()
        rev = list(reversed(features))
        for i in range(len(rev) - 1):
            cur, nxt = rev[i], rev[i + 1]
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose3d(cur, nxt, 2, stride=2),
                ResnetBlock(nxt * 2, nxt, time_emb_dim=dim),
            ]))
        self.outc = nn.Conv3d(features[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        temb = self.time_mlp(t.view(-1))

        h = self.inc(x)
        skips = [h]
        for res, down in self.downs:
            h = res(h, temb)
            h = down(h)
            skips.append(h)

        h = self.mid_block1(h, temb)
        if self.use_attention:
            h = self.mid_attn(h)
        h = self.mid_block2(h, temb)

        for i, (up, res) in enumerate(self.ups):
            h = up(h)
            h = torch.cat([h, skips[-(i + 2)]], dim=1)
            h = res(h, temb)

        # tanh keeps the synthetic image in the data range [-1, 1] and, unlike a hard
        # clamp, stays differentiable everywhere (a clamp would zero the gradient for any
        # saturated voxel -- the same dead-gradient trap found in the old perceptual loss).
        return torch.tanh(self.outc(h))


class SegHead(nn.Module):
    """Small 3D conv head: (synth 7T [, 3T]) -> 4-class tissue logits.

    Deliberately shallow. Its job is not to be a great segmenter in isolation -- it is the
    differentiable probe through which the anatomy/topology loss reaches the GENERATOR. If
    it were too powerful it could compensate for a bad image and let the generator off the
    hook, which is precisely the failure mode we are trying to eliminate.
    """

    def __init__(self, in_channels: int = 2, num_classes: int = 4, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, width, 3, padding=1),
            nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv3d(width, width, 3, padding=1),
            nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv3d(width, width, 3, padding=1, dilation=1),
            nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv3d(width, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CascadedSynthesisNet(nn.Module):
    """3T -> synth 7T -> seg, with the segmentation computed FROM the synthesised image.

    Returns:
        {"image": [B,1,D,H,W] synth 7T in [-1,1],
         "seg":   [B,C,D,H,W] tissue logits derived from that image}

    Because `seg` is a function of `image`, every segmentation/topology loss produces a
    non-zero gradient in the image decoder. `scripts/test_cascaded_gradient.py` asserts
    exactly that (the old parallel design gave 0.00e+00).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_classes: int = 4,
        features: Tuple[int, ...] = (32, 64, 128, 256),
        use_attention: bool = True,
        seg_sees_input: bool = True,
        seg_width: int = 32,
        detach_seg_input: bool = False,
    ):
        super().__init__()
        self.seg_sees_input = seg_sees_input
        # detach_seg_input=True severs the cascade (seg loss no longer reaches the
        # generator). Kept ONLY so the paper can ablate exactly this design choice and
        # show the parallel/detached variant is what fails.
        self.detach_seg_input = detach_seg_input

        self.generator = UNetGenerator(
            in_channels=in_channels, out_channels=out_channels,
            features=features, use_attention=use_attention,
        )
        self.seg_head = SegHead(
            in_channels=out_channels + (in_channels if seg_sees_input else 0),
            num_classes=num_classes, width=seg_width,
        )

    def forward(self, x3t: torch.Tensor) -> Dict[str, torch.Tensor]:
        img = self.generator(x3t)

        seg_in = img.detach() if self.detach_seg_input else img
        if self.seg_sees_input:
            seg_in = torch.cat([seg_in, x3t], dim=1)
        seg = self.seg_head(seg_in)
        return {"image": img, "seg": seg}

    @torch.no_grad()
    def n_params(self) -> Dict[str, int]:
        g = sum(p.numel() for p in self.generator.parameters())
        s = sum(p.numel() for p in self.seg_head.parameters())
        return {"generator": g, "seg_head": s, "total": g + s}
