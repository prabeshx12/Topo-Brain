"""
Acs & Zhuang's 3D FR-U-Net, reimplemented -- the SOTA baseline reviewers R1.2/R2.1 demanded.

  Acs R, Zhuang H. "Semi-supervised synthesis of 7T MRI from 3T using 3D FR-U-Net with
  anatomical segmentation consistency assessment." PLOS ONE 20(11):e0333499, 6 Nov 2025.
  doi:10.1371/journal.pone.0333499

WHY REIMPLEMENT RATHER THAN CITE. They report PSNR 23.25 / SSIM 0.737 on this exact dataset --
but under a protocol that is not ours (2D per-slice, all slices, full-head, unmasked, transverse
plane only). Quoting their number next to ours would be meaningless: we measured that the
convention alone is worth 17.15 dB (revision/PROTOCOL_SENSITIVITY.md). The ONLY defensible
comparison is to train their architecture on OUR splits and score it with OUR metrics.

They release NO code and NO weights, so reimplementation from the text is the only route.

WHAT THE PAPER SPECIFIES (implemented faithfully)
  * Encoder: 4 levels, 3D conv + PReLU + layer norm, max-pool stride 2.
        32 ch @ 64^3 -> 64 @ 32^3 -> 128 @ 16^3 -> 256 @ 8^3 (bottleneck)
  * Decoder: transposed conv (stride 2) at 128 / 64 / 32 ch, skip-concatenation.
  * Residual block per decoder stage: "two (3 x 3 x 3) convolutional layers with PReLU
    activations, and a skip connection that adds the input directly to the output."
  * Multi-scale fusion (MSF) per decoder stage: "three parallel 3D convolutional branches ...
    using (1 x 1 x 1), (3 x 3 x 3), and (5 x 5 x 5) kernels, followed by channel-wise
    concatenation."
  * Output: "1 x 1 x 1 convolution followed by a sigmoid activation function"  -> [0, 1].
  * Patches 64^3, stride 32.  Loss: MSE + 0.7 * SSIM.  Adam, lr 2e-5, batch 8.

WHAT THE PAPER DOES **NOT** SPECIFY  -- every one of these is an inference, flagged in code and
reported in the paper as a reproduction caveat. We do not get to pretend we ran *their* model.
  (i)   PARAMETER COUNT. Never stated ("million" appears zero times). We compute and report it.
  (ii)  LAYER-NORM AXIS. "Layer normalization" only. Keras' LayerNormalization defaults to the
        CHANNEL axis, so that is what we implement (see _LayerNormChannels). A conv-style
        GroupNorm is selectable for a sensitivity check.
  (iii) MSF CHANNEL BOOKKEEPING. Three branches are concatenated, but the output width is not
        given. We use the standard inception form: each branch -> C_out, concat -> 3*C_out,
        then a 1x1x1 projection back to C_out.
  (iv)  DECODER STAGE ORDER. We use: transposed-conv -> concat skip -> MSF -> residual.
  (v)   PATCH REASSEMBLY BLENDING. Not stated ("Gaussian"/"blend"/"average" never appear in a
        reassembly context). We reuse OUR tiled Tukey-window blending, identically to our own
        model, so the baseline is not disadvantaged.
  (vi)  SSIM window/constants and the PSNR data_range. Not stated.
  (vii) The semi-supervised consistency weights (alpha, and the lambda of Eq 6) are not stated
        -- and the symbol lambda is reused for two different things in their text.

ON THE SEMI-SUPERVISED HALF. Their headline model adds a consistency loss on UNPAIRED 7T volumes
(20 extra subjects from Li et al.). We implement the SUPERVISED FR-U-Net. This does not
disadvantage them: **their own statistics show the semi-supervised variant is NOT significantly
better than their supervised one on ANY metric or orientation (all p > 0.28)**, and they say so.
Their supervised row is 23.173 dB vs the semi-supervised 23.254 -- a 0.08 dB difference.
"""
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FRUNet"]


class _LayerNormChannels(nn.Module):
    """LayerNorm over the CHANNEL axis -- what Keras' LayerNormalization does by default.

    The paper says only "layer normalization" and does not give the axis. Their implementation is
    TensorFlow/Keras, whose LayerNormalization normalises over the last axis; in channels-last
    NDHWC that is the channel axis, per spatial location. We reproduce that. (Selectable
    alternative: GroupNorm, the usual conv-net choice -- see FRUNet(norm=...).)
    """

    def __init__(self, c: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(c, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # [B,C,D,H,W]
        x = x.permute(0, 2, 3, 4, 1)                          # -> [B,D,H,W,C]
        x = self.ln(x)
        return x.permute(0, 4, 1, 2, 3).contiguous()          # -> [B,C,D,H,W]


def _norm(kind: str, c: int) -> nn.Module:
    if kind == "layer":
        return _LayerNormChannels(c)
    if kind == "group":
        return nn.GroupNorm(min(8, c), c)
    raise ValueError(f"unknown norm {kind!r}")


class _ConvBlock(nn.Module):
    """3D conv + PReLU + norm -- the encoder's basic unit, twice per level."""

    def __init__(self, cin: int, cout: int, norm: str):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1), _norm(norm, cout), nn.PReLU(cout),
            nn.Conv3d(cout, cout, 3, padding=1), _norm(norm, cout), nn.PReLU(cout),
        )

    def forward(self, x):
        return self.body(x)


class _ResBlock(nn.Module):
    """"two (3x3x3) convolutional layers with PReLU activations, and a skip connection that adds
    the input directly to the output" -- verbatim from the paper."""

    def __init__(self, c: int, norm: str):
        super().__init__()
        self.c1, self.n1, self.a1 = nn.Conv3d(c, c, 3, padding=1), _norm(norm, c), nn.PReLU(c)
        self.c2, self.n2, self.a2 = nn.Conv3d(c, c, 3, padding=1), _norm(norm, c), nn.PReLU(c)

    def forward(self, x):
        h = self.a1(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return self.a2(h + x)                                  # "adds the input directly"


class _MSF(nn.Module):
    """Multi-scale fusion: parallel 1x1x1 / 3x3x3 / 5x5x5 branches, channel-concatenated.

    CAVEAT (iii): the paper gives no output width for the concatenation. We use the standard
    inception form -- each branch produces cout, concat -> 3*cout, then a 1x1x1 projection back
    to cout. Any other bookkeeping would change the parameter count, which they never state.
    """

    def __init__(self, cin: int, cout: int, norm: str):
        super().__init__()
        self.b1 = nn.Conv3d(cin, cout, 1)
        self.b3 = nn.Conv3d(cin, cout, 3, padding=1)
        self.b5 = nn.Conv3d(cin, cout, 5, padding=2)
        self.proj = nn.Conv3d(3 * cout, cout, 1)
        self.norm = _norm(norm, cout)
        self.act = nn.PReLU(cout)

    def forward(self, x):
        h = torch.cat([self.b1(x), self.b3(x), self.b5(x)], dim=1)
        return self.act(self.norm(self.proj(h)))


class FRUNet(nn.Module):
    """3D FR-U-Net (Acs & Zhuang, PLOS ONE 2025). 3T patch in -> 7T patch out, sigmoid to [0,1].

    NOTE THE OUTPUT RANGE. Their final activation is a sigmoid and their volumes are min-maxed to
    [0,1]. Our pipeline works in [-1,1] (tanh). The training script converts, so the baseline is
    trained in ITS OWN native range -- reproducing their design rather than bending it to ours.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 features: Sequence[int] = (32, 64, 128, 256), norm: str = "layer"):
        super().__init__()
        f = list(features)
        self.encoders = nn.ModuleList()
        cin = in_channels
        for c in f[:-1]:
            self.encoders.append(_ConvBlock(cin, c, norm))
            cin = c
        self.bottleneck = _ConvBlock(cin, f[-1], norm)
        self.pool = nn.MaxPool3d(2)

        self.ups, self.msfs, self.resblocks = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(len(f) - 1, 0, -1):
            cout = f[i - 1]
            self.ups.append(nn.ConvTranspose3d(f[i], cout, 2, stride=2))
            self.msfs.append(_MSF(cout * 2, cout, norm))       # *2: skip concatenation
            self.resblocks.append(_ResBlock(cout, norm))

        self.out = nn.Conv3d(f[0], out_channels, 1)            # "1x1x1 convolution ..."
        self.act = nn.Sigmoid()                                # "... followed by a sigmoid"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)

        # CAVEAT (iv): stage order (upconv -> concat -> MSF -> residual) is not given in the paper.
        for up, msf, res in zip(self.ups, self.msfs, self.resblocks):
            x = up(x)
            s = skips.pop()
            if x.shape[-3:] != s.shape[-3:]:
                x = F.interpolate(x, size=s.shape[-3:], mode="trilinear", align_corners=False)
            x = msf(torch.cat([x, s], dim=1))
            x = res(x)
        return self.act(self.out(x))

    def n_params(self) -> int:
        """The paper never states this ('million' appears zero times in the full text)."""
        return sum(p.numel() for p in self.parameters())


def frunet_loss(pred: torch.Tensor, target: torch.Tensor, ssim_fn,
                lam_ssim: float = 0.7) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Their Eq (1): Hybrid Loss = MSE + lambda * SSIM_loss, "We set lambda = 0.7".

    Both tensors must be in [0,1] (their sigmoid range), NOT our [-1,1].
    """
    l_mse = F.mse_loss(pred, target)
    l_ssim = 1.0 - ssim_fn(pred, target)
    return l_mse + lam_ssim * l_ssim, l_mse, l_ssim
