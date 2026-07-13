"""
A REAL, differentiable topological loss: the Euler characteristic of a cubical complex.

WHY THIS AND NOT THE ALTERNATIVES
---------------------------------
* The old "topology loss" was NOT topological: edge-weighted CE + a Sobel edge-Dice. It
  encodes no connectivity, no Betti number, no persistence. Worse, it sat on a PARALLEL
  decoder, so it sent EXACTLY 0.00e+00 gradient to the image decoder (revision/AUDIT.md).
  FS-RWKV and LiteMamba-Synth already use the same L1+SSIM+Sobel recipe on this exact task
  and do not call it topology.
* clDice (Shit et al., CVPR 2021) is the WRONG TOOL: its guarantee requires foreground and
  background to be homotopy-equivalent to GRAPHS (tubular: vessels, roads, neurons). CSF/GM/WM
  are blobs and sheets. Using it here is an easy reviewer kill.
* Persistent homology (Hu 2019; Clough 2020; Byrne TMI 2022) is the principled choice but is
  expensive per step and notoriously unstable on the garbage predictions of early training.

The Euler characteristic is a genuine topological invariant,

    chi = b0 - b1 + b2          (components - handles + voids)

and for a CUBICAL complex it has an exact combinatorial form:

    chi = V - E + F - C         (vertices - edges + faces - cubes)

Each term is a count of grid elements covered by at least one foreground voxel, which is a
LOCAL "OR" over a small window -- so all four are computable with tiny convolutions, and the
probabilistic OR

    soft_or(p_1..p_k) = 1 - prod_i (1 - p_i)

is exact for binary inputs and smoothly differentiable for soft ones. The result is a real
topological loss that is cheap (7 small conv3d ops), 3D-native, multi-class, and stable.

Sanity check on a single foreground voxel:
    C = 1, F = 6, E = 12, V = 8   ->   chi = 8 - 12 + 6 - 1 = 1   (a solid ball)  OK

NOTE ON SCOPE. This is applied per 64^3 PATCH, so it regularises LOCAL topology. Patch-level
chi is not volume-level chi; the global Betti numbers are what we measure at evaluation
(src/metrics_honest.py). Patch-wise topological losses are standard practice (Hu et al. 2019).

Reference: Li, Ma, Ouyang, Paetzold, Rueckert, Kainz -- fast Euler-characteristic loss,
IEEE TMI 2025 (arXiv:2507.23763).
"""
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["euler_characteristic", "EulerTopologyLoss"]

_EPS = 1e-7

# (kernel, padding) for each element of the cubical complex.
# Zero-padding the LOG is exactly right: log(1 - 0) = 0 contributes nothing to the product,
# i.e. the world outside the patch is background.
_FACES = [((2, 1, 1), (1, 0, 0)),      # faces perpendicular to z
          ((1, 2, 1), (0, 1, 0)),      # ... to y
          ((1, 1, 2), (0, 0, 1))]      # ... to x
_EDGES = [((1, 2, 2), (0, 1, 1)),      # edges parallel to z  (OR over 2x2 in the y-x plane)
          ((2, 1, 2), (1, 0, 1)),      # ... parallel to y
          ((2, 2, 1), (1, 1, 0))]      # ... parallel to x
_VERTS = ((2, 2, 2), (1, 1, 1))        # vertices: OR over the 2x2x2 corner window


def _soft_or(log_q: torch.Tensor, kernel: Sequence[int], padding: Sequence[int]) -> torch.Tensor:
    """sum over the window of log(1-p)  ->  1 - exp(sum) = 1 - prod(1-p) = soft OR."""
    k = torch.ones((1, 1, *kernel), dtype=log_q.dtype, device=log_q.device)
    s = F.conv3d(log_q, k, padding=tuple(padding))
    return 1.0 - torch.exp(s)


def euler_characteristic(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable Euler characteristic of a soft 3D foreground mask.

    Args:
        p:   [B, 1, D, H, W] soft occupancy in [0, 1].
        eps: probabilities are LINEARLY rescaled into [eps, 1-eps] before the log.

    Returns:
        [B] tensor of chi, one per sample. Exact (to O(eps)) for binary p.

    WHY A LINEAR RESCALE AND NOT A CLAMP. The obvious `(1-p).clamp(min=eps)` is a
    DEAD-GRADIENT TRAP: torch.clamp has ZERO gradient below its floor, so as soon as the
    model becomes confident (p -> 1) the topology loss silently stops producing any gradient
    at all. Measured: |grad| = 0.0000 with logits at +/-10. That is exactly the failure mode
    we criticised in the old perceptual loss (`clamp(loss, max=10)`), and it would have made
    this loss a no-op on a trained model.

    The affine map  p -> p*(1-2*eps) + eps  keeps p strictly inside (0,1) with derivative
    (1-2*eps) ~= 1 everywhere, so gradients survive saturation.

    Note the residual bias this leaves: a background voxel sits at p=eps, and each one is an
    EXPECTED spurious component, so E[chi] carries a bias of ~ eps * (#background voxels).
    With eps=1e-6 on a 64^3 patch that is ~0.13 -- negligible against the Betti errors we are
    trying to correct (GM beta0 = 65 vs GT 1).
    """
    assert p.dim() == 5 and p.shape[1] == 1, f"expected [B,1,D,H,W], got {tuple(p.shape)}"
    p = p.clamp(0.0, 1.0) * (1.0 - 2.0 * eps) + eps   # affine -> [eps, 1-eps], grad preserved
    log_q = torch.log1p(-p)                            # log(1 - p), stable

    dims = (1, 2, 3, 4)
    C = p.sum(dim=dims)
    Fc = sum(_soft_or(log_q, k, pad).sum(dim=dims) for k, pad in _FACES)
    E = sum(_soft_or(log_q, k, pad).sum(dim=dims) for k, pad in _EDGES)
    V = _soft_or(log_q, *_VERTS).sum(dim=dims)
    return V - E + Fc - C


class EulerTopologyLoss(nn.Module):
    """Per-class Euler-characteristic matching loss.

    L = mean_c  |chi(pred_c) - chi(gt_c)| / (|chi(gt_c)| + 1)

    The relative denominator keeps the term O(1) across classes whose chi differ by orders of
    magnitude (cortical GM has thousands of handles; WM has tens), so no per-class weight
    tuning is needed.

    Args:
        num_classes: number of segmentation classes (background is class 0).
        include_background: if False (default), class 0 is skipped -- the background's topology
            is the complement of the foreground's and adds nothing but noise.
        detach_gt: chi of the ground truth is a constant; detached for clarity/safety.
    """

    def __init__(self, num_classes: int = 4, include_background: bool = False,
                 detach_gt: bool = True, sharpness: float = 20.0):
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.detach_gt = detach_gt
        self.sharpness = sharpness

    def _sharpen(self, p: torch.Tensor) -> torch.Tensor:
        """p -> sigmoid(k * (p - 0.5)). Concentrates the loss on the DECISION BOUNDARY.

        WHY THIS IS NECESSARY, not cosmetic. chi as computed above is an EXPECTED chi (soft_or
        is literally P(window has >=1 foreground)), so every uncertain background voxel is an
        expected spurious component and contributes ~ +1 * p to chi. On a 13^3 test volume with
        1854 background voxels at p_bg = 3.4e-4 that is +0.62 -- which is the same order as a
        REAL topological error. Measured consequence with no sharpening:

            solid cube  (topology CORRECT)   loss 0.2609
            cube + hole (topology BROKEN)    loss 0.2363   <-- the BROKEN shape scored BETTER

        The +0.62 of background speckle numerically cancelled the -1 of the missing handle. An
        unsharpened Euler loss will therefore happily TRADE a hole against speckle -- the exact
        b0/b1 conflation this invariant is vulnerable to, doing real damage.

        Sharpening pushes confidently-background voxels to ~0 so they stop manufacturing
        expected components (k=20: p_bg 3.4e-4 -> 4.5e-5, an 8x cut in the spurious term), while
        the gradient dp'/dp = k*sig*(1-sig) is LARGEST exactly at p = 0.5 -- the decision
        boundary, which is the only place topology can actually change. The same map is applied
        to the ground truth, so the residual bias cancels in |chi_pred - chi_gt|.
        """
        if not self.sharpness:
            return p
        return torch.sigmoid(self.sharpness * (p - 0.5))

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: [B, C, D, H, W] segmentation logits (NOT softmaxed).
            target: [B, D, H, W] integer labels.
        Returns:
            {"loss": scalar, "chi_pred": [B,C], "chi_gt": [B,C], "chi_err": [B,C]}
        """
        probs = torch.softmax(logits.float(), dim=1)
        tgt = F.one_hot(target.long(), self.num_classes).permute(0, 4, 1, 2, 3).float()

        probs = self._sharpen(probs)
        tgt = self._sharpen(tgt)          # the SAME map on the GT, so the residual bias cancels

        first = 0 if self.include_background else 1
        chis_p, chis_g = [], []
        for c in range(first, self.num_classes):
            chis_p.append(euler_characteristic(probs[:, c:c + 1]))
            with torch.no_grad():
                chis_g.append(euler_characteristic(tgt[:, c:c + 1]))

        chi_p = torch.stack(chis_p, dim=1)                    # [B, C']
        chi_g = torch.stack(chis_g, dim=1)
        if self.detach_gt:
            chi_g = chi_g.detach()

        err = (chi_p - chi_g).abs() / (chi_g.abs() + 1.0)     # relative, O(1)
        if weights is not None:
            err = err * weights.view(1, -1).to(err.dtype)
        return {"loss": err.mean(), "chi_pred": chi_p, "chi_gt": chi_g, "chi_err": err}
