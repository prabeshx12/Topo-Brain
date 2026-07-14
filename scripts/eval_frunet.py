"""
Evaluate the FR-U-Net baseline — with the range inverse map that did NOT previously exist.

THE TRAP THIS SCRIPT EXISTS TO DISARM.
FR-U-Net ends in a SIGMOID, so it predicts in [0,1] (their design). Our pipeline, our targets and
our metrics all live in [-1,1]: src/metrics_honest.py hardcodes `data_range=2.0` "because volumes
live in [-1,1]". train_frunet.py does the FORWARD map (`tgt01 = (x7+1)/2`) -- but the INVERSE map
existed NOWHERE in the repository.

So the obvious way to score the baseline was to feed a [0,1] prediction against a [-1,1] target.
Every voxel then carries a systematic +1.0 offset, MSE is inflated by ~1.0, and PSNR collapses to
about 10*log10(4/1) = 6 dB NO MATTER HOW WELL THE MODEL ACTUALLY FITS.

We would have "beaten the state of the art" by roughly 15 dB, on a baseline we broke ourselves.
That is the single most dangerous kind of bug in this whole project: it flatters us, it is
invisible, and it is exactly the class of error that got the original paper rejected.

The map is one line -- `pred = pred01 * 2 - 1` -- and it is asserted below, not assumed.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pkg = importlib.util.module_from_spec(importlib.util.spec_from_loader("src", loader=None))
pkg.__path__ = [str(ROOT / "src")]
sys.modules["src"] = pkg
for _n in ("model", "model_frunet", "metrics_honest"):
    _s = importlib.util.spec_from_file_location(f"src.{_n}", ROOT / "src" / f"{_n}.py")
    _m = importlib.util.module_from_spec(_s)
    sys.modules[f"src.{_n}"] = _m
    _s.loader.exec_module(_m)
FRUNet = sys.modules["src.model_frunet"].FRUNet
mh = sys.modules["src.metrics_honest"]

_tc = importlib.util.spec_from_file_location("_tc", ROOT / "scripts" / "eval_cascaded.py")


def to_model_range(pred01: np.ndarray) -> np.ndarray:
    """FR-U-Net's sigmoid [0,1] -> our [-1,1]. THE MAP THAT WAS MISSING."""
    return pred01.astype(np.float32) * 2.0 - 1.0


def dnorm(vol, clip_lo=0.5, clip_hi=99.5, p_lo=1.0, p_hi=99.0):
    """Identical to scripts/eval_cascaded.py and the Kaggle training kernel."""
    v = vol.astype(np.float32).copy()
    roi = v[v > 0]
    if roi.size == 0:
        return v
    lo_c, hi_c = np.percentile(roi, clip_lo), np.percentile(roi, clip_hi)
    v = np.clip(v, lo_c, hi_c)
    roi = v[v > 0]
    lo, hi = np.percentile(roi, p_lo), np.percentile(roi, p_hi)
    if hi > lo:
        v = np.clip((v - lo) / (hi - lo), 0, 1) * 2.0 - 1.0
    v[vol <= 0] = -1.0
    return v


def tukey(n: int, alpha: float = 0.5) -> np.ndarray:
    w = np.ones(n, dtype=np.float32)
    e = int(alpha * (n - 1) / 2)
    if e > 0:
        r = np.arange(e + 1)
        ramp = 0.5 * (1 + np.cos(np.pi * (2 * r / (alpha * (n - 1)) - 1)))
        w[:e + 1] = ramp
        w[-(e + 1):] = ramp[::-1]
    return np.maximum(w, 1e-3)          # never exactly 0 -> no zero-weight plane


@torch.no_grad()
def tiled(model, x3, dev, ps=64, ov=32):
    D, H, W = x3.shape
    acc = np.zeros((D, H, W), np.float32)
    wacc = np.zeros((D, H, W), np.float32)
    win = (tukey(ps)[:, None, None] * tukey(ps)[None, :, None] * tukey(ps)[None, None, :])
    step = ps - ov

    def starts(L):
        s = list(range(0, max(1, L - ps + 1), step))
        if s[-1] != L - ps:
            s.append(L - ps)
        return s

    for z in starts(D):
        for y in starts(H):
            for x in starts(W):
                p = x3[z:z + ps, y:y + ps, x:x + ps]
                t = torch.from_numpy(p)[None, None].to(dev)
                o01 = model(t)[0, 0].float().cpu().numpy()     # sigmoid -> [0,1]
                acc[z:z + ps, y:y + ps, x:x + ps] += o01 * win
                wacc[z:z + ps, y:y + ps, x:x + ps] += win
    assert (wacc > 0).all(), "every voxel must be covered by at least one patch"
    return acc / wacc                                          # still [0,1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dir", required=True, help="dir holding the subject's nii files")
    ap.add_argument("--subject", default="sub-07", help="TEST fold. Not sub-06 (that is VAL).")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = Path(a.dir)

    def find(pat):
        hits = sorted(d.rglob(pat))
        if not hits:
            raise FileNotFoundError(pat)
        return hits[0]

    s = a.subject
    x3 = dnorm(nib.load(find(f"{s}_ses-1_desc-preproc_T1w_registered.nii*")).get_fdata())
    ref = nib.load(find(f"{s}_ses-2_desc-preproc_T1w.nii*"))
    x7 = dnorm(ref.get_fdata())
    seg = np.rint(nib.load(find(f"{s}_ses-2_desc-preproc_T1w_seg.nii*")).get_fdata()).astype(int)
    brain = seg > 0

    # spacing from the HEADER, not a hardcoded default. The paper says 1.0 mm; the data is 0.65.
    zooms = tuple(float(z) for z in ref.header.get_zooms()[:3])
    print(f"  voxel spacing from header: {zooms} mm  (NOT the hardcoded 1.0)")

    ck = torch.load(a.ckpt, map_location=dev)
    model = FRUNet().to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    pred01 = tiled(model, x3.astype(np.float32), dev)
    assert 0.0 <= pred01.min() and pred01.max() <= 1.0, \
        f"FR-U-Net is sigmoid-bounded; got [{pred01.min():.3f}, {pred01.max():.3f}]"

    pred = to_model_range(pred01)                              # <-- THE MISSING MAP
    assert -1.001 <= pred.min() and pred.max() <= 1.001, "inverse map must land in [-1,1]"
    print(f"  pred [0,1] range {pred01.min():.3f}..{pred01.max():.3f}"
          f"  ->  [-1,1] range {pred.min():.3f}..{pred.max():.3f}")

    # Guard the exact failure this file exists to prevent.
    wrong = mh.masked_psnr(pred01, x7, brain, data_range=2.0)   # what a careless run would do
    right = mh.masked_psnr(pred, x7, brain, data_range=2.0)
    print(f"\n  brain PSNR WITHOUT the inverse map : {wrong:6.2f} dB   <-- the trap")
    print(f"  brain PSNR WITH    the inverse map : {right:6.2f} dB")
    print(f"  the map is worth {right - wrong:+.2f} dB. Scoring the baseline without it would")
    print(f"  have handed us a fake win.\n")

    res = mh.evaluate_synthesis(pred, x7, seg, spacing=zooms)
    res["_subject"] = s
    res["_spacing_mm"] = zooms
    res["_psnr_without_inverse_map"] = wrong
    Path(a.out, f"{s}_frunet_eval.json").write_text(json.dumps(res, indent=2, default=float))
    nib.save(nib.Nifti1Image(pred, ref.affine), Path(a.out, f"{s}_frunet_pred7T.nii.gz"))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("_")},
                     indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
