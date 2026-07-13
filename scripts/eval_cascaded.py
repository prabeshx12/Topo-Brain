"""
Honest full-volume evaluation of a CascadedSynthesisNet checkpoint.

Deterministic: ONE forward per patch (no diffusion sampling), Tukey-window stitching,
then src/metrics_honest -- brain-only SSIM/PSNR, segmentation Dice, surface-to-surface
HD95. Runs on CPU.

    python scripts/eval_cascaded.py --ckpt cascaded_16677.pt --dir <folder with sub-06 nii>
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# load src.model + src.model_cascaded + src.metrics_honest without src/__init__ (monai)
pkg = importlib.util.module_from_spec(importlib.util.spec_from_loader("src", loader=None))
pkg.__path__ = [str(ROOT / "src")]
sys.modules["src"] = pkg
for _n in ("model", "model_cascaded", "metrics_honest"):
    _s = importlib.util.spec_from_file_location(f"src.{_n}", ROOT / "src" / f"{_n}.py")
    _m = importlib.util.module_from_spec(_s)
    sys.modules[f"src.{_n}"] = _m
    _s.loader.exec_module(_m)
from src.model_cascaded import CascadedSynthesisNet  # noqa: E402
from src import metrics_honest as mh                 # noqa: E402


def dnorm(vol, clip_lo=0.5, clip_hi=99.5, p_lo=1.0, p_hi=99.0):
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


def tukey(n, alpha=0.5):
    w = np.ones(n)
    t = int(alpha * n / 2)
    if t > 0:
        w[:t] = 0.5 * (1 - np.cos(np.pi * np.arange(t) / t))
        w[-t:] = 0.5 * (1 - np.cos(np.pi * np.arange(t, 0, -1) / t))
    return w


@torch.no_grad()
def tiled(model, vol, dev, patch=64, overlap=32):
    D, H, W = vol.shape
    stride = patch - overlap
    acc = np.zeros((D, H, W), np.float64)
    sacc = np.zeros((4, D, H, W), np.float64)
    wacc = np.zeros((D, H, W), np.float64)
    w1 = tukey(patch, overlap / patch)
    win = (w1[None, None, :] * w1[None, :, None] * w1[:, None, None])

    def starts(L):
        s = list(range(0, L - patch + 1, stride))
        if s and s[-1] + patch < L:
            s.append(L - patch)
        return s or [0]

    ds, hs, ws = starts(D), starts(H), starts(W)
    tot, n = len(ds) * len(hs) * len(ws), 0
    for d in ds:
        for h in hs:
            for w in ws:
                p = vol[d:d + patch, h:h + patch, w:w + patch]
                sh = p.shape
                if sh != (patch,) * 3:
                    q = np.full((patch,) * 3, -1.0, np.float32)
                    q[:sh[0], :sh[1], :sh[2]] = p
                    p = q
                x = torch.from_numpy(p).float()[None, None].to(dev)
                out = model(x)
                pr = out["image"].cpu().numpy().squeeze()[:sh[0], :sh[1], :sh[2]]
                sg = out["seg"].cpu().numpy().squeeze()[:, :sh[0], :sh[1], :sh[2]]
                ww = win[:sh[0], :sh[1], :sh[2]]
                acc[d:d + sh[0], h:h + sh[1], w:w + sh[2]] += pr * ww
                wacc[d:d + sh[0], h:h + sh[1], w:w + sh[2]] += ww
                for c in range(4):
                    sacc[c, d:d + sh[0], h:h + sh[1], w:w + sh[2]] += sg[c] * ww
                n += 1
                if n % 100 == 0:
                    print(f"  patch {n}/{tot}", flush=True)
    m = wacc > 0
    img = np.full((D, H, W), -1.0, np.float32)
    img[m] = (acc[m] / wacc[m]).astype(np.float32)
    for c in range(4):
        sacc[c][m] /= wacc[m]
    return img, np.argmax(sacc, 0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--subject", default="sub-06")
    ap.add_argument("--spacing", type=float, nargs=3, default=(0.65, 0.65, 0.65))
    ap.add_argument("--use-ema", type=int, default=1)
    ap.add_argument("--no-topology", action="store_true")
    args = ap.parse_args()
    d = args.dir
    s = args.subject

    def find(pat):
        hits = sorted(d.rglob(pat))
        return hits[0] if hits else None

    i3 = find(f"{s}_ses-1_desc-preproc_T1w_registered.nii*")
    t7 = find(f"{s}_ses-2_desc-preproc_T1w.nii*")
    gs = find(f"{s}_ses-2_desc-preproc_T1w_seg.nii*")
    assert i3 and t7 and gs, "need 3T, 7T and GT seg"

    x3 = dnorm(nib.load(str(i3)).get_fdata().astype(np.float32))
    x7 = dnorm(nib.load(str(t7)).get_fdata().astype(np.float32))
    gt = np.rint(nib.load(str(gs)).get_fdata()).astype(np.int32)
    brain = gt > 0
    aff = nib.load(str(i3)).affine
    print(f"{s}: {x3.shape}  brain {100*brain.mean():.1f}%")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = st["config"]["model"]
    model = CascadedSynthesisNet(
        in_channels=1, out_channels=1, num_classes=int(cfg["num_classes"]),
        features=tuple(cfg["features"]), use_attention=cfg["use_attention"],
    ).to(dev)
    key = "ema" if (args.use_ema and "ema" in st) else "model"
    model.load_state_dict(st[key])
    model.eval()
    print(f"loaded {key} @ step {st['step']} | params {model.n_params()['total']:,} | dev {dev}")

    img, seg = tiled(model, x3, dev)

    r = mh.evaluate_synthesis(img, x7, brain, pred_seg=seg, gt_seg=gt,
                              spacing=tuple(args.spacing), connectivity=26,
                              with_topology=not args.no_topology)

    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    wv_ssim = float(ssim(x7, img, data_range=2.0))
    wv_psnr = float(psnr(x7, img, data_range=2.0))
    r["whole_volume_no_paste"] = {"ssim": round(wv_ssim, 4), "psnr": round(wv_psnr, 2)}
    r["step"] = int(st["step"])

    print("\n" + "=" * 66)
    print(f"CASCADED @ step {st['step']}  ({s}, honest metrics)")
    print("=" * 66)
    print(f"  [A] brain-only   SSIM {r['image']['ssim_brain']:.4f}  PSNR {r['image']['psnr_brain']:6.2f} dB")
    print(f"  [B] whole-volume SSIM {wv_ssim:.4f}  PSNR {wv_psnr:6.2f} dB   (field convention)")
    if "tissue" in r:
        print(f"\n  {'tissue':6} {'Dice':>7} {'HD95mm':>8} {'nCC':>7} {'b0':>7}")
        for t, v in r["tissue"].items():
            b0 = v.get("betti_pred", [None])[0]
            print(f"  {t:6} {v['dice']:7.4f} {v['hd95_mm']:8.2f} {v['n_components']:7d} {str(b0):>7}")
        print(f"  brain  {r['brain']['dice']:7.4f} {r['brain']['hd95_mm']:8.2f}")

    print("\n  --- reference (same subject, honest metrics) ---")
    print(f"  B1 regression : brain SSIM 0.4436  PSNR 14.12 dB | whole-vol 0.8840 / 20.24 dB")
    print(f"  TopoBrain 110k: brain SSIM 0.0710  PSNR  8.27 dB | whole-vol 0.1618 /  9.45 dB")

    nib.save(nib.Nifti1Image(img, aff), str(d / f"{s}_cascaded_pred7T.nii.gz"))
    nib.save(nib.Nifti1Image(seg, aff), str(d / f"{s}_cascaded_predseg.nii.gz"))
    (d / "cascaded_eval.json").write_text(json.dumps(r, indent=2, default=float))
    print(f"\nwrote {d / 'cascaded_eval.json'}")


if __name__ == "__main__":
    sys.exit(main())
