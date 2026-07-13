"""
Honest evaluation of the Phase-3 cascaded checkpoint, on GPU.

Deterministic: ONE forward per patch (no diffusion sampling), Tukey-window stitching, then
src/metrics_honest -- brain-only SSIM/PSNR, segmentation Dice, surface-to-surface HD95,
true Betti numbers.

Evaluates the VAL subject (sub-06, fold 0). The TEST subject (sub-07, fold 1) is deliberately
NOT touched -- it is held out for the final result, since the old pipeline's fatal flaw was
selecting checkpoints on the test subject.

Reference (same subject, same honest metrics):
    B1 regression  : brain SSIM 0.4436 / 14.12 dB | whole-vol 0.8840 / 20.24 dB
    TopoBrain 110k : brain SSIM 0.0710 /  8.27 dB | whole-vol 0.1618 /  9.45 dB
    benchmark      : Acs & Zhuang (PLOS ONE 2025) PSNR 23.25 / SSIM 0.737 (their convention)
"""
import glob
import json
import os
import subprocess
import sys
import zipfile

import torch as _t


def log(*a):
    print(*a, flush=True)


_cap = _t.cuda.get_device_capability(0) if _t.cuda.is_available() else None
log(f"torch {_t.__version__} | cuda={_t.cuda.is_available()} | cap={_cap}")
if _t.cuda.is_available() and _cap and _cap[0] < 7 and not os.environ.get("TORCH_FIXED"):
    log("Pascal GPU -> installing cu121 torch, re-exec")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
                    "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121"],
                   check=False)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchvision"],
                   check=False)
    os.environ["TORCH_FIXED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gudhi"], check=False)

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402
import torch  # noqa: E402

hits = glob.glob("/kaggle/input/**/src/model_cascaded.py", recursive=True)
if hits:
    CODE = os.path.dirname(os.path.dirname(hits[0]))
else:
    z = glob.glob("/kaggle/input/**/*.zip", recursive=True)[0]
    CODE = "/kaggle/working/repo"
    os.makedirs(CODE, exist_ok=True)
    zipfile.ZipFile(z).extractall(CODE)
sys.path.insert(0, CODE)
log("code:", CODE)

from src.model_cascaded import CascadedSynthesisNet  # noqa: E402
from src import metrics_honest as mh  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ = (torch.randn(64, 64, device=DEV) @ torch.randn(64, 64, device=DEV)).sum().item()
log("device:", DEV)

SUB = "sub-06"


def find(pat):
    h = sorted(glob.glob(f"/kaggle/input/**/{pat}", recursive=True))
    return h[0] if h else None


i3 = find(f"{SUB}_ses-1_desc-preproc_T1w_registered.nii*")
t7 = find(f"{SUB}_ses-2_desc-preproc_T1w.nii*")
gs = find(f"{SUB}_ses-2_desc-preproc_T1w_seg.nii*")
ck = sorted(glob.glob("/kaggle/input/**/cascaded_*.pt", recursive=True),
            key=lambda p: int(p.split("_")[-1].split(".")[0]))
assert i3 and t7 and gs and ck, "missing inputs"
CKPT = ck[-1]
log("ckpt:", CKPT)


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


x3 = dnorm(nib.load(i3).get_fdata().astype(np.float32))
x7 = dnorm(nib.load(t7).get_fdata().astype(np.float32))
gt = np.rint(nib.load(gs).get_fdata()).astype(np.int32)
brain = gt > 0
aff = nib.load(i3).affine
sp = tuple(np.abs(np.diag(aff)[:3]))
log(f"{SUB} {x3.shape} | brain {100*brain.mean():.1f}% | spacing {sp}")

st = torch.load(CKPT, map_location=DEV, weights_only=False)
mc = st["config"]["model"]
model = CascadedSynthesisNet(
    in_channels=1, out_channels=1, num_classes=int(mc["num_classes"]),
    features=tuple(mc["features"]), use_attention=mc["use_attention"],
).to(DEV)
model.load_state_dict(st["ema"] if "ema" in st else st["model"])
model.eval()
log(f"loaded EMA @ step {st['step']} | params {model.n_params()['total']:,}")


def tukey(n, a=0.5):
    w = np.ones(n)
    t = int(a * n / 2)
    if t > 0:
        w[:t] = 0.5 * (1 - np.cos(np.pi * np.arange(t) / t))
        w[-t:] = 0.5 * (1 - np.cos(np.pi * np.arange(t, 0, -1) / t))
    return w


@torch.no_grad()
def tiled(vol, patch=64, ov=32):
    D, H, W = vol.shape
    stride = patch - ov
    acc = np.zeros((D, H, W), np.float64)
    sacc = np.zeros((4, D, H, W), np.float64)
    wacc = np.zeros((D, H, W), np.float64)
    w1 = tukey(patch, ov / patch)
    win = w1[None, None, :] * w1[None, :, None] * w1[:, None, None]

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
                out = model(torch.from_numpy(p).float()[None, None].to(DEV))
                pr = out["image"].cpu().numpy().squeeze()[:sh[0], :sh[1], :sh[2]]
                sg = out["seg"].cpu().numpy().squeeze()[:, :sh[0], :sh[1], :sh[2]]
                ww = win[:sh[0], :sh[1], :sh[2]]
                acc[d:d + sh[0], h:h + sh[1], w:w + sh[2]] += pr * ww
                wacc[d:d + sh[0], h:h + sh[1], w:w + sh[2]] += ww
                for c in range(4):
                    sacc[c, d:d + sh[0], h:h + sh[1], w:w + sh[2]] += sg[c] * ww
                n += 1
                if n % 100 == 0:
                    log(f"  patch {n}/{tot}")
    m = wacc > 0
    img = np.full((D, H, W), -1.0, np.float32)
    img[m] = (acc[m] / wacc[m]).astype(np.float32)
    for c in range(4):
        sacc[c][m] /= wacc[m]
    return img, np.argmax(sacc, 0).astype(np.uint8)


log("\n=== tiled inference (1 forward/patch) ===")
img, seg = tiled(x3)

r = mh.evaluate_synthesis(img, x7, brain, pred_seg=seg, gt_seg=gt,
                          spacing=sp, connectivity=26, with_topology=True)
from skimage.metrics import peak_signal_noise_ratio as psnr  # noqa: E402
from skimage.metrics import structural_similarity as ssim  # noqa: E402
r["whole_volume_no_paste"] = {"ssim": round(float(ssim(x7, img, data_range=2.0)), 4),
                              "psnr": round(float(psnr(x7, img, data_range=2.0)), 2)}
r["step"] = int(st["step"])

log("\n" + "=" * 70)
log(f"CASCADED @ step {st['step']}  ({SUB}, honest metrics)")
log("=" * 70)
log(f"  [A] brain-only   SSIM {r['image']['ssim_brain']:.4f}   PSNR {r['image']['psnr_brain']:6.2f} dB")
log(f"  [B] whole-volume SSIM {r['whole_volume_no_paste']['ssim']:.4f}   "
    f"PSNR {r['whole_volume_no_paste']['psnr']:6.2f} dB   (field convention)")
log(f"\n  {'tissue':6} {'Dice':>7} {'HD95mm':>8} {'nCC':>7} {'b0':>7} {'b0 GT':>7}")
for t, v in r["tissue"].items():
    log(f"  {t:6} {v['dice']:7.4f} {v['hd95_mm']:8.2f} {v['n_components']:7d} "
        f"{v['betti_pred'][0]:7d} {v['betti_gt'][0]:7d}")
log(f"  brain  {r['brain']['dice']:7.4f} {r['brain']['hd95_mm']:8.2f}")

log("\n  --- reference, same subject, same yardstick ---")
log("  B1 regression  : brain SSIM 0.4436 / 14.12 dB | whole-vol 0.8840 / 20.24 dB")
log("  TopoBrain 110k : brain SSIM 0.0710 /  8.27 dB | whole-vol 0.1618 /  9.45 dB")
log("  benchmark      : Acs & Zhuang 2025  PSNR 23.25 / SSIM 0.737 (their convention)")

d_ssim = r["image"]["ssim_brain"] - 0.4436
d_psnr = r["image"]["psnr_brain"] - 14.12
log(f"\n  vs B1 (brain-only): SSIM {d_ssim:+.4f}   PSNR {d_psnr:+.2f} dB")

nib.save(nib.Nifti1Image(img, aff), f"/kaggle/working/{SUB}_cascaded_pred7T.nii.gz")
nib.save(nib.Nifti1Image(seg, aff), f"/kaggle/working/{SUB}_cascaded_predseg.nii.gz")
with open("/kaggle/working/cascaded_eval.json", "w") as f:
    json.dump(r, f, indent=2, default=float)
log("\nwrote cascaded_eval.json")
