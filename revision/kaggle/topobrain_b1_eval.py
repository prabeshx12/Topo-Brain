"""
HEAD-TO-HEAD: B1 regression baseline vs TopoBrain (110k diffusion) on sub-06.

Reviewer R1.2 / R2.6. Same subject, same normalization, same tiling/stitching,
same metrics as the TopoBrain eval -- the ONLY difference is the model.

B1 inference is a single forward per patch (no diffusion sampling), so this is
fast. Computes SSIM/PSNR/Dice/HD95, per-tissue Dice, and true Betti numbers so
the comparison covers image quality AND anatomy AND topology.

TopoBrain (110k, DDIM-50) reference, from revision/kaggle/results_110k_sub06.md:
    SSIM 0.8998 | PSNR 20.85 | fgHD95 3.31 | CSF/GM/WM Dice 0.681/0.782/0.882
    Betti0 pred CSF/GM/WM = 753 / 5213 / 1393   (GT 5 / 1 / 15)
"""
import glob
import importlib.util
import json
import os
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gudhi"], check=False)


def log(*a):
    print(*a, flush=True)


import torch as _t  # noqa: E402

_cap = _t.cuda.get_device_capability(0) if _t.cuda.is_available() else None
log(f"torch {_t.__version__} | cuda={_t.cuda.is_available()} | cap={_cap}")
if _t.cuda.is_available() and _cap and _cap[0] < 7 and not os.environ.get("TORCH_FIXED"):
    log("installing cu121 torch for Pascal, re-exec...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
                    "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchvision"], check=False)
    os.environ["TORCH_FIXED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from scipy import ndimage  # noqa: E402

hits = glob.glob("/kaggle/input/**/src/model.py", recursive=True)
CODE = os.path.dirname(os.path.dirname(hits[0]))
sys.path.insert(0, CODE)
from src.model import AnatomyGuidedUNet  # noqa: E402
import scripts.evaluate_full_volume as ev  # noqa: E402

spec = importlib.util.spec_from_file_location("cb", f"{CODE}/scripts/compute_betti.py")
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)
betti_3d = cb.betti_3d

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cuda":
    _ = (torch.randn(64, 64, device=DEV) @ torch.randn(64, 64, device=DEV)).sum().item()
    log("CUDA OK")
CFG = yaml.safe_load(open(f"{CODE}/configs/train_diffusion.yaml"))


def find_one(p):
    h = sorted(glob.glob(f"/kaggle/input/**/{p}", recursive=True))
    return h[0] if h else None


IN_3T = find_one("sub-06_ses-1_desc-preproc_T1w_registered.nii*")
TGT_7T = find_one("sub-06_ses-2_desc-preproc_T1w.nii*")
GT_SEG = find_one("sub-06_ses-2_desc-preproc_T1w_seg.nii*")

# newest baseline checkpoint from the training kernel's output
ckpts = sorted(glob.glob("/kaggle/input/**/baseline_reg_*.pt", recursive=True),
               key=lambda p: int(p.split("_")[-1].split(".")[0]))
assert ckpts, "no baseline_reg_*.pt found"
CKPT = ckpts[-1]
log("baseline ckpt:", CKPT)


def diffusion_normalize(vol, clip_lo=0.5, clip_hi=99.5, p_lo=1.0, p_hi=99.0):
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


inp_nii = nib.load(IN_3T)
input_vol = diffusion_normalize(inp_nii.get_fdata().astype(np.float32))
affine = inp_nii.affine
target_vol = diffusion_normalize(nib.load(TGT_7T).get_fdata().astype(np.float32))
gt_seg = np.rint(nib.load(GT_SEG).get_fdata()).astype(np.int32)
spacing = tuple(np.abs(np.diag(affine)[:3]))
log("shape", input_vol.shape, "| norm ranges", float(input_vol.min()), float(input_vol.max()))

model = AnatomyGuidedUNet(
    in_channels=CFG["model"]["in_channels"],
    cond_channels=CFG["model"].get("cond_channels", 1),
    out_channels=CFG["model"]["out_channels"],
    num_classes=CFG["model"]["num_classes"],
    features=CFG["model"]["features"],
    use_attention=CFG["model"]["use_attention"],
).to(DEV)
st = torch.load(CKPT, map_location=DEV, weights_only=False)
model.load_state_dict(st["ema"])
model.eval()
log(f"loaded EMA @ step {st['step']} | baseline={st.get('baseline')}")


@torch.no_grad()
def tiled_regression(vol, patch=64, overlap=32):
    """Same Tukey-window tiling/stitching as evaluate_full_volume, but a single
    deterministic forward per patch (no diffusion sampling)."""
    D, H, W = vol.shape
    stride = patch - overlap
    acc = np.zeros((D, H, W), np.float64)
    sacc = np.zeros((4, D, H, W), np.float64)
    wacc = np.zeros((D, H, W), np.float64)
    w1 = ev._tukey_window_1d(patch, alpha=overlap / patch)
    win = (w1[None, None, :] * w1[None, :, None] * w1[:, None, None]).astype(np.float64)

    def starts(L):
        s = list(range(0, L - patch + 1, stride))
        if s and s[-1] + patch < L:
            s.append(L - patch)
        return s or [max(0, (L - patch) // 2)]

    ds, hs, ws = starts(D), starts(H), starts(W)
    tot = len(ds) * len(hs) * len(ws)
    n = 0
    for d in ds:
        for h in hs:
            for w in ws:
                p = vol[d:d+patch, h:h+patch, w:w+patch]
                sh = p.shape
                if sh != (patch, patch, patch):
                    q = np.zeros((patch, patch, patch), np.float32)
                    q[:sh[0], :sh[1], :sh[2]] = p
                    p = q
                x3 = torch.from_numpy(p).float()[None, None].to(DEV)
                out = model(torch.zeros_like(x3), torch.zeros(1, device=DEV, dtype=torch.long), x3)
                pr = out["prediction"].cpu().numpy().squeeze()[:sh[0], :sh[1], :sh[2]]
                sg = out["segmentation"].cpu().numpy().squeeze()[:, :sh[0], :sh[1], :sh[2]]
                ww = win[:sh[0], :sh[1], :sh[2]]
                acc[d:d+sh[0], h:h+sh[1], w:w+sh[2]] += pr * ww
                wacc[d:d+sh[0], h:h+sh[1], w:w+sh[2]] += ww
                for c in range(4):
                    sacc[c, d:d+sh[0], h:h+sh[1], w:w+sh[2]] += sg[c] * ww
                n += 1
                if n % 100 == 0:
                    log(f"  patch {n}/{tot}")
    m = wacc > 0
    outv = np.zeros_like(acc, np.float32)
    outv[m] = (acc[m] / wacc[m]).astype(np.float32)
    for c in range(4):
        sacc[c][m] /= wacc[m]
    return outv, np.argmax(sacc, 0).astype(np.uint8)


log("\n=== B1 tiled regression inference (1 forward/patch) ===")
pred_vol, pred_seg = tiled_regression(input_vol)

masks = [("Threshold(-0.95)", ev.make_brain_mask(input_vol, -0.95)),
         ("SegMaskGT(seg>0)", gt_seg > 0)]
res = {"checkpoint": CKPT, "step": int(st["step"]), "image_metrics": {}}
log("\n=== IMAGE METRICS (B1) ===")
for lbl, mk in masks:
    m = ev.compute_metric_set(lbl, mk, pred_vol, target_vol, input_vol, spacing)
    res["image_metrics"][lbl] = {"ssim": round(float(m["ssim_masked"]), 4),
                                 "psnr": round(float(m["psnr_masked"]), 2),
                                 "dice": round(float(m["dice"]), 4),
                                 "hd95_mm": round(float(m["hd95_mm"]), 2)}
    log(f"  [{lbl}] SSIM={m['ssim_masked']:.4f} PSNR={m['psnr_masked']:.2f} "
        f"Dice={m['dice']:.4f} HD95={m['hd95_mm']:.2f}")

TIS = {1: "CSF", 2: "GM", 3: "WM"}
log("\n=== PER-TISSUE DICE (B1) ===")
res["tissue_dice"] = {}
for lb, nm in TIS.items():
    a, b = pred_seg == lb, gt_seg == lb
    d = float(2 * np.logical_and(a, b).sum() / (a.sum() + b.sum())) if (a.sum() + b.sum()) else float("nan")
    res["tissue_dice"][nm] = round(d, 4)
    log(f"  {nm}: {d:.4f}")

log("\n=== TRUE TOPOLOGY (B1) ===")
res["topology"] = {}
for lb, nm in TIS.items():
    b = pred_seg == lb
    lab, _ = ndimage.label(b)
    cnt = np.bincount(lab.ravel())
    cnt[0] = 0
    ncc = int((cnt > 0).sum())
    lcc = float(cnt.max() / b.sum()) if b.sum() else 0.0
    b0, b1, b2 = betti_3d(b)
    res["topology"][nm] = {"ncc": ncc, "largest_cc": round(lcc, 4),
                           "b0": int(b0), "b1": int(b1), "b2": int(b2)}
    log(f"  {nm}: ncc={ncc} lcc={lcc:.3f} b0={b0} b1={b1} b2={b2}")

nib.save(nib.Nifti1Image(pred_vol, affine), "/kaggle/working/sub-06_B1_pred7T.nii.gz")
nib.save(nib.Nifti1Image(pred_seg, affine), "/kaggle/working/sub-06_B1_predseg.nii.gz")
with open("/kaggle/working/b1_eval_results.json", "w") as f:
    json.dump(res, f, indent=2)

log("\n" + "=" * 66)
log("HEAD-TO-HEAD  (sub-06)                B1-regression   TopoBrain-110k")
log("=" * 66)
tb = {"ssim": 0.8998, "psnr": 20.85, "hd95": 3.31,
      "CSF": 0.6811, "GM": 0.7823, "WM": 0.8821,
      "b0": {"CSF": 753, "GM": 5213, "WM": 1393}}
sm = res["image_metrics"].get("SegMaskGT(seg>0)", {})
th = res["image_metrics"].get("Threshold(-0.95)", {})
log(f"  SSIM (SegMaskGT)                   {sm.get('ssim'):>10}   {tb['ssim']:>10}")
log(f"  PSNR (SegMaskGT)                   {sm.get('psnr'):>10}   {tb['psnr']:>10}")
log(f"  fg HD95 (Threshold)                {th.get('hd95_mm'):>10}   {tb['hd95']:>10}")
for nm in ("CSF", "GM", "WM"):
    log(f"  Dice {nm:<3}                           {res['tissue_dice'][nm]:>10}   {tb[nm]:>10}")
for nm in ("CSF", "GM", "WM"):
    log(f"  Betti0 {nm:<3} (lower=better)        {res['topology'][nm]['b0']:>10}   {tb['b0'][nm]:>10}")
log("\nWhoever wins, this is reported as-is.")
