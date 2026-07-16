"""
Evaluate a cascaded checkpoint on CERN, using the EXACT normalised data training used
(pairs_cern.csv -> norm/{s}_3t/7t/seg, already [-1,1] with the COMPLETE whole-brain segs).

WHY NOT eval_cascaded.py: that script globs the OLD BIDS-named files and applies dnorm, and it
would find the OLD cerebrum-only seg (we only overwrote norm/*_seg.nii.gz). Running it on CERN
would silently score against the wrong segmentation. This reads the normalised pairs directly, so
the evaluation GT is byte-identical to the training GT. It also reads voxel spacing from the
NIfTI header (0.65 mm) instead of hard-coding it (audit finding M1).

Usage:
  python scripts/eval_cern.py --ckpt <fold/cascaded_40000.pt> \
      --pairs-csv /eos/user/p/ppokhrel/topobrain/data/norm/pairs_cern.csv \
      --subject sub-06 --out <dir>
"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# reuse the PROVEN tiled inference from eval_cascaded, and the honest metrics
pkg = importlib.util.module_from_spec(importlib.util.spec_from_loader("src", loader=None))
pkg.__path__ = [str(ROOT / "src")]
sys.modules.setdefault("src", pkg)
for _n in ("model", "model_cascaded", "metrics_honest"):
    _s = importlib.util.spec_from_file_location(f"src.{_n}", ROOT / "src" / f"{_n}.py")
    _m = importlib.util.module_from_spec(_s)
    sys.modules[f"src.{_n}"] = _m
    _s.loader.exec_module(_m)
mh = sys.modules["src.metrics_honest"]
CascadedSynthesisNet = sys.modules["src.model_cascaded"].CascadedSynthesisNet

_ec = importlib.util.spec_from_file_location("_ec", ROOT / "scripts" / "eval_cascaded.py")
_ecm = importlib.util.module_from_spec(_ec)
# eval_cascaded imports `from src...`; those are already in sys.modules, so exec is safe
_ec.loader.exec_module(_ecm)
tiled = _ecm.tiled


def row_for(pairs_csv, subject):
    with open(pairs_csv) as f:
        for row in csv.DictReader(f):
            if row["subject"] == subject:
                return row
    raise SystemExit(f"subject {subject} not in {pairs_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--subject", default="sub-06")
    ap.add_argument("--use-ema", type=int, default=1)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    row = row_for(a.pairs_csv, a.subject)
    # these are ALREADY normalised to [-1,1] with the complete seg -- no dnorm, no BIDS glob
    x3_img = nib.load(row["input_3t"])
    x3 = x3_img.get_fdata().astype(np.float32)
    x7 = nib.load(row["target_7t"]).get_fdata().astype(np.float32)
    gt = np.rint(nib.load(row["seg"]).get_fdata()).astype(np.int32)
    brain = gt > 0
    aff = x3_img.affine
    spacing = tuple(float(z) for z in x3_img.header.get_zooms()[:3])   # from header, not hardcoded
    assert x3.min() >= -1.01 and x3.max() <= 1.01, "input must already be normalised to [-1,1]"
    print(f"{a.subject}: {x3.shape}  brain {100*brain.mean():.1f}%  spacing {spacing} mm "
          f"(labels {sorted(np.unique(gt).tolist())})")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg = st["config"]["model"]
    model = CascadedSynthesisNet(
        in_channels=1, out_channels=1, num_classes=int(cfg["num_classes"]),
        features=tuple(cfg["features"]), use_attention=cfg["use_attention"]).to(dev)
    key = "ema" if (a.use_ema and "ema" in st) else "model"
    model.load_state_dict(st[key])
    model.eval()
    print(f"loaded {key} @ step {st['step']} | dev {dev}")

    img, seg = tiled(model, x3, dev)
    r = mh.evaluate_synthesis(img, x7, brain, pred_seg=seg, gt_seg=gt,
                              spacing=spacing, connectivity=26, with_topology=True)

    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    r["whole_volume_no_paste"] = {"ssim": round(float(ssim(x7, img, data_range=2.0)), 4),
                                  "psnr": round(float(psnr(x7, img, data_range=2.0)), 2)}
    r["step"] = int(st["step"])
    r["_subject"] = a.subject
    r["_spacing_mm"] = spacing

    print("\n" + "=" * 66)
    print(f"CASCADED @ step {st['step']}  ({a.subject}, COMPLETE segs, honest)")
    print("=" * 66)
    print(f"  brain-only   SSIM {r['image']['ssim_brain']:.4f}  PSNR {r['image']['psnr_brain']:6.2f} dB")
    print(f"  whole-volume SSIM {r['whole_volume_no_paste']['ssim']:.4f}  "
          f"PSNR {r['whole_volume_no_paste']['psnr']:6.2f} dB")
    if "tissue" in r:
        print(f"\n  {'tissue':6} {'Dice':>7} {'HD95mm':>8} {'b0pred':>7} {'b0gt':>6}")
        for t, v in r["tissue"].items():
            b0p = v.get("betti_pred", [None])[0]
            b0g = v.get("betti_gt", [None])[0]
            print(f"  {t:6} {v['dice']:7.4f} {v['hd95_mm']:8.2f} {str(b0p):>7} {str(b0g):>6}")

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(img, aff), str(outdir / f"{a.subject}_pred7T.nii.gz"))
    (outdir / "cascaded_eval.json").write_text(json.dumps(r, indent=2, default=float))
    print(f"\nwrote {outdir / 'cascaded_eval.json'}")


if __name__ == "__main__":
    main()
