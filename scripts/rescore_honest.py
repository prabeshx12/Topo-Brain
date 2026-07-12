"""
PHASE 1 — Re-score the EXISTING predictions with honest metrics.

The published numbers (SSIM 0.8991 / PSNR 20.00 / Dice 0.7682 / HD95 3.31) were produced
by a metric harness that (a) pasted the ground truth into 80.5% of the volume before
computing SSIM/PSNR, (b) computed "Dice"/"HD95" on INTENSITY thresholds rather than
segmentations, and (c) used a surface->object distance that scores a fragmented prediction
as near-perfect. See revision/AUDIT.md.

This re-scores the SAME saved predictions with src/metrics_honest.py, so we finally learn
what the real numbers are -- for TopoBrain (110k diffusion) and for the B1 regression
baseline, on the same subject, with the same yardstick.

No inference. CPU. Run:
    python scripts/rescore_honest.py --dir <folder with the .nii.gz predictions>
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mh", ROOT / "src" / "metrics_honest.py")
mh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mh)


def dnorm(vol, clip_lo=0.5, clip_hi=99.5, p_lo=1.0, p_hi=99.0):
    """The exact 'diffusion' normalization used in training (src/preprocessing.py)."""
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


def old_metrics(pred, target, mask):
    """Verbatim reimplementation of the OLD (published) harness, for the comparison."""
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    p = pred.copy()
    p[~mask] = target[~mask]                       # paste the ground truth  <-- THE CHEAT
    return (float(ssim(target, p, data_range=2.0)),  # whole-volume average
            float(psnr(target, p, data_range=2.0)))


def wholevol_metrics(pred, target):
    """Whole-volume SSIM/PSNR WITHOUT pasting -- the field's usual convention.

    This is NOT cheating: the model genuinely has to reproduce the (-1) background too.
    It is simply an easier task than the brain interior, so the number sits well above the
    brain-only value. We report it because the published competitors on this dataset
    (FS-RWKV; LiteMamba-Synth; Acs & Zhuang) almost certainly use this convention, and we
    must compare on equal terms rather than against a number we invented.
    """
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    return (float(ssim(target, pred, data_range=2.0)),
            float(psnr(target, pred, data_range=2.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--spacing", type=float, nargs=3, default=(0.65, 0.65, 0.65))
    ap.add_argument("--no-topology", action="store_true", help="skip Betti (memory-heavy)")
    args = ap.parse_args()
    d = args.dir

    def find(pat):
        hits = sorted(d.rglob(pat))
        return hits[0] if hits else None

    tgt_p = find("sub-06_ses-2_desc-preproc_T1w.nii*")
    gt_seg_p = find("sub-06_ses-2_desc-preproc_T1w_seg.nii*")
    assert tgt_p and gt_seg_p, "need the 7T target and the GT segmentation"

    target = dnorm(nib.load(str(tgt_p)).get_fdata().astype(np.float32))
    gt_seg = np.rint(nib.load(str(gt_seg_p)).get_fdata()).astype(np.int32)
    brain = gt_seg > 0                       # the real brain mask (from the GT segmentation)
    print(f"target {target.shape} | brain {brain.sum():,} / {brain.size:,} "
          f"= {100*brain.mean():.1f}% of the volume")
    print(f"=> pasting GT into the background gives a free "
          f"{10*np.log10(1/brain.mean()):.2f} dB of PSNR\n")

    models = {
        "TopoBrain-110k (diffusion)": (find("sub-06_pred7T_ddim.nii*"),
                                       find("sub-06_predseg_ddim.nii*")),
        "B1 (regression)":            (find("sub-06_B1_pred7T.nii*"),
                                       find("sub-06_B1_predseg.nii*")),
    }

    results = {}
    for name, (img_p, seg_p) in models.items():
        if img_p is None:
            print(f"[skip] {name}: prediction not found")
            continue
        pred = nib.load(str(img_p)).get_fdata().astype(np.float32)   # already in [-1,1]
        pred_seg = (np.rint(nib.load(str(seg_p)).get_fdata()).astype(np.int32)
                    if seg_p else None)

        r = mh.evaluate_synthesis(
            pred_img=pred, target_img=target, brain_mask=brain,
            pred_seg=pred_seg, gt_seg=gt_seg,
            spacing=tuple(args.spacing), connectivity=26,
            with_topology=not args.no_topology,
        )
        o_ssim, o_psnr = old_metrics(pred, target, brain)
        w_ssim, w_psnr = wholevol_metrics(pred, target)
        r["old_harness"] = {"ssim": round(o_ssim, 4), "psnr": round(o_psnr, 2)}
        r["whole_volume_no_paste"] = {"ssim": round(w_ssim, 4), "psnr": round(w_psnr, 2)}
        results[name] = r

        print("=" * 74)
        print(name)
        print("=" * 74)
        print("  IMAGE")
        print(f"    [A] brain-only (HONEST)        SSIM {r['image']['ssim_brain']:.4f}"
              f"   PSNR {r['image']['psnr_brain']:6.2f} dB")
        print(f"    [B] whole-volume, no paste     SSIM {w_ssim:.4f}"
              f"   PSNR {w_psnr:6.2f} dB   <- the field's convention (comparable to lit.)")
        print(f"    [C] OLD harness (GT pasted)    SSIM {o_ssim:.4f}"
              f"   PSNR {o_psnr:6.2f} dB   <- INVALID; +{o_psnr - r['image']['psnr_brain']:.2f} dB free")
        if "tissue" in r:
            print(f"  ANATOMY (on the SEGMENTATION, surface-to-surface HD95)")
            print(f"    {'tissue':6} {'Dice':>7} {'HD95mm':>8} {'ASSDmm':>8} {'nCC':>7} {'b0':>7}")
            for t, v in r["tissue"].items():
                b0 = v.get("betti_pred", [None])[0]
                print(f"    {t:6} {v['dice']:7.4f} {v['hd95_mm']:8.2f} {v['assd_mm']:8.2f} "
                      f"{v['n_components']:7d} {str(b0):>7}")
            print(f"    brain  {r['brain']['dice']:7.4f} {r['brain']['hd95_mm']:8.2f} "
                  f"{r['brain']['assd_mm']:8.2f}")
        print()

    out = d / "honest_rescore.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"wrote {out}")

    if len(results) == 2:
        a, b = list(results.items())
        print("\n" + "=" * 74)
        print("HEAD-TO-HEAD under HONEST metrics")
        print("=" * 74)
        print(f"  {'metric':28} {a[0][:20]:>20} {b[0][:20]:>20}")
        print(f"  {'SSIM (brain)':28} {a[1]['image']['ssim_brain']:>20.4f} "
              f"{b[1]['image']['ssim_brain']:>20.4f}")
        print(f"  {'PSNR (brain, dB)':28} {a[1]['image']['psnr_brain']:>20.2f} "
              f"{b[1]['image']['psnr_brain']:>20.2f}")
        if "tissue" in a[1] and "tissue" in b[1]:
            for t in ("CSF", "GM", "WM"):
                print(f"  {'Dice ' + t:28} {a[1]['tissue'][t]['dice']:>20.4f} "
                      f"{b[1]['tissue'][t]['dice']:>20.4f}")
                print(f"  {'HD95 ' + t + ' (mm)':28} {a[1]['tissue'][t]['hd95_mm']:>20.2f} "
                      f"{b[1]['tissue'][t]['hd95_mm']:>20.2f}")


if __name__ == "__main__":
    sys.exit(main())
