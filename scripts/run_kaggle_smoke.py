"""
End-to-end Kaggle smoke test runner.

One CLI that:
    1. Stages ADNI 3T NIfTI inputs from a Kaggle Dataset into a writable dir
    2. Builds a cohort manifest (`pairs_adni_smoke.csv`) from the filenames
    3. Runs scripts/preprocess_adni.py (HD-BET + N4 + diffusion-normalize)
    4. Zips the preprocessed bundle (resume checkpoint — download before
       synthesis in case the kernel dies)
    5. Loops scripts/evaluate_full_volume.py per subject for synthesis
    6. Zips synthesis outputs

Designed to be runnable as `!python` from a Kaggle notebook, so you don't
have to copy-paste cells. All defaults are tuned for Kaggle's T4 GPU + the
sub-06 145k checkpoint.

Quickstart in a Kaggle notebook (3 lines):

    !git clone -b feat/train-with-masks https://github.com/prabeshx12/Topo-Brain.git /kaggle/working/Topo-Brain 2>/dev/null || (cd /kaggle/working/Topo-Brain && git pull)
    !pip install -q HD-BET scikit-image
    !python /kaggle/working/Topo-Brain/scripts/run_kaggle_smoke.py \
        --checkpoint /kaggle/input/<your-ckpt>/checkpoint_145000.pt \
        --adni-dir /kaggle/input/<your-adni-dataset>

Use --skip-preprocess if you've already produced /kaggle/working/adni_preprocessed/
(e.g. by attaching a previous bundle as a Kaggle Dataset). Use --skip-synthesis
to stop after preprocessing.
"""
import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _find_adni_root(adni_dir: Path) -> Path:
    """Find the dir containing AD/ and CN/ subfolders. Tolerates layouts where
    the user uploaded as either `<root>/adni_nifti/{AD,CN}` or `<root>/{AD,CN}`."""
    for cand in (adni_dir / "adni_nifti", adni_dir):
        if any(cand.glob("[AC][DN]/*_T1w.nii*")):
            return cand
    raise FileNotFoundError(
        f"Couldn't find adni_nifti/AD/<ptid>_T1w.nii(.gz) under {adni_dir}. "
        f"Make sure your Kaggle dataset has the expected layout."
    )


def _stage_inputs(adni_src: Path, work_dir: Path) -> Path:
    """Copy NIfTIs from a read-only Kaggle input to a writable working dir."""
    target = work_dir / "adni_nifti"
    if target.exists() and any(target.rglob("*_T1w.nii*")):
        logging.info("Inputs already staged at %s — reusing", target)
        return target
    logging.info("Staging inputs %s -> %s", adni_src, target)
    shutil.copytree(str(adni_src), str(target), dirs_exist_ok=True)
    return target


def _build_cohort_csv(adni_local: Path, csv_path: Path) -> int:
    """Reconstruct the cohort manifest from filenames. Returns row count."""
    rows = []
    for grp in ("AD", "CN"):
        for f in sorted((adni_local / grp).glob("*_T1w.nii*")):
            rows.append(
                f"{f.name.split('_T1w.')[0]},{grp},{f}"
            )
    if not rows:
        raise RuntimeError(f"No inputs found under {adni_local}")
    with open(csv_path, "w") as fh:
        fh.write("ptid,group,nifti_path\n")
        fh.write("\n".join(rows) + "\n")
    logging.info("Wrote cohort CSV: %s (%d rows)", csv_path, len(rows))
    return len(rows)


def _run_preprocess(repo_dir: Path, cohort_csv: Path, pre_dir: Path,
                    pre_csv: Path, device: str) -> None:
    cmd = [
        sys.executable, "scripts/preprocess_adni.py",
        "--cohort-csv", str(cohort_csv),
        "--output-dir", str(pre_dir),
        "--output-csv", str(pre_csv),
        "--device", device,
    ]
    logging.info("Running preprocessing (%s)", " ".join(cmd[1:]))
    subprocess.check_call(cmd, cwd=str(repo_dir))


def _resolve_skip_preprocess_inputs(work: Path, adni_dir: Path) -> Path:
    """Set up `--skip-preprocess` mode regardless of how the user staged things.

    Handles every variation we've actually hit:
      - Bundle uploaded as a Kaggle Dataset and Kaggle stripped the `.gz` from
        every `*.nii.gz` on auto-extract (now `.nii`).
      - User attached the bundle but never copied it to /kaggle/working/.
      - User copied to /kaggle/working/ but the CSV still references paths
        from a previous session that no longer exist.

    Strategy: locate the preprocessed CSV anywhere under work or adni_dir,
    then for each ptid in the CSV, pick the best matching NIfTI file by
    (a) the recorded path if it still exists, else (b) globbing for
    `<ptid>_T1w_preprocessed.nii*` under work / adni_dir. Stage everything
    into work/ and rewrite the CSV with actual paths. Returns the final
    CSV path under work/.
    """
    import pandas as pd

    target_csv = work / "pairs_adni_smoke_preprocessed.csv"

    # ---- 1) locate the source CSV ----
    csv_candidates = [
        target_csv,
        adni_dir / "pairs_adni_smoke_preprocessed.csv",
    ] + sorted(adni_dir.rglob("pairs_adni_smoke_preprocessed.csv"))
    src_csv = next((c for c in csv_candidates if c.exists()), None)
    if src_csv is None:
        raise FileNotFoundError(
            f"--skip-preprocess: no pairs_adni_smoke_preprocessed.csv found under "
            f"{work} or {adni_dir}. Make sure the preprocessed bundle is attached."
        )
    if src_csv != target_csv:
        target_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(src_csv), str(target_csv))
        logging.info("Staged CSV %s -> %s", src_csv, target_csv)

    # ---- 2) build a ptid -> actual file index across both locations ----
    by_ptid: dict = {}
    for root in (work, adni_dir):
        if not root.exists():
            continue
        for f in root.rglob("*_T1w_preprocessed.nii*"):
            ptid = f.name.split("_T1w_preprocessed")[0]
            # Prefer files actually under work/ (writable) when both exist.
            if ptid not in by_ptid or root == work:
                by_ptid[ptid] = f.resolve()

    if not by_ptid:
        raise FileNotFoundError(
            f"--skip-preprocess: no *_T1w_preprocessed.nii* files found under "
            f"{work} or {adni_dir}. Did the preprocessed bundle ingest correctly?"
        )
    logging.info("Found %d preprocessed files (across .nii and .nii.gz)", len(by_ptid))

    # ---- 3) rewrite CSV with real paths ----
    df = pd.read_csv(target_csv)
    df["preprocessed_path"] = df["ptid"].astype(str).map(
        lambda p: str(by_ptid[p]) if p in by_ptid else "")
    missing = df[df["preprocessed_path"] == ""]
    if len(missing):
        raise RuntimeError(
            f"--skip-preprocess: missing files for ptids: {missing['ptid'].tolist()}. "
            f"Found {len(by_ptid)} files in total but none matching these subjects."
        )
    df.to_csv(target_csv, index=False)
    logging.info("Rewrote %d preprocessed_path entries to actual file locations", len(df))
    return target_csv


def _zip_bundle(label: str, src_dirs_or_files: list, zip_path: Path) -> None:
    """Bundle a set of dirs/files into a zip with clean POSIX paths."""
    stage = Path(tempfile.mkdtemp(prefix=f"{label}_"))
    try:
        for src in src_dirs_or_files:
            src = Path(src)
            if not src.exists():
                logging.warning("  skipping missing %s", src)
                continue
            dst = stage / src.name
            if src.is_dir():
                shutil.copytree(str(src), str(dst))
            else:
                shutil.copy(str(src), str(dst))
        shutil.make_archive(str(zip_path).replace(".zip", ""), "zip", str(stage))
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    size_mb = zip_path.stat().st_size / 1e6
    logging.info("Zipped %s -> %s (%.1f MB)", label, zip_path, size_mb)


def _run_synthesis(repo_dir: Path, checkpoint: Path, pre_csv: Path,
                   results_dir: Path, sampler: str, ddim_steps: int,
                   n_samples: int, start_index: int = 1,
                   end_index: int | None = None) -> None:
    import pandas as pd

    df = pd.read_csv(pre_csv)
    ok = df[df["preprocess_status"].isin(["ok", "skipped_exists"])].reset_index(drop=True)
    if ok.empty:
        raise RuntimeError("No subjects with preprocess_status ok in {pre_csv}")

    total_ok = len(ok)
    if start_index < 1:
        raise ValueError("--start-index must be 1 or greater")
    if start_index > total_ok:
        raise ValueError(f"--start-index {start_index} is beyond the {total_ok} synthesizable subjects")
    if end_index is not None and end_index < start_index:
        raise ValueError("--end-index must be greater than or equal to --start-index")

    start_pos = start_index - 1
    end_pos = min(end_index, total_ok) if end_index is not None else total_ok
    selected = ok.iloc[start_pos:end_pos].reset_index(drop=True)

    logging.info("Synthesizing rows %d-%d of %d with %s (%d steps, n_samples=%d)",
                 start_index, end_pos, total_ok, sampler, ddim_steps, n_samples)
    results_dir.mkdir(parents=True, exist_ok=True)

    for local_i, sub in selected.iterrows():
        global_i = start_pos + local_i
        ptid = sub["ptid"]
        out_subj = results_dir / ptid
        out_subj.mkdir(parents=True, exist_ok=True)
        if (out_subj / "predicted_7T.nii.gz").exists():
            logging.info("[%d/%d] %s -- already synthesized, skipping",
                         global_i + 1, total_ok, ptid)
            continue
        logging.info("[%d/%d] %s (%s) synthesizing",
                     global_i + 1, total_ok, ptid, sub["group"])
        t0 = time.time()
        subprocess.check_call([
            sys.executable, "scripts/evaluate_full_volume.py",
            "--checkpoint", str(checkpoint),
            "--input", sub["preprocessed_path"],
            "--output_dir", str(out_subj),
            "--sampler", sampler,
            "--ddim-steps", str(ddim_steps),
            "--n-samples", str(n_samples),
        ], cwd=str(repo_dir))
        logging.info("    -> %s done in %.1fs", ptid, time.time() - t0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the trained checkpoint .pt")
    parser.add_argument("--adni-dir", required=True,
                        help="Kaggle Dataset path. In default mode: contains "
                             "adni_nifti/{AD,CN}/ raw inputs. With --skip-preprocess: "
                             "may contain the preprocessed bundle (adni_preprocessed/ + "
                             "pairs_adni_smoke_preprocessed.csv) — script auto-stages.")
    parser.add_argument("--work-dir", default="/kaggle/working",
                        help="Writable workspace (default: /kaggle/working)")
    parser.add_argument("--repo-dir", default=None,
                        help="Topo-Brain repo location (default: parent of this script)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device for HD-BET (auto-falls-back if CUDA missing)")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Use existing preprocessed CSV; skip the slow N4 step")
    parser.add_argument("--skip-synthesis", action="store_true",
                        help="Stop after preprocessing")
    parser.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Per-tile sample averaging for synthesis (1 = no averaging)")
    parser.add_argument("--start-index", type=int, default=1,
                        help="1-based subject row to start synthesis from after filtering "
                             "preprocess_status to ok/skipped_exists. Use 18 to resume "
                             "after rows 1-17 completed.")
    parser.add_argument("--end-index", type=int, default=None,
                        help="Optional 1-based inclusive subject row to stop at after "
                             "filtering preprocess_status to ok/skipped_exists.")
    args = parser.parse_args()

    _setup_logging()

    checkpoint = Path(args.checkpoint).resolve()
    adni_dir = Path(args.adni_dir).resolve()
    work = Path(args.work_dir).resolve()
    repo = Path(args.repo_dir).resolve() if args.repo_dir else Path(__file__).resolve().parent.parent

    if not checkpoint.exists():
        logging.error("Checkpoint not found: %s", checkpoint)
        return 2
    if not adni_dir.exists():
        logging.error("ADNI dir not found: %s", adni_dir)
        return 2
    work.mkdir(parents=True, exist_ok=True)

    cohort_csv = work / "pairs_adni_smoke.csv"
    pre_dir    = work / "adni_preprocessed"
    pre_csv    = work / "pairs_adni_smoke_preprocessed.csv"
    pre_zip    = work / "adni_preprocessed_bundle.zip"
    results    = work / "results_adni_smoke"
    results_zip = work / "adni_smoke_results.zip"

    # ---- Preprocessing stage ----
    if args.skip_preprocess:
        # Stage CSV + index actual NIfTI files across /kaggle/input and /kaggle/working,
        # tolerating .nii vs .nii.gz and rewriting the CSV with real paths.
        try:
            pre_csv = _resolve_skip_preprocess_inputs(work, adni_dir)
        except (FileNotFoundError, RuntimeError) as e:
            logging.error("%s", e)
            return 2
        logging.info("Skipping preprocessing — using %s", pre_csv)
    else:
        adni_src = _find_adni_root(adni_dir)
        adni_local = _stage_inputs(adni_src, work)
        _build_cohort_csv(adni_local, cohort_csv)
        _run_preprocess(repo, cohort_csv, pre_dir, pre_csv, args.device)
        _zip_bundle("preprocessed",
                    [pre_dir, cohort_csv, pre_csv],
                    pre_zip)
        logging.info("==> Download %s before synthesis (resume checkpoint)",
                     pre_zip)

    if args.skip_synthesis:
        logging.info("Skipping synthesis (--skip-synthesis)")
        return 0

    # ---- Synthesis stage ----
    _run_synthesis(repo, checkpoint, pre_csv, results,
                   args.sampler, args.ddim_steps, args.n_samples,
                   args.start_index, args.end_index)
    _zip_bundle("results", [results], results_zip)

    # ---- Summary ----
    logging.info("=" * 60)
    logging.info("Done. Outputs:")
    logging.info("  preprocessed bundle:  %s", pre_zip)
    logging.info("  synthesis results:    %s", results)
    logging.info("  synthesis bundle zip: %s", results_zip)
    logging.info("Download both zips from Kaggle's right-side Output panel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
