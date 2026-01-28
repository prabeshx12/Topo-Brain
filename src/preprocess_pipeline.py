"""
Modular, CLI-driven preprocessing pipeline for BIDS 3T/7T MRI datasets.
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
from tqdm import tqdm

from .bids import BIDSFile, discover_bids_files
from .preprocessing import N4BiasFieldCorrection, IntensityNormalization


logger = logging.getLogger(__name__)


@dataclass
class SkullStripConfig:
    method: str = "hd-bet"
    device: str = "cuda"
    mode: str = "accurate"
    bet_threshold: float = 0.5
    use_existing_masks: bool = True
    mask_globs: Tuple[str, ...] = (
        "{image_dir}/*brain_mask.nii.gz",
        "{image_dir}/*brainmask.nii.gz",
        "{image_dir}/*mask.nii.gz",
        "{data_root}/derivatives/**/{subject}/**/brainmask.nii*",
        "{data_root}/derivatives/**/{subject}/**/brainmask.mgz",
    )


@dataclass
class BiasCorrectionConfig:
    enabled: bool = True
    n4_iterations: int = 50
    n4_convergence_threshold: float = 0.001


@dataclass
class NormalizationConfig:
    method: str = "zscore"
    percentile_lower: float = 1.0
    percentile_upper: float = 99.0
    clip_lower_percentile: float = 0.5
    clip_upper_percentile: float = 99.5


@dataclass
class ResampleConfig:
    target_spacing: Optional[Tuple[float, float, float]] = None
    interpolation: str = "linear"


@dataclass
class QCConfig:
    enabled: bool = True
    max_samples: int = 5
    output_dir: Optional[Path] = None


@dataclass
class PipelineConfig:
    data_root: Path
    output_root: Path
    modalities: Tuple[str, ...] = ("T1w", "T2w")
    session_3t: str = "ses-1"
    session_7t: str = "ses-2"
    prefer_aligned: bool = True
    require_aligned: bool = False
    aligned_keywords: Tuple[str, ...] = ("aligned",)
    include_derivatives: bool = False
    output_suffix: str = "desc-preproc"
    overwrite: bool = False
    resume: bool = True
    manifest_path: Optional[Path] = None
    pairs_manifest_path: Optional[Path] = None
    seed: int = 42
    skull_strip: SkullStripConfig = field(default_factory=SkullStripConfig)
    bias_correction: BiasCorrectionConfig = field(default_factory=BiasCorrectionConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    resample: ResampleConfig = field(default_factory=ResampleConfig)
    qc: QCConfig = field(default_factory=QCConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PipelineConfig":
        skull_strip = SkullStripConfig(**data.get("skull_strip", {}))
        bias_correction = BiasCorrectionConfig(**data.get("bias_correction", {}))
        normalization = NormalizationConfig(**data.get("normalization", {}))
        resample = ResampleConfig(**data.get("resample", {}))
        qc = QCConfig(**data.get("qc", {}))

        config = cls(
            data_root=Path(data.get("data_root", "data")),
            output_root=Path(data.get("output_root", "derivatives/topobrain-preproc")),
            modalities=tuple(data.get("modalities", ("T1w", "T2w"))),
            session_3t=data.get("session_3t", "ses-1"),
            session_7t=data.get("session_7t", "ses-2"),
            prefer_aligned=data.get("prefer_aligned", True),
            require_aligned=data.get("require_aligned", False),
            aligned_keywords=tuple(data.get("aligned_keywords", ("aligned",))),
            include_derivatives=data.get("include_derivatives", False),
            output_suffix=data.get("output_suffix", "desc-preproc"),
            overwrite=data.get("overwrite", False),
            resume=data.get("resume", True),
            manifest_path=Path(data["manifest_path"]) if data.get("manifest_path") else None,
            pairs_manifest_path=Path(data["pairs_manifest_path"]) if data.get("pairs_manifest_path") else None,
            seed=int(data.get("seed", 42)),
            skull_strip=skull_strip,
            bias_correction=bias_correction,
            normalization=normalization,
            resample=resample,
            qc=qc,
        )
        return config


class BIDSPreprocessingPipeline:
    """End-to-end preprocessing pipeline aligned with the research blueprint."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._configure_paths()
        self._configure_seed()
        self._init_components()

    def _configure_paths(self) -> None:
        self.config.data_root = self.config.data_root.expanduser().resolve()
        if not self.config.output_root.is_absolute():
            self.config.output_root = (self.config.data_root / self.config.output_root).resolve()

        if self.config.manifest_path is None:
            self.config.manifest_path = self.config.output_root / "manifest.csv"
        if self.config.pairs_manifest_path is None:
            self.config.pairs_manifest_path = self.config.output_root / "pairs.csv"

        if self.config.qc.output_dir is None:
            self.config.qc.output_dir = self.config.output_root / "qc"

        self.config.output_root.mkdir(parents=True, exist_ok=True)
        if self.config.qc.output_dir:
            self.config.qc.output_dir.mkdir(parents=True, exist_ok=True)

    def _configure_seed(self) -> None:
        np.random.seed(self.config.seed)

    def _init_components(self) -> None:
        self.bias_corrector = None
        if self.config.bias_correction.enabled:
            self.bias_corrector = N4BiasFieldCorrection(
                num_iterations=self.config.bias_correction.n4_iterations,
                convergence_threshold=self.config.bias_correction.n4_convergence_threshold,
            )

        self.normalizer = IntensityNormalization(
            method=self.config.normalization.method,
            percentile_lower=self.config.normalization.percentile_lower,
            percentile_upper=self.config.normalization.percentile_upper,
            clip_lower=self.config.normalization.clip_lower_percentile,
            clip_upper=self.config.normalization.clip_upper_percentile,
        )

        self._hdbet_predictor = None
        self._hdbet_device = None

    def run(self) -> List[Dict[str, object]]:
        logger.info("Discovering dataset in %s", self.config.data_root)
        bids_files = discover_bids_files(
            data_root=self.config.data_root,
            modalities=self.config.modalities,
            session_3t=self.config.session_3t,
            session_7t=self.config.session_7t,
            prefer_aligned=self.config.prefer_aligned,
            require_aligned=self.config.require_aligned,
            aligned_keywords=self.config.aligned_keywords,
            include_derivatives=self.config.include_derivatives,
        )

        if not bids_files:
            logger.error("No BIDS files found under %s", self.config.data_root)
            return []

        bids_files = sorted(bids_files, key=lambda x: (x.subject, x.session, x.modality))

        manifest: List[Dict[str, object]] = []
        qc_remaining = self.config.qc.max_samples if self.config.qc.enabled else 0

        for entry in tqdm(bids_files, desc="Preprocessing volumes"):
            output_path = self._build_output_path(entry, self.config.output_suffix)
            mask_output_path = self._build_output_path(entry, "desc-brainmask")

            if output_path.exists() and self.config.resume and not self.config.overwrite:
                logger.info("Skipping %s (already exists)", entry.path.name)
                manifest.append(self._manifest_entry(entry, output_path, mask_output_path, status="skipped"))
                continue

            try:
                mask_path = self._resolve_mask(entry)
                if mask_path is None and self.config.skull_strip.method != "existing":
                    mask_path = self._generate_mask(entry, mask_output_path)
                if mask_path is None:
                    raise RuntimeError("No brain mask available for skull stripping")

                image = nib.load(str(entry.path))
                image_array = image.get_fdata().astype(np.float32)
                affine = image.affine

                mask_array = self._load_mask(mask_path, image.shape)

                mask_for_n4 = mask_path
                if mask_path.suffix.lower() == ".mgz":
                    nib.save(nib.Nifti1Image(mask_array.astype(np.uint8), affine), str(mask_output_path))
                    mask_for_n4 = mask_output_path

                if self.bias_corrector is not None:
                    image_array = self.bias_corrector(entry.path, mask_for_n4)

                # NOTE: Do NOT apply mask before normalization - normalization handles background
                # image_array = self._apply_mask(image_array, mask_array)

                if self.config.resample.target_spacing is not None:
                    image_array, affine, mask_array = self._resample_image_and_mask(
                        image_array,
                        mask_array,
                        affine,
                        self.config.resample.target_spacing,
                        self.config.resample.interpolation,
                    )

                # Normalization now handles setting background to -1.0
                image_array = self.normalizer(image_array, mask_array)

                output_path.parent.mkdir(parents=True, exist_ok=True)
                nib.save(nib.Nifti1Image(image_array, affine), str(output_path))

                if mask_array is not None:
                    nib.save(nib.Nifti1Image(mask_array.astype(np.uint8), affine), str(mask_output_path))

                stats = self._compute_statistics(image_array)
                self._save_metadata(entry, output_path, mask_path, stats)

                if qc_remaining > 0 and mask_array is not None and self.config.qc.output_dir:
                    self._save_mask_overlay(entry, image_array, mask_array, self.config.qc.output_dir)
                    qc_remaining -= 1

                manifest.append(
                    self._manifest_entry(
                        entry,
                        output_path,
                        mask_output_path,
                        status="ok",
                        stats=stats,
                        mask_source=str(mask_path) if mask_path else None,
                    )
                )
            except Exception as exc:
                logger.error("Failed to preprocess %s: %s", entry.path, exc)
                manifest.append(
                    self._manifest_entry(
                        entry,
                        output_path,
                        mask_output_path,
                        status="failed",
                        error=str(exc),
                    )
                )

        self._write_manifest(manifest, self.config.manifest_path)
        self._write_pairs_manifest(manifest, self.config.pairs_manifest_path)
        return manifest

    def _resolve_mask(self, entry: BIDSFile) -> Optional[Path]:
        if not self.config.skull_strip.use_existing_masks:
            return None

        subject_dir = self.config.data_root / entry.subject
        session_dir = subject_dir / entry.session
        image_dir = entry.path.parent

        for pattern in self.config.skull_strip.mask_globs:
            expanded = pattern.format(
                data_root=self.config.data_root,
                subject=entry.subject,
                session=entry.session,
                modality=entry.modality,
                image_dir=image_dir,
                subject_dir=subject_dir,
                session_dir=session_dir,
            )
            from glob import glob

            matches = sorted(Path(p) for p in glob(expanded, recursive=True))
            if matches:
                return matches[0]

        return None

    def _generate_mask(self, entry: BIDSFile, mask_output_path: Path) -> Optional[Path]:
        method = self.config.skull_strip.method
        if method == "hd-bet":
            return self._run_hdbet(entry.path, mask_output_path)
        if method == "synthstrip":
            return self._run_synthstrip(entry.path, mask_output_path)
        if method == "fsl-bet":
            return self._run_fsl_bet(entry.path, mask_output_path)
        return None

    def _run_hdbet(self, image_path: Path, mask_output_path: Path) -> Optional[Path]:
        try:
            import torch
            from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict
            from HD_BET.checkpoint_download import maybe_download_parameters
        except ImportError as exc:
            raise RuntimeError("HD-BET is not installed") from exc

        maybe_download_parameters()

        device = self.config.skull_strip.device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU for HD-BET")
            device = "cpu"

        if self._hdbet_predictor is None or self._hdbet_device != device:
            self._hdbet_predictor = get_hdbet_predictor(
                use_tta=(self.config.skull_strip.mode == "accurate"),
                device=torch.device(device),
                verbose=False,
            )
            self._hdbet_device = device

        output_brain = mask_output_path.with_name(mask_output_path.name.replace("desc-brainmask", "desc-brain"))
        output_brain.parent.mkdir(parents=True, exist_ok=True)

        hdbet_predict(
            str(image_path),
            str(output_brain),
            predictor=self._hdbet_predictor,
            keep_brain_mask=True,
            compute_brain_extracted_image=True,
        )

        bet_mask = output_brain.parent / f"{self._strip_nii_extension(output_brain)}_bet.nii.gz"
        if bet_mask.exists():
            bet_mask.replace(mask_output_path)
            return mask_output_path

        if mask_output_path.exists():
            return mask_output_path

        raise RuntimeError(f"HD-BET mask not found for {image_path}")

    def _run_synthstrip(self, image_path: Path, mask_output_path: Path) -> Optional[Path]:
        import subprocess

        output_brain = mask_output_path.with_name(mask_output_path.name.replace("desc-brainmask", "desc-brain"))
        output_brain.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "mri_synthstrip",
            "-i",
            str(image_path),
            "-o",
            str(output_brain),
            "-m",
            str(mask_output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"SynthStrip failed: {result.stderr.strip()}")
        return mask_output_path

    def _run_fsl_bet(self, image_path: Path, mask_output_path: Path) -> Optional[Path]:
        import subprocess

        output_brain = mask_output_path.with_name(mask_output_path.name.replace("desc-brainmask", "desc-brain"))
        output_brain.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "bet",
            str(image_path),
            str(output_brain),
            "-f",
            str(self.config.skull_strip.bet_threshold),
            "-m",
            "-R",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FSL BET failed: {result.stderr.strip()}")

        expected_mask = output_brain.parent / f"{self._strip_nii_extension(output_brain)}_mask.nii.gz"
        if expected_mask.exists():
            expected_mask.replace(mask_output_path)
            return mask_output_path

        raise RuntimeError(f"FSL BET mask not found for {image_path}")

    def _load_mask(self, mask_path: Path, image_shape: Tuple[int, ...]) -> np.ndarray:
        mask_img = nib.load(str(mask_path))
        mask_array = mask_img.get_fdata().astype(np.float32)
        if mask_array.shape != image_shape:
            logger.warning("Mask shape mismatch for %s", mask_path)
        return (mask_array > 0).astype(np.uint8)

    def _apply_mask(self, image_array: np.ndarray, mask_array: Optional[np.ndarray]) -> np.ndarray:
        if mask_array is None:
            return image_array
        return image_array * (mask_array > 0)

    def _resample_image_and_mask(
        self,
        image_array: np.ndarray,
        mask_array: Optional[np.ndarray],
        affine: np.ndarray,
        target_spacing: Tuple[float, float, float],
        interpolation: str,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        order = 1 if interpolation == "linear" else 0
        original_affine = affine
        image_nii = nib.Nifti1Image(image_array, original_affine)
        resampled_image = resample_to_output(image_nii, voxel_sizes=target_spacing, order=order)
        image_array = resampled_image.get_fdata().astype(np.float32)
        affine = resampled_image.affine

        if mask_array is not None:
            mask_nii = nib.Nifti1Image(mask_array.astype(np.float32), original_affine)
            resampled_mask = resample_to_output(mask_nii, voxel_sizes=target_spacing, order=0)
            mask_array = (resampled_mask.get_fdata() > 0.5).astype(np.uint8)

        return image_array, affine, mask_array

    def _compute_statistics(self, image_array: np.ndarray) -> Dict[str, float]:
        nonzero = image_array[image_array != 0]
        if len(nonzero) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
        return {
            "mean": float(np.mean(nonzero)),
            "std": float(np.std(nonzero)),
            "min": float(np.min(nonzero)),
            "max": float(np.max(nonzero)),
            "median": float(np.median(nonzero)),
            "p01": float(np.percentile(nonzero, 1)),
            "p99": float(np.percentile(nonzero, 99)),
        }

    def _save_metadata(
        self,
        entry: BIDSFile,
        output_path: Path,
        mask_path: Optional[Path],
        stats: Dict[str, float],
    ) -> None:
        metadata = {
            "input_path": str(entry.path),
            "output_path": str(output_path),
            "subject": entry.subject,
            "session": entry.session,
            "modality": entry.modality,
            "field_strength": entry.field_strength,
            "aligned": entry.aligned,
            "mask_path": str(mask_path) if mask_path else None,
            "preprocessing": {
                "skull_stripping": self.config.skull_strip.method,
                "bias_correction": self.config.bias_correction.enabled,
                "normalization": self.config.normalization.method,
                "target_spacing": self.config.resample.target_spacing,
            },
            "statistics": stats,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }

        sidecar = output_path.with_suffix("")
        if sidecar.name.endswith(".nii"):
            sidecar = sidecar.with_suffix(".json")
        else:
            sidecar = output_path.with_suffix(".json")

        with open(sidecar, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    def _save_mask_overlay(
        self,
        entry: BIDSFile,
        image_array: np.ndarray,
        mask_array: np.ndarray,
        output_dir: Path,
    ) -> None:
        import matplotlib.pyplot as plt

        slice_idx = image_array.shape[2] // 2
        image_slice = image_array[:, :, slice_idx].T
        mask_slice = mask_array[:, :, slice_idx].T

        plt.figure(figsize=(6, 6))
        plt.imshow(image_slice, cmap="gray")
        plt.imshow(mask_slice, cmap="autumn", alpha=0.4)
        plt.axis("off")
        plt.title(f"{entry.subject} {entry.session} {entry.modality}")

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{entry.subject}_{entry.session}_{entry.modality}_mask_overlay.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _build_output_path(self, entry: BIDSFile, suffix: str) -> Path:
        output_dir = self.config.output_root / entry.subject / entry.session / "anat"
        output_dir.mkdir(parents=True, exist_ok=True)

        name = entry.path.name
        stem = name.replace(".nii.gz", "").replace(".nii", "")
        for token_to_drop in ("_defaced", "_brainmask", "_brain_mask", "_brain"):
            if token_to_drop in stem:
                stem = stem.replace(token_to_drop, "")
        if "_desc-" in stem:
            stem = stem.split("_desc-")[0] + "_" + stem.split("_desc-")[-1].split("_", 1)[-1]

        token = f"_{entry.modality}"
        if token in stem and suffix not in stem:
            stem = stem.replace(token, f"_{suffix}{token}")
        elif suffix not in stem:
            stem = f"{stem}_{suffix}"

        return output_dir / f"{stem}.nii.gz"

    @staticmethod
    def _strip_nii_extension(path: Path) -> str:
        name = path.name
        if name.endswith(".nii.gz"):
            return name[:-7]
        if name.endswith(".nii"):
            return name[:-4]
        return path.stem

    def _manifest_entry(
        self,
        entry: BIDSFile,
        output_path: Path,
        mask_output_path: Path,
        status: str,
        stats: Optional[Dict[str, float]] = None,
        mask_source: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Dict[str, object]:
        return {
            "subject": entry.subject,
            "session": entry.session,
            "modality": entry.modality,
            "field_strength": entry.field_strength,
            "aligned": entry.aligned,
            "input_path": str(entry.path),
            "output_path": str(output_path),
            "mask_output_path": str(mask_output_path),
            "mask_source": mask_source,
            "status": status,
            "error": error,
            "stats": stats or {},
        }

    def _write_manifest(self, manifest: List[Dict[str, object]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not manifest:
            return

        fieldnames = list(manifest[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest)

        jsonl_path = path.with_suffix(".jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as handle:
            for entry in manifest:
                handle.write(json.dumps(entry) + "\n")

    def _write_pairs_manifest(self, manifest: List[Dict[str, object]], path: Path) -> None:
        by_subject: Dict[str, Dict[str, Dict[str, object]]] = {}
        for entry in manifest:
            if entry.get("status") != "ok":
                continue
            if entry.get("modality") not in self.config.modalities:
                continue
            subject = entry["subject"]
            session = entry["session"]
            field_strength = entry.get("field_strength")
            label = None
            if field_strength == "3T" or session == self.config.session_3t:
                label = "3T"
            elif field_strength == "7T" or session == self.config.session_7t:
                label = "7T"
            if label is None:
                continue
            by_subject.setdefault(subject, {})[f"{label}_{entry['modality']}"] = entry

        pairs: List[Dict[str, object]] = []
        for subject, entries in by_subject.items():
            key_3t = f"3T_{self.config.modalities[0]}"
            key_7t = f"7T_{self.config.modalities[0]}"
            if key_3t in entries and key_7t in entries:
                pair = {
                    "subject": subject,
                    "modality": self.config.modalities[0],
                    "input_3t": entries[key_3t]["output_path"],
                    "target_7t": entries[key_7t]["output_path"],
                }
                t2_key = f"3T_T2w"
                if t2_key in entries:
                    pair["input_3t_t2"] = entries[t2_key]["output_path"]
                pairs.append(pair)
            else:
                logger.warning("Missing 3T/7T pair after preprocessing: %s", subject)

        path.parent.mkdir(parents=True, exist_ok=True)
        if pairs:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(pairs[0].keys()))
                writer.writeheader()
                writer.writerows(pairs)

            jsonl_path = path.with_suffix(".jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as handle:
                for entry in pairs:
                    handle.write(json.dumps(entry) + "\n")
