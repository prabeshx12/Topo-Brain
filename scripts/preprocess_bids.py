"""
CLI entry point for the BIDS preprocessing pipeline.
"""
import argparse
import logging
from pathlib import Path
from typing import Dict

import yaml

from src.preprocess_pipeline import BIDSPreprocessingPipeline, PipelineConfig


def _load_config(path: Path) -> Dict[str, object]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
            return data if isinstance(data, dict) else {}
    return {}


def _apply_overrides(config: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    if args.data_root:
        config["data_root"] = args.data_root
    if args.output_root:
        config["output_root"] = args.output_root
    if args.overwrite:
        config["overwrite"] = True
    if args.no_resume:
        config["resume"] = False

    if args.skull_strip_method:
        config.setdefault("skull_strip", {})["method"] = args.skull_strip_method
    if args.skull_strip_device:
        config.setdefault("skull_strip", {})["device"] = args.skull_strip_device
    if args.skull_strip_mode:
        config.setdefault("skull_strip", {})["mode"] = args.skull_strip_mode

    if args.no_bias_correction:
        config.setdefault("bias_correction", {})["enabled"] = False

    if args.target_spacing:
        config.setdefault("resample", {})["target_spacing"] = args.target_spacing
    if args.include_derivatives:
        config["include_derivatives"] = True

    return config


def _setup_logging(output_root: Path) -> None:
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "preprocess.log"),
            logging.StreamHandler(),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="BIDS preprocessing pipeline")
    parser.add_argument("--config", type=str, default="configs/preprocess.yaml", help="Path to YAML config")
    parser.add_argument("--data-root", type=str, help="BIDS dataset root")
    parser.add_argument("--output-root", type=str, help="Output root for preprocessed data")
    parser.add_argument("--skull-strip-method", type=str, choices=["hd-bet", "synthstrip", "fsl-bet", "existing"])
    parser.add_argument("--skull-strip-device", type=str, choices=["cuda", "cpu"])
    parser.add_argument("--skull-strip-mode", type=str, choices=["accurate", "fast"])
    parser.add_argument("--target-spacing", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--no-bias-correction", action="store_true", help="Disable N4 bias correction")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip existing outputs")
    parser.add_argument("--include-derivatives", action="store_true", help="Include derivatives in discovery")

    args = parser.parse_args()

    config_path = Path(args.config)
    config_data = _load_config(config_path)
    config_data = _apply_overrides(config_data, args)

    pipeline_config = PipelineConfig.from_dict(config_data)
    pipeline = BIDSPreprocessingPipeline(pipeline_config)
    _setup_logging(pipeline.config.output_root)

    logging.info("Running preprocessing with config: %s", pipeline.config)
    pipeline.run()


if __name__ == "__main__":
    main()
