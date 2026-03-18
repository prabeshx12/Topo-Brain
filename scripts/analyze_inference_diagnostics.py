"""
Analyze ADNI inference diagnostics and generate a CSV risk report.

Inputs supported:
1) Root output directory from infer_adni_batch.py (contains inference_summary.json)
2) A direct path to inference_summary.json

Outputs:
- diagnostics_risk_report.csv
- diagnostics_risk_report.json

Usage:
    python scripts/analyze_inference_diagnostics.py \
      --input /path/to/adni_inference_results

    python scripts/analyze_inference_diagnostics.py \
      --input /path/to/adni_inference_results/inference_summary.json \
      --output /path/to/reports
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _find_summary_path(input_path: Path) -> Path:
    if input_path.is_file() and input_path.name == "inference_summary.json":
        return input_path

    if input_path.is_dir():
        candidate = input_path / "inference_summary.json"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not locate inference_summary.json from input: {input_path}"
    )


def _float_tuple(values) -> Tuple[float, float, float]:
    if not values or len(values) < 3:
        return (0.0, 0.0, 0.0)
    return (float(values[0]), float(values[1]), float(values[2]))


def assess_case(case: Dict) -> Dict:
    """Assign risk score/level and concrete warnings per case."""
    subject = case.get("subject", "unknown")
    session = case.get("session", "unknown")
    status = case.get("status", "failed")

    warnings: List[str] = []
    risk_score = 0

    if status != "ok":
        warnings.append("inference_failed")
        risk_score += 100
        return {
            "subject": subject,
            "session": session,
            "status": status,
            "risk_score": risk_score,
            "risk_level": "critical",
            "warnings": ";".join(warnings),
            "shape": "",
            "voxel_size": "",
            "input_range": "",
            "range_ok": False,
            "recommended_action": "inspect logs and rerun subject",
            "error": case.get("error", ""),
        }

    shape = case.get("shape", [])
    voxel_size = case.get("voxel_size", [])
    input_range = case.get("input_range", [])
    range_ok = bool(case.get("range_ok", False))

    # 1) Intensity range compatibility with expected [-1, 1]
    if not range_ok:
        warnings.append("intensity_not_in_expected_range")
        risk_score += 35

    if input_range and len(input_range) == 2:
        vmin, vmax = float(input_range[0]), float(input_range[1])
        if vmin < -2.0 or vmax > 2.0:
            warnings.append("extreme_intensity_range")
            risk_score += 25

    # 2) Shape compatibility with patch inference assumptions
    if shape and len(shape) == 3:
        d, h, w = int(shape[0]), int(shape[1]), int(shape[2])
        min_dim = min(d, h, w)
        if min_dim < 64:
            warnings.append("small_dimension_below_patch_size")
            risk_score += 40
        elif min_dim < 96:
            warnings.append("small_dimension_low_context")
            risk_score += 12

    # 3) Voxel anisotropy risk
    vx, vy, vz = _float_tuple(voxel_size)
    voxel_vals = [v for v in (vx, vy, vz) if v > 0]
    if voxel_vals:
        anisotropy_ratio = max(voxel_vals) / min(voxel_vals)
        if anisotropy_ratio > 2.0:
            warnings.append("high_voxel_anisotropy")
            risk_score += 22
        elif anisotropy_ratio > 1.5:
            warnings.append("moderate_voxel_anisotropy")
            risk_score += 10

    # 4) Conservative severity bins
    if risk_score >= 50:
        level = "high"
        action = "manual QC before using in downstream analysis"
    elif risk_score >= 20:
        level = "medium"
        action = "review visually and monitor morphology drift"
    else:
        level = "low"
        action = "acceptable for routine batch inference"

    return {
        "subject": subject,
        "session": session,
        "status": status,
        "risk_score": risk_score,
        "risk_level": level,
        "warnings": ";".join(warnings) if warnings else "none",
        "shape": "x".join(str(x) for x in shape) if shape else "",
        "voxel_size": "x".join(f"{float(v):.4f}" for v in voxel_size) if voxel_size else "",
        "input_range": f"[{float(input_range[0]):.4f}, {float(input_range[1]):.4f}]" if input_range else "",
        "range_ok": range_ok,
        "recommended_action": action,
        "error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate risk report from ADNI inference diagnostics")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to inference output root or inference_summary.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output folder for reports (default: alongside summary)",
    )
    args = parser.parse_args()

    summary_path = _find_summary_path(Path(args.input))

    with open(summary_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if not isinstance(cases, list):
        raise ValueError(f"Expected a list in {summary_path}")

    assessed = [assess_case(c) for c in cases]
    assessed.sort(key=lambda x: x["risk_score"], reverse=True)

    output_dir = Path(args.output) if args.output else summary_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "diagnostics_risk_report.csv"
    json_path = output_dir / "diagnostics_risk_report.json"

    fieldnames = [
        "subject",
        "session",
        "status",
        "risk_score",
        "risk_level",
        "warnings",
        "shape",
        "voxel_size",
        "input_range",
        "range_ok",
        "recommended_action",
        "error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(assessed)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(assessed, f, indent=2)

    total = len(assessed)
    critical = sum(1 for x in assessed if x["risk_level"] == "critical")
    high = sum(1 for x in assessed if x["risk_level"] == "high")
    medium = sum(1 for x in assessed if x["risk_level"] == "medium")
    low = sum(1 for x in assessed if x["risk_level"] == "low")

    print("Risk analysis complete")
    print(f"Summary source: {summary_path}")
    print(f"CSV report: {csv_path}")
    print(f"JSON report: {json_path}")
    print(f"Cases: total={total}, critical={critical}, high={high}, medium={medium}, low={low}")


if __name__ == "__main__":
    main()
