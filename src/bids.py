"""
BIDS dataset discovery and pairing utilities for 3T/7T MRI.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


logger = logging.getLogger(__name__)

_SUBJECT_RE = re.compile(r"sub-[a-zA-Z0-9]+")
_SESSION_RE = re.compile(r"ses-[a-zA-Z0-9]+")


@dataclass(frozen=True)
class BIDSFile:
    """Representation of a discovered BIDS NIfTI file."""
    path: Path
    subject: str
    session: str
    modality: str
    field_strength: Optional[str]
    json_path: Optional[Path]
    aligned: bool
    metadata: Dict[str, object]


def _find_entity(parts: Iterable[str], pattern: re.Pattern) -> Optional[str]:
    for part in parts:
        match = pattern.search(part)
        if match:
            return match.group(0)
    return None


def _detect_modality(filename: str, modalities: Iterable[str]) -> Optional[str]:
    for modality in modalities:
        if f"_{modality}" in filename:
            return modality
    return None


def _sidecar_json(path: Path) -> Optional[Path]:
    if path.suffix == ".gz" and path.name.endswith(".nii.gz"):
        return path.with_suffix("").with_suffix(".json")
    if path.suffix == ".nii":
        return path.with_suffix(".json")
    return None


def _read_json_metadata(json_path: Optional[Path]) -> Dict[str, object]:
    if json_path is None or not json_path.exists():
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        logger.warning("Failed to read JSON metadata: %s (%s)", json_path, exc)
        return {}


def _infer_field_strength(
    session: str,
    metadata: Dict[str, object],
    session_3t: str,
    session_7t: str,
) -> Optional[str]:
    strength = metadata.get("MagneticFieldStrength")
    if isinstance(strength, (int, float)):
        if strength >= 6:
            return "7T"
        if strength >= 2:
            return "3T"
    if session == session_3t:
        return "3T"
    if session == session_7t:
        return "7T"
    return None


def discover_bids_files(
    data_root: Path,
    modalities: Iterable[str],
    session_3t: str = "ses-1",
    session_7t: str = "ses-2",
    file_pattern: Optional[str] = None,
    prefer_aligned: bool = True,
    require_aligned: bool = False,
    aligned_keywords: Iterable[str] = ("aligned",),
    include_derivatives: bool = False,
) -> List[BIDSFile]:
    """
    Discover BIDS NIfTI files and return a de-duplicated list.

    Args:
        data_root: Root of the BIDS dataset.
        modalities: Modalities to include (e.g., T1w, T2w).
        session_3t: Session label for 3T scans.
        session_7t: Session label for 7T scans.
        file_pattern: Optional glob pattern (relative to data_root).
        prefer_aligned: Prefer files with aligned keywords when duplicates exist.
        require_aligned: If True, only include aligned files.
        aligned_keywords: Keywords that indicate aligned data in the path.
    """
    data_root = Path(data_root)
    if file_pattern:
        candidates = data_root.rglob(file_pattern)
    else:
        candidates = data_root.rglob("*.nii*")

    raw_entries: List[BIDSFile] = []
    aligned_keywords = tuple(k.lower() for k in aligned_keywords)

    for path in candidates:
        if not path.is_file():
            continue

        if not include_derivatives:
            derivatives_root = data_root / "derivatives"
            if derivatives_root.exists():
                try:
                    path.relative_to(derivatives_root)
                    continue
                except ValueError:
                    pass

        name_lower = path.name.lower()
        if not (name_lower.endswith(".nii") or name_lower.endswith(".nii.gz")):
            continue

        if any(tag in name_lower for tag in ("brainmask", "_mask", "_seg", "_label")):
            continue

        modality = _detect_modality(path.name, modalities)
        if modality is None:
            continue

        subject = _find_entity(path.parts, _SUBJECT_RE)
        session = _find_entity(path.parts, _SESSION_RE)
        # Try to infer session if missing
        if session is None:
             # If we can't determine session from path, we need to check metadata or assume based on file structure
             # For now, let's look at field strength first if possible
             json_path_temp = _sidecar_json(path)
             metadata_temp = _read_json_metadata(json_path_temp)
             field_strength_temp = _infer_field_strength(
                 session="", # Unknown yet
                 metadata=metadata_temp,
                 session_3t=session_3t,
                 session_7t=session_7t,
             )
             
             if field_strength_temp == "3T":
                 session = session_3t
             elif field_strength_temp == "7T":
                 session = session_7t
             
             # If still none, and we are in a non-strict mode, maybe we can assume 3T/ses-1 as default?
             # Or just skip if we really can't tell.
             if session is None:
                 # Last ditch: check if 'ses-' is in any parent folder even if regex missed it (unlikely with regex)
                 # Or just assign default 3T session if it looks like a subject folder
                 session = session_3t 

        if subject is None:
             continue

        json_path = _sidecar_json(path)
        metadata = _read_json_metadata(json_path)

        field_strength = _infer_field_strength(
            session=session,
            metadata=metadata,
            session_3t=session_3t,
            session_7t=session_7t,
        )

        aligned = any(keyword in part.lower() for keyword in aligned_keywords for part in path.parts)
        if require_aligned and not aligned:
            continue

        raw_entries.append(
            BIDSFile(
                path=path,
                subject=subject,
                session=session,
                modality=modality,
                field_strength=field_strength,
                json_path=json_path if json_path and json_path.exists() else None,
                aligned=aligned,
                metadata=metadata,
            )
        )

    deduped: Dict[Tuple[str, str, str], BIDSFile] = {}
    for entry in raw_entries:
        key = (entry.subject, entry.session, entry.modality)
        if key not in deduped:
            deduped[key] = entry
            continue
        current = deduped[key]
        if prefer_aligned and entry.aligned and not current.aligned:
            deduped[key] = entry

    return list(deduped.values())


def create_3t_7t_pairs(
    bids_files: Iterable[BIDSFile],
    modality: str = "T1w",
    session_3t: str = "ses-1",
    session_7t: str = "ses-2",
) -> List[Dict[str, Path]]:
    """
    Create paired 3T/7T entries per subject for a specific modality.
    """
    by_subject: Dict[str, Dict[str, BIDSFile]] = {}
    for entry in bids_files:
        if entry.modality != modality:
            continue
        if entry.field_strength == "3T" or entry.session == session_3t:
            by_subject.setdefault(entry.subject, {})["3T"] = entry
        elif entry.field_strength == "7T" or entry.session == session_7t:
            by_subject.setdefault(entry.subject, {})["7T"] = entry

    pairs: List[Dict[str, Path]] = []
    for subject, group in by_subject.items():
        if "3T" in group and "7T" in group:
            pairs.append(
                {
                    "subject": subject,
                    "modality": modality,
                    "input_3t": group["3T"].path,
                    "target_7t": group["7T"].path,
                }
            )
        else:
            logger.warning("Missing 3T/7T pair for subject %s (%s)", subject, modality)

    return pairs
