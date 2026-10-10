"""Read the shared patient-level manifest produced by T1/T2."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class CanonicalSplit:
    train: list[str]
    val: list[str]
    test: list[str]
    manifest_path: Path
    exploratory: bool = False


def load_canonical_split(path: str | Path, exploratory: bool = False) -> CanonicalSplit:
    """The split file is a `thyroid-detection-v1` JSON manifest.

    Exploratory mode allows name-derived proxy groups during development. It always enforces
    that a group occurs in exactly one split; it does not make those groups verified patients.
    """
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if document.get("schema") != "thyroid-detection-v1":
        raise ValueError("Expected manifest schema thyroid-detection-v1")
    if document.get("patient_identity_verified") is not True and not exploratory:
        raise ValueError("Patient identities are unverified; pass --exploratory for development only")
    rows = document.get("records") or []
    by_split = {name: [] for name in ("train", "val", "test")}
    seen_ids, seen_groups = set(), {}
    for row in rows:
        image_id, group, split = row.get("id"), row.get("patient_id"), row.get("split")
        if not isinstance(image_id, str) or not image_id or image_id in seen_ids:
            raise ValueError("Manifest image IDs must be unique nonempty strings")
        if not isinstance(group, str) or not group:
            raise ValueError("Each record requires a pseudonymized patient_id")
        if split not in by_split:
            raise ValueError(f"Unknown split value: {split!r}")
        if group in seen_groups and seen_groups[group] != split:
            raise ValueError("The same patient group appears in multiple splits")
        seen_ids.add(image_id)
        seen_groups[group] = split
        by_split[split].append(image_id)
    if not all(by_split.values()):
        raise ValueError("Train, val, and test splits must all contain records")
    return CanonicalSplit(**by_split, manifest_path=path, exploratory=exploratory)
