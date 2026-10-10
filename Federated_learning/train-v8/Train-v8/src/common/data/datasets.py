"""Dataset factory for the pseudonymized, shared `thyroid-detection-v1` manifest."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def build_dataset(data_root: str | Path, ids: list[str], cfg: Any):
    from .manifest import ManifestDataset, read_manifest

    split_path = Path(cfg["data"]["manifest_path"])
    document, _ = read_manifest(
        split_path, data_root, exploratory=bool(cfg["data"].get("exploratory", False))
    )
    requested = set(ids)
    rows = [row for row in document["records"] if row["id"] in requested]
    if len(rows) != len(requested):
        raise ValueError("The manifest does not contain every requested split ID")
    return ManifestDataset(rows, data_root, cfg)
