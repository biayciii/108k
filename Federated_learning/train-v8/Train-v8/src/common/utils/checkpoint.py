"""Lưu/khôi phục checkpoint — interface chung dùng cho cả 3 kiến trúc.

Mỗi Trainer cụ thể (`src/engine/trainer.py`) tự quyết định `state` chứa gì (model
state_dict, optimizer, epoch, config...); module này chỉ lo phần I/O.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    import torch

    try:
        return torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        return torch.load(Path(path), map_location=map_location)
