"""Load + merge YAML config, override từ CLI, và LUÔN ghi lại resolved config ra run dir.

Tự viết nhỏ gọn thay vì phụ thuộc OmegaConf/Hydra (xem Train/init.md mục 4 — quyết định
"chưa làm", ưu tiên ít dependency khi chỉ cần merge 2-3 cấp). Nguyên tắc thiết kế #4 trong
Train/init.md: mọi lần train phải ghi lại full config đã dùng, khắc phục đúng lỗi "Faster-RCNN
không có cấu hình huấn luyện nào được lưu lại" (proposal.md mục 2.4).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dict truy cập được bằng thuộc tính (cfg.seed thay vì cfg["seed"])."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def merge(self, other: "Config") -> "Config":
        """Merge đệ quy: key trùng ở `other` ghi đè `self`; dict con được merge lồng nhau."""
        merged = Config(copy.deepcopy(dict(self)))
        for key, value in other.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = Config(merged[key]).merge(Config(value))
            else:
                merged[key] = value
        return merged


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy config {path}. Xem Train/init.md mục 1 — cần "
            f"configs/base.yaml và configs/<arch>.yaml trước khi train."
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config(data)


def override_from_cli(cfg: Config, args: Any) -> Config:
    """Override các key cấp cao (training.*, data.brightness_level) bằng CLI arg tương ứng nếu
    arg đó không phải None. Nguyên tắc thiết kế #6 (Train/init.md): brightness_level là
    control-variable dùng chung — override có chủ đích, không phải mặc định ngầm khác nhau
    giữa các model."""
    merged = Config(copy.deepcopy(dict(cfg)))
    if getattr(args, "epochs", None) is not None:
        merged.setdefault("training", Config())["epochs"] = args.epochs
    if getattr(args, "batch_size", None) is not None:
        merged.setdefault("training", Config())["batch_size"] = args.batch_size
    if getattr(args, "lr", None) is not None:
        merged.setdefault("training", Config())["lr"] = args.lr
    if getattr(args, "brightness_level", None) is not None:
        merged.setdefault("data", Config())["brightness_level"] = args.brightness_level
    return merged


def _to_plain(value: Any) -> Any:
    """Chuyển `Config` (dict subclass) lồng nhau về `dict`/`list` thuần — PyYAML's SafeDumper
    tra representer theo type() chính xác, không nhận diện subclass của dict."""
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    return value


def save_resolved_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_to_plain(cfg), f, allow_unicode=True, sort_keys=False)
