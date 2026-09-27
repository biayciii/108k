"""Interface `Trainer` chung — mỗi kiến trúc implement 1 subclass. `build_trainer()` là
registry để `scripts/train.py` không cần if/else theo `--arch` (nguyên tắc thiết kế #1,
Train/init.md).

Các subclass dưới đây CHỈ là khung — `.fit()`/`.evaluate()` thật cần `src/models/<arch>` wiring
xong (đang chặn bởi vấn đề packaging nêu trong `src/models/detr/__init__.py`) và
`src/common/data/datasets.py` xong (đang chặn bởi T1-T3 trong plan.csv).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Trainer(ABC):
    def __init__(self, cfg: Any, resume: Path | None = None):
        self.cfg = cfg
        self.resume = resume

    @abstractmethod
    def fit(self, train_set: Any, val_set: Any, output_dir: Path) -> None: ...

    @abstractmethod
    def evaluate(self, dataset: Any, output_dir: Path) -> dict: ...


class DetrTrainer(Trainer):
    def fit(self, train_set: Any, val_set: Any, output_dir: Path) -> None:
        raise NotImplementedError(
            "Chờ src/models/detr wiring xong (xem __init__.py ở đó) + "
            "src/common/data/datasets.py (chờ T1-T3, plan.csv)."
        )

    def evaluate(self, dataset: Any, output_dir: Path) -> dict:
        raise NotImplementedError("Chờ common/metrics/detection.py (mAP) chốt công thức chung.")


class FasterRcnnTrainer(Trainer):
    def fit(self, train_set: Any, val_set: Any, output_dir: Path) -> None:
        raise NotImplementedError(
            "Chờ src/models/faster_rcnn wiring xong + cấu hình huấn luyện T5 (plan.csv) — "
            "Faster-RCNN hiện chưa có epoch/batch/lr nào được xác nhận."
        )

    def evaluate(self, dataset: Any, output_dir: Path) -> dict:
        raise NotImplementedError("Chờ common/metrics/detection.py (mAP) chốt công thức chung.")


class Yolov7Trainer(Trainer):
    def fit(self, train_set: Any, val_set: Any, output_dir: Path) -> None:
        raise NotImplementedError(
            "Chờ src/models/yolov7 wiring xong (bỏ sys.path hack, xem __init__.py ở đó)."
        )

    def evaluate(self, dataset: Any, output_dir: Path) -> dict:
        raise NotImplementedError("Chờ common/metrics/detection.py (mAP) chốt công thức chung.")


_REGISTRY: dict[str, type[Trainer]] = {
    "detr": DetrTrainer,
    "faster_rcnn": FasterRcnnTrainer,
    "yolov7": Yolov7Trainer,
}


def build_trainer(arch: str, cfg: Any, resume: Path | None = None) -> Trainer:
    if arch not in _REGISTRY:
        raise ValueError(f"Kiến trúc không hỗ trợ: {arch}. Chọn 1 trong {list(_REGISTRY)}")
    return _REGISTRY[arch](cfg, resume=resume)
