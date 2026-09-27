"""Entrypoint đánh giá — load resolved_config.yaml + checkpoint từ một run dir đã train,
chạy `Trainer.evaluate()` trên tập test.

CHƯA implement được đầy đủ: phụ thuộc `src/common/metrics/detection.py` (mAP, chưa chốt công
thức chung) và `src/engine/trainer.py.evaluate()` (chưa wiring model thật). Giữ CLI parse ở
đây trước để `scripts/train.py`/`scripts/evaluate.py` có cùng quy ước tham số ngay từ đầu.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.data.datasets import build_dataset  # noqa: E402
from src.common.data.split import load_canonical_split  # noqa: E402
from src.common.utils.config import load_config  # noqa: E402
from src.engine.trainer import build_trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", required=True, choices=["detr", "faster_rcnn", "yolov7"])
    p.add_argument("--run-dir", type=Path, required=True, help="Thư mục output của lần train")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None, help="Mặc định: <run-dir>/best.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.run_dir / "resolved_config.yaml")

    split = load_canonical_split(args.split_file)
    test_set = build_dataset(args.data_root, split.test, cfg)

    checkpoint = args.checkpoint or (args.run_dir / "best.pt")
    trainer = build_trainer(args.arch, cfg, resume=checkpoint)
    metrics = trainer.evaluate(test_set, output_dir=args.run_dir)
    print(metrics)


if __name__ == "__main__":
    main()
