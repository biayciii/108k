"""Entrypoint huấn luyện duy nhất cho cả 3 kiến trúc — nguyên tắc thiết kế #1 (Train/init.md).

Chạy được ở MỨC KHUNG ngay bây giờ: parse args, merge config, ghi resolved config ra
output-dir. Sẽ dừng có chủ đích ở bước load split/dataset/trainer thật vì những phần đó còn
NotImplementedError (chờ T1-T3, T4-T6 trong plan.csv — xem thông báo lỗi tương ứng).

Ví dụ chạy:
    python scripts/train.py --arch yolov7 \\
        --data-root ../Data/yolov7-data \\
        --split-file ../Data/canonical_split.json \\
        --output-dir runs/yolov7_run01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Cho phép chạy script này từ bất kỳ đâu (không chỉ khi cwd == Train/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.data.split import load_canonical_split  # noqa: E402
from src.common.data.datasets import build_dataset  # noqa: E402
from src.common.utils.config import (  # noqa: E402
    load_config,
    override_from_cli,
    save_resolved_config,
)
from src.common.utils.seed import set_seed  # noqa: E402
from src.engine.trainer import build_trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", required=True, choices=["detr", "faster_rcnn", "yolov7"])
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Override YAML riêng kiến trúc; mặc định configs/<arch>.yaml",
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument(
        "--split-file",
        type=Path,
        required=True,
        help="Split canonical từ T1/T2 (plan.csv), dùng chung cho cả 3 kiến trúc",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=None, help="Override config nếu có")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument(
        "--brightness-level",
        type=int,
        default=None,
        help="Control-variable dùng chung — KHÔNG để mặc định khác nhau giữa model",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    root = Path(__file__).resolve().parents[1]
    base_cfg = load_config(root / "configs" / "base.yaml")
    arch_cfg = load_config(args.config or root / "configs" / f"{args.arch}.yaml")
    cfg = base_cfg.merge(arch_cfg)
    cfg = override_from_cli(cfg, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, args.output_dir / "resolved_config.yaml")  # nguyên tắc #4
    print(f"[train] Đã ghi resolved config: {args.output_dir / 'resolved_config.yaml'}")

    split = load_canonical_split(args.split_file)  # nguyên tắc #5 — NotImplementedError hiện tại
    train_set = build_dataset(args.data_root, split.train, cfg)
    val_set = build_dataset(args.data_root, split.val, cfg)

    trainer = build_trainer(args.arch, cfg, resume=args.resume)  # nguyên tắc #1
    trainer.fit(train_set, val_set, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
