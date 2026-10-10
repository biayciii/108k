"""Entrypoint chung cho ba detector qua manifest `thyroid-detection-v1`.

Training requires a prepared pseudonymized manifest, a patient-level split, and explicit
resolved values for the shared data/training settings. See Train/init.md for its schema.

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
        help="Shared thyroid-detection-v1 manifest produced from the canonical T1/T2 split",
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
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--init-weights", type=Path, default=None,
                   help="Optional initialization checkpoint; a fresh optimizer is used")
    p.add_argument("--exploratory", action="store_true",
                   help="Permit proxy patient groups while developing; not for final evaluation")
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
    import torch
    cfg["device"] = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    cfg["num_workers"] = args.num_workers
    cfg["initial_weights"] = str(args.init_weights.resolve()) if args.init_weights else None
    cfg.setdefault("data", {})["manifest_path"] = str(args.split_file.resolve())
    cfg["data"]["exploratory"] = args.exploratory

    required = {
        "training.epochs": cfg["training"].get("epochs"),
        "training.batch_size": cfg["training"].get("batch_size"),
        "training.lr": cfg["training"].get("lr"),
        "data.image_size": cfg["data"].get("image_size"),
        "data.brightness_level": cfg["data"].get("brightness_level"),
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError("Set these unresolved config values before training: " + ", ".join(missing))
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, args.output_dir / "resolved_config.yaml")  # nguyên tắc #4
    print(f"[train] Đã ghi resolved config: {args.output_dir / 'resolved_config.yaml'}")

    split = load_canonical_split(args.split_file, exploratory=args.exploratory)
    train_set = build_dataset(args.data_root, split.train, cfg)
    val_set = build_dataset(args.data_root, split.val, cfg)

    trainer = build_trainer(args.arch, cfg, resume=args.resume)  # nguyên tắc #1
    trainer.fit(train_set, val_set, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
