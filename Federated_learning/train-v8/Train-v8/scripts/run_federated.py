"""Run patient-preserving FedAvg or FedProx simulation from a shared manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.data.manifest import ManifestDataset, read_manifest
from src.common.data.partition import partition_by_gap
from src.common.data.split import load_canonical_split
from src.common.utils.checkpoint import load_checkpoint, save_checkpoint
from src.common.utils.config import load_config, override_from_cli, save_resolved_config
from src.common.utils.seed import set_seed
from src.engine.federated import run_federated_round
from src.engine.trainer import build_trainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=("detr", "faster_rcnn", "yolov7"))
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Shared patient-level thyroid-detection-v1 manifest")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("fedavg", "fedprox"), default="fedavg")
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--local-epochs", type=int, required=True)
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--covariate", default="gap")
    parser.add_argument("--prox-mu", type=float, default=0.01)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--brightness-level", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--init-weights", type=Path,
                        help="Optional common initialization weights; optimizer starts fresh")
    parser.add_argument("--exploratory", action="store_true",
                        help="Allow proxy patient groups for development only")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def main():
    args = parse_args()
    if args.rounds < 1 or args.local_epochs < 1 or args.num_clients < 2:
        raise ValueError("rounds/local-epochs must be positive and num-clients at least two")
    if args.prox_mu < 0 or args.num_workers < 0:
        raise ValueError("prox-mu and num-workers must be nonnegative")
    set_seed(args.seed)
    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs" / "base.yaml")
    arch = load_config(args.config or root / "configs" / f"{args.arch}.yaml")
    cfg = override_from_cli(base.merge(arch), args)
    cfg["device"] = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    cfg["seed"] = args.seed
    cfg["num_workers"] = args.num_workers
    cfg["initial_weights"] = str(args.init_weights.resolve()) if args.init_weights else None
    cfg.setdefault("data", {})["manifest_path"] = str(args.manifest.resolve())
    cfg["data"]["exploratory"] = args.exploratory
    if args.image_size is not None:
        cfg["data"]["image_size"] = args.image_size
    required = ("training.batch_size", "training.lr", "data.image_size",
                "data.brightness_level")
    missing = [key for key in required if cfg.get(key.split(".")[0], {}).get(key.split(".")[1]) is None]
    if missing:
        raise ValueError("Set unresolved config values before running FL: " + ", ".join(missing))

    split = load_canonical_split(args.manifest, exploratory=args.exploratory)
    document, manifest_info = read_manifest(args.manifest, args.data_root,
                                            exploratory=args.exploratory)
    train_ids, val_ids = set(split.train), set(split.val)
    train_rows = [row for row in document["records"] if row["id"] in train_ids]
    val_rows = [row for row in document["records"] if row["id"] in val_ids]
    clients_rows, partition_summary = partition_by_gap(
        train_rows, args.num_clients, covariate=args.covariate
    )
    clients = {client: ManifestDataset(rows, args.data_root, cfg)
               for client, rows in clients_rows.items()}
    val_set = ManifestDataset(val_rows, args.data_root, cfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg["federated"] = {"method": args.method, "rounds": args.rounds,
                        "local_epochs": args.local_epochs, "num_clients": args.num_clients,
                        "covariate": args.covariate, "prox_mu": args.prox_mu,
                        "manifest_path": str(args.manifest.resolve()),
                        "manifest_sha256": manifest_info["sha256"]}
    cfg["training"]["epochs"] = args.local_epochs
    save_resolved_config(cfg, args.output_dir / "resolved_config.yaml")
    write_json(args.output_dir / "partition_summary.json", partition_summary)

    initial_cfg = dict(cfg)
    if args.resume:
        initial_cfg["initial_weights"] = None
    coordinator = build_trainer(args.arch, initial_cfg)
    start_round, best = 0, float("-inf")
    if args.resume:
        checkpoint = load_checkpoint(args.resume, map_location="cpu")
        if checkpoint.get("arch") != args.arch or checkpoint.get("method") != args.method:
            raise ValueError("Resume checkpoint architecture/method does not match this run")
        if checkpoint.get("manifest_sha256") != manifest_info["sha256"]:
            raise ValueError("Resume checkpoint was created from a different data manifest")
        coordinator.model.load_state_dict(checkpoint["model"], strict=True)
        start_round = int(checkpoint["round"])
        best = float(checkpoint.get("best_map50", best))
    global_state = {key: value.detach().cpu().clone()
                    for key, value in coordinator.model.state_dict().items()}
    log_path = args.output_dir / "log.jsonl"
    for round_index in range(start_round, args.rounds):
        def local_update(client_id, dataset, weights, mu):
            print(f"round {round_index + 1}: {client_id}, {len(dataset)} train images", flush=True)
            local_cfg = dict(cfg)
            local_cfg["initial_weights"] = None
            worker = build_trainer(args.arch, local_cfg)
            state, summary = worker.fit_client(dataset, dict(weights), args.local_epochs, mu)
            return state, len(dataset), summary

        global_state, details = run_federated_round(
            global_state, clients, local_update, method=args.method, prox_mu=args.prox_mu
        )
        coordinator.model.load_state_dict(global_state, strict=True)
        validation = coordinator.evaluate(val_set)
        if validation["map50"] is not None and validation["map50"] > best:
            best = validation["map50"]
        row = {"round": round_index + 1, **details, "validation": validation}
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(f"[{args.method}/{args.arch}] round {round_index + 1}/{args.rounds} "
              f"train_images={details['n_train']} val_map50={validation['map50']}", flush=True)
        state = {"arch": args.arch, "method": args.method, "round": round_index + 1,
                 "model": global_state, "best_map50": best,
                 "manifest_sha256": manifest_info["sha256"], "config": dict(cfg)}
        save_checkpoint(state, args.output_dir / "last.pth")
        if validation["map50"] is not None and validation["map50"] == best:
            save_checkpoint(state, args.output_dir / "best.pth")
    print(f"COMPLETED {args.method}: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
