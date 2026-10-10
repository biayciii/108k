"""Shared supervised trainer for the DETR, Faster R-CNN and YOLOv7 adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..common.data.manifest import collate
from ..common.metrics.coco import detection_metrics
from ..common.utils.checkpoint import load_checkpoint, save_checkpoint
from ..models.adapters import build_adapter


class Trainer(ABC):
    def __init__(self, cfg: Any, resume: Path | None = None):
        self.cfg = cfg
        self.resume = Path(resume) if resume else None
        requested = str(cfg.get("device", "auto"))
        self.device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                                   else "cpu" if requested == "auto" else requested)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {self.device}")
        self.cfg["device"] = str(self.device)
        self.adapter = build_adapter(self.arch, cfg)
        self.model = self.adapter.model.to(self.device)
        if hasattr(self.adapter, "criterion") and isinstance(self.adapter.criterion, torch.nn.Module):
            self.adapter.criterion.to(self.device)
        if self.resume is None:
            self._load_initial_weights()

    @property
    @abstractmethod
    def arch(self) -> str: ...

    def _load_initial_weights(self) -> None:
        path = self.cfg.get("initial_weights")
        if not path:
            return
        checkpoint = load_checkpoint(path, map_location="cpu")
        candidates = [checkpoint]
        if isinstance(checkpoint, dict):
            candidates += [checkpoint.get(k) for k in ("model", "state_dict", "ema")]
            for candidate in list(candidates):
                if isinstance(candidate, dict):
                    candidates += [candidate.get(k) for k in ("model", "state_dict")]
        normalized = []
        for candidate in candidates:
            if isinstance(candidate, torch.nn.Module):
                candidate = candidate.state_dict()
            if isinstance(candidate, dict):
                normalized.append(candidate)
                if "state_dict" in candidate and isinstance(candidate["state_dict"], dict):
                    normalized.append(candidate["state_dict"])
        state = next((x for x in normalized if x and all(isinstance(k, str) for k in x)
                      and any(isinstance(v, torch.Tensor) for v in x.values())), None)
        if state is None:
            raise ValueError(f"No model state_dict found in initial checkpoint: {path}")
        current = self.model.state_dict()
        compatible = {}
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            candidates = [key]
            for prefix in ("model.", "module.", "model.model."):
                if key.startswith(prefix):
                    candidates.append(key[len(prefix):])
            match = next((name for name in candidates if name in current and
                          current[name].shape == value.shape), None)
            if match is not None:
                compatible[match] = value
        if not compatible:
            raise ValueError(f"No matching model tensors in initial checkpoint: {path}")
        strict = bool(self.cfg.get("strict_initial_weights", False))
        if strict and set(compatible) != set(current):
            raise ValueError(f"Checkpoint missing/mismatched tensors: {sorted(set(current)-set(compatible))[:10]}")
        result = self.model.load_state_dict(compatible, strict=strict)
        matched = len(compatible) / max(1, len(current))
        if matched < 0.5:
            raise ValueError(f"Only {matched:.0%} of model tensors matched; refusing this checkpoint")
        print(f"[train] Initialized {matched:.1%} of model tensors from {path}; "
              f"missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=int(self.cfg["training"]["batch_size"]),
                          shuffle=shuffle, num_workers=int(self.cfg.get("num_workers", 0)),
                          collate_fn=collate, pin_memory=self.device.type == "cuda")

    def _optimizer(self):
        training = self.cfg["training"]
        lr = float(training["lr"])
        if self.arch == "detr" and hasattr(self.adapter, "model"):
            backbone = list(self.model.backbone.parameters())
            backbone_ids = {id(p) for p in backbone}
            other = [p for p in self.model.parameters() if id(p) not in backbone_ids]
            groups = [{"params": other, "lr": lr},
                      {"params": backbone, "lr": float(training.get("lr_backbone", lr))}]
        else:
            groups = self.model.parameters()
        name = str(training.get("optimizer", "adamw")).lower()
        if name == "sgd":
            return torch.optim.SGD(groups, lr=lr, momentum=float(training.get("momentum", 0.9)),
                                   weight_decay=float(training.get("weight_decay", 0.0)))
        if name == "adamw":
            return torch.optim.AdamW(groups, lr=lr,
                                     weight_decay=float(training.get("weight_decay", 1e-4)))
        raise ValueError(f"Unsupported optimizer {name!r}; choose adamw or sgd")

    def _loss_epoch(self, loader, optimizer=None, proximal_reference=None, prox_mu=0.0):
        training = optimizer is not None
        self.model.train(training)
        if hasattr(self.adapter, "criterion") and isinstance(self.adapter.criterion, torch.nn.Module):
            self.adapter.criterion.train(training)
        losses, count = 0.0, 0
        for images, targets in loader:
            images = [image.to(self.device, non_blocking=True) for image in images]
            targets = [{k: v.to(self.device, non_blocking=True) for k, v in row.items()}
                       for row in targets]
            with torch.set_grad_enabled(training):
                loss = self.adapter.loss(images, targets)
                if training and prox_mu:
                    from .federated import fedprox_penalty
                    loss = loss + float(prox_mu) * fedprox_penalty(self.model, proximal_reference)
            if loss.ndim:
                loss = loss.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {self.arch} loss: {loss.item()}")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip = self.cfg["training"].get("grad_clip_norm")
                if clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(clip))
                optimizer.step()
            losses += float(loss.detach()) * len(images)
            count += len(images)
        if not count:
            raise ValueError("Data loader is empty")
        return losses / count

    def fit_client(self, train_set: Any, global_state: dict[str, torch.Tensor],
                   local_epochs: int, prox_mu: float = 0.0) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Train one client from a round-start state and return CPU weights and loss."""
        if local_epochs < 1 or prox_mu < 0:
            raise ValueError("local_epochs must be positive and prox_mu nonnegative")
        self.model.load_state_dict(global_state, strict=True)
        reference = {name: value.detach().to(self.device).clone()
                     for name, value in self.model.named_parameters()}
        optimizer = self._optimizer()
        loader = self._loader(train_set, shuffle=True)
        losses = [self._loss_epoch(loader, optimizer, reference, prox_mu)
                  for _ in range(local_epochs)]
        state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        return state, {"train_loss": sum(losses) / len(losses)}

    @torch.inference_mode()
    def evaluate(self, dataset: Any, output_dir: Path | None = None) -> dict:
        self.model.eval()
        predictions, targets = [], []
        for images, batch_targets in self._loader(dataset, shuffle=False):
            images = [image.to(self.device, non_blocking=True) for image in images]
            batch_predictions = self.adapter.predict(images)
            for prediction, target in zip(batch_predictions, batch_targets):
                predictions.append({k: v.detach().cpu() for k, v in prediction.items()})
                targets.append({k: v.detach().cpu() for k, v in target.items()})
        return detection_metrics(predictions, targets)

    def fit(self, train_set: Any, val_set: Any, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        training = self.cfg["training"]
        epochs = int(training["epochs"])
        if epochs < 1:
            raise ValueError("training.epochs must be at least 1")
        optimizer = self._optimizer()
        start_epoch, best = 0, float("-inf")
        if self.resume:
            checkpoint = load_checkpoint(self.resume, map_location=str(self.device))
            if checkpoint.get("arch") != self.arch:
                raise ValueError("Resume checkpoint architecture does not match this run")
            self.model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best = float(checkpoint.get("best_map50", best))
        if start_epoch >= epochs:
            raise ValueError(f"Checkpoint already reached epoch {start_epoch}; requested total is {epochs}")

        train_loader = self._loader(train_set, shuffle=True)
        log_path = output_dir / "log.jsonl"
        for epoch in range(start_epoch, epochs):
            started = time.time()
            train_loss = self._loss_epoch(train_loader, optimizer)
            metrics = self.evaluate(val_set)
            row = {"epoch": epoch + 1, "train_loss": train_loss, **metrics,
                   "lr": optimizer.param_groups[0]["lr"], "elapsed_sec": time.time() - started}
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if metrics["map50"] is not None and metrics["map50"] > best:
                best = metrics["map50"]
            fmt = lambda value: "NA" if value is None else f"{value:.5f}"
            print(f"[{self.arch}] epoch {epoch + 1}/{epochs} loss={train_loss:.5f} "
                  f"val_map50={fmt(metrics['map50'])} val_map50_95={fmt(metrics['map50_95'])}",
                  flush=True)
            state = {"arch": self.arch, "epoch": epoch, "model": self.model.state_dict(),
                     "optimizer": optimizer.state_dict(), "best_map50": best,
                     "config": dict(self.cfg)}
            save_checkpoint(state, output_dir / "last.pth")
            if metrics["map50"] is not None and metrics["map50"] == best:
                save_checkpoint(state, output_dir / "best.pth")


class DetrTrainer(Trainer):
    arch = "detr"


class FasterRcnnTrainer(Trainer):
    arch = "faster_rcnn"


class Yolov7Trainer(Trainer):
    arch = "yolov7"


_REGISTRY: dict[str, type[Trainer]] = {
    "detr": DetrTrainer,
    "faster_rcnn": FasterRcnnTrainer,
    "yolov7": Yolov7Trainer,
}


def build_trainer(arch: str, cfg: Any, resume: Path | None = None) -> Trainer:
    if arch not in _REGISTRY:
        raise ValueError(f"Kiến trúc không hỗ trợ: {arch}. Chọn 1 trong {list(_REGISTRY)}")
    return _REGISTRY[arch](cfg, resume=resume)
