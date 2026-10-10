"""Common loss/prediction interface; reuse all three legacy model definitions."""
from pathlib import Path
from types import SimpleNamespace

import torch

from .legacy import LEGACY, detr_module, frcnn_module, yolo_module


class DetrAdapter:
    def __init__(self, cfg):
        arch = cfg.get("architecture", {})
        args = SimpleNamespace(
            device=cfg["device"], dataset_file="thyroid", masks=False, frozen_weights=None,
            backbone=arch.get("backbone", "resnet50"), dilation=False, position_embedding="sine",
            lr_backbone=cfg["training"].get("lr_backbone", 1e-5),
            brighness_levels=cfg["data"]["brightness_level"], pretrained_backbone=False,
            enc_layers=arch.get("enc_layers", 6), dec_layers=arch.get("dec_layers", 6),
            dim_feedforward=arch.get("dim_feedforward", 2048), hidden_dim=arch.get("hidden_dim", 256),
            dropout=arch.get("dropout", 0.1), nheads=arch.get("nheads", 8), pre_norm=False,
            num_queries=arch.get("num_queries", 75), aux_loss=True,
            set_cost_class=1, set_cost_bbox=5, set_cost_giou=2,
            bbox_loss_coef=5, giou_loss_coef=2, eos_coef=0.1)
        for key, value in arch.get("legacy_args", {}).items():
            if hasattr(args, key) and key not in {"device", "pretrained_backbone", "frozen_weights", "brighness_levels", "lr_backbone"}:
                setattr(args, key, value)
        self.model, self.criterion, self.post = detr_module("models").build_model(args)
        self.model.backbone[0].body.conv1.weight.requires_grad_(True)
        if arch.get("train_all_backbone", False):
            # Random initialization needs the early residual stage to learn too.
            self.model.backbone.requires_grad_(True)

    def inputs(self, images):
        return [(image - 0.4) / 0.2 for image in images]

    def loss(self, images, targets):
        converted = []
        for image, target in zip(images, targets):
            h, w = image.shape[-2:]
            boxes = target["boxes"]
            xy, wh = (boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]
            converted.append({"boxes": torch.cat((xy, wh), 1) / boxes.new_tensor([w, h, w, h]),
                              "labels": target["labels"]})
        values = self.criterion(self.model(self.inputs(images)), converted)
        return sum(values[k] * weight for k, weight in self.criterion.weight_dict.items() if k in values)

    def predict(self, images):
        sizes = torch.tensor([x.shape[-2:] for x in images], device=images[0].device)
        return self.post["bbox"](self.model(self.inputs(images)), sizes)


class FasterRcnnAdapter:
    def __init__(self, cfg):
        detector = frcnn_module().ThyroidDetector(
            num_level=cfg["data"]["brightness_level"], pretrained=False, iou_thresholds=[0.5])
        self.model = detector.model
        size = cfg["data"]["image_size"]
        self.model.transform.min_size = (size,)
        self.model.transform.max_size = size

    def loss(self, images, targets):
        return sum(self.model(images, targets).values())

    def predict(self, images):
        return self.model(images)


class Yolov7Adapter:
    def __init__(self, cfg):
        import yaml
        model_cfg = cfg.get("architecture", {}).get("model_yaml")
        self.model = yolo_module("models.yolo").Model(
            model_cfg or str(LEGACY / "yolov7/cfg/training/yolov7.yaml"),
            ch=cfg["data"]["brightness_level"], nc=2)
        hyp_path = Path(cfg.get("architecture", {}).get("hyp_yaml") or
                        LEGACY.parents[1] / "Data/yolov7-data/hyp.scratch.p5.yaml")
        with hyp_path.open(encoding="utf-8") as stream:
            self.model.hyp = yaml.safe_load(stream)
        self.model.gr = 1.0
        self.model.names = ["shoulder", "thyroid"]
        self.loss_type = yolo_module("utils.loss").ComputeLoss
        self.nms = yolo_module("utils.general").non_max_suppression
        self.criterion = None

    def loss(self, images, targets):
        if self.criterion is None:
            self.criterion = self.loss_type(self.model)
        rows = []
        for index, (image, target) in enumerate(zip(images, targets)):
            h, w = image.shape[-2:]
            boxes = target["boxes"]
            xywh = torch.cat(((boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]), 1)
            xywh = xywh / boxes.new_tensor([w, h, w, h])
            rows.append(torch.cat((boxes.new_full((len(boxes), 1), index),
                                   (target["labels"] - 1).to(boxes.dtype)[:, None], xywh), 1))
        return self.criterion(self.model(torch.stack(images)), torch.cat(rows))[0] / len(images)

    def predict(self, images):
        values = self.nms(self.model(torch.stack(images))[0], conf_thres=0.001, iou_thres=0.65)
        return [{"boxes": v[:, :4], "scores": v[:, 4], "labels": v[:, 5].long() + 1} for v in values]


def build_adapter(arch, cfg):
    return {"detr": DetrAdapter, "faster_rcnn": FasterRcnnAdapter, "yolov7": Yolov7Adapter}[arch](cfg)
