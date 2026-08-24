import time
from typing import Any, List

# from torchvision.references.detection.engine import train_one_epoch, evaluate
import lightning.pytorch as pl
import torch
from torch import nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform


class ThyroidDetector(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.initial_model()

        # Train metrics
        self.train_loss = MeanMetric()
        # Val metrics
        self.val_map = MeanAveragePrecision(
            box_format="xyxy", iou_type="bbox", iou_thresholds=self.hparams.iou_thresholds, class_metrics=True
        )
        self.best_val_map = MaxMetric()
        # Test metrics
        self.test_map = MeanAveragePrecision(
            box_format="xyxy", iou_type="bbox", iou_thresholds=self.hparams.iou_thresholds, class_metrics=True
        )

    def initial_model(self):
        # Setup FasterRCNN
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        self.model.transform = GeneralizedRCNNTransform(256, 1024, [0.4] * self.hparams.num_level, [0.2] * self.hparams.num_level)
        self.model.backbone.body.conv1 = nn.Conv2d(self.hparams.num_level, 64, 7, stride=2, padding=3, bias=False)

        self.num_classes = 3
        in_features = self.model.roi_heads.box_predictor.bbox_pred.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_channels=in_features, num_classes=self.num_classes
        )

        # # Setup RetinaNet
        # weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        # self.model = retinanet_resnet50_fpn_v2(weights=weights)
        # in_channels = self.model.head.classification_head.cls_logits.in_channels
        # num_anchors = self.model.head.classification_head.num_anchors
        # self.num_classes = 3
        # cls_logits = torch.nn.Conv2d(in_channels=in_channels, out_channels=num_anchors*self.num_classes, kernel_size=3, stride=1, padding=1)
        # torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
        # torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))
        # self.model.head.classification_head.cls_logits = cls_logits

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def validation_step(self, batch: Any, batch_idx: int):
        _, imgs, targets = batch
        preds = self.model(imgs)
        self.val_map.update(preds, targets)
        # metrics, precision, recall = self.val_map.compute()
        # log_d = {f"val/{k}": v.item() for k, v in metrics.items()}
        # self.log_dict(log_d, on_epoch=True, on_step=False, prog_bar=True, batch_size=len(batch))
        # return log_d

    def on_train_start(self):
        self.best_val_map.reset()

    def on_validation_epoch_end(self):
        cr_map, precisions, recalls = self.val_map.compute()
        # self.log_dict(
        #     {f"val/{k}": v for k, v in cr_map.items()},
        #     prog_bar=True,
        # )
        self.log_dict(
            {
                'val/map_50': cr_map['map_50'].item(),
                'val/map_50_class_0': cr_map['map_per_class'][0].item(),
                'val/map_50_class_1': cr_map['map_per_class'][1].item(),  
                'val/map_small': cr_map['map_small'].item(),
                'val/map_medium': cr_map['map_medium'].item(),
                'val/map_large': cr_map['map_large'].item()
            }, 
            prog_bar=True,
        )
        self.best_val_map.update(cr_map["map_50"].item())
        # start_time = time.time()
        self.log("best_val_map_50", self.best_val_map.compute(), prog_bar=True)
        # print('Time computing mAP50: ', time.time() - start_time)
        self.val_map.reset()

    def test_step(self, batch: Any, batch_idx: int):
        _, imgs, targets = batch
        preds = self.model(imgs)
        self.test_map.update(preds, targets)
        # metrics, precisions, recalls = self.test_map.compute()
        # log_d = {f"test/{k}": v.item() for k, v in metrics.items()}
        # self.log_dict(log_d, on_epoch=True, on_step=False, prog_bar=True, batch_size=len(batch))
        # return log_d

    def on_test_epoch_end(self):
        metrics, precisions, recalls = self.test_map.compute()
        # self.log_dict(
        #     {f"test/{k}": v for k, v in metrics.items()},
        #     prog_bar=True,
        # )
        self.log_dict(
            {
                'test/map_50': metrics['map_50'].item(),
                'test/map_class_0': metrics['map_per_class'][0].item(),
                'test/map_class_1': metrics['map_per_class'][1].item(),  
                'test/map_small': metrics['map_small'].item(),
                'test/map_medium': metrics['map_medium'].item(),
                'test/map_large': metrics['map_large'].item()
            }, 
            prog_bar=True,
        )
        self.test_map.reset()

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "best_val_map_50",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        _, imgs, targets = batch
        loss_dict = self.model(imgs, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.train_loss.update(loss)
        self.log(
            "train/loss",
            self.train_loss.compute().item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch)
        )
        return {"loss": loss}
