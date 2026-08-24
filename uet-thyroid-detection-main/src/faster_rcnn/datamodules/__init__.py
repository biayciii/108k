import albumentations as A
import torch
from albumentations.augmentations.geometric.resize import Resize
from albumentations.pytorch.transforms import ToTensorV2

TARGET_SIZE = 256


def get_train_transforms():
    return A.Compose(
        [
            Resize(TARGET_SIZE, TARGET_SIZE),
            ToTensorV2(p=1.0),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.1),
    )


def get_test_transforms():
    return A.Compose(
        [
            Resize(TARGET_SIZE, TARGET_SIZE),
            ToTensorV2(p=1.0),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.1),
    )


def collate_fn(batch):
    img_ids, imgs, targets = zip(*batch)
    imgs = torch.stack(imgs)
    return list(img_ids), imgs, list(targets)


def format_prediction_string(labels, boxes, scores):
    pred_strings = []
    for j in zip(labels, scores, boxes):
        pred_strings.append(
            f"{j[0]} {j[1]:.4f} {j[2][0]} {j[2][1]} {j[2][2]} {j[2][3]}"
        )
    return " ".join(pred_strings)
