import os
import shutil
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from easydict import EasyDict as edict
from split_train_test import split_train_test
from torch.utils.data import Dataset


class ThyroidDataset(Dataset):
    def __init__(self, root_dir):
        """

        Args:
                root_dir (str): data folder path
        """
        super().__init__()
        self.root_dir = root_dir
        self.setup()

    def setup(self):
        # Read both ANTORI
        # OR and POSTORIOR dicom image, but here we only annotate POST image.
        dcm_dataset = []
        for dcm_fn in os.listdir(self.root_dir):
            dcm_sample = {
                "dcm": pydicom.dcmread(os.path.join(self.root_dir, dcm_fn)),
                "name": dcm_fn.rsplit(".", 1)[0],
            }
            dcm_dataset.append(edict(dcm_sample))

        self.dcm_dataset = dcm_dataset

    def __len__(self):
        return len(self.dcm_dataset)

    def __getitem__(self, idx):
        dcm_item = self.dcm_dataset[idx].dcm
        name = self.dcm_dataset[idx].name
        # Crop irrelevance pixels (headneck only)
        bbox_dict = self.get_bboxes_from_roidb(dcm_item[0x0057, 0x1001])
        img = dcm_item.pixel_array
        #        img = dcm_item.pixel_array[:int(bbox_dict[1].bboxes[-1])]
        # img = dcm_item.pixel_array[:300:]
        return img, name, bbox_dict

    def get_bboxes_from_roidb(self, roidb):
        """

        Args:
                roidb (pydicom dataset): ROI dataset read from dicom image

        Returns:
                list: list of bounding boxes sorted by top left y axis, thus 1st element is thyroid's \
                ROI, 2nd element is shoulder's ROID.
        """
        gt_boxes = []
        top_left_ys = []
        for roi_dataset in roidb:
            bbox_info = list(roi_dataset[0x0057, 0x105B])
            color = roi_dataset[0x0057, 0x1047].value
            x_min, x_max, y_min, y_max = (
                np.min(bbox_info[0::3]),
                np.max(bbox_info[0::3]),
                np.min(bbox_info[1::3]),
                np.max(bbox_info[1::3]),
            )

            gt_boxes.append(
                edict({"color": color, "bboxes": np.array([x_min, y_min, x_max, y_max])})
            )
            top_left_ys.append(y_min)

        sorted_gt_bboxes = list(map(gt_boxes.__getitem__, np.argsort(top_left_ys)))
        return sorted_gt_bboxes


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="Command help to extract bbox of shoulder, \
                                     thyroid ROI from dicom image then save annotation as csv file",
        add_help=add_help,
    )
    parser.add_argument(
        "--root_dir",
        help="Path to data directory which contains images/ and labels/ folders",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--out",
        help="Path to output folder which places generated annotation file",
        type=str,
        default=None,
    )
    return parser


def main(args):
    root_dir = args.root_dir
    thyroid_roidb = ThyroidDataset(root_dir)
    out_dir = Path(args.out)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Evaludate bounding box
    dataset_df = pd.DataFrame(columns=["img_id", "width", "height", "class", "bbox"])
    for img, name, bbox_dict in thyroid_roidb:
        roi_bbox, shoulder_bbox = bbox_dict[0].bboxes, bbox_dict[-1].bboxes

        roi_bbox_coor = [
            roi_bbox[0],
            roi_bbox[1],
            (roi_bbox[2] - roi_bbox[0]),
            (roi_bbox[3] - roi_bbox[1]),
        ]
        shoulder_bbox_coor = [
            shoulder_bbox[0],
            shoulder_bbox[1],
            (shoulder_bbox[2] - shoulder_bbox[0]),
            (shoulder_bbox[3] - shoulder_bbox[1]),
        ]

        dataset_df.loc[len(dataset_df)] = [
            name,
            img.shape[1],
            img.shape[0],
            "thyroid",
            roi_bbox_coor,
        ]
        dataset_df.loc[len(dataset_df)] = [
            name,
            img.shape[1],
            img.shape[0],
            "shoulder",
            shoulder_bbox_coor,
        ]

        # fig, ax = plt.subplots()
        # ax.imshow(img*255, cmap="Greys")
        # colors = [(0, 1, 0), (0, 1, 0), (1, 0, 0)]

        # rect_thyroi = patches.Rectangle((roi_bbox_coor[0], roi_bbox_coor[1]), roi_bbox_coor[2], roi_bbox_coor[3], edgecolor=colors[0], facecolor="none")
        # rect_gt_sld = patches.Rectangle((shoulder_bbox_coor[0], shoulder_bbox_coor[1]), shoulder_bbox_coor[2], shoulder_bbox_coor[3], edgecolor=colors[0], facecolor="none")

        # ax.add_patch(rect_thyroi)
        # ax.add_patch(rect_gt_sld)

        # ax.axis("off")
        # fig.savefig(f"outputs/{name}.png", bbox_inches="tight")

    dataset_df.to_csv(out_dir / "data.csv")
    # split_train_test(f"{out_dir.resolve()}/data.csv", out_dir.resolve())
    # os.remove(out_dir / "data.csv")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
