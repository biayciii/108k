import ast
import os
from typing import Any, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from scipy import misc, ndimage, signal
from torch.utils.data import Dataset


class ThyroidDataset(Dataset):
    def __init__(self, image_root_dir, label_file, width, height, num_level, transforms=None):
        super().__init__()
        self.image_root_dir = image_root_dir
        self.label_file = label_file
        self.transforms = transforms
        self.width = width
        self.height = height
        self.cutting_threshold = 256
        self.num_level = num_level

        self.setup()

    def setup(self):
        self.img_fns = os.listdir(os.path.join(self.image_root_dir))
        self.ann_tb = pd.read_csv(os.path.join(self.label_file), index_col=False)
        self.ann_tb["bbox"] = self.ann_tb["bbox"].apply(lambda x: ast.literal_eval(x))
        self.classes = ["_", "shoulder", "thyroid"]
        # self.classes = ["_", "thyroid"]

    def __len__(self):
        return len(self.img_fns)

    def get_dcm_im(self, dcm_path):
        img = pydicom.dcmread(dcm_path).pixel_array
        # img = np.concatenate([img[..., None]] * 3, axis=2)
        return img

    def irrelevant_cutting(self, im: np.ndarray, cutting_threshold):
        # Cut irrelevant part of images
        if im.shape[:2] != (512, 512):
            im = im[:cutting_threshold, :cutting_threshold]

        return im

    def prepare_standard_data_format(self, img_id, img, boxes, labels):
        """

        Args:
            img_id (str): image id
            img (np.ndarray): image shape HxWx3
            boxes (list): (x_min, y_min, x_max, y_max) without normalized
            labels (list): list of labels corresponding each bboxes

        Returns:
            dict: PASCAL VOC dictionary
        """
        # Potential line
        #####################################################
        boxes[:, 2:4] += boxes[:, 0:2]
        # boxes[:, [0, 2]] *= self.width / img.shape[1]
        # boxes[:, [1, 3]] *= self.height / img.shape[0]
        #####################################################

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        # PASCAL POC Data format for fasterrcnn
        target = {}
        target["width"] = img.shape[2]
        target["height"] = img.shape[1]
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target

    # def gray_to_pil(self, im):
    #     im = Image.fromarray(np.uint16(im))
    #     return im

    # def plt_show(im):
    #     plt.imshow(gray_to_pil(im), cmap='gray')

    def increase_count(self, im, factor=1):
        # scharr = np.array([[1, 1, 1],

        #                 [1, 2, 1],

        #                 [1, 1, 1]])*factor
        
        if factor <= 0:
            return im

        scharr = (
            np.array(
                [
                    [1, 1, 1, 1, 1],
                    [1, 2, 2, 2, 1],
                    [1, 2, 4, 2, 1],
                    [1, 2, 2, 2, 1],
                    [1, 1, 1, 1, 1],
                ]
            )
            * factor
        )

        im = signal.convolve2d(im, scharr, boundary="symm", mode="same")
        # im = ndimage.gaussian_filter(im.astype(np.float32)*1000*factor, sigma=1, mode='reflect').astype(np.uint16)
        im[im < 0] = 0
        im[im > 255] = 255
        return im

    def create_imbatch(self, im: np.ndarray, brighness_levels: int):
        # im shape HxW
        bright_factors = [2**i for i in range(brighness_levels - 1)]
        bright_factors.insert(0, 0) # Original image
        
        im_batch = np.vstack([self.increase_count(im, factor) for factor in bright_factors]).reshape(brighness_levels, *im.shape)
        im_batch = im_batch.transpose(1, 2, 0) # transpose to HxWxnum_level
        return im_batch.astype(np.uint8)

    # def save_im(im, path):
    #     cv2.imwrite(path, im)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_id = self.img_fns[index].rsplit(".", 1)[0]
        records = self.ann_tb[self.ann_tb["img_id"] == img_id]
        img = self.get_dcm_im(os.path.join(self.image_root_dir, self.img_fns[index]))
        img = self.irrelevant_cutting(img, self.cutting_threshold)  # HxW

        ################Create multi level increased#########################
        img = self.create_imbatch(img, self.num_level)  # HxWxnum_level#
        ################Create multi level increased#########################

        labels, boxes = (
            records["class"].values.tolist(),
            records["bbox"].values.tolist(),
        )
        labels = [self.classes.index(l) for l in labels]
        labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        img = np.asarray(img, dtype=np.float32) / 255
        target = self.prepare_standard_data_format(img_id, img, boxes, labels)
        
        if self.transforms:
            params = {"image": img, "bboxes": target["boxes"], "labels": labels}
            transformed = self.transforms(**params)
            img = transformed["image"]
            target["boxes"] = torch.as_tensor(transformed["bboxes"])
        else:
            img = torch.as_tensor(img, dtype=torch.float32)

        return img_id, img, target
