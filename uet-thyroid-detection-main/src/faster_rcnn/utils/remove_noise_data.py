import argparse
import ast
import copy
import os

import pandas as pd


def main(args):
    data_dir = args.data_dir
    ann_file_p = f"{data_dir}/anns.csv"
    annotation_tb = pd.read_csv(ann_file_p, index_col=False)
    annotation_tb["bbox"] = annotation_tb["bbox"].apply(lambda x: ast.literal_eval(x))

    org_annotation_tb = copy.deepcopy(annotation_tb)
    annotation_tb["img_aspect_ratio"] = annotation_tb["width"] / annotation_tb["height"]
    annotation_tb["box_aspect_ratio"] = annotation_tb["bbox"].apply(lambda x: x[2] / x[3])

    desired_sz = (480, 480)

    annotation_tb["resized_bbox"] = annotation_tb.apply(
        lambda x: resize_bbox(x, desired_sz), axis=1
    )
    annotation_tb["new_box_aspect_ratio"] = annotation_tb["resized_bbox"].apply(
        lambda x: x[2] / x[3]
    )
    annotation_tb["resized_bbox_sz"] = annotation_tb["resized_bbox"].apply(lambda x: int(x[2]))

    # Selected image ids
    removed_ids = []
    removed_ids.extend(
        list(annotation_tb[annotation_tb["new_box_aspect_ratio"] > 3]["img_id"].unique())
    )
    removed_ids.extend(
        list(annotation_tb[annotation_tb["resized_bbox_sz"] >= 100]["img_id"].unique())
    )
    # Update annotation file
    org_annotation_tb = org_annotation_tb[~org_annotation_tb["img_id"].isin(removed_ids)]
    org_annotation_tb.to_csv(f"{data_dir}/anns.csv", index=False)

    img_dir = f"{data_dir}/imgs/"
    for img_id in removed_ids:
        os.remove(f"{img_dir}/{img_id}.dcm")


def resize_bbox(record, desired_sz):
    ratio_x, ratio_y = desired_sz[0] / record.width, desired_sz[1] / record.height
    bbox = copy.deepcopy(record.bbox)
    bbox[0] *= ratio_x
    bbox[2] *= ratio_x
    bbox[1] *= ratio_y
    bbox[3] *= ratio_y

    return bbox


def get_args():
    parser = argparse.ArgumentParser(description="Command for remove noise data")
    parser.add_argument(
        "--data_dir", type=str, help="Path to train/test folder", default="data/train"
    )
    return parser


if __name__ == "__main__":
    args = get_args().parse_args()
    main(args)
