import argparse
import random


def split_train_test(csv_file, out_dir):
    import os
    from shutil import move

    import pandas as pd

    data_tb = pd.read_csv(csv_file, index_col=False)

    random.seed(42)
    test_ids = random.sample(data_tb["img_id"].unique().tolist(), k=20)

    # train_tb = data_tb.sample(79, random_state=42)
    train_tb = data_tb[~data_tb["img_id"].isin(test_ids)]
    test_tb = data_tb[data_tb["img_id"].isin(test_ids)]

    os.makedirs(f"{out_dir}/test/imgs", exist_ok=True)
    os.makedirs(f"{out_dir}/train/imgs", exist_ok=True)

    test_tb[test_tb.columns[1:]].to_csv(f"{out_dir}/test/anns.csv")
    train_tb[train_tb.columns[1:]].to_csv(f"{out_dir}/train/anns.csv")

    for dcm_fn in test_tb["img_id"].unique():
        move(f"{out_dir}/train/imgs/{dcm_fn}.dcm", f"{out_dir}/test/imgs")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str)
    parser.add_argument("--output_dir", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    split_train_test(args.csv_file, args.output_dir)
