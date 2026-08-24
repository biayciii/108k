import argparse
import os
import re
from shutil import copyfile


def count_and_copy_valid_folder(root_dir, des):
    count = 0
    for fp in os.listdir(root_dir):
        valid = 0
        for fn in os.listdir(os.path.join(root_dir, fp)):
            if re.match(r".*(ANT|POST).*_dup.*\.dcm", fn):
                valid = 1
                suffix = fn[fn.rfind(".") :]
                copyfile(os.path.join(root_dir, fp, fn), os.path.join(des, f"{fp}{suffix}"))
                break
        if valid == 0:
            print(fp)
        count += valid

    return count


def get_args():
    argparser = argparse.ArgumentParser(
        description="CLI for create standard format dicom file from source dataset"
    )
    argparser.add_argument(
        "--root_dir", type=str, help="Path to src folder contains img folders of patients"
    )
    argparser.add_argument("--out", type=str, help="Path to output folder")
    return argparser


def main(args):
    os.makedirs(args.out, exist_ok=True)
    print(count_and_copy_valid_folder(args.root_dir, args.out))


if __name__ == "__main__":
    args = get_args()
    main(args.parse_args())
