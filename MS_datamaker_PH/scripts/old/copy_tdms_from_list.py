#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import shutil
from pathlib import Path


def detect_filename_col(df):
    for c in ["tdms_filename", "filename", "tdms_file", "file"]:
        if c in df.columns:
            return c
    raise ValueError("TDMS filename column not found")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--list_csv", required=True,
                        help="tdms list csv")

    parser.add_argument("--src_dir", required=True,
                        help="original tdms directory")

    parser.add_argument("--dest_dir", required=True,
                        help="destination directory")

    args = parser.parse_args()

    df = pd.read_csv(args.list_csv)

    file_col = detect_filename_col(df)

    src_dir = Path(args.src_dir)
    dest_dir = Path(args.dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0

    for fname in df[file_col]:

        src = src_dir / fname
        dst = dest_dir / fname

        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"[WARN] missing: {src}")
            missing += 1

    print(f"[INFO] copied files : {copied}")
    print(f"[INFO] missing files: {missing}")
    print(f"[INFO] destination : {dest_dir}")


if __name__ == "__main__":
    main()