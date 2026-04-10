#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import shutil
from pathlib import Path


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--list_csv", required=True,
                        help="tdms list csv")

    parser.add_argument("--dest_dir", required=True,
                        help="destination directory")

    args = parser.parse_args()

    df = pd.read_csv(args.list_csv)

    if "tdms_filename" not in df.columns or "tdms_filepath" not in df.columns:
        raise ValueError("CSV must contain tdms_filename and tdms_filepath")

    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0

    for _, row in df.iterrows():

        src = Path(row["tdms_filepath"])
        dst = dest_dir / row["tdms_filename"]

        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"[WARN] missing file: {src}")
            missing += 1

    print(f"[INFO] copied  : {copied}")
    print(f"[INFO] missing : {missing}")
    print(f"[INFO] dest    : {dest_dir}")


if __name__ == "__main__":
    main()