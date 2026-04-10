#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match_csv", required=True)
    parser.add_argument("--dest_dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.match_csv)

    matched = df[df["matched"] == True]

    os.makedirs(args.dest_dir, exist_ok=True)

    total = len(matched)
    copied = 0

    for i, row in matched.iterrows():

        src = row["tdms_filepath"]
        fname = os.path.basename(src)
        dst = os.path.join(args.dest_dir, fname)

        if not os.path.exists(src):
            print(f"[WARN] missing: {src}")
            continue

        if os.path.exists(dst):
            continue

        shutil.copy2(src, dst)
        copied += 1

        if copied % 50 == 0:
            print(f"[INFO] copied {copied}/{total}")

    print(f"[DONE] copied {copied} files")


if __name__ == "__main__":
    main()

'''
python scripts/copy_matched_tdms.py \
  --match_csv outputs/catalog_tdms_intersection.csv \
  --dest_dir data/raw_tdms
'''