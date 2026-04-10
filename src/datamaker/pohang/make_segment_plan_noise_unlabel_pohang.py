#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import math
import hashlib
import random
import pandas as pd
import numpy as np
from nptdms import TdmsFile


SEGMENT_SEC = 2.0
MAX_NOISE_SEGMENTS_PER_FILE = 15
MAX_UNLABEL_SEGMENTS_PER_FILE = 15

CHANNEL_CONFIG = {
    "pohang": {
        "ch_start": 243,
        "ch_end": 648,
    }
}

VALID_EXTS = {".tdms"}


def get_tdms_duration_sec(tdms_path, fs):
    tdms = TdmsFile.read(tdms_path)

    max_len = 0
    for group in tdms.groups():
        for ch in group.channels():
            try:
                n = len(ch)
                if n > max_len:
                    max_len = n
            except Exception:
                continue

    if max_len == 0:
        raise RuntimeError(f"Could not read TDMS channel length: {tdms_path}")

    return float(max_len) / float(fs)


def make_file_seed(file_path, suffix):
    key = f"{file_path}|{suffix}".encode("utf-8")
    h = hashlib.md5(key).hexdigest()
    return int(h[:8], 16)


def build_random_segment_times(duration_sec, segment_sec, max_segments, rng):
    if duration_sec < segment_sec:
        return []

    max_start_int = int(math.floor(duration_sec - segment_sec))
    candidates = list(range(max_start_int + 1))

    if len(candidates) == 0:
        return [(0.0, float(segment_sec))]

    n_pick = min(max_segments, len(candidates))
    picked = sorted(rng.sample(candidates, n_pick))

    times = []
    for start_int in picked:
        start_sec = float(start_int)
        end_sec = start_sec + float(segment_sec)
        times.append((start_sec, end_sec))

    return times


def make_rows(inv_row, duration_sec, data_type, label, max_segments_per_file):
    rows = []

    file_path = str(inv_row["file_path"])
    file_name = str(inv_row["file_name"])
    file_stem = str(inv_row["file_stem"])
    group_id = str(inv_row["group_id"])
    original_fs = float(inv_row["original_fs"])

    rng = random.Random(make_file_seed(file_path, data_type))

    seg_times = build_random_segment_times(
        duration_sec=duration_sec,
        segment_sec=SEGMENT_SEC,
        max_segments=max_segments_per_file,
        rng=rng,
    )

    ch_cfg = CHANNEL_CONFIG["pohang"]

    for start_sec, end_sec in seg_times:
        rows.append({
            "segment_id": None,
            "group_id": group_id,
            "dataset_id": "pohang",
            "site": "pohang",
            "view": "pohang",
            "data_type": data_type,
            "label": label,
            "file_path": file_path,
            "file_name": file_name,
            "file_stem": file_stem,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "ch_start": int(ch_cfg["ch_start"]),
            "ch_end": int(ch_cfg["ch_end"]),
            "original_fs": original_fs,
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory_csv", required=True)
    parser.add_argument("--out_noise_csv", required=True)
    parser.add_argument("--out_unlabel_csv", required=True)
    args = parser.parse_args()

    inv_df = pd.read_csv(args.inventory_csv)
    inv_df = inv_df[inv_df["ext"].astype(str).str.lower().isin(VALID_EXTS)].copy()

    noise_inv = inv_df[inv_df["data_type"] == "noise"].copy()
    unlabel_inv = inv_df[inv_df["data_type"] == "unlabel"].copy()

    noise_rows = []
    unlabel_rows = []

    print(f"[INFO] noise source files   : {len(noise_inv)}")
    print(f"[INFO] unlabel source files: {len(unlabel_inv)}")

    for _, row in noise_inv.iterrows():
        try:
            duration_sec = get_tdms_duration_sec(row["file_path"], row["original_fs"])
            rows = make_rows(
                inv_row=row,
                duration_sec=duration_sec,
                data_type="noise",
                label=0,
                max_segments_per_file=MAX_NOISE_SEGMENTS_PER_FILE,
            )
            noise_rows.extend(rows)
            print(f"[NOISE] {row['file_name']} -> {len(rows)} segments")
        except Exception as e:
            print(f"[WARN][NOISE] skip: {row['file_path']} | {e}")

    for _, row in unlabel_inv.iterrows():
        try:
            duration_sec = get_tdms_duration_sec(row["file_path"], row["original_fs"])
            rows = make_rows(
                inv_row=row,
                duration_sec=duration_sec,
                data_type="unlabel",
                label=2,
                max_segments_per_file=MAX_UNLABEL_SEGMENTS_PER_FILE,
            )
            unlabel_rows.extend(rows)
            print(f"[UNLABEL] {row['file_name']} -> {len(rows)} segments")
        except Exception as e:
            print(f"[WARN][UNLABEL] skip: {row['file_path']} | {e}")

    noise_df = pd.DataFrame(noise_rows)
    unlabel_df = pd.DataFrame(unlabel_rows)

    if len(noise_df) > 0:
        noise_df = noise_df.reset_index(drop=True)
        noise_df["segment_id"] = np.arange(len(noise_df), dtype=int)
    if len(unlabel_df) > 0:
        unlabel_df = unlabel_df.reset_index(drop=True)
        unlabel_df["segment_id"] = np.arange(len(unlabel_df), dtype=int)

    Path(args.out_noise_csv).parent.mkdir(parents=True, exist_ok=True)
    noise_df.to_csv(args.out_noise_csv, index=False, encoding="utf-8-sig")
    unlabel_df.to_csv(args.out_unlabel_csv, index=False, encoding="utf-8-sig")

    print(f"[SAVED] {args.out_noise_csv}")
    print(f"[TOTAL NOISE SEGMENTS] {len(noise_df)}")
    if len(noise_df) > 0:
        print(noise_df.groupby(["site", "view"]).size())

    print(f"[SAVED] {args.out_unlabel_csv}")
    print(f"[TOTAL UNLABEL SEGMENTS] {len(unlabel_df)}")
    if len(unlabel_df) > 0:
        print(unlabel_df.groupby(["site", "view"]).size())


if __name__ == "__main__":
    main()