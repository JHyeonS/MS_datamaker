#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import argparse
import math
import struct
import hashlib
import random
import pandas as pd
import numpy as np

CHANNEL_CONFIG = {
    "utah_2019_1": {"ch_start": 679, "ch_end": 1084},
    "utah_2019_2": {"ch_start": 273, "ch_end": 678},
}

def _read_be_u2(buf, offset):
    return struct.unpack(">H", buf[offset:offset + 2])[0]

def _read_segy_basic_info(segy_path):
    with open(segy_path, "rb") as f:
        text_header = f.read(3200)
        if len(text_header) != 3200:
            raise RuntimeError(f"Invalid SEG-Y text header: {segy_path}")
        bin_header = f.read(400)
        if len(bin_header) != 400:
            raise RuntimeError(f"Invalid SEG-Y binary header: {segy_path}")
    return {
        "sample_interval_us": _read_be_u2(bin_header, 16),
        "n_samples_bin": _read_be_u2(bin_header, 20),
    }

def _read_first_trace_n_samples(segy_path):
    with open(segy_path, "rb") as f:
        f.seek(3600)
        trace_header = f.read(240)
    if len(trace_header) != 240:
        raise RuntimeError(f"Cannot read first trace header: {segy_path}")
    return _read_be_u2(trace_header, 114)

def get_segy_duration_sec(segy_path, fs=None):
    info = _read_segy_basic_info(segy_path)
    n_samples = 0
    try:
        n_samples = _read_first_trace_n_samples(segy_path)
    except Exception:
        n_samples = 0
    if n_samples <= 0:
        n_samples = int(info["n_samples_bin"])
    if n_samples <= 0:
        raise RuntimeError(f"Could not read SEG-Y sample count: {segy_path}")
    if fs is None:
        if info["sample_interval_us"] > 0:
            fs = 1e6 / float(info["sample_interval_us"])
        else:
            raise RuntimeError(f"Could not infer fs: {segy_path}")
    return float(n_samples) / float(fs)

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
    return [(float(s), float(s) + float(segment_sec)) for s in picked]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory_csv", required=True)
    ap.add_argument("--out_noise_csv", required=True)
    ap.add_argument("--out_unlabel_csv", required=True)
    ap.add_argument("--segment_sec", type=float, default=2.0)
    ap.add_argument("--max_noise_per_file", type=int, default=15)
    ap.add_argument("--max_unlabel_per_file", type=int, default=15)
    args = ap.parse_args()

    inv = pd.read_csv(args.inventory_csv)
    noise_rows = []
    unlabel_rows = []

    for _, row in inv.iterrows():
        data_type = str(row["data_type"])
        if data_type not in {"noise", "unlabel"}:
            continue

        file_path = str(row["file_path"])
        fs = float(row["original_fs"])
        duration_sec = get_segy_duration_sec(file_path, fs=fs)

        max_seg = args.max_noise_per_file if data_type == "noise" else args.max_unlabel_per_file
        rng = random.Random(make_file_seed(file_path, data_type))
        seg_times = build_random_segment_times(duration_sec, args.segment_sec, max_seg, rng)

        for view_name, ch_cfg in CHANNEL_CONFIG.items():
            for s, e in seg_times:
                out_row = {
                    "segment_id": None,
                    "group_id": row["group_id"],
                    "dataset_id": "utah_2019",
                    "site": "utah_2019",
                    "view": view_name,
                    "data_type": data_type,
                    "label": 0 if data_type == "noise" else 2,
                    "file_path": row["file_path"],
                    "file_name": row["file_name"],
                    "file_stem": row["file_stem"],
                    "start_sec": s,
                    "end_sec": e,
                    "ch_start": int(ch_cfg["ch_start"]),
                    "ch_end": int(ch_cfg["ch_end"]),
                    "original_fs": float(row["original_fs"]),
                }
                if data_type == "noise":
                    noise_rows.append(out_row)
                else:
                    unlabel_rows.append(out_row)

    noise_df = pd.DataFrame(noise_rows).reset_index(drop=True)
    if len(noise_df) > 0:
        noise_df["segment_id"] = np.arange(len(noise_df), dtype=int)

    unlabel_df = pd.DataFrame(unlabel_rows).reset_index(drop=True)
    if len(unlabel_df) > 0:
        unlabel_df["segment_id"] = np.arange(len(unlabel_df), dtype=int)

    Path(args.out_noise_csv).parent.mkdir(parents=True, exist_ok=True)
    noise_df.to_csv(args.out_noise_csv, index=False, encoding="utf-8-sig")
    unlabel_df.to_csv(args.out_unlabel_csv, index=False, encoding="utf-8-sig")

    print(f"[DONE] noise={len(noise_df)} unlabel={len(unlabel_df)}")
    if len(noise_df) > 0:
        print(noise_df.groupby(['site', 'view']).size())
    if len(unlabel_df) > 0:
        print(unlabel_df.groupby(['site', 'view']).size())

if __name__ == "__main__":
    main()