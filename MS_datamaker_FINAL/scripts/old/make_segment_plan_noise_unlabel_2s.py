#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import math
import struct
import hashlib
import random
import pandas as pd
import numpy as np

ROOT = Path("/home/ted1204/MS_datamaker_merge")

INVENTORY_CSV = ROOT / "outputs" / "file_inventory.csv"

OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NOISE_CSV = OUT_DIR / "segment_plan_noise.csv"
OUT_UNLABEL_CSV = OUT_DIR / "segment_plan_unlabel.csv"

SEGMENT_SEC = 2.0
FS_CONFIG = {"pohang": 1000, "utah": 1000}

CHANNEL_CONFIG = {
    "pohang": {"ch_start": 243, "ch_end": 648},
    "utah_2019_1": {"ch_start": 679, "ch_end": 1084},
    "utah_2019_2": {"ch_start": 273, "ch_end": 678},
}

VALID_EXTS = {".tdms", ".sgy", ".segy"}
MAX_NOISE_SEGMENTS_PER_FILE = 15
MAX_UNLABEL_SEGMENTS_PER_FILE = 15

def get_tdms_duration_sec(tdms_path, fs):
    from nptdms import TdmsFile
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

def _read_be_u2(buf, offset):
    return struct.unpack(">H", buf[offset:offset + 2])[0]

def _read_segy_basic_info(segy_path):
    segy_path = Path(segy_path)
    with open(segy_path, "rb") as f:
        text_header = f.read(3200)
        if len(text_header) != 3200:
            raise RuntimeError(f"Invalid SEG-Y text header: {segy_path}")
        bin_header = f.read(400)
        if len(bin_header) != 400:
            raise RuntimeError(f"Invalid SEG-Y binary header: {segy_path}")
    return {"sample_interval_us": _read_be_u2(bin_header, 16), "n_samples_bin": _read_be_u2(bin_header, 20)}

def _read_first_trace_n_samples(segy_path):
    segy_path = Path(segy_path)
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

def get_duration_sec(file_path, site):
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    fs = FS_CONFIG[site]
    if ext == ".tdms":
        return get_tdms_duration_sec(file_path, fs)
    if ext in {".sgy", ".segy"}:
        return get_segy_duration_sec(file_path, fs)
    raise ValueError(f"Unsupported extension: {file_path}")

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
    return [(float(start_int), float(start_int) + float(segment_sec)) for start_int in picked]

def make_rows(inv_row, duration_sec, data_type, label, max_segments_per_file):
    rows = []
    site = str(inv_row["site"])
    file_path = str(inv_row["file_path"])
    file_name = str(inv_row["file_name"])
    file_stem = str(inv_row["file_stem"])
    group_id = int(inv_row["group_id"])
    rng = random.Random(make_file_seed(file_path, data_type))
    seg_times = build_random_segment_times(duration_sec, SEGMENT_SEC, max_segments_per_file, rng)

    if site == "pohang":
        ch_cfg = CHANNEL_CONFIG["pohang"]
        for start_sec, end_sec in seg_times:
            rows.append({
                "segment_id": None, "group_id": group_id, "site": "pohang", "view": "pohang",
                "data_type": data_type, "label": label, "file_path": file_path, "file_name": file_name,
                "file_stem": file_stem, "start_sec": start_sec, "end_sec": end_sec,
                "ch_start": int(ch_cfg["ch_start"]), "ch_end": int(ch_cfg["ch_end"]),
            })

    elif site == "utah":
        for view_name in ["utah_2019_1", "utah_2019_2"]:
            ch_cfg = CHANNEL_CONFIG[view_name]
            for start_sec, end_sec in seg_times:
                rows.append({
                    "segment_id": None, "group_id": group_id, "site": "utah_2019", "view": view_name,
                    "data_type": data_type, "label": label, "file_path": file_path, "file_name": file_name,
                    "file_stem": file_stem, "start_sec": start_sec, "end_sec": end_sec,
                    "ch_start": int(ch_cfg["ch_start"]), "ch_end": int(ch_cfg["ch_end"]),
                })
    return rows

def main():
    inv_df = pd.read_csv(INVENTORY_CSV)
    inv_df = inv_df[inv_df["ext"].astype(str).str.lower().isin(VALID_EXTS)].copy()
    noise_inv = inv_df[inv_df["data_type"] == "noise"].copy()
    unlabel_inv = inv_df[inv_df["data_type"] == "unlabel"].copy()

    noise_rows = []; unlabel_rows = []
    print(f"[INFO] noise source files   : {len(noise_inv)}")
    print(f"[INFO] unlabel source files: {len(unlabel_inv)}")

    for _, row in noise_inv.iterrows():
        try:
            duration_sec = get_duration_sec(row["file_path"], row["site"])
            rows = make_rows(row, duration_sec, "noise", 0, MAX_NOISE_SEGMENTS_PER_FILE)
            noise_rows.extend(rows)
            print(f"[NOISE] {row['file_name']} -> {len(rows)} segments")
        except Exception as e:
            print(f"[WARN][NOISE] skip: {row['file_path']} | {e}")

    for _, row in unlabel_inv.iterrows():
        try:
            duration_sec = get_duration_sec(row["file_path"], row["site"])
            rows = make_rows(row, duration_sec, "unlabel", -1, MAX_UNLABEL_SEGMENTS_PER_FILE)
            unlabel_rows.extend(rows)
            print(f"[UNLABEL] {row['file_name']} -> {len(rows)} segments")
        except Exception as e:
            print(f"[WARN][UNLABEL] skip: {row['file_path']} | {e}")

    noise_df = pd.DataFrame(noise_rows)
    unlabel_df = pd.DataFrame(unlabel_rows)

    if len(noise_df) > 0:
        noise_df = noise_df.reset_index(drop=True)
        noise_df["segment_id"] = np.arange(len(noise_df), dtype=int)
        noise_df.to_csv(OUT_NOISE_CSV, index=False)
        print(f"[SAVED] {OUT_NOISE_CSV}")
        print(f"[TOTAL NOISE SEGMENTS] {len(noise_df)}")
        print(noise_df.groupby(["site", "view"]).size())
    else:
        print("[WARN] noise segment count is 0.")

    if len(unlabel_df) > 0:
        unlabel_df = unlabel_df.reset_index(drop=True)
        unlabel_df["segment_id"] = np.arange(len(unlabel_df), dtype=int)
        unlabel_df.to_csv(OUT_UNLABEL_CSV, index=False)
        print(f"[SAVED] {OUT_UNLABEL_CSV}")
        print(f"[TOTAL UNLABEL SEGMENTS] {len(unlabel_df)}")
        print(unlabel_df.groupby(["site", "view"]).size())
    else:
        print("[WARN] unlabel segment count is 0.")

if __name__ == "__main__":
    main()
