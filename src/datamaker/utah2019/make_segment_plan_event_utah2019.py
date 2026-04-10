#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

CHANNEL_CONFIG = {
    "utah_2019_1": {"ch_start": 679, "ch_end": 1084},
    "utah_2019_2": {"ch_start": 273, "ch_end": 678},
}

def normalize_name(x):
    if pd.isna(x):
        return None
    return Path(str(x).strip().replace("\\", "/")).name

def normalize_stem(x):
    if pd.isna(x):
        return None
    return Path(str(x).strip().replace("\\", "/")).stem

def find_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def auto_find_file_col(df):
    candidates = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["file", "path", "name", "sgy", "segy"]):
            candidates.append(c)
    if not candidates:
        raise ValueError("Could not detect file column.")
    return candidates[0]

def auto_find_label_col(df):
    col = find_col(df, ["is_event", "event", "label", "class", "target"])
    return col

def auto_find_time_cols(df):
    start_col = find_col(df, ["start_sec", "start_s", "start"])
    end_col = find_col(df, ["end_sec", "end_s", "end"])
    center_col = find_col(df, ["center_sec", "time_sec", "event_sec", "sec", "time"])
    return start_col, end_col, center_col

def is_event_value(x):
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "event"}:
        return True
    try:
        return float(s) == 1.0
    except Exception:
        return False

def make_rows(label_df: pd.DataFrame, inv_df: pd.DataFrame, view_name: str, source_label_csv: str, segment_sec: float):
    file_col = auto_find_file_col(label_df)
    label_col = auto_find_label_col(label_df)
    start_col, end_col, center_col = auto_find_time_cols(label_df)

    event_inv = inv_df[inv_df["data_type"] == "event"].copy()
    event_inv["file_name_norm"] = event_inv["file_name"].map(normalize_name)
    event_inv["file_stem_norm"] = event_inv["file_stem"].map(normalize_stem)

    ch_cfg = CHANNEL_CONFIG[view_name]
    rows = []

    for _, r in label_df.iterrows():
        if label_col is not None and not is_event_value(r[label_col]):
            continue

        f_name = normalize_name(r[file_col])
        f_stem = normalize_stem(r[file_col])

        hit = event_inv[event_inv["file_name_norm"] == f_name]
        if len(hit) == 0:
            hit = event_inv[event_inv["file_stem_norm"] == f_stem]
        if len(hit) == 0:
            continue

        inv_row = hit.iloc[0]

        if start_col is not None and end_col is not None:
            seg_start = float(r[start_col])
            seg_end = float(r[end_col])
        elif center_col is not None:
            c = float(r[center_col])
            seg_start = c - segment_sec / 2.0
            seg_end = c + segment_sec / 2.0
        else:
            raise ValueError("Could not find time columns.")

        if np.isnan(seg_start) or np.isnan(seg_end):
            continue
        seg_start = max(0.0, seg_start)
        if seg_end <= seg_start:
            seg_end = seg_start + segment_sec

        rows.append({
            "segment_id": None,
            "group_id": inv_row["group_id"],
            "dataset_id": "utah_2019",
            "site": "utah_2019",
            "view": view_name,
            "data_type": "event",
            "label": 1,
            "file_path": inv_row["file_path"],
            "file_name": inv_row["file_name"],
            "file_stem": inv_row["file_stem"],
            "start_sec": seg_start,
            "end_sec": seg_end,
            "ch_start": int(ch_cfg["ch_start"]),
            "ch_end": int(ch_cfg["ch_end"]),
            "original_fs": float(inv_row["original_fs"]),
            "source_label_csv": source_label_csv,
        })

    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory_csv", required=True)
    ap.add_argument("--lower_label_csv", required=True)
    ap.add_argument("--upper_label_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--segment_sec", type=float, default=2.0)
    args = ap.parse_args()

    inv_df = pd.read_csv(args.inventory_csv)
    lower_df = pd.read_csv(args.lower_label_csv)
    upper_df = pd.read_csv(args.upper_label_csv)

    rows = []
    rows.extend(make_rows(lower_df, inv_df, "utah_2019_1", Path(args.lower_label_csv).name, args.segment_sec))
    rows.extend(make_rows(upper_df, inv_df, "utah_2019_2", Path(args.upper_label_csv).name, args.segment_sec))

    out_df = pd.DataFrame(rows).reset_index(drop=True)
    out_df["segment_id"] = np.arange(len(out_df), dtype=int)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print(f"[DONE] saved: {args.out_csv}")
    print(f"[INFO] rows: {len(out_df)}")
    if len(out_df) > 0:
        print(out_df.groupby(['site', 'view']).size())

if __name__ == "__main__":
    main()