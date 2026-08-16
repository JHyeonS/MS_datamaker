#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import re
from pathlib import Path
import pandas as pd


SEGMENT_SEC = 2.0


def normalize_name(x):
    if pd.isna(x):
        return None
    x = str(x).strip().replace("\\", "/")
    name = Path(x).name

    # 예: 0000_RAW_20230719_014658.140.sgy -> RAW_20230719_014658.140.sgy
    name = re.sub(r"^\d+_(RAW_.*)$", r"\1", name)

    return name


def normalize_stem(x):
    if pd.isna(x):
        return None
    x = str(x).strip().replace("\\", "/")
    stem = Path(x).stem
    stem = re.sub(r"^\d+_(RAW_.*)$", r"\1", stem)
    return stem


def lower_cols(df: pd.DataFrame) -> dict:
    return {c.lower(): c for c in df.columns}


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    colmap = lower_cols(df)
    for cand in candidates:
        if cand.lower() in colmap:
            return colmap[cand.lower()]
    return None


def auto_find_file_col(label_df: pd.DataFrame, inv_df: pd.DataFrame) -> str:
    cand_cols = []
    for c in label_df.columns:
        lc = c.lower()
        if any(k in lc for k in ["shot", "file", "path", "name", "sgy", "segy", "raw"]):
            cand_cols.append(c)

    if not cand_cols:
        cand_cols = list(label_df.columns)

    inv_names = set(inv_df["file_name"].astype(str).tolist())
    inv_stems = set(inv_df["file_stem"].astype(str).tolist())

    best_col = None
    best_score = -1

    for c in cand_cols:
        score = 0
        series = label_df[c].dropna().astype(str).head(300)

        for v in series:
            if normalize_name(v) in inv_names:
                score += 2
            elif normalize_stem(v) in inv_stems:
                score += 1

        if score > best_score:
            best_score = score
            best_col = c

    if best_col is None:
        raise ValueError("Could not find file/path column in label csv.")

    return best_col


def auto_find_label_col(df: pd.DataFrame) -> str:
    preferred = [
        "label", "label_name", "class_name", "class", "target",
        "category", "type", "is_event"
    ]
    col = find_col(df, preferred)
    if col is None:
        raise ValueError(f"Could not find label column. columns={df.columns.tolist()}")
    return col


def auto_find_time_cols(df: pd.DataFrame):
    start_col = find_col(df, [
        "start_sec", "start_s", "start", "t_start",
        "begin_sec", "onset_sec"
    ])
    end_col = find_col(df, [
        "end_sec", "end_s", "end", "t_end",
        "stop_sec", "offset_sec"
    ])

    if start_col is not None and end_col is not None:
        return start_col, end_col, None

    center_col = find_col(df, [
        "center_sec", "time_sec", "sec", "time",
        "timestamp_sec", "event_sec"
    ])
    if center_col is not None:
        return None, None, center_col

    raise ValueError("Could not find time columns.")


def auto_find_channel_cols(df: pd.DataFrame):
    ch_start_col = find_col(df, [
        "ch_start", "channel_start", "start_ch", "c0"
    ])
    ch_end_col = find_col(df, [
        "ch_end", "channel_end", "end_ch", "c1"
    ])

    if ch_start_col is None or ch_end_col is None:
        raise ValueError(
            "Could not find channel columns. "
            "Need something like ch_start/ch_end."
        )

    return ch_start_col, ch_end_col


def parse_label_value(x):
    if pd.isna(x):
        return None, None

    s = str(x).strip().lower()

    if s in {"1", "event", "microseismic", "ms", "true", "yes", "y"}:
        return 1, "event"

    if s in {"0", "noise", "false", "no", "n"}:
        return 0, "noise"

    if s in {"2", "-1", "unlabel", "unlabeled", "unlabelled", "ignore", "ignored"}:
        return 2, "unlabel"

    try:
        v = int(float(s))
        if v == 1:
            return 1, "event"
        if v == 0:
            return 0, "noise"
        if v in {-1, 2}:
            return 2, "unlabel"
    except Exception:
        pass

    return None, None


def build_inventory_lookup(inv_df: pd.DataFrame):
    by_name = {}
    by_stem = {}

    for _, row in inv_df.iterrows():
        by_name.setdefault(str(row["file_name"]), []).append(row)
        by_stem.setdefault(str(row["file_stem"]), []).append(row)

    return by_name, by_stem


def match_inventory_row(raw_file_value, inv_by_name, inv_by_stem):
    name = normalize_name(raw_file_value)
    stem = normalize_stem(raw_file_value)

    candidates = []
    if name in inv_by_name:
        candidates.extend(inv_by_name[name])

    if not candidates and stem in inv_by_stem:
        candidates.extend(inv_by_stem[stem])

    if not candidates:
        return None

    return candidates[0]


def build_all_segments(inventory_df: pd.DataFrame, label_df: pd.DataFrame, source_label_csv: str) -> pd.DataFrame:
    inv_by_name, inv_by_stem = build_inventory_lookup(inventory_df)

    file_col = auto_find_file_col(label_df, inventory_df)
    label_col = auto_find_label_col(label_df)
    start_col, end_col, center_col = auto_find_time_cols(label_df)
    ch_start_col, ch_end_col = auto_find_channel_cols(label_df)

    rows = []

    for _, r in label_df.iterrows():
        label_id, data_type = parse_label_value(r[label_col])
        if label_id is None:
            continue

        inv_row = match_inventory_row(
            raw_file_value=r[file_col],
            inv_by_name=inv_by_name,
            inv_by_stem=inv_by_stem,
        )
        if inv_row is None:
            continue

        if center_col is not None:
            try:
                center = float(r[center_col])
            except Exception:
                continue
            seg_start = center - SEGMENT_SEC / 2.0
            seg_end = center + SEGMENT_SEC / 2.0
        else:
            try:
                seg_start = float(r[start_col])
                seg_end = float(r[end_col])
            except Exception:
                continue

        try:
            ch_start = int(r[ch_start_col])
            ch_end = int(r[ch_end_col])
        except Exception:
            continue

        if np.isnan(seg_start) or np.isnan(seg_end):
            continue

        seg_start = max(0.0, seg_start)
        if seg_end <= seg_start:
            seg_end = seg_start + SEGMENT_SEC

        if ch_end <= ch_start:
            continue

        rows.append({
            "segment_id": None,
            "group_id": str(inv_row["group_id"]),
            "dataset_id": "utah_2023",
            "site": str(inv_row["site"]),
            "view": str(inv_row["view"]),
            "data_type": data_type,
            "label": int(label_id),
            "file_path": str(inv_row["file_path"]),
            "file_name": str(inv_row["file_name"]),
            "file_stem": str(inv_row["file_stem"]),
            "start_sec": float(seg_start),
            "end_sec": float(seg_end),
            "ch_start": int(ch_start),
            "ch_end": int(ch_end),
            "original_fs": float(inv_row["original_fs"]),
            "source_label_csv": source_label_csv,
        })

    out_df = pd.DataFrame(rows)
    if len(out_df) == 0:
        raise RuntimeError("No valid segments found from label csv and inventory matching.")

    out_df = out_df.reset_index(drop=True)
    out_df["segment_id"] = np.arange(len(out_df), dtype=int)
    return out_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory_csv", required=True)
    parser.add_argument("--label_csv", required=True)
    parser.add_argument("--out_event_csv", required=True)
    parser.add_argument("--out_noise_csv", required=True)
    parser.add_argument("--out_unlabel_csv", required=True)
    parser.add_argument("--out_all_csv", default=None)
    args = parser.parse_args()

    inventory_df = pd.read_csv(args.inventory_csv)
    label_df = pd.read_csv(args.label_csv)

    all_df = build_all_segments(
        inventory_df=inventory_df,
        label_df=label_df,
        source_label_csv=Path(args.label_csv).name,
    )

    event_df = all_df[all_df["label"] == 1].copy().reset_index(drop=True)
    noise_df = all_df[all_df["label"] == 0].copy().reset_index(drop=True)
    unlabel_df = all_df[all_df["label"] == 2].copy().reset_index(drop=True)

    if len(event_df) > 0:
        event_df["segment_id"] = np.arange(len(event_df), dtype=int)
    if len(noise_df) > 0:
        noise_df["segment_id"] = np.arange(len(noise_df), dtype=int)
    if len(unlabel_df) > 0:
        unlabel_df["segment_id"] = np.arange(len(unlabel_df), dtype=int)

    out_dir = Path(args.out_event_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    event_df.to_csv(args.out_event_csv, index=False, encoding="utf-8-sig")
    noise_df.to_csv(args.out_noise_csv, index=False, encoding="utf-8-sig")
    unlabel_df.to_csv(args.out_unlabel_csv, index=False, encoding="utf-8-sig")

    if args.out_all_csv is not None:
        all_df.to_csv(args.out_all_csv, index=False, encoding="utf-8-sig")

    print(f"[SAVED] {args.out_event_csv} | n={len(event_df)}")
    print(f"[SAVED] {args.out_noise_csv} | n={len(noise_df)}")
    print(f"[SAVED] {args.out_unlabel_csv} | n={len(unlabel_df)}")
    if args.out_all_csv is not None:
        print(f"[SAVED] {args.out_all_csv} | n={len(all_df)}")


if __name__ == "__main__":
    main()
