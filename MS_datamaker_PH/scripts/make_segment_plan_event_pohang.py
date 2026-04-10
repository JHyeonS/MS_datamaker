#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


SEGMENT_SEC = 2.0

CHANNEL_CONFIG = {
    "pohang": {
        "ch_start": 243,
        "ch_end": 648,
    }
}


def normalize_name(x):
    if pd.isna(x):
        return None
    x = str(x).strip().replace("\\", "/")
    return Path(x).name


def normalize_stem(x):
    if pd.isna(x):
        return None
    x = str(x).strip().replace("\\", "/")
    return Path(x).stem


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
        if any(k in lc for k in ["file", "path", "name", "tdms"]):
            cand_cols.append(c)

    if not cand_cols:
        cand_cols = list(label_df.columns)

    inv_names = set(inv_df["file_name"].astype(str).tolist())
    inv_stems = set(inv_df["file_stem"].astype(str).tolist())

    best_col = None
    best_score = -1

    for c in cand_cols:
        score = 0
        series = label_df[c].dropna().astype(str).head(200)

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


def auto_find_label_col(df: pd.DataFrame) -> str | None:
    preferred = ["is_event", "event", "label", "class", "target", "category", "type"]
    col = find_col(df, preferred)
    if col is not None:
        return col

    for c in df.columns:
        s = df[c].dropna().astype(str).str.lower()
        values = set(s.unique().tolist())
        if any(v in values for v in ["event", "noise", "unlabel", "microseismic", "ms"]):
            return c

    return None


def auto_find_time_cols(df: pd.DataFrame):
    start_col = find_col(df, ["start_sec", "start_s", "start", "t_start", "begin_sec", "onset_sec"])
    end_col = find_col(df, ["end_sec", "end_s", "end", "t_end", "stop_sec", "offset_sec"])

    if start_col is not None and end_col is not None:
        return start_col, end_col, None

    center_col = find_col(df, ["center_sec", "time_sec", "sec", "time", "timestamp_sec", "event_sec"])
    return start_col, end_col, center_col


def is_event_value(x) -> bool:
    if pd.isna(x):
        return False

    s = str(x).strip().lower()

    if s in {"1", "true", "yes", "y", "event", "microseismic", "ms"}:
        return True

    if s in {"0", "false", "no", "n", "noise", "unlabel", "unknown", "ignore", "ignored"}:
        return False

    try:
        return float(s) == 1.0
    except Exception:
        return False


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


def make_event_rows(
    label_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    source_name: str,
) -> list[dict]:
    event_inventory_df = inventory_df[inventory_df["data_type"] == "event"].copy()
    inv_by_name, inv_by_stem = build_inventory_lookup(event_inventory_df)

    file_col = auto_find_file_col(label_df, event_inventory_df)
    label_col = auto_find_label_col(label_df)
    start_col, end_col, center_col = auto_find_time_cols(label_df)

    rows = []

    for _, r in label_df.iterrows():
        if label_col is not None and not is_event_value(r[label_col]):
            continue

        inv_row = match_inventory_row(
            raw_file_value=r[file_col],
            inv_by_name=inv_by_name,
            inv_by_stem=inv_by_stem,
        )

        if inv_row is None:
            continue

        if start_col is not None and end_col is not None:
            try:
                seg_start = float(r[start_col])
                seg_end = float(r[end_col])
            except Exception:
                continue
        elif center_col is not None:
            try:
                center = float(r[center_col])
            except Exception:
                continue
            seg_start = center - SEGMENT_SEC / 2.0
            seg_end = center + SEGMENT_SEC / 2.0
        else:
            raise ValueError(f"{source_name}: could not find time columns.")

        if np.isnan(seg_start) or np.isnan(seg_end):
            continue

        if seg_end <= seg_start:
            seg_end = seg_start + SEGMENT_SEC

        seg_start = max(0.0, seg_start)

        ch_cfg = CHANNEL_CONFIG["pohang"]

        rows.append({
            "segment_id": None,
            "group_id": str(inv_row["group_id"]),
            "dataset_id": "pohang",
            "site": "pohang",
            "view": "pohang",
            "data_type": "event",
            "label": 1,
            "file_path": str(inv_row["file_path"]),
            "file_name": str(inv_row["file_name"]),
            "file_stem": str(inv_row["file_stem"]),
            "start_sec": float(seg_start),
            "end_sec": float(seg_end),
            "ch_start": int(ch_cfg["ch_start"]),
            "ch_end": int(ch_cfg["ch_end"]),
            "original_fs": float(inv_row["original_fs"]),
            "source_label_csv": source_name,
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory_csv", required=True)
    parser.add_argument("--label_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    inventory_df = pd.read_csv(args.inventory_csv)
    label_df = pd.read_csv(args.label_csv)

    all_rows = make_event_rows(
        label_df=label_df,
        inventory_df=inventory_df,
        source_name=Path(args.label_csv).name,
    )

    out_df = pd.DataFrame(all_rows)

    if len(out_df) == 0:
        raise RuntimeError("No event segments found.")

    out_df = out_df.reset_index(drop=True)
    out_df["segment_id"] = np.arange(len(out_df), dtype=int)

    bad_len = out_df["end_sec"] <= out_df["start_sec"]
    out_df.loc[bad_len, "end_sec"] = out_df.loc[bad_len, "start_sec"] + SEGMENT_SEC

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print(f"[SAVED] {args.out_csv}")
    print(f"[TOTAL EVENT SEGMENTS] {len(out_df)}")
    print(out_df.groupby(["site", "view"]).size())


if __name__ == "__main__":
    main()