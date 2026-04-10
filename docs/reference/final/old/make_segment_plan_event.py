#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path("/home/ted1204/MS_datamaker_merge")

INVENTORY_CSV = ROOT / "outputs" / "file_inventory.csv"

POHANG_LABEL_CSV = ROOT / "csv" / "pohang_labels_2s.csv"
UTAH_LOWER_LABEL_CSV = ROOT / "csv" / "utah_2019_labels_2s_1.csv"
UTAH_UPPER_LABEL_CSV = ROOT / "csv" / "utah_2019_labels_2s_2.csv"

OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "segment_plan_event.csv"

SEGMENT_SEC = 2.0

CHANNEL_CONFIG = {
    "pohang": {
        "ch_start": 243,
        "ch_end": 648,
    },
    "utah_lower": {
        "ch_start": 679,
        "ch_end": 1084,
    },
    "utah_upper": {
        "ch_start": 273,
        "ch_end": 678,
    },
}


def normalize_name(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    x = x.replace("\\", "/")
    return Path(x).name


def normalize_stem(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    x = x.replace("\\", "/")
    return Path(x).stem


def lower_cols(df):
    return {c.lower(): c for c in df.columns}


def find_col(df, candidates):
    colmap = lower_cols(df)
    for cand in candidates:
        if cand.lower() in colmap:
            return colmap[cand.lower()]
    return None


def auto_find_file_col(label_df, inv_df):
    cand_cols = []
    for c in label_df.columns:
        lc = c.lower()
        if any(k in lc for k in ["file", "path", "name", "tdms", "sgy", "segy"]):
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
        raise ValueError("file/path 컬럼을 찾지 못했습니다.")

    return best_col


def auto_find_label_col(df):
    preferred = [
        "label", "class", "target", "category", "type",
        "event_label", "is_event"
    ]
    col = find_col(df, preferred)
    if col is not None:
        return col

    for c in df.columns:
        s = df[c].dropna().astype(str).str.lower()
        values = set(s.unique().tolist())
        if any(v in values for v in ["event", "noise", "unlabel", "microseismic", "ms"]):
            return c

    return None


def auto_find_time_cols(df):
    start_col = find_col(df, [
        "start_sec", "start_s", "start", "t_start", "begin_sec", "onset_sec"
    ])
    end_col = find_col(df, [
        "end_sec", "end_s", "end", "t_end", "stop_sec", "offset_sec"
    ])

    if start_col is not None and end_col is not None:
        return start_col, end_col, None

    center_col = find_col(df, [
        "center_sec", "time_sec", "sec", "time", "timestamp_sec", "event_sec"
    ])

    return start_col, end_col, center_col


def is_event_value(x):
    if pd.isna(x):
        return False

    s = str(x).strip().lower()

    if s in {"1", "true", "yes", "y", "event", "microseismic", "ms"}:
        return True

    if s in {"0", "false", "no", "n", "noise", "unlabel", "unknown"}:
        return False

    try:
        return float(s) == 1.0
    except Exception:
        return False


def build_inventory_lookup(inv_df):
    by_name = {}
    by_stem = {}

    for _, row in inv_df.iterrows():
        by_name.setdefault(str(row["file_name"]), []).append(row)
        by_stem.setdefault(str(row["file_stem"]), []).append(row)

    return by_name, by_stem


def match_inventory_row(raw_file_value, inv_by_name, inv_by_stem, site=None):
    name = normalize_name(raw_file_value)
    stem = normalize_stem(raw_file_value)

    candidates = []

    if name in inv_by_name:
        candidates.extend(inv_by_name[name])

    if not candidates and stem in inv_by_stem:
        candidates.extend(inv_by_stem[stem])

    if not candidates:
        return None

    if site is not None:
        site_filtered = [r for r in candidates if str(r["site"]) == site]
        if len(site_filtered) == 1:
            return site_filtered[0]
        if len(site_filtered) > 1:
            candidates = site_filtered

    return candidates[0]


def make_event_rows(label_df, inventory_df, source_name, view_name):
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
            site="pohang" if view_name == "pohang" else "utah",
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
            raise ValueError(
                f"{source_name}: time 컬럼(start/end 또는 center time)을 찾지 못했습니다."
            )

        if np.isnan(seg_start) or np.isnan(seg_end):
            continue

        if seg_end <= seg_start:
            seg_end = seg_start + SEGMENT_SEC

        ch_cfg = CHANNEL_CONFIG[view_name]

        rows.append({
            "segment_id": None,
            "group_id": int(inv_row["group_id"]),
            "site": "pohang" if view_name == "pohang" else "utah",
            "view": view_name,
            "data_type": "event",
            "label": 1,
            "file_path": str(inv_row["file_path"]),
            "file_name": str(inv_row["file_name"]),
            "file_stem": str(inv_row["file_stem"]),
            "start_sec": float(seg_start),
            "end_sec": float(seg_end),
            "ch_start": int(ch_cfg["ch_start"]),
            "ch_end": int(ch_cfg["ch_end"]),
            "source_label_csv": source_name,
        })

    return rows


def main():
    inventory_df = pd.read_csv(INVENTORY_CSV)

    pohang_df = pd.read_csv(POHANG_LABEL_CSV)
    utah_lower_df = pd.read_csv(UTAH_LOWER_LABEL_CSV)
    utah_upper_df = pd.read_csv(UTAH_UPPER_LABEL_CSV)

    all_rows = []

    all_rows.extend(
        make_event_rows(
            label_df=pohang_df,
            inventory_df=inventory_df,
            source_name="pohang_labels_2s.csv",
            view_name="pohang",
        )
    )

    all_rows.extend(
        make_event_rows(
            label_df=utah_lower_df,
            inventory_df=inventory_df,
            source_name="utah_2019_labels_2s_1.csv",
            view_name="utah_lower",
        )
    )

    all_rows.extend(
        make_event_rows(
            label_df=utah_upper_df,
            inventory_df=inventory_df,
            source_name="utah_2019_labels_2_2s.csv",
            view_name="utah_upper",
        )
    )

    out_df = pd.DataFrame(all_rows)

    if len(out_df) == 0:
        raise RuntimeError("event segment가 0개입니다. file/time/label 컬럼명을 확인하세요.")

    out_df = out_df.reset_index(drop=True)
    out_df["segment_id"] = np.arange(len(out_df), dtype=int)

    out_df["start_sec"] = out_df["start_sec"].clip(lower=0.0)

    bad_len = out_df["end_sec"] <= out_df["start_sec"]
    out_df.loc[bad_len, "end_sec"] = out_df.loc[bad_len, "start_sec"] + SEGMENT_SEC

    out_df.to_csv(OUT_CSV, index=False)

    print(f"[SAVED] {OUT_CSV}")
    print(f"[TOTAL EVENT SEGMENTS] {len(out_df)}")
    print(out_df.groupby(["site", "view"]).size())


if __name__ == "__main__":
    main()