#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert 1-second sliding labels to 2-second sliding labels.

Example
-------
Input 1s windows:
  seg0: [0,1)
  seg1: [1,2)
  seg2: [2,3)
  seg3: [3,4)

Output 2s windows:
  seg0_2s: [0,2) -> event if seg0 or seg1 is event
  seg1_2s: [1,3) -> event if seg1 or seg2 is event
  seg2_2s: [2,4) -> event if seg2 or seg3 is event

So the new label is:
  new_is_event(i) = old_is_event(i) OR old_is_event(i+1)

Expected input columns:
  - start_sec
  - end_sec
and one event column among:
  - is_event / event / label / class / target
and one file id column among:
  - tdms_file / sgy_file / file_name / shot_id / file_stem / file_path

Output columns:
  - same metadata columns as much as possible
  - segment_index
  - start_sec
  - end_sec
  - is_event
  - src_segment_indices
  - src_event_flags
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import pandas as pd


ID_CANDIDATES = ["tdms_file", "sgy_file", "file_name", "shot_id", "file_stem", "file_path"]
EVENT_CANDIDATES = ["is_event", "event", "label", "class", "target"]


def find_existing_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find any of columns: {candidates}")


def to_event_flag(x) -> int:
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "event"}:
        return 1
    if s in {"0", "false", "no", "n", "noise", "unlabel", "unlabeled", "ignore", "ignored"}:
        return 0
    try:
        return 1 if float(s) == 1.0 else 0
    except Exception:
        raise ValueError(f"Could not parse event flag from value: {x}")


def convert_one_group(g: pd.DataFrame, id_col: str, event_col: str) -> pd.DataFrame:
    g = g.copy().sort_values(["start_sec", "end_sec"]).reset_index(drop=True)

    rows = []
    if len(g) < 2:
        return pd.DataFrame(rows)

    for i in range(len(g) - 1):
        r0 = g.iloc[i]
        r1 = g.iloc[i + 1]

        start0 = float(r0["start_sec"])
        end0 = float(r0["end_sec"])
        start1 = float(r1["start_sec"])
        end1 = float(r1["end_sec"])

        # Require contiguous 1-second sliding windows: [t,t+1), [t+1,t+2)
        if abs(end0 - start1) > 1e-8:
            continue

        e0 = to_event_flag(r0[event_col])
        e1 = to_event_flag(r1[event_col])
        new_event = 1 if (e0 == 1 or e1 == 1) else 0

        out = {}

        # keep other metadata from first row when available
        for c in g.columns:
            if c in {"segment_index", "start_sec", "end_sec", event_col, "is_event",
                     "src_segment_indices", "src_event_flags"}:
                continue
            out[c] = r0[c]

        out["segment_index"] = i
        out["start_sec"] = start0
        out["end_sec"] = end1
        out["is_event"] = new_event
        if "label" in g.columns and event_col != "label":
            # preserve original label column only if desired in metadata;
            # new canonical flag is is_event
            pass
        out["src_segment_indices"] = f"{i},{i+1}"
        out["src_event_flags"] = f"{e0},{e1}"

        rows.append(out)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)

    df = pd.read_csv(in_csv)

    for c in ["start_sec", "end_sec"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    id_col = find_existing_col(df, ID_CANDIDATES)
    event_col = find_existing_col(df, EVENT_CANDIDATES)

    out_frames = []
    for _, g in df.groupby(id_col, sort=False):
        out_frames.append(convert_one_group(g, id_col=id_col, event_col=event_col))

    out_df = pd.concat(out_frames, axis=0, ignore_index=True) if out_frames else pd.DataFrame()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[DONE] saved: {out_csv}")
    print(f"[INFO] input rows : {len(df)}")
    print(f"[INFO] output rows: {len(out_df)}")
    if len(out_df) > 0:
        print("[INFO] is_event counts:")
        print(out_df["is_event"].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()
