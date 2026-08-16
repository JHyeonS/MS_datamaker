#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

from datamaker.final.raw_metadata import infer_raw_sampling_rate

VALID_EXTS = {".tdms", ".sgy", ".segy"}

def scan_one(root_dir: Path, data_type: str, site: str, view: str,
             original_fs: float, infer_header_fs: bool, ch_start: int, ch_end: int) -> list[dict]:
    rows = []
    files = sorted(
        p for p in root_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )

    for p in files:
        fs = infer_raw_sampling_rate(p, fallback_fs=original_fs) if infer_header_fs else float(original_fs)
        rows.append({
            "dataset_id": "pohang",
            "site": site,
            "view": view,
            "data_type": data_type,
            "file_path": str(p.resolve()),
            "file_name": p.name,
            "file_stem": p.stem,
            "ext": p.suffix.lower(),
            "group_id": f"pohang__{p.stem}",
            "original_fs": float(fs),
            "ch_start": int(ch_start),
            "ch_end": int(ch_end),
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_dir", required=True)
    ap.add_argument("--noise_dir", required=True)
    ap.add_argument("--unlabel_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--site", default="pohang")
    ap.add_argument("--view", default="pohang")
    ap.add_argument("--original_fs", type=float, default=1000.0,
                    help="Logical sampling rate used for segment-plan time coordinates.")
    ap.add_argument("--infer_header_fs", action="store_true",
                    help="Opt in to raw-header fs inference. Default keeps the legacy/logical fs convention.")
    ap.add_argument("--ch_start", type=int, default=243)
    ap.add_argument("--ch_end", type=int, default=648)
    args = ap.parse_args()

    rows = []
    rows += scan_one(Path(args.event_dir), "event", args.site, args.view,
                     args.original_fs, args.infer_header_fs, args.ch_start, args.ch_end)
    rows += scan_one(Path(args.noise_dir), "noise", args.site, args.view,
                     args.original_fs, args.infer_header_fs, args.ch_start, args.ch_end)
    rows += scan_one(Path(args.unlabel_dir), "unlabel", args.site, args.view,
                     args.original_fs, args.infer_header_fs, args.ch_start, args.ch_end)

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] saved: {args.out_csv}")
    print(df["data_type"].value_counts())
    print(df["original_fs"].value_counts().sort_index())

if __name__ == "__main__":
    main()
