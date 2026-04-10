#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd

VALID_EXTS = {".sgy", ".segy"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--site", default="utah_2023")
    parser.add_argument("--view", default="utah_2023")
    parser.add_argument("--original_fs", type=float, default=1000.0)
    parser.add_argument("--ch_start", type=int, required=True)
    parser.add_argument("--ch_end", type=int, required=True)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    files = sorted(
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )

    rows = []
    for p in files:
        rows.append({
            "dataset_id": "utah_2023",
            "site": args.site,
            "view": args.view,
            "file_path": str(p.resolve()),
            "file_name": p.name,
            "file_stem": p.stem,
            "ext": p.suffix.lower(),
            "group_id": f"utah_2023__{p.stem}",
            "original_fs": float(args.original_fs),
            "ch_start": int(args.ch_start),
            "ch_end": int(args.ch_end),
        })

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print(f"[DONE] saved: {args.out_csv}")
    print(f"[INFO] n_files: {len(df)}")


if __name__ == "__main__":
    main()