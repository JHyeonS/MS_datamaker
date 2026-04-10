#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

VALID_EXTS = {".sgy", ".segy"}

def scan_one(root_dir: Path, data_type: str) -> list[dict]:
    rows = []
    files = sorted(
        p for p in root_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )

    for p in files:
        rows.append({
            "dataset_id": "utah_2019",
            "site": "utah_2019",
            "data_type": data_type,
            "file_path": str(p.resolve()),
            "file_name": p.name,
            "file_stem": p.stem,
            "ext": p.suffix.lower(),
            "group_id": f"utah_2019__{p.stem}",
            "original_fs": 1000.0,
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_dir", required=True)
    ap.add_argument("--noise_dir", required=True)
    ap.add_argument("--unlabel_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows = []
    rows += scan_one(Path(args.event_dir), "event")
    rows += scan_one(Path(args.noise_dir), "noise")
    rows += scan_one(Path(args.unlabel_dir), "unlabel")

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print(f"[DONE] saved: {args.out_csv}")
    print(df["data_type"].value_counts())

if __name__ == "__main__":
    main()