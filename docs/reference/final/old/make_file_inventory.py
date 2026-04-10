#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


OUT_DIR = Path("/home/ted1204/MS_datamaker_merge/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCAN_TARGETS = [
    ("pohang", "event",   Path("/home/ted1204/MS_datamaker_PH/data/event")),
    ("pohang", "noise",   Path("/home/ted1204/MS_datamaker_PH/data/noise")),
    ("pohang", "unlabel", Path("/home/ted1204/MS_datamaker_PH/data/unlabel")),

    ("utah",   "event",   Path("/home/ted1204/MS_datamaker_UTAH/data/event")),
    ("utah",   "noise",   Path("/home/ted1204/MS_datamaker_UTAH/data/noise")),
    ("utah",   "unlabel", Path("/home/ted1204/MS_datamaker_UTAH/data/unlabel")),
]

VALID_EXTS = {".tdms", ".sgy", ".segy", ".h5", ".hdf5"}


def scan_files(site, data_type, root_dir, start_group_id):

    rows = []
    group_id = start_group_id

    if not root_dir.exists():
        print(f"[WARN] not found: {root_dir}")
        return rows, group_id

    files = sorted(
        p for p in root_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )

    for p in files:

        rows.append({
            "group_id": group_id,
            "site": site,
            "data_type": data_type,
            "file_path": str(p.resolve()),
            "file_name": p.name,
            "file_stem": p.stem,
            "ext": p.suffix.lower(),
            "parent_dir": p.parent.name,
        })

        group_id += 1

    print(f"[INFO] {site} | {data_type} -> {len(rows)} files")

    return rows, group_id


def main():

    all_rows = []
    next_group_id = 0

    for site, data_type, root_dir in SCAN_TARGETS:

        rows, next_group_id = scan_files(
            site=site,
            data_type=data_type,
            root_dir=root_dir,
            start_group_id=next_group_id,
        )

        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    out_csv = OUT_DIR / "file_inventory.csv"

    df.to_csv(out_csv, index=False)

    print("\n[SAVED]", out_csv)
    print("[TOTAL FILES]", len(df))
    print("[LAST GROUP ID]", next_group_id - 1 if len(df) > 0 else -1)


if __name__ == "__main__":
    main()