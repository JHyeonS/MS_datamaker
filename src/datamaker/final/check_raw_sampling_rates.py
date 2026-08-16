#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from datamaker.final.raw_metadata import infer_raw_sampling_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Inventory or segment-plan CSV with file_path and original_fs columns.")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("--max_files", type=int, default=None, help="Optional smoke-test limit over unique raw files.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    required = {"file_path", "original_fs"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cols = [c for c in ["dataset_id", "site", "data_type", "label", "file_path", "original_fs"] if c in df.columns]
    unique = df[cols].drop_duplicates("file_path").copy()
    if args.max_files is not None:
        unique = unique.head(args.max_files).copy()

    rows = []
    for row in unique.itertuples(index=False):
        item = row._asdict()
        file_path = item["file_path"]
        metadata_fs = float(item["original_fs"])
        try:
            header_fs = infer_raw_sampling_rate(file_path)
            error = ""
        except Exception as exc:
            header_fs = float("nan")
            error = repr(exc)
        diff = abs(float(header_fs) - metadata_fs) if pd.notna(header_fs) else float("nan")
        item.update({
            "metadata_fs": metadata_fs,
            "header_fs": header_fs,
            "fs_abs_diff": diff,
            "fs_match": bool(pd.notna(header_fs) and diff <= args.tolerance),
            "error": error,
        })
        rows.append(item)

    out = pd.DataFrame(rows)
    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[DONE] saved: {out_path}")

    print(f"[INFO] checked unique files: {len(out)}")
    print("[INFO] metadata_fs counts")
    print(out["metadata_fs"].value_counts(dropna=False).sort_index())
    print("[INFO] header_fs counts")
    print(out["header_fs"].value_counts(dropna=False).sort_index())
    print("[INFO] fs_match counts")
    print(out["fs_match"].value_counts(dropna=False))

    mismatches = out[~out["fs_match"]]
    if len(mismatches):
        print("[WARN] mismatches by site/data_type")
        group_cols = [c for c in ["site", "data_type", "label"] if c in mismatches.columns]
        if group_cols:
            print(mismatches.groupby(group_cols).size().sort_index())
        raise SystemExit(2)


if __name__ == "__main__":
    main()
