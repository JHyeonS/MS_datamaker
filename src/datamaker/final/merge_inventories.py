#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import json
import pandas as pd


REQUIRED_BASE_COLS = [
    "file_path",
    "group_id",
    "original_fs",
]

TARGET_COLS = [
    "dataset_id",
    "site",
    "view",
    "file_path",
    "file_name",
    "file_stem",
    "ext",
    "group_id",
    "original_fs",
    "__source_inventory__",
]


def fill_missing_columns(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    df = df.copy()

    missing_base = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
    if missing_base:
        raise ValueError(f"{csv_path} missing required base columns: {missing_base}")

    # dataset_id
    if "dataset_id" not in df.columns:
        if "site" in df.columns:
            df["dataset_id"] = df["site"].astype(str)
        else:
            df["dataset_id"] = csv_path.parent.name

    # site
    if "site" not in df.columns:
        if "dataset_id" in df.columns:
            df["site"] = df["dataset_id"].astype(str)
        else:
            df["site"] = csv_path.parent.name

    # view
    if "view" not in df.columns:
        df["view"] = df["site"].astype(str)

    # file_name
    if "file_name" not in df.columns:
        df["file_name"] = df["file_path"].astype(str).map(lambda x: Path(x).name)

    # file_stem
    if "file_stem" not in df.columns:
        df["file_stem"] = df["file_path"].astype(str).map(lambda x: Path(x).stem)

    # ext
    if "ext" not in df.columns:
        df["ext"] = df["file_path"].astype(str).map(lambda x: Path(x).suffix.lower())

    df["__source_inventory__"] = str(csv_path.resolve())

    # reorder / trim
    for col in TARGET_COLS:
        if col not in df.columns:
            df[col] = ""

    return df[TARGET_COLS].copy()


def load_one(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = fill_missing_columns(df, csv_path)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, help="FINAL/input root")
    parser.add_argument("--out_csv", required=True, help="Merged inventory csv path")
    parser.add_argument("--out_report_json", required=True, help="Summary json path")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    inv_paths = sorted(
        p for p in input_root.glob("*/inventory.csv")
        if p.parent.name != "final"
    )

    if len(inv_paths) == 0:
        raise RuntimeError(f"No inventory.csv found under: {input_root}")

    frames = []
    for p in inv_paths:
        df = load_one(p)
        frames.append(df)
        print(f"[LOAD] {p} | rows={len(df)}")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    report = {
        "n_files": int(len(merged)),
        "n_sources": int(len(inv_paths)),
        "sources": [str(p.resolve()) for p in inv_paths],
        "site_counts": merged["site"].astype(str).value_counts().to_dict(),
        "dataset_counts": merged["dataset_id"].astype(str).value_counts().to_dict(),
        "view_counts": merged["view"].astype(str).value_counts().to_dict(),
        "ext_counts": merged["ext"].astype(str).value_counts().to_dict(),
        "duplicated_file_path_count": int(merged["file_path"].astype(str).duplicated().sum()),
        "duplicated_group_id_count": int(merged["group_id"].astype(str).duplicated().sum()),
    }

    out_report = Path(args.out_report_json)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[DONE] merged inventory saved to: {out_csv}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
