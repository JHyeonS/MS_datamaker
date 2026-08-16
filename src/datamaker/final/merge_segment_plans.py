#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import json
import pandas as pd


REQUIRED_COLS = [
    "group_id",
    "dataset_id",
    "site",
    "view",
    "data_type",
    "label",
    "file_path",
    "file_name",
    "file_stem",
    "start_sec",
    "end_sec",
    "ch_start",
    "ch_end",
    "original_fs",
]


PLAN_NAMES = [
    ("segment_plan_event.csv", "event"),
    ("segment_plan_noise.csv", "noise"),
    ("segment_plan_unlabel.csv", "unlabel"),
]


def load_one(csv_path: Path, expected_type: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    df = df.copy()
    df["__source_segment_plan__"] = str(csv_path.resolve())

    # data_type sanity
    if "data_type" in df.columns:
        bad = df["data_type"].astype(str) != expected_type
        if bad.any():
            print(f"[WARN] {csv_path} has rows with data_type != {expected_type}")

    return df


def save_with_new_segment_ids(df: pd.DataFrame, out_csv: Path) -> None:
    df = df.reset_index(drop=True).copy()
    df["segment_id"] = range(len(df))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, help="FINAL/input root")
    parser.add_argument("--out_dir", required=True, help="FINAL/output dir")
    parser.add_argument("--out_report_json", required=True, help="Merge report json")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_by_type = {}
    report = {"sources": {}, "merged_counts": {}}

    for file_name, dtype in PLAN_NAMES:
        paths = sorted(
            p for p in input_root.glob(f"*/{file_name}")
            if p.parent.name != "final"
        )
        if len(paths) == 0:
            print(f"[WARN] no files found for {file_name}")
            merged_by_type[dtype] = pd.DataFrame()
            continue

        frames = []
        source_counts = {}
        for p in paths:
            df = load_one(p, expected_type=dtype)
            frames.append(df)
            source_counts[str(p.resolve())] = int(len(df))
            print(f"[LOAD] {p} | rows={len(df)}")

        merged = pd.concat(frames, ignore_index=True).reset_index(drop=True)
        merged_by_type[dtype] = merged
        report["sources"][dtype] = source_counts
        report["merged_counts"][dtype] = int(len(merged))

    out_event = out_dir / "segment_plan_event.csv"
    out_noise = out_dir / "segment_plan_noise.csv"
    out_unlabel = out_dir / "segment_plan_unlabel.csv"
    out_all = out_dir / "segment_plan_all.csv"

    save_with_new_segment_ids(merged_by_type["event"], out_event)
    save_with_new_segment_ids(merged_by_type["noise"], out_noise)
    save_with_new_segment_ids(merged_by_type["unlabel"], out_unlabel)

    all_df = pd.concat(
        [
            merged_by_type["event"],
            merged_by_type["noise"],
            merged_by_type["unlabel"],
        ],
        ignore_index=True,
    ).reset_index(drop=True)
    save_with_new_segment_ids(all_df, out_all)

    report["merged_counts"]["all"] = int(len(all_df))
    report["site_counts_all"] = all_df["site"].astype(str).value_counts().to_dict() if len(all_df) > 0 else {}
    report["label_counts_all"] = all_df["label"].value_counts().to_dict() if len(all_df) > 0 else {}

    out_report = Path(args.out_report_json)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[DONE] merged plans saved under: {out_dir}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
