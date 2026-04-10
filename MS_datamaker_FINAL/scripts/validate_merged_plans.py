#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import json
import pandas as pd


VALID_LABELS = {0, 1, 2}
VALID_DATA_TYPES = {"noise", "event", "unlabel"}


def validate_plan(df: pd.DataFrame, plan_name: str) -> tuple[dict, pd.DataFrame]:
    errors = []

    for idx, row in df.iterrows():
        row_errors = []

        try:
            label = int(row["label"])
            if label not in VALID_LABELS:
                row_errors.append(f"invalid_label:{label}")
        except Exception:
            row_errors.append("invalid_label_parse")

        data_type = str(row["data_type"])
        if data_type not in VALID_DATA_TYPES:
            row_errors.append(f"invalid_data_type:{data_type}")

        try:
            start_sec = float(row["start_sec"])
            end_sec = float(row["end_sec"])
            if end_sec <= start_sec:
                row_errors.append("invalid_time_range")
        except Exception:
            row_errors.append("invalid_time_parse")

        try:
            ch_start = int(row["ch_start"])
            ch_end = int(row["ch_end"])
            if ch_end <= ch_start:
                row_errors.append("invalid_channel_range")
        except Exception:
            row_errors.append("invalid_channel_parse")

        file_path = str(row["file_path"])
        if not Path(file_path).exists():
            row_errors.append("missing_file_path")

        group_id = str(row["group_id"]).strip()
        if group_id == "" or group_id.lower() == "nan":
            row_errors.append("empty_group_id")

        site = str(row["site"]).strip()
        if site == "" or site.lower() == "nan":
            row_errors.append("empty_site")

        try:
            original_fs = float(row["original_fs"])
            if original_fs <= 0:
                row_errors.append("invalid_original_fs")
        except Exception:
            row_errors.append("invalid_original_fs_parse")

        if row_errors:
            errors.append({
                "plan_name": plan_name,
                "row_index": int(idx),
                "file_path": file_path,
                "file_name": row.get("file_name", ""),
                "site": row.get("site", ""),
                "dataset_id": row.get("dataset_id", ""),
                "data_type": row.get("data_type", ""),
                "label": row.get("label", ""),
                "error_list": "|".join(row_errors),
            })

    summary = {
        "plan_name": plan_name,
        "n_rows": int(len(df)),
        "n_errors": int(len(errors)),
        "site_counts": df["site"].astype(str).value_counts().to_dict() if len(df) > 0 else {},
        "label_counts": df["label"].value_counts().to_dict() if len(df) > 0 else {},
    }

    return summary, pd.DataFrame(errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_dir", required=True)
    parser.add_argument("--out_report_json", required=True)
    parser.add_argument("--out_error_csv", required=True)
    args = parser.parse_args()

    plan_dir = Path(args.plan_dir)

    plan_files = [
        plan_dir / "segment_plan_event.csv",
        plan_dir / "segment_plan_noise.csv",
        plan_dir / "segment_plan_unlabel.csv",
        plan_dir / "segment_plan_all.csv",
    ]

    summaries = {}
    error_frames = []

    for p in plan_files:
        if not p.exists():
            print(f"[WARN] missing plan file: {p}")
            continue

        df = pd.read_csv(p)
        summary, err_df = validate_plan(df, p.name)
        summaries[p.name] = summary
        if len(err_df) > 0:
            error_frames.append(err_df)

        print(f"[CHECK] {p.name} | rows={summary['n_rows']} | errors={summary['n_errors']}")

    all_errors = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()

    out_error = Path(args.out_error_csv)
    out_error.parent.mkdir(parents=True, exist_ok=True)
    all_errors.to_csv(out_error, index=False, encoding="utf-8-sig")

    final_report = {
        "plans": summaries,
        "total_error_rows": int(len(all_errors)),
    }

    out_report = Path(args.out_report_json)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(f"[DONE] validation report saved to: {out_report}")
    print(f"[DONE] error csv saved to: {out_error}")
    print(json.dumps(final_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()