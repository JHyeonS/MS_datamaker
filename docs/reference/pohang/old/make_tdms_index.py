#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from datetime import datetime, timezone, timedelta

import pandas as pd


PATTERN = re.compile(r'_(\d{8})_(\d{6}\.\d{3})\.tdms$', re.IGNORECASE)


def parse_start_utc(filename: str) -> datetime:
    m = PATTERN.search(filename)
    if not m:
        raise ValueError(f"datetime parse failed: {filename}")

    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S.%f")

    if "UTC+0900" in filename:
        kst = timezone(timedelta(hours=9))
        dt = dt.replace(tzinfo=kst).astimezone(timezone.utc)
    elif "_UTC_" in filename:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        # 애매한 포맷은 일단 UTC로 간주
        dt = dt.replace(tzinfo=timezone.utc)

    return dt


def main():
    parser = argparse.ArgumentParser(description="Make TDMS index CSV from filenames only")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory containing TDMS files")
    parser.add_argument("--out_csv", type=str, required=True,
                        help="Output CSV path")
    parser.add_argument("--out_bad_csv", type=str, default=None,
                        help="Optional CSV path for failed filenames")
    args = parser.parse_args()

    rows = []
    bad_rows = []
    count = 0

    for root, _, files in os.walk(args.root_dir):
        for fname in files:
            if not fname.lower().endswith(".tdms"):
                continue

            count += 1
            fpath = os.path.join(root, fname)

            try:
                start_utc = parse_start_utc(fname)
                rows.append({
                    "filename": fname,
                    "filepath": fpath,
                    "starttimeUTC": start_utc,
                })
            except Exception as e:
                bad_rows.append({
                    "filename": fname,
                    "filepath": fpath,
                    "error": str(e),
                })

            if count % 10000 == 0:
                print(f"[INFO] processed {count} tdms files")

    df = pd.DataFrame(rows)

    if df.empty:
        print("[WARN] No TDMS files found.")
        df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
        return

    df = df.sort_values("starttimeUTC").reset_index(drop=True)

    # 다음 파일의 starttimeUTC를 현재 파일의 endtimeUTC로 사용
    df["endtimeUTC"] = df["starttimeUTC"].shift(-1)

    # duration_sec도 같이 계산
    df["duration_sec"] = (
        df["endtimeUTC"] - df["starttimeUTC"]
    ).dt.total_seconds()

    # 마지막 파일은 다음 starttime이 없으므로 NaN
    df["starttimeUTC"] = df["starttimeUTC"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    df["endtimeUTC"] = df["endtimeUTC"].apply(
        lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%fZ") if pd.notnull(x) else None
    )

    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] saved {len(df)} rows to {args.out_csv}")

    if args.out_bad_csv is not None and len(bad_rows) > 0:
        pd.DataFrame(bad_rows).to_csv(args.out_bad_csv, index=False, encoding="utf-8-sig")
        print(f"[DONE] saved {len(bad_rows)} bad rows to {args.out_bad_csv}")


if __name__ == "__main__":
    main()


'''
python scripts/make_tdms_index.py \
  --root_dir /data2/Hyeonsu \
  --out_csv outputs/tdms_file_index.csv \
  --out_bad_csv outputs/tdms_file_index_bad.csv
  '''