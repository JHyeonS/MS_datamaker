#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd


def load_tdms_index(tdms_csv: str) -> pd.DataFrame:
    df = pd.read_csv(tdms_csv)
    df["starttimeUTC"] = pd.to_datetime(df["starttimeUTC"], utc=True, errors="coerce")
    df["endtimeUTC"] = pd.to_datetime(df["endtimeUTC"], utc=True, errors="coerce")
    df = df.dropna(subset=["starttimeUTC", "endtimeUTC"]).sort_values("starttimeUTC").reset_index(drop=True)
    return df


def load_catalog(xlsx_path: str, time_col: str = "UTC") -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, skiprows=1)
    print(df.head())
    print(df.columns)
    
    if time_col not in df.columns:
        raise ValueError(f"'{time_col}' 컬럼이 없음. 사용 가능한 컬럼: {list(df.columns)}")

    df["event_time_utc"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["event_time_utc"]).reset_index(drop=True)
    return df


def match_events(catalog_df: pd.DataFrame, tdms_df: pd.DataFrame) -> pd.DataFrame:
    starts = tdms_df["starttimeUTC"].values
    ends = tdms_df["endtimeUTC"].values

    matched_rows = []
    for _, row in catalog_df.iterrows():
        t = row["event_time_utc"]

        cond = (tdms_df["starttimeUTC"] <= t) & (t < tdms_df["endtimeUTC"])
        hit = tdms_df[cond]

        if len(hit) > 0:
            td = hit.iloc[0]
            out = row.to_dict()
            out["matched"] = True
            out["tdms_filename"] = td["filename"]
            out["tdms_filepath"] = td["filepath"]
            out["tdms_starttimeUTC"] = td["starttimeUTC"]
            out["tdms_endtimeUTC"] = td["endtimeUTC"]
            matched_rows.append(out)
        else:
            out = row.to_dict()
            out["matched"] = False
            out["tdms_filename"] = None
            out["tdms_filepath"] = None
            out["tdms_starttimeUTC"] = None
            out["tdms_endtimeUTC"] = None
            matched_rows.append(out)

    return pd.DataFrame(matched_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog_xlsx", type=str, required=True)
    parser.add_argument("--tdms_csv", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--time_col", type=str, default="UTC",
                        help="catalog에서 시간으로 사용할 컬럼명. 기본값: UTC")
    args = parser.parse_args()

    tdms_df = load_tdms_index(args.tdms_csv)
    catalog_df = load_catalog(args.catalog_xlsx, time_col=args.time_col)
    result_df = match_events(catalog_df, tdms_df)

    result_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    n_total = len(result_df)
    n_matched = int(result_df["matched"].sum())
    n_unmatched = n_total - n_matched

    print(f"[DONE] total={n_total}, matched={n_matched}, unmatched={n_unmatched}")
    print(f"[DONE] saved to {args.out_csv}")


if __name__ == "__main__":
    main()

'''
python scripts/match_catalog_tdms.py \
  --catalog_xlsx csv/iDAS_detection_korean_microeqrthquake_list_20190501_20191001.xlsx \
  --tdms_csv outputs/tdms_file_index.csv \
  --out_csv outputs/catalog_tdms_intersection.csv \
  --time_col UTC
  '''