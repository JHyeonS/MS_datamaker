#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd

def detect_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"컬럼을 찾지 못함. 현재 컬럼: {df.columns.tolist()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_tdms_csv", required=True)
    parser.add_argument("--event_tdms_csv", required=True)
    parser.add_argument("--out_non_event_csv", required=True)
    parser.add_argument("--out_unlabel_csv", required=True)
    parser.add_argument("--non_event_n", type=int, default=50)
    parser.add_argument("--unlabel_n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df_all = pd.read_csv(args.all_tdms_csv)
    df_event = pd.read_csv(args.event_tdms_csv)

    all_name_col = detect_col(df_all, ["tdms_filename", "filename", "tdms_file", "file"])
    all_path_col = detect_col(df_all, ["tdms_filepath", "filepath", "path", "file_path"])
    event_name_col = detect_col(df_event, ["tdms_filename", "filename", "tdms_file", "file"])

    event_files = set(df_event[event_name_col].astype(str).str.strip().unique())
    df_candidates = df_all[~df_all[all_name_col].astype(str).str.strip().isin(event_files)].copy()

    df_candidates = df_candidates[[all_name_col, all_path_col]].rename(columns={
        all_name_col: "tdms_filename",
        all_path_col: "tdms_filepath"
    })

    df_candidates = df_candidates.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    need = args.non_event_n + args.unlabel_n
    if len(df_candidates) < need:
        raise ValueError(f"후보 파일 부족: candidates={len(df_candidates)}, needed={need}")

    df_non_event = df_candidates.iloc[:args.non_event_n].copy()
    df_unlabel = df_candidates.iloc[args.non_event_n:args.non_event_n + args.unlabel_n].copy()

    df_non_event.to_csv(args.out_non_event_csv, index=False)
    df_unlabel.to_csv(args.out_unlabel_csv, index=False)

    print("[INFO] df_all columns   :", df_all.columns.tolist())
    print("[INFO] df_event columns :", df_event.columns.tolist())
    print("[INFO] non_event saved :", len(df_non_event))
    print("[INFO] unlabel saved   :", len(df_unlabel))

if __name__ == "__main__":
    main()