#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True, help="Input all_samples.csv. Repeat this flag.")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    frames = []
    for p in args.csv:
        df = pd.read_csv(p)
        df["__source_csv__"] = str(Path(p).resolve())
        frames.append(df)

    out = pd.concat(frames, axis=0, ignore_index=True)

    if "site" in out.columns:
        out["site"] = out["site"].astype(str).str.strip()
    if "label" in out.columns:
        out["label"] = out["label"].astype(int)
    if "label_name" in out.columns:
        out["label_name"] = out["label_name"].astype(str).str.strip()

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] saved: {out_path}")
    print("[INFO] rows:", len(out))
    if "site" in out.columns:
        print("[INFO] site counts:")
        print(out["site"].value_counts())
    if "label_name" in out.columns:
        print("[INFO] label_name counts:")
        print(out["label_name"].value_counts())

if __name__ == "__main__":
    main()
