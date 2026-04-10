#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample_poly

try:
    import segyio
except Exception:
    segyio = None


def parse_args():
    p = argparse.ArgumentParser(description="Build NPY patches and metadata_raw.csv from Utah SGY files")
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing .sgy files")
    p.add_argument("--out_dir", type=str, required=True, help="Output root directory")
    p.add_argument("--site", type=str, default="utah_2023", help="Site/domain name")
    p.add_argument("--fs", type=float, default=1000.0, help="Original sampling rate")
    p.add_argument("--target_fs", type=float, default=None, help="Optional target sampling rate")
    p.add_argument("--win_sec", type=float, default=2.0, help="Window length in seconds")
    p.add_argument("--stride_sec", type=float, default=1.0, help="Stride in seconds")
    p.add_argument("--splits", type=str, required=True,
                   help='Channel splits, e.g. "400-805/850-1255/1300-1705/1750-2155"')
    p.add_argument("--expected_channels", type=int, default=None,
                   help="If set, require patch.shape[0] == expected_channels")
    p.add_argument("--expected_samples", type=int, default=None,
                   help="If set, require patch.shape[1] == expected_samples after resampling")
    p.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "float64"])
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--glob_pattern", type=str, default="*.sgy")
    p.add_argument("--input_channel_base", type=int, default=0,
                   help="0 if splits are zero-based inclusive indices, 1 if one-based inclusive")
    return p.parse_args()


def parse_splits(s: str, input_channel_base: int = 0) -> List[Tuple[int, int, int]]:
    out = []
    parts = [x.strip() for x in s.split("/") if x.strip()]
    for i, part in enumerate(parts, start=1):
        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", part)
        if not m:
            raise ValueError(f"Invalid split spec: {part}")
        a = int(m.group(1))
        b = int(m.group(2))
        if b < a:
            raise ValueError(f"Invalid split range: {part}")
        if input_channel_base == 1:
            a -= 1
            b -= 1
        ch_start = a
        ch_end_excl = b + 1
        out.append((i, ch_start, ch_end_excl))
    return out


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def maybe_resample(arr: np.ndarray, original_fs: float, target_fs: Optional[float]) -> np.ndarray:
    if target_fs is None or np.isclose(original_fs, target_fs):
        return arr

    from fractions import Fraction
    frac = Fraction(float(target_fs) / float(original_fs)).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator

    arr_rs = resample_poly(arr, up=up, down=down, axis=1)
    expected_len = int(round(arr.shape[1] * float(target_fs) / float(original_fs)))
    if arr_rs.shape[1] != expected_len:
        if arr_rs.shape[1] > expected_len:
            arr_rs = arr_rs[:, :expected_len]
        else:
            pad = expected_len - arr_rs.shape[1]
            arr_rs = np.pad(arr_rs, ((0, 0), (0, pad)), mode="edge")
    return arr_rs


def read_full_sgy(file_path: str) -> np.ndarray:
    if segyio is None:
        raise ImportError("segyio is not installed")
    with segyio.open(file_path, "r", ignore_geometry=True) as f:
        trace_count = f.tracecount
        if trace_count <= 0:
            raise ValueError(f"No traces found: {file_path}")
        traces = [np.asarray(f.trace[i], dtype=np.float32) for i in range(trace_count)]
    return np.stack(traces, axis=0)


def save_npy(arr: np.ndarray, out_path: Path, dtype: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr.astype(dtype, copy=False))


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    npy_root = out_dir / "samples" / args.site
    metadata_dir = out_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    split_cfg = parse_splits(args.splits, input_channel_base=args.input_channel_base)

    sgy_files = sorted([p for p in Path(args.data_dir).glob(args.glob_pattern) if p.is_file()], key=lambda p: natural_key(p.name))
    sgy_files = [p for p in sgy_files if not p.name.startswith("._")]
    if args.max_files is not None:
        sgy_files = sgy_files[:args.max_files]

    win_len = int(round(args.win_sec * args.fs))
    stride_len = int(round(args.stride_sec * args.fs))
    if win_len <= 0 or stride_len <= 0:
        raise ValueError("win_sec and stride_sec must be positive")

    rows = []
    error_rows = []

    for sgy_path in tqdm(sgy_files, desc="Building NPY", ncols=120):
        shot_id = sgy_path.stem
        group_id = f"{args.site}_{shot_id}"

        try:
            full = read_full_sgy(str(sgy_path))  # (C, T)
            n_channels, n_samples = full.shape

            if n_samples < win_len:
                raise ValueError(f"Too short for one window: n_samples={n_samples}, win_len={win_len}")

            seg_starts = list(range(0, n_samples - win_len + 1, stride_len))
            for seg_idx, s0 in enumerate(seg_starts):
                s1 = s0 + win_len
                start_sec = s0 / args.fs
                end_sec = s1 / args.fs

                for split_id, ch_start, ch_end in split_cfg:
                    if ch_start < 0 or ch_end > n_channels:
                        raise ValueError(
                            f"Split out of bounds in {sgy_path.name}: "
                            f"split_id={split_id}, range=[{ch_start},{ch_end}), n_channels={n_channels}"
                        )

                    patch = full[ch_start:ch_end, s0:s1]
                    patch = maybe_resample(patch, args.fs, args.target_fs)

                    if args.expected_channels is not None and patch.shape[0] != args.expected_channels:
                        raise ValueError(
                            f"Unexpected channel count in {sgy_path.name}, seg={seg_idx}, split={split_id}: "
                            f"{patch.shape[0]} != {args.expected_channels}"
                        )
                    if args.expected_samples is not None and patch.shape[1] != args.expected_samples:
                        raise ValueError(
                            f"Unexpected sample count in {sgy_path.name}, seg={seg_idx}, split={split_id}: "
                            f"{patch.shape[1]} != {args.expected_samples}"
                        )

                    split_name = f"split_{split_id:02d}_ch{ch_start:04d}_{(ch_end - 1):04d}"
                    out_path = npy_root / shot_id / f"seg_{seg_idx:03d}" / f"{split_name}.npy"

                    if (not out_path.exists()) or args.overwrite:
                        save_npy(patch, out_path, args.dtype)

                    rows.append({
                        "site": args.site,
                        "domain": args.site,
                        "shot_id": shot_id + ".sgy",
                        "segment_index": seg_idx,
                        "split_id": split_id,
                        "split_name": split_name,
                        "group_id": group_id,
                        "raw_file_path": str(sgy_path.resolve()),
                        "file_stem": shot_id,
                        "npy_path": str(out_path.resolve()),
                        "start_sec": float(start_sec),
                        "end_sec": float(end_sec),
                        "duration_sec": float(end_sec - start_sec),
                        "ch_start": int(ch_start),
                        "ch_end": int(ch_end),
                        "num_channels": int(ch_end - ch_start),
                        "original_fs": float(args.fs),
                        "saved_fs": float(args.target_fs if args.target_fs is not None else args.fs),
                        "num_samples": int(patch.shape[1]),
                        "shape0": int(patch.shape[0]),
                        "shape1": int(patch.shape[1]),
                    })

        except Exception as e:
            error_rows.append({
                "raw_file_path": str(sgy_path),
                "error_type": type(e).__name__,
                "error_message": str(e),
            })

    meta_df = pd.DataFrame(rows)
    meta_csv = metadata_dir / "metadata_raw.csv"
    meta_df.to_csv(meta_csv, index=False, encoding="utf-8-sig")

    if error_rows:
        pd.DataFrame(error_rows).to_csv(metadata_dir / "build_errors.csv", index=False, encoding="utf-8-sig")

    summary = {
        "site": args.site,
        "data_dir": str(Path(args.data_dir).resolve()),
        "out_dir": str(out_dir.resolve()),
        "n_files": len(sgy_files),
        "n_rows": int(len(meta_df)),
        "n_errors": int(len(error_rows)),
        "fs": float(args.fs),
        "target_fs": None if args.target_fs is None else float(args.target_fs),
        "win_sec": float(args.win_sec),
        "stride_sec": float(args.stride_sec),
        "splits": args.splits,
        "expected_channels": args.expected_channels,
        "expected_samples": args.expected_samples,
        "dtype": args.dtype,
    }
    with open(metadata_dir / "build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[DONE] metadata saved: {meta_csv}")
    print(f"[DONE] rows: {len(meta_df)}")
    print(f"[DONE] errors: {len(error_rows)}")


if __name__ == "__main__":
    main()
