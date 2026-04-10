#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt
from obspy import read


def bandpass_filter(x, fs, fmin=5.0, fmax=80.0, order=4):
    nyq = 0.5 * fs
    low = fmin / nyq
    high = fmax / nyq
    if not (0 < low < high < 1):
        raise ValueError(f"Invalid bandpass range: fmin={fmin}, fmax={fmax}, fs={fs}")
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x, axis=1)


def robust_normalize(x, eps=1e-6):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return (x - med) / (1.4826 * mad + eps)


def make_image(x, out_png, cmap="gray", clip=3.0):
    img = x.T
    img = np.clip(img, -clip, clip)
    plt.figure(figsize=(12, 12))
    plt.imshow(img, cmap=cmap, aspect="auto", origin="lower", vmin=-clip, vmax=clip)
    plt.axis("off")
    plt.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def read_sgy_selected_all_obspy(sgy_path, ch_start, ch_end):
    st = read(sgy_path, format="SEGY")
    n_traces = len(st)

    if ch_start < 0 or ch_end >= n_traces or ch_start > ch_end:
        raise ValueError(
            f"Invalid channel range: ch_start={ch_start}, ch_end={ch_end}, total_traces={n_traces}"
        )

    selected = st[ch_start:ch_end + 1]
    traces = [np.asarray(tr.data, dtype=np.float32) for tr in selected]
    return np.stack(traces, axis=0)  # (C, T)


def process_and_save_segments(x, out_root, folder_name, sgy_name, fs, win_len,
                              fmin, fmax, bp_order, cmap, clip):
    event_dir = Path(out_root) / folder_name
    event_dir.mkdir(parents=True, exist_ok=True)

    _, n_t = x.shape
    n_seg = n_t // win_len
    if n_seg == 0:
        return []

    seg_rows = []
    for seg_idx in range(n_seg):
        s = seg_idx * win_len
        e = s + win_len
        x_seg = x[:, s:e]

        x_seg = bandpass_filter(x_seg, fs=fs, fmin=fmin, fmax=fmax, order=bp_order)
        x_seg = robust_normalize(x_seg)

        seg_name = f"seg_{seg_idx:03d}"
        out_png = event_dir / f"{seg_name}.png"
        make_image(x_seg, str(out_png), cmap=cmap, clip=clip)

        seg_rows.append({
            "sgy_file": sgy_name,
            "folder": folder_name,
            "segment_index": seg_idx,
            "png_relpath": f"{folder_name}/{seg_name}.png",
            "sample_start": s,
            "sample_end": e,
            "start_sec": s / fs,
            "end_sec": e / fs,
        })

    pd.DataFrame(seg_rows).to_csv(event_dir / "segments.csv", index=False)
    return seg_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_files", type=int, default=None)

    parser.add_argument("--fs", type=float, default=1000.0)
    parser.add_argument("--win_sec", type=float, default=1.0)

    parser.add_argument("--upper_ch_start", type=int, required=True)
    parser.add_argument("--upper_ch_end", type=int, required=True)
    parser.add_argument("--lower_ch_start", type=int, required=True)
    parser.add_argument("--lower_ch_end", type=int, required=True)

    parser.add_argument("--fmin", type=float, default=5.0)
    parser.add_argument("--fmax", type=float, default=80.0)
    parser.add_argument("--bp_order", type=int, default=4)

    parser.add_argument("--clip", type=float, default=3.0)
    parser.add_argument("--cmap", type=str, default="gray")
    args = parser.parse_args()

    upper_out = Path(args.out_dir) / "upper"
    lower_out = Path(args.out_dir) / "lower"
    upper_out.mkdir(parents=True, exist_ok=True)
    lower_out.mkdir(parents=True, exist_ok=True)

    sgy_files = sorted(Path(args.data_dir).glob("*.sgy"))
    if args.max_files is not None:
        sgy_files = sgy_files[:args.max_files]

    win_len = int(args.fs * args.win_sec)
    upper_summary = []
    lower_summary = []

    print(f"[INFO] num_files={len(sgy_files)}")

    for file_idx, sgy_path in enumerate(sgy_files):
        stem = sgy_path.stem

        try:
            x_upper = read_sgy_selected_all_obspy(
                str(sgy_path), args.upper_ch_start, args.upper_ch_end
            )
            upper_folder = f"{file_idx:04d}_{stem}"
            rows_upper = process_and_save_segments(
                x_upper, upper_out, upper_folder, sgy_path.name,
                args.fs, win_len, args.fmin, args.fmax,
                args.bp_order, args.cmap, args.clip
            )
            upper_summary.extend(rows_upper)

            x_lower = read_sgy_selected_all_obspy(
                str(sgy_path), args.lower_ch_start, args.lower_ch_end
            )
            lower_folder = f"{file_idx:04d}_{stem}"
            rows_lower = process_and_save_segments(
                x_lower, lower_out, lower_folder, sgy_path.name,
                args.fs, win_len, args.fmin, args.fmax,
                args.bp_order, args.cmap, args.clip
            )
            lower_summary.extend(rows_lower)

            print(f"[INFO] done {file_idx+1}/{len(sgy_files)}: {sgy_path.name}")

        except Exception as e:
            print(f"[WARN] failed: {sgy_path.name} | {e}")

    if upper_summary:
        pd.DataFrame(upper_summary).to_csv(upper_out / "summary.csv", index=False)
    if lower_summary:
        pd.DataFrame(lower_summary).to_csv(lower_out / "summary.csv", index=False)

    print("[DONE]")


if __name__ == "__main__":
    main()