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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--fs", type=float, default=1000.0)
    parser.add_argument("--win_sec", type=float, default=1.0)
    parser.add_argument("--ch_start", type=int, default=273)
    parser.add_argument("--ch_end", type=int, default=678)   # 406 channels
    parser.add_argument("--fmin", type=float, default=5.0)
    parser.add_argument("--fmax", type=float, default=80.0)
    parser.add_argument("--bp_order", type=int, default=4)
    parser.add_argument("--clip", type=float, default=3.0)
    parser.add_argument("--cmap", type=str, default="gray")
    parser.add_argument("--save_npy", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sgy_files = sorted(Path(args.data_dir).glob("*.sgy"))
    if args.max_files is not None:
        sgy_files = sgy_files[:args.max_files]

    win_len = int(args.fs * args.win_sec)
    summary_rows = []

    print(f"[INFO] num_files={len(sgy_files)}")

    for file_idx, sgy_path in enumerate(sgy_files):
        stem = sgy_path.stem
        event_dir = Path(args.out_dir) / f"{file_idx:04d}_{stem}"
        event_dir.mkdir(parents=True, exist_ok=True)

        try:
            x = read_sgy_selected_all_obspy(str(sgy_path), args.ch_start, args.ch_end)  # (C,T)
            _, n_t = x.shape
            n_seg = n_t // win_len

            if n_seg == 0:
                print(f"[SKIP] too short: {sgy_path.name} | n_t={n_t}")
                continue

            seg_rows = []

            for seg_idx in range(n_seg):
                s = seg_idx * win_len
                e = s + win_len
                x_seg = x[:, s:e]

                x_seg = bandpass_filter(x_seg, fs=args.fs, fmin=args.fmin, fmax=args.fmax, order=args.bp_order)
                x_seg = robust_normalize(x_seg)

                seg_name = f"seg_{seg_idx:03d}"
                out_png = event_dir / f"{seg_name}.png"
                make_image(x_seg, str(out_png), cmap=args.cmap, clip=args.clip)

                if args.save_npy:
                    np.save(event_dir / f"{seg_name}.npy", x_seg.astype(np.float32))

                row = {
                    "file_index": file_idx,
                    "sgy_file": sgy_path.name,
                    "folder": event_dir.name,
                    "segment_index": seg_idx,
                    "png_relpath": f"{event_dir.name}/{seg_name}.png",
                    "npy_relpath": f"{event_dir.name}/{seg_name}.npy" if args.save_npy else "",
                    "sample_start": s,
                    "sample_end": e,
                    "start_sec": s / args.fs,
                    "end_sec": e / args.fs,
                }
                seg_rows.append(row)
                summary_rows.append(row)

            pd.DataFrame(seg_rows).to_csv(event_dir / "segments.csv", index=False)
            print(f"[INFO] done {file_idx+1}/{len(sgy_files)}: {sgy_path.name} -> {n_seg} segments")

        except Exception as e:
            print(f"[WARN] failed: {sgy_path.name} | {e}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(Path(args.out_dir) / "summary.csv", index=False)

    print("[DONE]")


if __name__ == "__main__":
    main()