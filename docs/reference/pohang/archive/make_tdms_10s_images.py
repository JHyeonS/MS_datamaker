#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from nptdms import TdmsFile


def bandpass_filter(x, fs, fmin=5.0, fmax=80.0, order=4):
    """
    x: (C, T)
    채널별 시간축(axis=1)으로 bandpass
    """
    nyq = 0.5 * fs
    low = fmin / nyq
    high = fmax / nyq

    if low <= 0 or high >= 1 or low >= high:
        raise ValueError(
            f"Invalid bandpass range: fmin={fmin}, fmax={fmax}, fs={fs}"
        )

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x, axis=1)


def robust_normalize(x, eps=1e-6):
    """
    x: (C, T)
    전체 segment 기준 robust normalization
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return (x - med) / (1.4826 * mad + eps)


def make_image(x, out_png, cmap="gray", clip=3.0):
    """
    x: (C, T) -> img: (T, C)
    X-T domain image 저장
    """
    img = x.T  # (T, C)

    # 시각화 안정화를 위한 clipping
    img = np.clip(img, -clip, clip)

    plt.figure(figsize=(12, 12))
    plt.imshow(
        img,
        cmap=cmap,
        aspect="auto",
        origin="lower",
        vmin=-clip,
        vmax=clip
    )
    plt.axis("off")
    plt.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def read_tdms_selected_channels(tdms_path, ch_start=214, ch_end=620):
    """
    channel windowing
    """
    tdms = TdmsFile.read(tdms_path)
    groups = tdms.groups()
    if len(groups) == 0:
        raise RuntimeError(f"No groups in {tdms_path}")

    group = groups[0]
    channels = group.channels()
    if len(channels) == 0:
        raise RuntimeError(f"No channels in {tdms_path}")

    if ch_start < 0 or ch_end >= len(channels) or ch_start > ch_end:
        raise ValueError(
            f"Invalid channel range: ch_start={ch_start}, ch_end={ch_end}, total_channels={len(channels)}"
        )

    selected = channels[ch_start:ch_end + 1]
    data = [np.asarray(ch[:], dtype=np.float32) for ch in selected]
    return np.stack(data, axis=0)  # (C, T)


def preprocess_segment(x_seg, fs, fmin, fmax, order):
    """
    segment 단위 전처리:
    1) bandpass
    2) robust normalization
    """
    x_seg = bandpass_filter(x_seg, fs=fs, fmin=fmin, fmax=fmax, order=order)
    x_seg = robust_normalize(x_seg)
    return x_seg


def main():
    parser = argparse.ArgumentParser(description="Make DAS images from matched TDMS files")
    parser.add_argument("--match_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    # time slicing
    parser.add_argument("--segment_sec", type=float, default=10.0)
    parser.add_argument("--fs", type=float, default=1000.0)

    parser.add_argument("--max_events", type=int, default=200)

    # channel windowing
    parser.add_argument("--ch_start", type=int, default=214)
    parser.add_argument("--ch_end", type=int, default=620)

    # bandpass
    parser.add_argument("--fmin", type=float, default=5.0)
    parser.add_argument("--fmax", type=float, default=80.0)
    parser.add_argument("--bp_order", type=int, default=4)

    # image
    parser.add_argument("--cmap", type=str, default="gray")
    parser.add_argument("--clip", type=float, default=3.0)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.match_csv)
    df = df[df["matched"] == True].copy().reset_index(drop=True)

    total_targets = min(len(df), args.max_events)
    seg_len = int(args.segment_sec * args.fs)

    saved_files, skipped_files = 0, 0

    for i, row in df.iloc[:total_targets].iterrows():
        tdms_path = row["tdms_filepath"]
        tdms_name = os.path.basename(tdms_path)
        stem = os.path.splitext(tdms_name)[0]

        event_dir = os.path.join(args.out_dir, f"{i:04d}_{stem}")
        os.makedirs(event_dir, exist_ok=True)

        try:
            # 1) channel windowing
            x = read_tdms_selected_channels(
                tdms_path,
                ch_start=args.ch_start,
                ch_end=args.ch_end
            )

            _, n_t = x.shape
            n_seg = n_t // seg_len

            if n_seg == 0:
                print(f"[SKIP] too short: {tdms_path} | n_t={n_t}")
                skipped_files += 1
                continue

            # 2) time slicing
            for seg_idx in range(n_seg):
                s = seg_idx * seg_len
                e = s + seg_len
                x_seg = x[:, s:e]  # (C, T)

                # 3) bandpass + robust normalization
                x_seg = preprocess_segment(
                    x_seg,
                    fs=args.fs,
                    fmin=args.fmin,
                    fmax=args.fmax,
                    order=args.bp_order
                )

                # 4) X-T domain + gray scale image
                out_png = os.path.join(event_dir, f"seg_{seg_idx:03d}.png")
                make_image(
                    x_seg,
                    out_png,
                    cmap=args.cmap,
                    clip=args.clip
                )

            saved_files += 1
            print(f"[INFO] done {i+1}/{total_targets}: {tdms_name} -> {n_seg} images")

        except Exception as ex:
            print(f"[WARN] failed: {tdms_path} | {ex}")
            skipped_files += 1

    print(f"[DONE] processed={total_targets}, saved={saved_files}, skipped={skipped_files}")


if __name__ == "__main__":
    main()