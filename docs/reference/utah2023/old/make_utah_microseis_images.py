#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pathlib import Path

try:
    import segyio
except ImportError:
    raise ImportError("Please install segyio: pip install segyio")


def bandpass_filter(x, fs, fmin=5.0, fmax=80.0, order=4):
    """
    x: (C, T)
    """
    nyq = 0.5 * fs
    low = fmin / nyq
    high = fmax / nyq
    if not (0 < low < high < 1):
        raise ValueError(f"Invalid bandpass range: fmin={fmin}, fmax={fmax}, fs={fs}")

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x, axis=1)


def robust_normalize(x, eps=1e-6):
    """
    전체 segment 기준 robust normalization
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return (x - med) / (1.4826 * mad + eps)


def make_image(x, out_png, cmap="gray", clip=3.0):
    """
    x: (C, T) -> img: (T, C)
    """
    img = x.T
    img = np.clip(img, -clip, clip)

    plt.figure(figsize=(12, 12))
    plt.imshow(
        img,
        cmap=cmap,
        aspect="auto",
        origin="lower",
        vmin=-clip,
        vmax=clip,
    )
    plt.axis("off")
    plt.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def parse_catalog_line(line):
    """
    예:
    FORGE_78-32_iDASv3-P11_UTC190427171923.sgy 13313
    """
    parts = line.strip().split()
    if len(parts) < 2:
        return None
    fname = parts[0]
    sample_idx = int(parts[1])
    return fname, sample_idx


def load_catalog(catalog_path, max_events=10):
    items = []
    with open(catalog_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("Filename"):
                continue
            if s.upper().startswith("PERFORATIONS"):
                break
            parsed = parse_catalog_line(s)
            if parsed is None:
                continue
            items.append(parsed)
            if len(items) >= max_events:
                break
    return items


def find_sgy_file(data_dir, fname):
    p = Path(data_dir) / fname
    if p.exists():
        return str(p)

    matches = list(Path(data_dir).rglob(fname))
    if len(matches) == 0:
        return None
    return str(matches[0])


def read_sgy_selected_crop(sgy_path, ch_start, ch_end, sample_start, win_len):
    """
    반환 shape: (C, T)
    주의: SEG-Y에서 trace를 channel처럼 사용
    """
    with segyio.open(sgy_path, "r", ignore_geometry=True) as f:
        n_traces = f.tracecount
        if ch_start < 0 or ch_end >= n_traces or ch_start > ch_end:
            raise ValueError(
                f"Invalid channel range: ch_start={ch_start}, ch_end={ch_end}, total_traces={n_traces}"
            )

        sample_axis = np.asarray(f.samples)
        n_samples = len(sample_axis)

        if sample_start < 0:
            raise ValueError(f"sample_start < 0: {sample_start}")
        if sample_start + win_len > n_samples:
            raise ValueError(
                f"crop exceeds sample length: start={sample_start}, win_len={win_len}, total={n_samples}"
            )

        traces = []
        for tr_idx in range(ch_start, ch_end + 1):
            tr = np.asarray(f.trace[tr_idx], dtype=np.float32)
            traces.append(tr[sample_start:sample_start + win_len])

        x = np.stack(traces, axis=0)  # (C, T)
    return x


def main():
    parser = argparse.ArgumentParser(description="Make Utah microseismic 1-sec images from catalog sample index")
    parser.add_argument("--catalog_txt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--max_events", type=int, default=10)
    parser.add_argument("--fs", type=float, default=1000.0)
    parser.add_argument("--win_sec", type=float, default=1.0)

    parser.add_argument("--ch_start", type=int, default=214)
    parser.add_argument("--ch_end", type=int, default=619)   # 214~619 => 406 channels

    parser.add_argument("--fmin", type=float, default=5.0)
    parser.add_argument("--fmax", type=float, default=80.0)
    parser.add_argument("--bp_order", type=int, default=4)

    parser.add_argument("--clip", type=float, default=3.0)
    parser.add_argument("--cmap", type=str, default="gray")
    parser.add_argument("--save_npy", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    win_len = int(args.fs * args.win_sec)

    items = load_catalog(args.catalog_txt, max_events=args.max_events)
    print(f"[INFO] loaded {len(items)} catalog entries")

    done, skipped = 0, 0

    for idx, (fname, sample_idx) in enumerate(items):
        sgy_path = find_sgy_file(args.data_dir, fname)
        if sgy_path is None:
            print(f"[WARN] file not found: {fname}")
            skipped += 1
            continue

        stem = Path(fname).stem
        event_dir = os.path.join(args.out_dir, f"{idx:04d}_{stem}")
        os.makedirs(event_dir, exist_ok=True)

        try:
            x = read_sgy_selected_crop(
                sgy_path=sgy_path,
                ch_start=args.ch_start,
                ch_end=args.ch_end,
                sample_start=sample_idx,
                win_len=win_len,
            )

            x = bandpass_filter(x, fs=args.fs, fmin=args.fmin, fmax=args.fmax, order=args.bp_order)
            x = robust_normalize(x)

            out_png = os.path.join(event_dir, "event_1s.png")
            make_image(x, out_png, cmap=args.cmap, clip=args.clip)

            if args.save_npy:
                out_npy = os.path.join(event_dir, "event_1s.npy")
                np.save(out_npy, x.astype(np.float32))

            done += 1
            print(f"[INFO] done {idx+1}/{len(items)}: {fname} sample={sample_idx}")

        except Exception as e:
            print(f"[WARN] failed: {fname} | {e}")
            skipped += 1

    print(f"[DONE] success={done}, skipped={skipped}")


if __name__ == "__main__":
    main()