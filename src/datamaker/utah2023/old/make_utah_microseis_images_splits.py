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


MACOS_PREFIXES = ("._", ".DS_Store")


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



def _clean_label(s):
    return str(s).replace("/", "_").replace(" ", "")



def parse_split_ranges(split_text):
    if split_text is None or not str(split_text).strip():
        return None

    text = str(split_text).strip()
    for sep in ["/", ";", "|"]:
        text = text.replace(sep, ",")
    chunks = [c.strip() for c in text.split(",") if c.strip()]

    ranges = []
    for idx, chunk in enumerate(chunks):
        chunk = chunk.replace("~", "-").replace(":", "-")
        parts = [p.strip() for p in chunk.split("-") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"Invalid split range: '{chunk}'. Example: 200-605,650-1055")
        start, end = int(parts[0]), int(parts[1])
        if start > end:
            raise ValueError(f"Invalid split range: start > end in '{chunk}'")
        ranges.append((start, end))
    return ranges



def prompt_split_ranges():
    print("[INPUT] split ranges를 입력하세요.")
    print("        예시: 200-605/650-1055/1100-1505/1550-1955")
    text = input("split ranges: ").strip()
    return parse_split_ranges(text)



def list_sgy_files(data_dir, max_files=None):
    files = sorted(
        p for p in Path(data_dir).rglob("*")
        if p.is_file() and p.suffix.lower() == ".sgy" and not any(p.name.startswith(pref) for pref in MACOS_PREFIXES)
    )
    if max_files is not None:
        files = files[:max_files]
    return files



def load_channel_mapping(csv_path):
    if csv_path is None:
        return None

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"channel mapping csv not found: {csv_path}")

    df = pd.read_csv(path)
    cols = list(df.columns)
    channel_col = next((c for c in cols if c.strip().lower() == "channel"), None)
    depth_col = next((c for c in cols if c.strip().lower() == "depth (m)"), None)
    fiber_col = next((c for c in cols if c.strip().lower() == "fiber distance (m)"), None)

    if channel_col is None:
        raise ValueError(f"'Channel' column not found in mapping csv: {cols}")

    if depth_col is not None and df[depth_col].notna().sum() > 1:
        value_col = depth_col
        value_name = "Depth (m)"
    elif fiber_col is not None and df[fiber_col].notna().sum() > 1:
        value_col = fiber_col
        value_name = "Fiber distance (m)"
    else:
        raise ValueError(
            "Mapping csv exists but neither usable 'Depth (m)' nor 'Fiber distance (m)' columns were found."
        )

    mapping = {
        int(ch): float(val)
        for ch, val in zip(df[channel_col], df[value_col])
        if pd.notna(ch) and pd.notna(val)
    }

    return {
        "raw_df": df,
        "channel_col": channel_col,
        "value_col": value_col,
        "value_name": value_name,
        "mapping": mapping,
    }



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



def get_axis_endpoint_labels(ch_start, ch_end, mapping_info, input_channel_base=0):
    # Returned values correspond to visual top/bottom labels.
    if mapping_info is None:
        top_label = f"ch {ch_start}"
        bottom_label = f"ch {ch_end}"
        ylabel = "Channel"
        return top_label, bottom_label, ylabel

    channel_map = mapping_info["mapping"]
    ylabel = mapping_info["value_name"]

    csv_ch_start = ch_start + (1 - input_channel_base)
    csv_ch_end = ch_end + (1 - input_channel_base)

    if csv_ch_start not in channel_map or csv_ch_end not in channel_map:
        top_label = f"ch {ch_start}"
        bottom_label = f"ch {ch_end}"
        return top_label, bottom_label, ylabel

    v_start = channel_map[csv_ch_start]
    v_end = channel_map[csv_ch_end]

    # First row is ch_start and is drawn at the top (origin='upper').
    top_val = v_start
    bottom_val = v_end

    top_label = f"{top_val:.1f}"
    bottom_label = f"{bottom_val:.1f}"
    return top_label, bottom_label, ylabel



def make_image(x, out_png, fs, cmap="gray", clip=3.0,
               top_label=None, bottom_label=None, ylabel="Depth (m)"):
    # x shape: (channels/depth, time)
    img = np.clip(x, -clip, clip)
    n_ch, n_t = img.shape
    duration_sec = n_t / fs

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(
        img,
        cmap=cmap,
        aspect="auto",
        origin="upper",  # shallow on top if the selected channel order is shallow->deep
        vmin=-clip,
        vmax=clip,
        extent=[0.0, duration_sec, 0, n_ch - 1],
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, duration_sec)

    if n_ch > 1:
        ax.set_yticks([0, n_ch - 1])
        ax.set_yticklabels([
            top_label if top_label is not None else "top",
            bottom_label if bottom_label is not None else "bottom",
        ])
    else:
        ax.set_yticks([0])
        ax.set_yticklabels([top_label if top_label is not None else "single"])

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)



def process_and_save_segments(x, out_root, folder_name, sgy_name, split_name, split_start, split_end,
                              fs, win_len, stride_len, fmin, fmax, bp_order, cmap, clip,
                              mapping_info=None, input_channel_base=0):
    event_dir = Path(out_root) / split_name / folder_name
    event_dir.mkdir(parents=True, exist_ok=True)

    _, n_t = x.shape
    if win_len <= 0:
        raise ValueError(f"win_len must be positive, got {win_len}")
    if stride_len <= 0:
        raise ValueError(f"stride_len must be positive, got {stride_len}")
    if n_t < win_len:
        return []

    start_indices = list(range(0, n_t - win_len + 1, stride_len))
    n_seg = len(start_indices)

    top_label, bottom_label, ylabel = get_axis_endpoint_labels(
        split_start, split_end, mapping_info, input_channel_base=input_channel_base
    )

    seg_rows = []
    for seg_idx, s in enumerate(start_indices):
        e = s + win_len
        x_seg = x[:, s:e]

        x_seg = bandpass_filter(x_seg, fs=fs, fmin=fmin, fmax=fmax, order=bp_order)
        x_seg = robust_normalize(x_seg)

        seg_name = f"seg_{seg_idx:03d}"
        out_png = event_dir / f"{seg_name}.png"
        make_image(
            x_seg,
            str(out_png),
            fs=fs,
            cmap=cmap,
            clip=clip,
            top_label=top_label,
            bottom_label=bottom_label,
            ylabel=ylabel,
        )

        seg_rows.append({
            "sgy_file": sgy_name,
            "split_name": split_name,
            "split_ch_start": split_start,
            "split_ch_end": split_end,
            "folder": folder_name,
            "segment_index": seg_idx,
            "png_relpath": f"{split_name}/{folder_name}/{seg_name}.png",
            "sample_start": s,
            "sample_end": e,
            "start_sec": s / fs,
            "end_sec": e / fs,
            "axis_top": top_label,
            "axis_bottom": bottom_label,
            "axis_name": ylabel,
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
    parser.add_argument(
        "--stride_sec",
        type=float,
        default=None,
        help="Sliding window stride in seconds. Default=None means non-overlap (same as win_sec). 예: win_sec=2, stride_sec=1 -> 0~2,1~3,2~4...",
    )

    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help="예: '200-605,650-1055,1100-1505,1550-1955' 또는 '200-605/650-1055/...'",
    )
    parser.add_argument(
        "--input_channel_base",
        type=int,
        default=0,
        choices=[0, 1],
        help="입력 split range의 channel 시작 기준. 기본값 0(파이썬 인덱스)",
    )
    parser.add_argument(
        "--channel_map_csv",
        type=str,
        default=None,
        help="Channel-Depth/FiberDistance 매핑 CSV 경로",
    )

    parser.add_argument("--fmin", type=float, default=5.0)
    parser.add_argument("--fmax", type=float, default=80.0)
    parser.add_argument("--bp_order", type=int, default=4)

    parser.add_argument("--clip", type=float, default=3.0)
    parser.add_argument("--cmap", type=str, default="gray")
    args = parser.parse_args()

    split_ranges = parse_split_ranges(args.splits)
    if split_ranges is None:
        split_ranges = prompt_split_ranges()
    if not split_ranges:
        raise ValueError("No valid split ranges were provided.")

    mapping_info = None
    if args.channel_map_csv is not None:
        mapping_info = load_channel_mapping(args.channel_map_csv)
        print(f"[INFO] mapping csv loaded: {args.channel_map_csv}")
        print(f"[INFO] y-axis values: {mapping_info['value_name']}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sgy_files = list_sgy_files(args.data_dir, max_files=args.max_files)
    win_len = int(args.fs * args.win_sec)
    stride_sec = args.win_sec if args.stride_sec is None else args.stride_sec
    stride_len = int(args.fs * stride_sec)
    if stride_len <= 0:
        raise ValueError(f"stride_sec is too small: {stride_sec}")
    summary_rows = []

    print(f"[INFO] data_dir={args.data_dir}")
    print(f"[INFO] num_files={len(sgy_files)}")
    print(f"[INFO] splits={split_ranges}")
    print(f"[INFO] win_sec={args.win_sec}, stride_sec={stride_sec}")

    for file_idx, sgy_path in enumerate(sgy_files):
        stem = sgy_path.stem
        for split_idx, (split_start, split_end) in enumerate(split_ranges):
            split_name = f"split_{split_idx+1:02d}_ch{split_start:04d}_{split_end:04d}"
            folder = f"{file_idx:04d}_{stem}"
            try:
                x = read_sgy_selected_all_obspy(str(sgy_path), split_start, split_end)
                rows = process_and_save_segments(
                    x=x,
                    out_root=out_dir,
                    folder_name=folder,
                    sgy_name=sgy_path.name,
                    split_name=split_name,
                    split_start=split_start,
                    split_end=split_end,
                    fs=args.fs,
                    win_len=win_len,
                    stride_len=stride_len,
                    fmin=args.fmin,
                    fmax=args.fmax,
                    bp_order=args.bp_order,
                    cmap=args.cmap,
                    clip=args.clip,
                    mapping_info=mapping_info,
                    input_channel_base=args.input_channel_base,
                )
                summary_rows.extend(rows)
                print(
                    f"[INFO] done {file_idx+1}/{len(sgy_files)} | {sgy_path.name} | {split_name}"
                )
            except Exception as e:
                print(f"[WARN] failed: {sgy_path.name} | {split_name} | {e}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)

    split_meta = []
    for split_idx, (split_start, split_end) in enumerate(split_ranges):
        split_name = f"split_{split_idx+1:02d}_ch{split_start:04d}_{split_end:04d}"
        top_label, bottom_label, ylabel = get_axis_endpoint_labels(
            split_start, split_end, mapping_info, input_channel_base=args.input_channel_base
        )
        split_meta.append({
            "split_name": split_name,
            "split_ch_start": split_start,
            "split_ch_end": split_end,
            "axis_name": ylabel,
            "axis_top": top_label,
            "axis_bottom": bottom_label,
        })
    pd.DataFrame(split_meta).to_csv(out_dir / "split_metadata.csv", index=False)

    print("[DONE]")


if __name__ == "__main__":
    main()
