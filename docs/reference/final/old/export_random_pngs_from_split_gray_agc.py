#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Randomly sample rows from a split CSV and export .npy arrays as grayscale PNGs
using bandpass + AGC preprocessing.

Expected .npy shape:
- (C, T)
or
- (1, C, T)

Example
-------
python src/export_random_pngs_from_split_gray_agc.py \
  --csv /home/ted1204/MS_datamaker_merge/outputs_npy/metadata/experiments/stage1_utah_only/test.csv \
  --out_dir /home/ted1204/MS_datamaker_merge/outputs_npy/preview_pngs_gray/stage1_utah_test \
  --num_samples 12 \
  --seed 42 \
  --fs 2000 \
  --fmin 5 \
  --fmax 80 \
  --agc_win_sec 0.05
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_npy_column(df: pd.DataFrame) -> str:
    for col in ["npy_path", "path"]:
        if col in df.columns:
            return col
    raise ValueError("CSV must contain either 'npy_path' or 'path' column.")


def sanitize_text(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


def load_npy_2d(npy_path: Path) -> np.ndarray:
    x = np.load(npy_path)

    if x.ndim == 3:
        if x.shape[0] == 1:
            x = x[0]
        else:
            raise ValueError(f"Expected (1,C,T) or (C,T), got shape={x.shape} for {npy_path}")
    elif x.ndim != 2:
        raise ValueError(f"Expected 2D or 3D array, got shape={x.shape} for {npy_path}")

    return x.astype(np.float32)


def bandpass(x: np.ndarray, fs: float, fmin: float, fmax: float, order: int = 4) -> np.ndarray:
    """
    x: (C, T)
    """
    nyq = 0.5 * fs
    low = fmin / nyq
    high = fmax / nyq

    if not (0.0 < low < high < 1.0):
        raise ValueError(
            f"Invalid bandpass range: fs={fs}, fmin={fmin}, fmax={fmax}, "
            f"normalized=({low:.4f}, {high:.4f})"
        )

    b, a = butter(order, [low, high], btype="band")
    y = filtfilt(b, a, x, axis=1)
    return y.astype(np.float32)


def robust_norm_channelwise(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    x: (C, T)
    channel-wise robust normalization
    """
    med = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=1, keepdims=True)
    return (x - med) / (1.4826 * mad + eps)


def agc(x: np.ndarray, win: int = 100, eps: float = 1e-6) -> np.ndarray:
    """
    Simple channel-wise AGC.
    x: (C, T)
    """
    if win < 1:
        return x

    x64 = x.astype(np.float64)
    power = x64 ** 2

    kernel = np.ones(win, dtype=np.float64) / float(win)
    out = np.empty_like(x64)

    for c in range(x.shape[0]):
        env = np.sqrt(np.convolve(power[c], kernel, mode="same"))
        out[c] = x64[c] / (env + eps)

    return out.astype(np.float32)


def global_clip_to_uint8(x: np.ndarray, clip_sigma: float = 3.0) -> np.ndarray:
    """
    Convert processed float array to uint8 grayscale image.
    x: (C, T)
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-8
    sigma = 1.4826 * mad

    vmin = med - clip_sigma * sigma
    vmax = med + clip_sigma * sigma

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(np.min(x))
        vmax = float(np.max(x))
        if vmin == vmax:
            vmax = vmin + 1e-6

    y = np.clip(x, vmin, vmax)
    y = (y - vmin) / (vmax - vmin + 1e-8)
    y = (255.0 * y).astype(np.uint8)
    return y


def preprocess_to_gray_image(
    x: np.ndarray,
    fs: float,
    fmin: float,
    fmax: float,
    order: int,
    agc_win_sec: float,
    clip_sigma: float,
) -> np.ndarray:
    """
    x: (C, T)
    return uint8 grayscale image: (C, T)
    """
    y = bandpass(x, fs=fs, fmin=fmin, fmax=fmax, order=order)
    y = robust_norm_channelwise(y)

    agc_win = max(1, int(round(agc_win_sec * fs)))
    y = agc(y, win=agc_win)

    img = global_clip_to_uint8(y, clip_sigma=clip_sigma)
    return img


def save_gray_png(
    img: np.ndarray,
    save_path: Path,
    title: str,
    dpi: int = 150,
) -> None:
    """
    img: uint8 (C, T)
    """
    c, t = img.shape
    fig_w = max(6.0, t / 300.0)
    fig_h = max(4.0, c / 80.0)

    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(
        img,
        cmap="gray",
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=0,
        vmax=255,
    )
    plt.xlabel("Time")
    plt.ylabel("Channel")
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Input split CSV")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for PNGs")
    ap.add_argument("--num_samples", type=int, default=10, help="Number of samples to export")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--label_filter", type=int, default=None, choices=[0, 1, 2], help="Optional label filter")

    ap.add_argument("--fs", type=float, required=True, help="Sampling rate in Hz")
    ap.add_argument("--fmin", type=float, default=5.0, help="Bandpass low cutoff")
    ap.add_argument("--fmax", type=float, default=80.0, help="Bandpass high cutoff")
    ap.add_argument("--filter_order", type=int, default=4, help="Butterworth filter order")
    ap.add_argument("--agc_win_sec", type=float, default=0.05, help="AGC window in seconds")
    ap.add_argument("--clip_sigma", type=float, default=3.0, help="Robust clip sigma")
    ap.add_argument("--dpi", type=int, default=150, help="PNG dpi")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = pd.read_csv(csv_path)
    npy_col = find_npy_column(df)

    if args.label_filter is not None:
        if "label" not in df.columns:
            raise ValueError("label_filter was provided, but CSV has no 'label' column.")
        df = df[df["label"] == args.label_filter].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No rows available after filtering.")

    rng = random.Random(args.seed)
    n = min(args.num_samples, len(df))
    indices = rng.sample(list(range(len(df))), n)

    print(f"[INFO] csv={csv_path}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] total_rows={len(df)}")
    print(f"[INFO] export_n={n}")

    for rank, idx in enumerate(indices, start=1):
        row = df.iloc[idx]
        npy_path = Path(row[npy_col])

        if not npy_path.exists():
            print(f"[WARN] missing file: {npy_path}")
            continue

        try:
            x = load_npy_2d(npy_path)
            img = preprocess_to_gray_image(
                x=x,
                fs=args.fs,
                fmin=args.fmin,
                fmax=args.fmax,
                order=args.filter_order,
                agc_win_sec=args.agc_win_sec,
                clip_sigma=args.clip_sigma,
            )

            label = row["label"] if "label" in row.index else "na"
            label_name = row["label_name"] if "label_name" in row.index else f"label_{label}"
            site = row["site"] if "site" in row.index else "unknown"
            group_id = row["group_id"] if "group_id" in row.index else "na"
            stem = npy_path.stem

            title = f"{site} | {label_name} | group={group_id} | {stem}"
            save_name = (
                f"{rank:03d}__{sanitize_text(site)}__{sanitize_text(str(label_name))}"
                f"__g{sanitize_text(str(group_id))}__{sanitize_text(stem)}.png"
            )
            save_path = out_dir / save_name

            save_gray_png(
                img=img,
                save_path=save_path,
                title=title,
                dpi=args.dpi,
            )
            print(f"[SAVED] {save_path}")

        except Exception as e:
            print(f"[FAILED] idx={idx} path={npy_path} error={e}")

    print("[DONE]")


if __name__ == "__main__":
    main()