#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import math
import argparse
import logging
from fractions import Fraction
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample_poly
import scipy.ndimage as ndimage

try:
    from nptdms import TdmsFile
except Exception:
    TdmsFile = None

try:
    import segyio
except Exception:
    segyio = None


LABEL_NAME_TO_ID = {
    "noise": 0,
    "event": 1,
    "unlabel": 2,
}

LABEL_ID_TO_DIR = {
    0: "0_noise",
    1: "1_event",
    2: "2_unlabel",
}


DEFAULT_FILTER_SPECS = {
    "pohang": {"low_hz": None, "high_hz": 50.0},
    "utah_2019": {"low_hz": None, "high_hz": 200.0},
    "utah_2023": {"low_hz": None, "high_hz": 500.0},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build NPY dataset from segment plan CSVs")

    parser.add_argument("--csv", action="append", required=True,
                        help="Input segment plan csv path. Can be used multiple times.")
    parser.add_argument("--out_root", type=str, required=True,
                        help="Output root directory")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "float64"],
                        help="Saved numpy dtype")
    parser.add_argument("--tdms-group", type=str, default=None,
                        help="TDMS group name. If omitted, first group is used.")
    parser.add_argument("--tdms-channel", type=str, default=None,
                        help="TDMS channel name. If omitted, first channel is used.")

    # old fs meaning: original fs for crop if row/site fs not available
    parser.add_argument("--fs", type=float, default=None,
                        help="Fallback original sampling rate for crop/resample")

    parser.add_argument("--target_fs", type=float, default=None,
                        help="If provided, resample each cropped sample to target_fs before saving")
    parser.add_argument("--site_fs", action="append", default=[],
                        help="Site-specific original fs mapping. Example: --site_fs pohang:1000 --site_fs utah:4000")

    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing npy files")
    parser.add_argument("--log_every", type=int, default=100,
                        help="Write progress log every N rows")
    parser.add_argument("--site_col", type=str, default="site",
                        help="Site column name")
    parser.add_argument("--label_col", type=str, default=None,
                        help="Optional explicit label column name. If omitted, inferred.")
    parser.add_argument("--preprocess", type=str, default="none",
                        choices=["none", "raw_rms", "median_filter_rms", "filter_rms", "filter_logenv_rms"],
                        help="Optional fixed preprocessing before saving NPY samples.")
    parser.add_argument("--target_rms", type=float, default=0.15,
                        help="Target RMS used by preprocessing modes that end with RMS normalization.")
    parser.add_argument("--rms_scope", type=str, default="site",
                        choices=["global", "site"],
                        help="RMS normalization scope for preprocessing modes that end with RMS normalization.")
    parser.add_argument("--filter_config", type=str, default=None,
                        help=(
                            "Optional JSON file with per-site filters, e.g. "
                            "{\"pohang\": {\"low_hz\": null, \"high_hz\": 50}}."
                        ))
    parser.add_argument("--filter_spec", action="append", default=[],
                        help=(
                            "Override site filter as site:low_hz:high_hz. "
                            "Use none for open cutoffs, e.g. pohang:none:50."
                        ))
    parser.add_argument("--default_filter", type=str, default="none:none",
                        help="Fallback filter as low_hz:high_hz for sites without --filter_spec.")
    parser.add_argument("--filter_order_low", type=float, default=3.0,
                        help="FFT high-pass transition order for the low cutoff.")
    parser.add_argument("--filter_decay_low", type=float, default=3.0,
                        help="FFT high-pass transition decay for the low cutoff.")
    parser.add_argument("--filter_order_high", type=float, default=1.0,
                        help="FFT low-pass transition order for the high cutoff.")
    parser.add_argument("--filter_decay_high", type=float, default=1.0,
                        help="FFT low-pass transition decay for the high cutoff.")
    parser.add_argument("--filter_pad_length", type=int, default=500,
                        help="Mirror/taper padding samples for FFT filtering.")
    parser.add_argument("--filter_taper_length", type=int, default=300,
                        help="Taper samples inside the FFT filtering padding.")
    parser.add_argument("--filter_zero_dc", action=argparse.BooleanOptionalAction, default=True,
                        help="Zero the DC FFT bin during filtering.")
    parser.add_argument("--log_base", type=float, default=1.0,
                        help="Log-envelope scaling base added to the envelope before log10.")
    parser.add_argument("--log_smooth_sigma_channel", type=float, default=1.0,
                        help="Gaussian smoothing sigma along channel axis for log-envelope scaling.")
    parser.add_argument("--log_smooth_sigma_time", type=float, default=0.5,
                        help="Gaussian smoothing sigma along time axis for log-envelope scaling.")
    parser.add_argument("--log_eps", type=float, default=1e-8,
                        help="Small epsilon used in log-envelope scaling.")
    parser.add_argument("--preprocess_run_name", type=str, default=None,
                        help="Optional run folder name used with --append_preprocess_run_name.")
    parser.add_argument("--append_preprocess_run_name", action="store_true",
                        help="Append a deterministic RMS/filter run folder under --out_root.")
    return parser.parse_args()


def setup_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("build_npy_dataset")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def parse_site_fs(site_fs_args: List[str]) -> Dict[str, float]:
    out = {}
    for item in site_fs_args:
        if ":" not in item:
            raise ValueError(f"Invalid --site_fs format: {item}. Use site:fs")
        site, fs = item.split(":", 1)
        out[site.strip().lower()] = float(fs)
    return out


def ensure_site(row: pd.Series, site_col: str) -> str:
    if site_col in row and pd.notna(row[site_col]):
        return str(row[site_col]).strip().lower()
    raise ValueError(f"Missing site column '{site_col}'")


def infer_label_from_row(row: pd.Series, explicit_label_col: Optional[str]) -> int:
    if explicit_label_col is not None and explicit_label_col in row and pd.notna(row[explicit_label_col]):
        return parse_label_value(row[explicit_label_col])

    for c in ["label", "label_id", "class_id", "target"]:
        if c in row.index and pd.notna(row[c]):
            return parse_label_value(row[c])

    for c in ["label_name", "class_name", "category", "split_type"]:
        if c in row.index and pd.notna(row[c]):
            return parse_label_value(str(row[c]))

    raise ValueError("Could not infer label from row")


def parse_label_value(v: Any) -> int:
    if isinstance(v, (int, np.integer)):
        iv = int(v)
        if iv == -1:
            return 2
        if iv in [0, 1, 2]:
            return iv
        raise ValueError(f"Unsupported integer label: {iv}")

    if isinstance(v, float):
        iv = int(v)
        if iv == -1:
            return 2
        if iv in [0, 1, 2]:
            return iv
        raise ValueError(f"Unsupported float label: {v}")

    s = str(v).strip().lower()
    if s in LABEL_NAME_TO_ID:
        return LABEL_NAME_TO_ID[s]
    if s in ["-1", "unlabeled", "unlabelled"]:
        return 2
    if s in ["0", "1", "2"]:
        return int(s)

    raise ValueError(f"Unsupported label value: {v}")


def resolve_file_path(row: pd.Series) -> str:
    for c in ["file_path", "filepath", "path", "raw_path", "source_path", "filename_full", "data_path"]:
        if c in row.index and pd.notna(row[c]):
            return str(row[c])
    raise ValueError("Could not find raw file path column")


def resolve_file_stem(row: pd.Series, raw_path: str) -> str:
    for c in ["file_stem", "filestem", "stem"]:
        if c in row.index and pd.notna(row[c]):
            return str(row[c])
    return Path(raw_path).stem


def resolve_group_id(row: pd.Series) -> str:
    for c in ["group_id", "groupid", "segment_group", "group"]:
        if c in row.index and pd.notna(row[c]):
            return str(row[c])
    return ""


def resolve_view(row: pd.Series) -> str:
    for c in ["view", "branch", "upper_lower", "position"]:
        if c in row.index and pd.notna(row[c]):
            return str(row[c])
    return ""


def resolve_start_end_sec(row: pd.Series) -> Tuple[float, float]:
    start_col = None
    end_col = None

    for c in ["start_sec", "start_s", "start", "t0", "start_time_sec"]:
        if c in row.index and pd.notna(row[c]):
            start_col = c
            break
    for c in ["end_sec", "end_s", "end", "t1", "end_time_sec"]:
        if c in row.index and pd.notna(row[c]):
            end_col = c
            break

    if start_col is None or end_col is None:
        raise ValueError("Could not find start/end seconds columns")

    start_sec = float(row[start_col])
    end_sec = float(row[end_col])
    if not (end_sec > start_sec):
        raise ValueError(f"Invalid time window: start={start_sec}, end={end_sec}")
    return start_sec, end_sec


def resolve_channel_range(row: pd.Series) -> Tuple[int, int]:
    ch_start = None
    ch_end = None

    for c in ["ch_start", "channel_start", "c0", "start_ch"]:
        if c in row.index and pd.notna(row[c]):
            ch_start = int(row[c])
            break
    for c in ["ch_end", "channel_end", "c1", "end_ch"]:
        if c in row.index and pd.notna(row[c]):
            ch_end = int(row[c])
            break

    if ch_start is None or ch_end is None:
        raise ValueError("Could not find channel range columns")
    if ch_end <= ch_start:
        raise ValueError(f"Invalid channel range: ch_start={ch_start}, ch_end={ch_end}")
    return ch_start, ch_end


def infer_fs_from_row(row: pd.Series) -> Optional[float]:
    for c in ["fs", "sample_rate", "sampling_rate", "sr", "original_fs", "raw_fs"]:
        if c in row.index and pd.notna(row[c]):
            return float(row[c])
    return None


def resolve_original_fs(row: pd.Series, site: str, site_fs_map: Dict[str, float], fallback_fs: Optional[float]) -> float:
    row_fs = infer_fs_from_row(row)
    if row_fs is not None:
        return float(row_fs)
    if site in site_fs_map:
        return float(site_fs_map[site])
    if fallback_fs is not None:
        return float(fallback_fs)
    raise ValueError(
        f"Could not resolve original fs for site='{site}'. "
        f"Need row fs column, --site_fs, or --fs."
    )


def sanitize_site_name(site: str) -> str:
    site = site.strip().lower()
    if site == "":
        raise ValueError("Empty site")
    return site


def read_tdms_window(
    file_path: str,
    start_sec: float,
    end_sec: float,
    ch_start: int,
    ch_end: int,
    original_fs: float,
    tdms_group: Optional[str] = None,
    tdms_channel: Optional[str] = None,
) -> np.ndarray:
    if TdmsFile is None:
        raise ImportError("nptdms is not installed")

    tdms = TdmsFile.read(file_path)
    groups = tdms.groups()
    if len(groups) == 0:
        raise ValueError(f"No TDMS groups found: {file_path}")

    # -----------------------------
    # 1) group 선택
    # -----------------------------
    if tdms_group is not None:
        group = None
        for g in groups:
            if g.name == tdms_group:
                group = g
                break
        if group is None:
            raise ValueError(f"TDMS group '{tdms_group}' not found in {file_path}")
    else:
        group = groups[0]

    channels = group.channels()
    if len(channels) == 0:
        raise ValueError(f"No TDMS channels found in group '{group.name}': {file_path}")

    # -----------------------------
    # 2) time index 계산
    # -----------------------------
    s0 = int(round(start_sec * original_fs))
    s1 = int(round(end_sec * original_fs))

    if s1 <= s0:
        raise ValueError(
            f"Invalid TDMS time window after index conversion: "
            f"start_sec={start_sec}, end_sec={end_sec}, s0={s0}, s1={s1}"
        )

    # -----------------------------
    # 3) special case:
    #    tdms_channel이 지정된 경우
    # -----------------------------
    if tdms_channel is not None:
        target_channel = None
        for ch in channels:
            if ch.name == tdms_channel:
                target_channel = ch
                break

        if target_channel is None:
            raise ValueError(f"TDMS channel '{tdms_channel}' not found in {file_path}")

        data = np.asarray(target_channel[:])

        # case A: channel 하나 안에 이미 2D 데이터가 들어있는 경우
        if data.ndim == 2:
            # data shape: (C, T) 라고 가정
            if ch_start < 0 or ch_end > data.shape[0]:
                raise ValueError(
                    f"Channel slice out of bounds for {file_path}: "
                    f"ch_start={ch_start}, ch_end={ch_end}, shape={data.shape}"
                )
            if s0 < 0 or s1 > data.shape[1]:
                raise ValueError(
                    f"Time slice out of bounds for {file_path}: "
                    f"s0={s0}, s1={s1}, shape={data.shape}"
                )
            return np.asarray(data[ch_start:ch_end, s0:s1])

        # case B: channel 하나가 1D 시계열인 경우
        elif data.ndim == 1:
            if ch_start != 0 or ch_end != 1:
                raise ValueError(
                    f"TDMS channel '{tdms_channel}' is 1D, "
                    f"but requested channel range [{ch_start}, {ch_end}). "
                    f"For standard DAS TDMS, do not set --tdms-channel."
                )
            if s0 < 0 or s1 > data.shape[0]:
                raise ValueError(
                    f"Time slice out of bounds for {file_path}: "
                    f"s0={s0}, s1={s1}, len={data.shape[0]}"
                )
            return data[s0:s1][None, :]  # (1, T)

        else:
            raise ValueError(
                f"Unsupported TDMS target channel ndim={data.ndim}, shape={data.shape}, file={file_path}"
            )

    # -----------------------------
    # 4) 일반 DAS TDMS:
    #    여러 TDMS channel을 stack해서 (C, T) 구성
    # -----------------------------
    if ch_start < 0 or ch_end > len(channels):
        raise ValueError(
            f"Channel slice out of bounds for {file_path}: "
            f"ch_start={ch_start}, ch_end={ch_end}, n_channels={len(channels)}"
        )

    traces = []
    selected_channels = channels[ch_start:ch_end]

    for ch in selected_channels:
        arr = np.asarray(ch[:])

        if arr.ndim != 1:
            raise ValueError(
                f"Expected 1D TDMS channel when stacking, but got "
                f"ndim={arr.ndim}, shape={arr.shape}, file={file_path}, channel={ch.name}"
            )

        if s0 < 0 or s1 > arr.shape[0]:
            raise ValueError(
                f"Time slice out of bounds for {file_path}: "
                f"s0={s0}, s1={s1}, len={arr.shape[0]}, channel={ch.name}"
            )

        traces.append(arr[s0:s1])

    if len(traces) == 0:
        raise ValueError(f"Empty TDMS slice from {file_path}")

    # 최종 shape = (C, T)
    out = np.stack(traces, axis=0)
    return out


def read_segy_window(
    file_path: str,
    start_sec: float,
    end_sec: float,
    ch_start: int,
    ch_end: int,
    original_fs: float,
) -> np.ndarray:
    if segyio is None:
        raise ImportError("segyio is not installed")

    with segyio.open(file_path, "r", ignore_geometry=True) as f:
        trace_count = f.tracecount
        if trace_count <= 0:
            raise ValueError(f"No traces in SEG-Y file: {file_path}")

        s0 = int(round(start_sec * original_fs))
        s1 = int(round(end_sec * original_fs))

        if ch_start < 0 or ch_end > trace_count:
            raise ValueError(f"Channel slice out of bounds for {file_path}: ch_start={ch_start}, ch_end={ch_end}, trace_count={trace_count}")

        traces = []
        for i in range(ch_start, ch_end):
            tr = np.asarray(f.trace[i], dtype=np.float32)
            if s1 > tr.shape[0]:
                raise ValueError(f"Time slice out of bounds for {file_path}: s1={s1}, trace_len={tr.shape[0]}")
            traces.append(tr[s0:s1])

        if len(traces) == 0:
            raise ValueError(f"Empty slice from SEG-Y file: {file_path}")

        return np.stack(traces, axis=0)


def read_raw_window(
    file_path: str,
    start_sec: float,
    end_sec: float,
    ch_start: int,
    ch_end: int,
    original_fs: float,
    tdms_group: Optional[str],
    tdms_channel: Optional[str],
) -> np.ndarray:
    ext = Path(file_path).suffix.lower()

    if ext == ".tdms":
        return read_tdms_window(
            file_path=file_path,
            start_sec=start_sec,
            end_sec=end_sec,
            ch_start=ch_start,
            ch_end=ch_end,
            original_fs=original_fs,
            tdms_group=tdms_group,
            tdms_channel=tdms_channel,
        )
    if ext in [".sgy", ".segy"]:
        return read_segy_window(
            file_path=file_path,
            start_sec=start_sec,
            end_sec=end_sec,
            ch_start=ch_start,
            ch_end=ch_end,
            original_fs=original_fs,
        )

    raise ValueError(f"Unsupported raw file extension: {ext}")


def maybe_resample(arr: np.ndarray, original_fs: float, target_fs: Optional[float]) -> np.ndarray:
    if target_fs is None:
        return arr

    if np.isclose(original_fs, target_fs):
        return arr

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



def parse_optional_hz(value: str) -> Optional[float]:
    s = str(value).strip().lower()
    if s in {"", "none", "null", "nan", "-"}:
        return None
    return float(s)


def parse_filter_pair(value: str) -> Dict[str, Optional[float]]:
    parts = str(value).split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid filter pair '{value}'. Use low_hz:high_hz")
    return {"low_hz": parse_optional_hz(parts[0]), "high_hz": parse_optional_hz(parts[1])}


def parse_filter_specs(
    filter_config: Optional[str],
    filter_args: List[str],
    default_filter: str,
) -> Tuple[Dict[str, Dict[str, Optional[float]]], Dict[str, Optional[float]]]:
    specs = {site: cfg.copy() for site, cfg in DEFAULT_FILTER_SPECS.items()}
    fallback = parse_filter_pair(default_filter)

    if filter_config is not None:
        config_path = Path(filter_config)
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        for site, cfg in loaded.items():
            site_key = str(site).strip().lower()
            specs[site_key] = {
                "low_hz": None if cfg.get("low_hz") is None else float(cfg.get("low_hz")),
                "high_hz": None if cfg.get("high_hz") is None else float(cfg.get("high_hz")),
            }

    for item in filter_args:
        parts = str(item).split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid --filter_spec '{item}'. Use site:low_hz:high_hz")
        site = parts[0].strip().lower()
        if site == "":
            raise ValueError(f"Invalid empty site in --filter_spec '{item}'")
        specs[site] = {"low_hz": parse_optional_hz(parts[1]), "high_hz": parse_optional_hz(parts[2])}

    return specs, fallback



def format_token_number(value: float) -> str:
    text = f"{float(value):g}"
    text = text.replace("-", "m").replace(".", "p")
    return text


def format_hz_token(value: Optional[float]) -> str:
    if value is None:
        return "none"
    return f"{format_token_number(value)}hz"


def filter_spec_token(spec: Dict[str, Optional[float]]) -> str:
    low = spec.get("low_hz")
    high = spec.get("high_hz")
    if low is None and high is None:
        return "nofilter"
    if low is None:
        return f"lp{format_hz_token(high)}"
    if high is None:
        return f"hp{format_hz_token(low)}"
    return f"bp{format_hz_token(low)}-{format_hz_token(high)}"


def sanitize_run_token(value: str) -> str:
    allowed = []
    for ch in str(value).strip().lower():
        if ch.isalnum() or ch in {"_", "-", "."}:
            allowed.append(ch)
        elif ch in {" ", "/", ":", "|"}:
            allowed.append("_")
    out = "".join(allowed).strip("_")
    return out or "run"


def make_preprocess_run_name(
    args: argparse.Namespace,
    filter_specs: Dict[str, Dict[str, Optional[float]]],
) -> str:
    if args.preprocess_run_name:
        return sanitize_run_token(args.preprocess_run_name)
    if args.preprocess == "none":
        return "raw"

    preferred_sites = [s for s in ["pohang", "utah_2019", "utah_2023"] if s in filter_specs]
    extra_sites = sorted(s for s in filter_specs if s not in preferred_sites)
    site_order = preferred_sites + extra_sites

    parts = [
        args.preprocess,
        f"rms{format_token_number(args.target_rms)}",
        f"scope-{args.rms_scope}",
    ]
    for site in site_order:
        site_token = sanitize_run_token(site).replace("utah_", "ut")
        parts.append(f"{site_token}_{filter_spec_token(filter_specs[site])}")
    return "__".join(parts)

def get_filter_spec_for_site(
    site: str,
    filter_specs: Dict[str, Dict[str, Optional[float]]],
    fallback_filter: Dict[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    return filter_specs.get(site, fallback_filter)


def mirror_padding(signal: np.ndarray, pad_length: int, taper_length: int, axis: int = -1) -> np.ndarray:
    if pad_length <= 0:
        return signal
    if taper_length < 0 or taper_length > pad_length:
        raise ValueError(f"Invalid taper_length={taper_length} for pad_length={pad_length}")

    axis = axis if axis >= 0 else signal.ndim + axis
    if signal.shape[axis] < taper_length:
        raise ValueError(
            f"Cannot mirror pad with taper_length={taper_length}; "
            f"axis length is {signal.shape[axis]}"
        )

    def _slice(arr: np.ndarray, start, stop):
        idx = [slice(None)] * arr.ndim
        idx[axis] = slice(start, stop)
        return arr[tuple(idx)]

    zero_length = pad_length - taper_length
    mirror = np.flip(signal, axis=axis)

    if taper_length > 0:
        taper = np.hanning(2 * taper_length)
        shape = [1] * signal.ndim
        shape[axis] = 2 * taper_length
        taper = taper.reshape(shape)
        taper_top = _slice(taper, None, taper_length)
        taper_bot = _slice(taper, taper_length, None)
        mirror_top = _slice(mirror, -taper_length, None) * taper_top
        mirror_bot = _slice(mirror, None, taper_length) * taper_bot
    else:
        shape_empty = list(signal.shape)
        shape_empty[axis] = 0
        mirror_top = np.empty(shape_empty, dtype=signal.dtype)
        mirror_bot = np.empty(shape_empty, dtype=signal.dtype)

    shape_zero = list(signal.shape)
    shape_zero[axis] = zero_length
    zeros = np.zeros(shape_zero, dtype=signal.dtype)
    return np.concatenate([zeros, mirror_top, signal, mirror_bot, zeros], axis=axis)


def f_filter(nt: int, dt: float, f_cut: float, order: float, decay: float, is_lowpass: bool, max_clip: float = 400.0) -> np.ndarray:
    f = np.abs(np.fft.fftfreq(nt, dt))
    x = f - f_cut if is_lowpass else f_cut - f
    x = np.clip(x, -max_clip, max_clip)
    m = -(decay / order) * np.log2(1 + 2 ** (order * x))
    return 2 ** m


def bandpass_filter_mask(
    nt: int,
    dt: float,
    f_low: Optional[float],
    order_low: float,
    decay_low: float,
    f_high: Optional[float],
    order_high: float,
    decay_high: float,
) -> np.ndarray:
    if f_low is None:
        m_low = np.ones(nt)
    else:
        m_low = f_filter(nt, dt, f_low, order_low, decay_low, is_lowpass=False)
    if f_high is None:
        m_high = np.ones(nt)
    else:
        m_high = f_filter(nt, dt, f_high, order_high, decay_high, is_lowpass=True)
    return m_low * m_high


def fft_filtering(data: np.ndarray, mask: np.ndarray, zero_dc: bool) -> np.ndarray:
    shape = data.shape
    nt = shape[-1]
    if len(mask) != nt:
        raise ValueError(f"Filter mask length {len(mask)} does not match time length {nt}")

    m_f = mask.copy()
    if zero_dc and len(m_f) > 0:
        m_f[0] = 0.0

    data_unfold = data.reshape((-1, nt)) if data.ndim > 1 else data[None, :]
    data_fft = np.fft.fft(data_unfold, axis=-1)
    data_fft *= m_f[None]
    out = np.real(np.fft.ifft(data_fft, axis=-1))
    return out.reshape(shape) if data.ndim > 1 else out.squeeze()


def median_detrend(arr: np.ndarray) -> np.ndarray:
    return arr - np.median(arr, axis=-1, keepdims=True)


def filter_signal_for_site(
    arr: np.ndarray,
    saved_fs: float,
    site: str,
    args: argparse.Namespace,
    filter_specs: Dict[str, Dict[str, Optional[float]]],
    fallback_filter: Dict[str, Optional[float]],
) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    spec = get_filter_spec_for_site(site, filter_specs, fallback_filter)
    if spec["low_hz"] is None and spec["high_hz"] is None:
        return out

    pad_length = int(args.filter_pad_length)
    taper_length = int(args.filter_taper_length)
    padded = mirror_padding(out, pad_length=pad_length, taper_length=taper_length, axis=-1)
    dt = 1.0 / float(saved_fs)
    mask = bandpass_filter_mask(
        nt=padded.shape[-1],
        dt=dt,
        f_low=spec["low_hz"],
        order_low=float(args.filter_order_low),
        decay_low=float(args.filter_decay_low),
        f_high=spec["high_hz"],
        order_high=float(args.filter_order_high),
        decay_high=float(args.filter_decay_high),
    )
    filtered = fft_filtering(padded, mask, zero_dc=bool(args.filter_zero_dc))
    if pad_length > 0:
        filtered = filtered[..., pad_length:-pad_length]
    return np.asarray(filtered, dtype=np.float32)


def envelope_1d_for_log_scaling(data: np.ndarray) -> np.ndarray:
    nt, _ = data.shape
    data_fft = np.fft.fftn(data, axes=(-2, -1))
    freqs_t = np.fft.fftfreq(nt)
    hilbert_filter_t = np.where(freqs_t > 0, 1j, np.where(freqs_t < 0, -1j, 0))[:, None]
    data_hilbert = np.fft.ifftn(hilbert_filter_t * data_fft, axes=(-2, -1))
    data_env = np.real(np.sqrt(data_hilbert * data_hilbert + data * data))
    return data_env.astype(np.float32)


def calculate_log_envelope_scale(
    data: np.ndarray,
    log_base: float,
    smooth_sigma: Tuple[float, float],
    eps: float,
) -> np.ndarray:
    data_env = envelope_1d_for_log_scaling(data) + float(log_base)
    data_env_log = np.log10(data_env)
    data_env_log -= data_env_log.min()
    data_env_log += float(eps)

    scale = data_env_log / data_env
    scale = np.log10(scale)
    scale = ndimage.gaussian_filter(scale, sigma=smooth_sigma)
    scale = np.power(10.0, scale)
    return scale.astype(np.float32)


def apply_log_envelope_scaling(arr: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    smooth_sigma = (float(args.log_smooth_sigma_channel), float(args.log_smooth_sigma_time))
    scale = calculate_log_envelope_scale(
        np.asarray(arr, dtype=np.float32),
        log_base=float(args.log_base),
        smooth_sigma=smooth_sigma,
        eps=float(args.log_eps),
    )
    return np.asarray(arr * scale, dtype=np.float32)


def apply_fixed_preprocess(
    arr: np.ndarray,
    saved_fs: float,
    site: str,
    args: argparse.Namespace,
    filter_specs: Dict[str, Dict[str, Optional[float]]],
    fallback_filter: Dict[str, Optional[float]],
) -> np.ndarray:
    if args.preprocess == "none":
        return arr

    out = np.asarray(arr, dtype=np.float32)
    if args.preprocess == "raw_rms":
        return out
    if args.preprocess == "median_filter_rms":
        out = median_detrend(out)
        return filter_signal_for_site(out, saved_fs, site, args, filter_specs, fallback_filter)
    if args.preprocess == "filter_rms":
        return filter_signal_for_site(out, saved_fs, site, args, filter_specs, fallback_filter)
    if args.preprocess == "filter_logenv_rms":
        out = filter_signal_for_site(out, saved_fs, site, args, filter_specs, fallback_filter)
        return apply_log_envelope_scaling(out, args)

    raise ValueError(f"Unsupported preprocess mode: {args.preprocess}")


def new_stats() -> Dict[str, Any]:
    return {"count": 0, "sum_sq": 0.0, "min": None, "max": None}



def new_frequency_stats(name: str, saved_fs: float, fft_n: int) -> Dict[str, Any]:
    return {
        "name": name,
        "saved_fs": float(saved_fs),
        "fft_n": int(fft_n),
        "sample_count": 0,
        "channel_count": 0,
        "sum_power": None,
    }


def update_frequency_stats(container: Dict[str, Dict[str, Any]], name: str, arr: np.ndarray, saved_fs: float) -> None:
    data = np.asarray(arr, dtype=np.float64)
    if data.ndim != 2 or data.size == 0:
        return

    n_time = int(data.shape[-1])
    key = f"{name}|fs={float(saved_fs):g}|n={n_time}"
    if key not in container:
        container[key] = new_frequency_stats(name=name, saved_fs=saved_fs, fft_n=n_time)

    stats = container[key]
    spec = np.fft.rfft(data, axis=-1)
    power = (np.abs(spec) ** 2) / float(n_time)
    mean_power_by_freq = power.mean(axis=0)

    if stats["sum_power"] is None:
        stats["sum_power"] = mean_power_by_freq
    else:
        stats["sum_power"] += mean_power_by_freq
    stats["sample_count"] += 1
    stats["channel_count"] += int(data.shape[0])


def frequency_stats_to_frame(container: Dict[str, Dict[str, Any]], group_col: str) -> pd.DataFrame:
    rows = []
    for stats in container.values():
        sample_count = int(stats["sample_count"])
        if sample_count <= 0 or stats["sum_power"] is None:
            continue
        saved_fs = float(stats["saved_fs"])
        fft_n = int(stats["fft_n"])
        freqs = np.fft.rfftfreq(fft_n, d=1.0 / saved_fs)
        mean_power = stats["sum_power"] / float(sample_count)
        total_power = float(np.sum(mean_power))
        for freq_hz, power_value in zip(freqs, mean_power):
            rows.append({
                group_col: stats["name"],
                "saved_fs": saved_fs,
                "fft_n": fft_n,
                "sample_count": sample_count,
                "channel_count_total": int(stats["channel_count"]),
                "freq_hz": float(freq_hz),
                "mean_power": float(power_value),
                "power_fraction": float(power_value / total_power) if total_power > 0 else 0.0,
            })
    return pd.DataFrame(rows)


def save_min_max_after_rms(
    metadata_dir: Path,
    final_stats_by_scope: Dict[str, Dict[str, Any]],
    final_stats_by_site_label: Dict[str, Dict[str, Any]],
    target_rms: float,
    rms_scope: str,
) -> None:
    payload = {
        "target_rms": float(target_rms),
        "rms_scope": rms_scope,
        "by_scope": final_stats_by_scope,
        "by_site_label": final_stats_by_site_label,
    }
    with open(metadata_dir / "min_max_after_rms.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def update_stats(stats: Dict[str, Any], arr: np.ndarray) -> None:
    arr64 = np.asarray(arr, dtype=np.float64)
    if arr64.size == 0:
        return
    stats["count"] += int(arr64.size)
    stats["sum_sq"] += float(np.sum(arr64 * arr64))
    vmin = float(np.min(arr64))
    vmax = float(np.max(arr64))
    stats["min"] = vmin if stats["min"] is None else min(float(stats["min"]), vmin)
    stats["max"] = vmax if stats["max"] is None else max(float(stats["max"]), vmax)


def finalize_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    count = int(stats["count"])
    rms = math.sqrt(float(stats["sum_sq"]) / count) if count > 0 else None
    return {
        "count": count,
        "sum_sq": float(stats["sum_sq"]),
        "rms": rms,
        "min": stats["min"],
        "max": stats["max"],
    }


def stats_key_for_row(site: str, rms_scope: str) -> str:
    if rms_scope == "global":
        return "global"
    if rms_scope == "site":
        return site
    raise ValueError(f"Unsupported rms_scope: {rms_scope}")


def load_window_for_row(
    row: pd.Series,
    args: argparse.Namespace,
    site_fs_map: Dict[str, float],
    filter_specs: Dict[str, Dict[str, Optional[float]]],
    fallback_filter: Dict[str, Optional[float]],
) -> Tuple[Dict[str, Any], np.ndarray]:
    site = sanitize_site_name(ensure_site(row, args.site_col))
    label_id = infer_label_from_row(row, args.label_col)
    raw_path = resolve_file_path(row)
    file_stem = resolve_file_stem(row, raw_path)
    group_id = resolve_group_id(row)
    view = resolve_view(row)
    start_sec, end_sec = resolve_start_end_sec(row)
    ch_start, ch_end = resolve_channel_range(row)
    original_fs = resolve_original_fs(row=row, site=site, site_fs_map=site_fs_map, fallback_fs=args.fs)

    if not Path(raw_path).exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    arr = read_raw_window(
        file_path=raw_path,
        start_sec=start_sec,
        end_sec=end_sec,
        ch_start=ch_start,
        ch_end=ch_end,
        original_fs=original_fs,
        tdms_group=args.tdms_group,
        tdms_channel=args.tdms_channel,
    )
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={arr.shape}")

    arr = maybe_resample(arr, original_fs=original_fs, target_fs=args.target_fs)
    saved_fs = args.target_fs if args.target_fs is not None else original_fs
    arr = apply_fixed_preprocess(
        arr=arr,
        saved_fs=saved_fs,
        site=site,
        args=args,
        filter_specs=filter_specs,
        fallback_filter=fallback_filter,
    )

    ctx = {
        "site": site,
        "label_id": int(label_id),
        "raw_path": raw_path,
        "file_stem": file_stem,
        "group_id": group_id,
        "view": view,
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "ch_start": int(ch_start),
        "ch_end": int(ch_end),
        "original_fs": float(original_fs),
        "saved_fs": float(saved_fs),
    }
    return ctx, arr

def save_npy(arr: np.ndarray, out_path: Path, dtype: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr.astype(dtype, copy=False))


def make_output_path(out_root: Path, site: str, label_id: int, sample_idx: int) -> Path:
    subdir = LABEL_ID_TO_DIR[label_id]
    return out_root / site / subdir / f"{sample_idx:07d}.npy"


def main():
    args = parse_args()

    base_out_root = Path(args.out_root)
    site_fs_map = parse_site_fs(args.site_fs)
    filter_specs, fallback_filter = parse_filter_specs(args.filter_config, args.filter_spec, args.default_filter)

    preprocess_run_name = None
    out_root = base_out_root
    if args.append_preprocess_run_name:
        preprocess_run_name = make_preprocess_run_name(args, filter_specs)
        out_root = base_out_root / preprocess_run_name

    metadata_dir = out_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(metadata_dir / "build_npy_dataset.log")

    logger.info("========== Build NPY Dataset Start ==========")
    logger.info(f"base_out_root  : {base_out_root}")
    logger.info(f"out_root       : {out_root}")
    logger.info(f"run_name       : {preprocess_run_name}")
    logger.info(f"csv inputs     : {args.csv}")
    logger.info(f"dtype          : {args.dtype}")
    logger.info(f"overwrite      : {args.overwrite}")
    logger.info(f"log_every      : {args.log_every}")
    logger.info(f"fallback fs    : {args.fs}")
    logger.info(f"target_fs      : {args.target_fs}")
    logger.info(f"site_fs_map    : {site_fs_map}")
    logger.info(f"preprocess     : {args.preprocess}")
    logger.info(f"target_rms     : {args.target_rms}")
    logger.info(f"rms_scope      : {args.rms_scope}")
    logger.info(f"filter_specs   : {filter_specs}")
    logger.info(f"log_base       : {args.log_base}")
    logger.info(f"log_smooth_sig : ({args.log_smooth_sigma_channel}, {args.log_smooth_sigma_time})")
    logger.info(f"log_eps        : {args.log_eps}")
    logger.info(f"fallback_filter: {fallback_filter}")
    logger.info("=============================================")

    input_copy_dir = metadata_dir / "input_segment_plans"
    input_copy_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    source_csv_counts = {}

    for csv_path in args.csv:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df["__source_csv__"] = str(csv_path.resolve())
        all_frames.append(df)
        source_csv_counts[str(csv_path)] = len(df)

        dst = input_copy_dir / csv_path.name
        if csv_path.resolve() != dst.resolve():
            df.to_csv(dst, index=False)

        logger.info(f"Loaded CSV: {csv_path} | rows={len(df)}")

    df_all = pd.concat(all_frames, axis=0, ignore_index=True)
    logger.info(f"Total merged rows: {len(df_all)}")

    rms_pre_stats = {}
    rms_by_scope = {}
    rms_scale_by_scope = {}
    stats_error_rows = []

    if args.preprocess != "none":
        logger.info("============== RMS Stats Pass ==============")
        stats_progress = tqdm(df_all.iterrows(), total=len(df_all), ncols=120, desc="RMS stats")
        for idx, row in stats_progress:
            try:
                ctx, arr = load_window_for_row(
                    row=row,
                    args=args,
                    site_fs_map=site_fs_map,
                    filter_specs=filter_specs,
                    fallback_filter=fallback_filter,
                )
                scope_key = stats_key_for_row(ctx["site"], args.rms_scope)
                if scope_key not in rms_pre_stats:
                    rms_pre_stats[scope_key] = new_stats()
                update_stats(rms_pre_stats[scope_key], arr)
            except Exception as e:
                stats_error_rows.append({
                    "pass": "rms_stats",
                    "global_index": int(idx),
                    "site": row.get(args.site_col, ""),
                    "raw_file_path": row.get("file_path", row.get("path", "")),
                    "source_csv": row.get("__source_csv__", ""),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                })
                if len(stats_error_rows) <= 20 or len(stats_error_rows) % args.log_every == 0:
                    logger.error(f"[RMS FAILED] idx={idx} | {type(e).__name__}: {e}")

        rms_pre_stats = {k: finalize_stats(v) for k, v in rms_pre_stats.items()}
        for scope_key, stats in rms_pre_stats.items():
            rms = stats["rms"]
            if rms is None or rms <= 0:
                logger.warning(f"[RMS] invalid rms for scope={scope_key}: {rms}")
                continue
            rms_by_scope[scope_key] = float(rms)
            rms_scale_by_scope[scope_key] = float(args.target_rms) / float(rms)
            logger.info(
                f"[RMS] scope={scope_key} rms={rms:.8g} "
                f"scale={rms_scale_by_scope[scope_key]:.8g} "
                f"min={stats['min']} max={stats['max']} count={stats['count']}"
            )

        with open(metadata_dir / "preprocess_rms_stats.json", "w", encoding="utf-8") as f:
            json.dump({
                "preprocess": args.preprocess,
                "target_rms": args.target_rms,
                "rms_scope": args.rms_scope,
                "rms_pre_stats": rms_pre_stats,
                "rms_by_scope": rms_by_scope,
                "rms_scale_by_scope": rms_scale_by_scope,
                "filter_specs": filter_specs,
                "fallback_filter": fallback_filter,
                "log_envelope_scaling": {
                    "enabled": args.preprocess == "filter_logenv_rms",
                    "log_base": float(args.log_base),
                    "smooth_sigma": [float(args.log_smooth_sigma_channel), float(args.log_smooth_sigma_time)],
                    "eps": float(args.log_eps),
                },
                "stats_error_count": len(stats_error_rows),
                "base_out_root": str(base_out_root),
                "effective_out_root": str(out_root),
                "preprocess_run_name": preprocess_run_name,
            }, f, indent=2, ensure_ascii=False)
        logger.info("============================================")

    sample_idx_counter = {}
    manifest_rows = []
    error_rows = list(stats_error_rows)

    success_count = 0
    fail_count = 0
    per_site_counts = {}
    per_label_counts = {0: 0, 1: 0, 2: 0}
    final_stats_by_scope = {}
    final_stats_by_site_label = {}
    frequency_stats_by_scope = {}
    frequency_stats_by_site_label = {}

    t0 = time.time()
    progress = tqdm(df_all.iterrows(), total=len(df_all), ncols=120, desc="Building NPY")

    for idx, row in progress:
        try:
            ctx, arr = load_window_for_row(
                row=row,
                args=args,
                site_fs_map=site_fs_map,
                filter_specs=filter_specs,
                fallback_filter=fallback_filter,
            )

            scope_key = stats_key_for_row(ctx["site"], args.rms_scope)
            rms_value = None
            scale_factor = 1.0
            if args.preprocess != "none":
                if scope_key not in rms_by_scope:
                    raise ValueError(f"Missing RMS value for scope='{scope_key}'")
                rms_value = float(rms_by_scope[scope_key])
                scale_factor = float(rms_scale_by_scope[scope_key])
                arr = arr * scale_factor

            label_id = int(ctx["label_id"])
            site = ctx["site"]
            key = f"{site}_{label_id}"
            if key not in sample_idx_counter:
                sample_idx_counter[key] = 0

            sample_idx = sample_idx_counter[key]
            out_path = make_output_path(out_root, site, label_id, sample_idx)

            if out_path.exists() and not args.overwrite:
                logger.debug(f"[SKIP EXISTING] {out_path}")
            else:
                save_npy(arr, out_path, args.dtype)

            shape0, shape1 = arr.shape[:2]
            sample_idx_counter[key] += 1
            success_count += 1
            per_label_counts[label_id] += 1
            per_site_counts[site] = per_site_counts.get(site, 0) + 1

            if scope_key not in final_stats_by_scope:
                final_stats_by_scope[scope_key] = new_stats()
            update_stats(final_stats_by_scope[scope_key], arr)

            site_label_key = f"{site}|{LABEL_ID_TO_DIR[label_id].split('_', 1)[1]}"
            if site_label_key not in final_stats_by_site_label:
                final_stats_by_site_label[site_label_key] = new_stats()
            update_stats(final_stats_by_site_label[site_label_key], arr)

            update_frequency_stats(frequency_stats_by_scope, scope_key, arr, ctx["saved_fs"])
            update_frequency_stats(frequency_stats_by_site_label, site_label_key, arr, ctx["saved_fs"])

            manifest_rows.append({
                "global_index": int(idx),
                "sample_index_within_class": int(sample_idx),
                "site": site,
                "label": label_id,
                "label_name": LABEL_ID_TO_DIR[label_id].split("_", 1)[1],
                "npy_path": str(out_path),
                "raw_file_path": str(ctx["raw_path"]),
                "file_name": Path(ctx["raw_path"]).name,
                "file_stem": ctx["file_stem"],
                "group_id": ctx["group_id"],
                "view": ctx["view"],
                "start_sec": float(ctx["start_sec"]),
                "end_sec": float(ctx["end_sec"]),
                "duration_sec": float(ctx["end_sec"] - ctx["start_sec"]),
                "ch_start": int(ctx["ch_start"]),
                "ch_end": int(ctx["ch_end"]),
                "num_channels": int(ctx["ch_end"] - ctx["ch_start"]),
                "original_fs": float(ctx["original_fs"]),
                "saved_fs": float(ctx["saved_fs"]),
                "num_samples": int(shape1),
                "shape0": int(shape0),
                "shape1": int(shape1),
                "preprocess": args.preprocess,
                "rms_scope_key": scope_key,
                "rms_value": rms_value,
                "rms_scale_factor": scale_factor,
                "source_csv": row["__source_csv__"],
            })

            if success_count % args.log_every == 0:
                elapsed = time.time() - t0
                speed = success_count / max(elapsed, 1e-9)
                logger.info(
                    f"processed={success_count}/{len(df_all)} "
                    f"({100.0 * success_count / len(df_all):.2f}%) | "
                    f"failed={fail_count} | speed={speed:.2f} samples/sec | "
                    f"elapsed={elapsed/60.0:.2f} min"
                )

        except Exception as e:
            fail_count += 1
            error_rows.append({
                "pass": "build",
                "global_index": int(idx),
                "site": row.get(args.site_col, ""),
                "raw_file_path": row.get("file_path", row.get("path", "")),
                "source_csv": row.get("__source_csv__", ""),
                "error_type": type(e).__name__,
                "error_message": str(e),
            })
            if fail_count <= 20 or fail_count % args.log_every == 0:
                logger.error(f"[FAILED] idx={idx} | {type(e).__name__}: {e}")

    elapsed_total = time.time() - t0

    manifest_df = pd.DataFrame(manifest_rows)
    all_csv_path = metadata_dir / "all_samples.csv"
    manifest_df.to_csv(all_csv_path, index=False)

    if len(error_rows) > 0:
        pd.DataFrame(error_rows).to_csv(metadata_dir / "build_errors.csv", index=False)

    for label_id, name in [(0, "noise"), (1, "event"), (2, "unlabel")]:
        if "label" in manifest_df.columns:
            out_manifest = manifest_df[manifest_df["label"] == label_id].copy()
        else:
            out_manifest = pd.DataFrame()
        out_manifest.to_csv(metadata_dir / f"manifest_{name}.csv", index=False)

    final_stats_by_scope = {k: finalize_stats(v) for k, v in final_stats_by_scope.items()}
    final_stats_by_site_label = {k: finalize_stats(v) for k, v in final_stats_by_site_label.items()}

    with open(metadata_dir / "preprocess_final_stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "preprocess": args.preprocess,
            "target_rms": args.target_rms,
            "rms_scope": args.rms_scope,
            "final_stats_by_scope": final_stats_by_scope,
            "final_stats_by_site_label": final_stats_by_site_label,
        }, f, indent=2, ensure_ascii=False)

    save_min_max_after_rms(
        metadata_dir=metadata_dir,
        final_stats_by_scope=final_stats_by_scope,
        final_stats_by_site_label=final_stats_by_site_label,
        target_rms=args.target_rms,
        rms_scope=args.rms_scope,
    )

    freq_scope_df = frequency_stats_to_frame(frequency_stats_by_scope, group_col="scope")
    freq_site_label_df = frequency_stats_to_frame(frequency_stats_by_site_label, group_col="site_label")
    freq_scope_df.to_csv(metadata_dir / "frequency_energy_by_scope.csv", index=False)
    freq_site_label_df.to_csv(metadata_dir / "frequency_energy_by_site_label.csv", index=False)

    summary = {
        "n_input_rows": int(len(df_all)),
        "n_success": int(success_count),
        "n_failed": int(fail_count),
        "n_stats_failed": int(len(stats_error_rows)),
        "success_rate": float(success_count / len(df_all)) if len(df_all) > 0 else 0.0,
        "elapsed_sec": float(elapsed_total),
        "elapsed_min": float(elapsed_total / 60.0),
        "avg_speed_samples_per_sec": float(success_count / max(elapsed_total, 1e-9)),
        "dtype": args.dtype,
        "base_out_root": str(base_out_root),
        "out_root": str(out_root),
        "preprocess_run_name": preprocess_run_name,
        "input_csvs": args.csv,
        "input_csv_row_counts": source_csv_counts,
        "target_fs": args.target_fs,
        "site_fs_map": site_fs_map,
        "preprocess": {
            "mode": args.preprocess,
            "target_rms": args.target_rms,
            "rms_scope": args.rms_scope,
            "rms_by_scope": rms_by_scope,
            "rms_scale_by_scope": rms_scale_by_scope,
            "rms_pre_stats": rms_pre_stats,
            "final_stats_by_scope": final_stats_by_scope,
            "final_stats_by_site_label": final_stats_by_site_label,
            "frequency_energy_by_scope_csv": str(metadata_dir / "frequency_energy_by_scope.csv"),
            "frequency_energy_by_site_label_csv": str(metadata_dir / "frequency_energy_by_site_label.csv"),
            "min_max_after_rms_json": str(metadata_dir / "min_max_after_rms.json"),
            "filter_specs": filter_specs,
            "fallback_filter": fallback_filter,
            "filter_order_low": args.filter_order_low,
            "filter_decay_low": args.filter_decay_low,
            "filter_order_high": args.filter_order_high,
            "filter_decay_high": args.filter_decay_high,
            "filter_pad_length": args.filter_pad_length,
            "filter_taper_length": args.filter_taper_length,
            "filter_zero_dc": args.filter_zero_dc,
            "log_envelope_scaling": {
                "enabled": args.preprocess == "filter_logenv_rms",
                "log_base": float(args.log_base),
                "smooth_sigma": [float(args.log_smooth_sigma_channel), float(args.log_smooth_sigma_time)],
                "eps": float(args.log_eps),
            },
        },
        "per_site_counts": per_site_counts,
        "per_label_counts": {
            "noise": int(per_label_counts[0]),
            "event": int(per_label_counts[1]),
            "unlabel": int(per_label_counts[2]),
        },
    }

    with open(metadata_dir / "build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("============== Build Finished ==============")
    logger.info(f"Saved manifest  : {all_csv_path}")
    logger.info(f"Success         : {success_count}")
    logger.info(f"Failed          : {fail_count}")
    logger.info(f"Stats failed    : {len(stats_error_rows)}")
    logger.info(f"Elapsed (min)   : {elapsed_total/60.0:.2f}")
    logger.info(f"Avg speed       : {success_count/max(elapsed_total,1e-9):.2f} samples/sec")
    logger.info(f"Per site counts : {per_site_counts}")
    logger.info(f"Per label counts: {summary['per_label_counts']}")
    logger.info(f"Final stats     : {final_stats_by_scope}")
    logger.info("============================================")


if __name__ == "__main__":
    main()
