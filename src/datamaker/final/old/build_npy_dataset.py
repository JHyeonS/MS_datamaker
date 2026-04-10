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


def save_npy(arr: np.ndarray, out_path: Path, dtype: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr.astype(dtype, copy=False))


def make_output_path(out_root: Path, site: str, label_id: int, sample_idx: int) -> Path:
    subdir = LABEL_ID_TO_DIR[label_id]
    return out_root / site / subdir / f"{sample_idx:07d}.npy"


def main():
    args = parse_args()

    out_root = Path(args.out_root)
    metadata_dir = out_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(metadata_dir / "build_npy_dataset.log")
    site_fs_map = parse_site_fs(args.site_fs)

    logger.info("========== Build NPY Dataset Start ==========")
    logger.info(f"out_root       : {out_root}")
    logger.info(f"csv inputs     : {args.csv}")
    logger.info(f"dtype          : {args.dtype}")
    logger.info(f"overwrite      : {args.overwrite}")
    logger.info(f"log_every      : {args.log_every}")
    logger.info(f"fallback fs    : {args.fs}")
    logger.info(f"target_fs      : {args.target_fs}")
    logger.info(f"site_fs_map    : {site_fs_map}")
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

    sample_idx_counter = {}
    manifest_rows = []
    error_rows = []

    success_count = 0
    fail_count = 0
    per_site_counts = {}
    per_label_counts = {0: 0, 1: 0, 2: 0}

    t0 = time.time()

    progress = tqdm(df_all.iterrows(), total=len(df_all), ncols=120, desc="Building NPY")

    for idx, row in progress:
        try:
            site = sanitize_site_name(ensure_site(row, args.site_col))
            label_id = infer_label_from_row(row, args.label_col)
            raw_path = resolve_file_path(row)
            file_stem = resolve_file_stem(row, raw_path)
            group_id = resolve_group_id(row)
            view = resolve_view(row)
            start_sec, end_sec = resolve_start_end_sec(row)
            ch_start, ch_end = resolve_channel_range(row)

            original_fs = resolve_original_fs(
                row=row,
                site=site,
                site_fs_map=site_fs_map,
                fallback_fs=args.fs,
            )

            key = f"{site}_{label_id}"
            if key not in sample_idx_counter:
                sample_idx_counter[key] = 0

            sample_idx = sample_idx_counter[key]
            out_path = make_output_path(out_root, site, label_id, sample_idx)

            if out_path.exists() and not args.overwrite:
                arr = np.load(out_path)
                shape0, shape1 = arr.shape[:2]
                saved_fs = args.target_fs if args.target_fs is not None else original_fs
            else:
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
                save_npy(arr, out_path, args.dtype)

                shape0, shape1 = arr.shape
                saved_fs = args.target_fs if args.target_fs is not None else original_fs

            sample_idx_counter[key] += 1
            success_count += 1
            per_label_counts[label_id] += 1
            per_site_counts[site] = per_site_counts.get(site, 0) + 1

            manifest_rows.append({
                "global_index": int(idx),
                "sample_index_within_class": int(sample_idx),
                "site": site,
                "label": int(label_id),
                "label_name": LABEL_ID_TO_DIR[label_id].split("_", 1)[1],
                "npy_path": str(out_path),
                "raw_file_path": str(raw_path),
                "file_name": Path(raw_path).name,
                "file_stem": file_stem,
                "group_id": group_id,
                "view": view,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "duration_sec": float(end_sec - start_sec),
                "ch_start": int(ch_start),
                "ch_end": int(ch_end),
                "num_channels": int(ch_end - ch_start),
                "original_fs": float(original_fs),
                "saved_fs": float(saved_fs),
                "num_samples": int(shape1),
                "shape0": int(shape0),
                "shape1": int(shape1),
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
        manifest_df[manifest_df["label"] == label_id].copy().to_csv(
            metadata_dir / f"manifest_{name}.csv", index=False
        )

    summary = {
        "n_input_rows": int(len(df_all)),
        "n_success": int(success_count),
        "n_failed": int(fail_count),
        "success_rate": float(success_count / len(df_all)) if len(df_all) > 0 else 0.0,
        "elapsed_sec": float(elapsed_total),
        "elapsed_min": float(elapsed_total / 60.0),
        "avg_speed_samples_per_sec": float(success_count / max(elapsed_total, 1e-9)),
        "dtype": args.dtype,
        "out_root": str(out_root),
        "input_csvs": args.csv,
        "input_csv_row_counts": source_csv_counts,
        "target_fs": args.target_fs,
        "site_fs_map": site_fs_map,
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
    logger.info(f"Elapsed (min)   : {elapsed_total/60.0:.2f}")
    logger.info(f"Avg speed       : {success_count/max(elapsed_total,1e-9):.2f} samples/sec")
    logger.info(f"Per site counts : {per_site_counts}")
    logger.info(f"Per label counts: {summary['per_label_counts']}")
    logger.info("============================================")


if __name__ == "__main__":
    main()