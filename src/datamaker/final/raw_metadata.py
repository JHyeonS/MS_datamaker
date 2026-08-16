#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def infer_raw_sampling_rate(path: Union[str, Path], fallback_fs: Optional[float] = None) -> float:
    """Infer raw sampling rate from supported DAS file headers.

    Falls back to ``fallback_fs`` only when the header cannot be read.
    """

    raw_path = Path(path)
    ext = raw_path.suffix.lower()

    try:
        if ext in {".sgy", ".segy"}:
            return infer_segy_sampling_rate(raw_path)
        if ext == ".tdms":
            return infer_tdms_sampling_rate(raw_path)
    except Exception:
        if fallback_fs is not None:
            return float(fallback_fs)
        raise

    if fallback_fs is not None:
        return float(fallback_fs)
    raise ValueError(f"Unsupported raw file extension for sampling-rate inference: {ext}")


def infer_segy_sampling_rate(path: Path) -> float:
    try:
        import segyio
    except Exception as exc:
        raise ImportError("segyio is required to infer SEG-Y sampling rate") from exc

    with segyio.open(str(path), "r", ignore_geometry=True) as f:
        interval_us = int(f.bin[segyio.BinField.Interval])
        if interval_us <= 0 and f.tracecount > 0:
            interval_us = int(f.header[0][segyio.TraceField.TRACE_SAMPLE_INTERVAL])
        if interval_us <= 0:
            raise ValueError(f"Invalid SEG-Y sample interval: {path}")
        return float(1_000_000.0 / float(interval_us))


def infer_tdms_sampling_rate(path: Path) -> float:
    try:
        from nptdms import TdmsFile
    except Exception as exc:
        raise ImportError("nptdms is required to infer TDMS sampling rate") from exc

    tdms = TdmsFile.read_metadata(str(path))
    props = dict(tdms.properties)
    for key in ("SamplingFrequency[Hz]", "Sampling Frequency [Hz]", "SamplingFrequency"):
        value = props.get(key)
        if value is not None:
            fs = float(value)
            if fs > 0:
                return fs

    groups = tdms.groups()
    if groups and groups[0].channels():
        ch_props = dict(groups[0].channels()[0].properties)
        wf_increment = ch_props.get("wf_increment")
        if wf_increment is not None:
            inc = float(wf_increment)
            if inc > 0:
                return float(1.0 / inc)

    raise ValueError(f"Could not infer TDMS sampling rate: {path}")
