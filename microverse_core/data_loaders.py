"""Loads power and ENF data into the project's contract record types.

Two real datasets feed this project: the NLR GenAI Workload Power
Profiles (per-node and whole-facility wattmeter logs) and the AFRL
ENF measurements. Neither ships in this repo. synthetic_power_profile()
and synthetic_enf() produce plausible-shaped traces for development
use only -- never for a result that goes in a paper.

Pipeline (ingestion path):
    load_enf()               ENF CSV -> list[float]
    discover_nlr_pairs()     scans a folder, pairs NVML+RAPL logs by node
    load_nlr_multi()         loads all pairs, auto-detects sample rate
    build_combined_records() merges ENF + all node windows into one
                              record per index
    write_combined_jsonl()   writes merged records to JSONL
    read_combined_jsonl()    generator that yields one record at a time

See .readme/data_loaders.md for the real dataset details, the combined
record's column layout, and the ENF cleaning pipeline's full
validation history.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .contracts import PowerSample, WorkloadClass


def load_nlr_profile(path: str) -> list[PowerSample]:
    """Loads one NLR power profile CSV into PowerSample records.

    Args:
        path: Path to a CSV file with timestamp, node_id, power_w,
            and workload columns.

    Returns:
        list[PowerSample]: One record per row.
    """
    samples: list[PowerSample] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            samples.append(
                PowerSample(
                    timestamp=float(row["timestamp"]),
                    node_id=str(row["node_id"]),
                    power_w=float(row["power_w"]),
                    workload_class=str(row["workload"]),
                )
            )
    return samples


def load_enf(path: str) -> list[float]:
    """Loads an ENF CSV file into a flat list of Hz readings.

    Args:
        path: Path to the ENF CSV file. The first row is metadata and
            is skipped; each subsequent row is (index, frequency_hz).

    Returns:
        list[float]: Frequency readings in Hz, ordered by ascending index.
    """
    values: list[float] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            if not row:
                continue
            try:
                values.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    return values


def clean_enf(
    values: list[float],
    physical_floor: float = 40.0,
    physical_ceiling: float = 80.0,
    hampel_window: int = 11,
    hampel_n_sigmas: float = 2.0,
) -> list[float]:
    """Detects and corrects implausible ENF readings via a two-stage process.

    Not called automatically by load_enf() -- call this explicitly so
    the transformation stays visible and auditable. Superseded by
    combined_smooth() for production use; kept for anyone who wants
    the lighter-weight version. See .readme/data_loaders.md for the
    full validation history and why interpolation is used instead of
    a flat median replacement.

    Stage 1 (physical-range check): any reading outside
    [physical_floor, physical_ceiling] Hz is marked bad
    unconditionally. Stage 2 (Hampel filter): for each point, compares
    against the median of a local window of already-known-good points,
    flagging it bad if it deviates more than
    `hampel_n_sigmas * 1.4826 * MAD` from that local median.

    Args:
        values: Raw ENF readings from load_enf().
        physical_floor: Lower sanity bound in Hz.
        physical_ceiling: Upper sanity bound in Hz.
        hampel_window: Samples considered on each side of the point
            being checked.
        hampel_n_sigmas: Outlier threshold in MAD-scaled "sigmas".
            Lower is more aggressive.

    Returns:
        list[float]: Same length as `values`. Detected-bad points are
        replaced via linear interpolation between the nearest good
        neighbors.
    """
    n = len(values)
    if n == 0:
        return values

    bad = [v < physical_floor or v > physical_ceiling for v in values]

    k = 1.4826
    for i in range(n):
        w_lo = max(0, i - hampel_window)
        w_hi = min(n, i + hampel_window + 1)
        window = [values[j] for j in range(w_lo, w_hi) if not bad[j]]
        if len(window) < 2:
            continue
        med = statistics.median(window)
        mad = statistics.median([abs(v - med) for v in window])
        threshold = hampel_n_sigmas * k * mad
        if threshold > 0 and abs(values[i] - med) > threshold:
            bad[i] = True

    result = list(values)
    bad_idx = [i for i, b in enumerate(bad) if b]
    for i in bad_idx:
        lo = i - 1
        while lo >= 0 and bad[lo]:
            lo -= 1
        hi = i + 1
        while hi < n and bad[hi]:
            hi += 1
        if lo >= 0 and hi < n:
            frac = (i - lo) / (hi - lo)
            result[i] = values[lo] + frac * (values[hi] - values[lo])
        elif lo >= 0:
            result[i] = values[lo]
        elif hi < n:
            result[i] = values[hi]

    return result


def hampel_correct(
    values: list[float],
    window: int = 11,
    n_sigmas: float = 2.0,
) -> list[float]:
    """Detects and corrects outliers via a rolling median/MAD Hampel filter.

    Same mechanism as clean_enf()'s Hampel stage, factored out so
    combined_smooth() can call it as a distinct stage before its
    lowpass filter.

    Args:
        values: Input readings.
        window: Samples considered on each side of the point being
            checked.
        n_sigmas: Outlier threshold in MAD-scaled "sigmas".

    Returns:
        list[float]: Same length as `values`, with detected outliers
        replaced via linear interpolation between nearest good
        neighbors.
    """
    n = len(values)
    if n == 0:
        return values
    bad = [False] * n
    k = 1.4826
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        w = [values[j] for j in range(lo, hi) if not bad[j]]
        if len(w) < 2:
            continue
        med = statistics.median(w)
        mad = statistics.median([abs(v - med) for v in w])
        threshold = n_sigmas * k * mad
        if threshold > 0 and abs(values[i] - med) > threshold:
            bad[i] = True

    result = list(values)
    bad_idx = [i for i, b in enumerate(bad) if b]
    for i in bad_idx:
        lo = i - 1
        while lo >= 0 and bad[lo]:
            lo -= 1
        hi = i + 1
        while hi < n and bad[hi]:
            hi += 1
        if lo >= 0 and hi < n:
            frac = (i - lo) / (hi - lo)
            result[i] = values[lo] + frac * (values[hi] - values[lo])
        elif lo >= 0:
            result[i] = values[lo]
        elif hi < n:
            result[i] = values[hi]

    return result


def lowpass_filter_enf(
    values: list[float],
    sample_rate_hz: float = 0.5,
    cutoff_hz: float = 0.02,
    order: int = 10,
    nominal: float = 60.0,
) -> list[float]:
    """Applies a zero-phase Butterworth lowpass filter to the ENF deviation series.

    Every output point is a weighted combination of real input
    points -- unlike clean_enf(), nothing is discarded or replaced
    with guesswork. Uses filtfilt (forward-backward) for zero phase
    shift, since a forward-only filter would introduce a time lag
    that would corrupt AnchorExtractor's window alignment.

    Args:
        values: ENF readings in Hz.
        sample_rate_hz: Sample rate of `values`.
        cutoff_hz: Filter cutoff, in Hz, applied to this time series
            (how fast the reading is allowed to change) -- not the
            same kind of quantity as clean_enf()'s physical bounds.
        order: Butterworth filter order.
        nominal: Nominal frequency the deviation is computed against.

    Returns:
        list[float]: Filtered ENF values, same length as `values`.
    """
    from scipy.signal import butter, filtfilt

    nyquist = sample_rate_hz / 2.0
    normalized_cutoff = min(cutoff_hz / nyquist, 0.99)

    deviation = [v - nominal for v in values]
    b, a = butter(order, normalized_cutoff, btype="low")
    smoothed_deviation = filtfilt(b, a, deviation)

    return [d + nominal for d in smoothed_deviation]


def combined_smooth(
    values: list[float],
    hampel_window: int = 11,
    hampel_n_sigmas: float = 2.0,
    lowpass_cutoff_hz: float = 0.02,
    lowpass_order: int = 10,
    sample_rate_hz: float = 0.5,
) -> list[float]:
    """Cleans ENF data via outlier correction followed by lowpass smoothing.

    The recommended, validated ENF cleaning step for production use --
    call this in place of clean_enf(). Requires scipy. Must be called
    exactly once, at ingestion, before the data goes anywhere near
    attack injection. See .readme/data_loaders.md for validation
    results.

    Args:
        values: Raw ENF readings from load_enf().
        hampel_window: Passed to hampel_correct().
        hampel_n_sigmas: Passed to hampel_correct().
        lowpass_cutoff_hz: Passed to lowpass_filter_enf().
        lowpass_order: Passed to lowpass_filter_enf().
        sample_rate_hz: Passed to lowpass_filter_enf().

    Returns:
        list[float]: Cleaned and smoothed ENF values.
    """
    corrected = hampel_correct(values, window=hampel_window, n_sigmas=hampel_n_sigmas)
    return lowpass_filter_enf(
        corrected,
        sample_rate_hz=sample_rate_hz,
        cutoff_hz=lowpass_cutoff_hz,
        order=lowpass_order,
    )
_NLR_MAX_WINDOWS = 1800
_NLR_DATETIME_FORMAT = "%Y-%m-%d_%H:%M:%S.%f"

_NODE_ID_PATTERN = re.compile(r'(x\d+c\d+s\d+b\d+n\d+)', re.IGNORECASE)
_SLURM_ID_PATTERN = re.compile(r'slurmid_(\d+)', re.IGNORECASE)

_GPU_LIMIT_W = 800.0
_CPU_LIMIT_W = 800.0


@dataclass
class AggregatedGPUWindow:
    """One ENF-aligned window of GPU telemetry, averaged across N raw samples.

    N = nlr_sample_rate_hz * enf_window_seconds. Column names match
    the original NVML log header. `index` aligns with the ENF list
    index for the same time window.

    Attributes:
        index: Window index.
        gpu0_w, gpu1_w, gpu2_w, gpu3_w: Averaged wattage per GPU,
            converted from mW.
        gpu0_c, gpu1_c, gpu2_c, gpu3_c: Averaged temperature per GPU.
    """
    index:  int
    gpu0_w: float
    gpu1_w: float
    gpu2_w: float
    gpu3_w: float
    gpu0_c: float
    gpu1_c: float
    gpu2_c: float
    gpu3_c: float

    def to_dict(self, prefix: str = "") -> dict:
        """Returns this window's fields as a flat dict of column names to values.

        Args:
            prefix: Optional prefix (e.g. a node ID) for every key.

        Returns:
            dict: e.g. {"gpu-0[W]": ..., "gpu-0[C]": ..., ...}.
        """
        p = prefix
        return {
            f"{p}gpu-0[W]": round(self.gpu0_w, 4),
            f"{p}gpu-1[W]": round(self.gpu1_w, 4),
            f"{p}gpu-2[W]": round(self.gpu2_w, 4),
            f"{p}gpu-3[W]": round(self.gpu3_w, 4),
            f"{p}gpu-0[C]": round(self.gpu0_c, 4),
            f"{p}gpu-1[C]": round(self.gpu1_c, 4),
            f"{p}gpu-2[C]": round(self.gpu2_c, 4),
            f"{p}gpu-3[C]": round(self.gpu3_c, 4),
        }


@dataclass
class AggregatedCPUWindow:
    """One ENF-aligned window of CPU telemetry, averaged across N raw samples.

    Column names match the original RAPL log header. `index` aligns
    with the ENF list index for the same time window.

    Attributes:
        index: Window index.
        cpu0_uj, cpu1_uj: Averaged socket-level energy, in uJ.
        cpu0_core_uj, cpu1_core_uj: Averaged core-domain energy, in uJ.
        cpu0_w, cpu1_w: Averaged socket-level wattage.
        cpu0_core_w, cpu1_core_w: Averaged core-domain wattage.
    """
    index:        int
    cpu0_uj:      float
    cpu0_core_uj: float
    cpu1_uj:      float
    cpu1_core_uj: float
    cpu0_w:       float
    cpu0_core_w:  float
    cpu1_w:       float
    cpu1_core_w:  float

    def to_dict(self, prefix: str = "") -> dict:
        """Returns this window's fields as a flat dict of column names to values.

        Args:
            prefix: Optional prefix (e.g. a node ID) for every key.

        Returns:
            dict: e.g. {"cpu-0[uJ]": ..., "cpu-0[W]": ..., ...}.
        """
        p = prefix
        return {
            f"{p}cpu-0[uJ]":      round(self.cpu0_uj, 4),
            f"{p}cpu-0-core[uJ]": round(self.cpu0_core_uj, 4),
            f"{p}cpu-1[uJ]":      round(self.cpu1_uj, 4),
            f"{p}cpu-1-core[uJ]": round(self.cpu1_core_uj, 4),
            f"{p}cpu-0[W]":       round(self.cpu0_w, 4),
            f"{p}cpu-0-core[W]":  round(self.cpu0_core_w, 4),
            f"{p}cpu-1[W]":       round(self.cpu1_w, 4),
            f"{p}cpu-1-core[W]":  round(self.cpu1_core_w, 4),
        }


@dataclass
class _RawGPURow:
    """One unaggregated raw GPU telemetry sample."""
    gpu0_mw: float
    gpu1_mw: float
    gpu2_mw: float
    gpu3_mw: float
    gpu0_c:  float
    gpu1_c:  float
    gpu2_c:  float
    gpu3_c:  float


@dataclass
class _RawCPURow:
    """One unaggregated raw CPU telemetry sample."""
    cpu0_uj:      float
    cpu0_core_uj: float
    cpu1_uj:      float
    cpu1_core_uj: float
    cpu0_w:       float
    cpu0_core_w:  float
    cpu1_w:       float
    cpu1_core_w:  float


def _nlr_parse_header(line: str) -> list[str]:
    """Parses one NLR log header line into a list of column names.

    Args:
        line: Raw header line, prefixed with "#".

    Returns:
        list[str]: Column names.
    """
    return line.lstrip("#").split()


def _nlr_mean(values: list[float]) -> float:
    """Returns the mean of a list of values, or 0.0 if empty.

    Args:
        values: Values to average.

    Returns:
        float: The mean, or 0.0 for an empty list.
    """
    return sum(values) / len(values) if values else 0.0


_RAPL_WRAP_CEILING_UJ = 65_500_000_000
_RAPL_WRAP_TOLERANCE_UJ = 2_000_000_000


def _unwrap_uj_series(
    values: list[float],
    wrap_ceiling: float = _RAPL_WRAP_CEILING_UJ,
    wrap_tolerance: float = _RAPL_WRAP_TOLERANCE_UJ,
) -> list[float]:
    """Converts a raw, periodically-wrapping RAPL energy counter sequence into a continuous one.

    Detects a wrap whenever a reading drops by roughly `wrap_ceiling`
    from the previous raw reading, and from that point on adds
    `(wrap_count * wrap_ceiling)` to every subsequent raw value.
    Handles any number of wraps across the sequence, in order. Must be
    called on each channel's full raw sample sequence before
    aggregating into ENF-aligned windows -- see .readme/data_loaders.md
    for why.

    Args:
        values: Raw, possibly-wrapping energy readings, in uJ.
        wrap_ceiling: The counter's wraparound ceiling.
        wrap_tolerance: Tolerance for detecting a wrap event.

    Returns:
        list[float]: Same length as `values`, monotonically
        non-decreasing.
    """
    if not values:
        return values

    unwrapped = [values[0]]
    wrap_count = 0
    for i in range(1, len(values)):
        prev_raw = values[i - 1]
        curr_raw = values[i]
        if curr_raw < prev_raw - wrap_tolerance:
            wrap_count += 1
        unwrapped.append(curr_raw + wrap_count * wrap_ceiling)

    return unwrapped


def _extract_node_id(filename: str) -> Optional[str]:
    """Extracts the node identifier from a wattmeter log filename.

    Matches the node ID's actual shape (xNcNsNbNnN) directly, rather
    than relying on what text happens to precede it -- robust across
    every real filename convention this project has seen.

    Args:
        filename: Log file path or name.

    Returns:
        Optional[str]: The node ID, or None if no match is found.
    """
    name = Path(filename).name
    m = _NODE_ID_PATTERN.search(name)
    return m.group(1) if m else None


def _detect_sample_rate_hz(path: str, n_samples: int = 30) -> float:
    """Auto-detects an NLR log file's sample rate from its first timestamps.

    Args:
        path: Path to the log file.
        n_samples: Number of timestamps to read before stopping.

    Returns:
        float: Detected sample rate in Hz, rounded to the nearest integer.

    Raises:
        RuntimeError: If fewer than 2 readable timestamps are found.
    """
    timestamps = []
    with open(path) as fh:
        for line in fh:
            if len(timestamps) >= n_samples:
                break
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            try:
                ts = datetime.strptime(parts[0], _NLR_DATETIME_FORMAT)
                timestamps.append(ts)
            except (ValueError, IndexError):
                continue

    if len(timestamps) < 2:
        raise RuntimeError(
            f"[detect_sample_rate] could not detect rate from {path} -- "
            f"fewer than 2 readable timestamps found"
        )

    intervals = [
        (timestamps[i + 1] - timestamps[i]).total_seconds()
        for i in range(len(timestamps) - 1)
    ]
    mean_interval = sum(intervals) / len(intervals)
    detected_hz = round(1.0 / mean_interval)

    print(f"[detect_sample_rate] {Path(path).name}: "
          f"mean interval={mean_interval * 1000:.1f}ms "
          f"-> {detected_hz} Hz")

    return float(detected_hz)


def _pad_windows(
    windows: list,
    target_length: int,
    label: str = "",
) -> list:
    """Pads a window list to a target length by repeating the last window.

    Used when an NLR recording is shorter than the ENF file it's
    aligned to -- fills the gap by holding the last real reading
    rather than crashing or producing empty records.

    Args:
        windows: Windows to pad.
        target_length: Desired final length.
        label: Used only in the printed status line.

    Returns:
        list: `windows`, padded to `target_length`. Returns `windows`
        unchanged if it's already empty or long enough.
    """
    if not windows or len(windows) >= target_length:
        return windows

    n_real = len(windows)
    n_pad = target_length - n_real
    last = windows[-1]

    print(f"[pad_windows] {label}: {n_real} real windows, "
          f"padding {n_pad} with last reading to reach {target_length}")

    padded = list(windows)
    for extra_idx in range(n_real, target_length):
        import copy
        w = copy.copy(last)
        w.index = extra_idx
        padded.append(w)

    return padded


def discover_nlr_pairs(
    folder: str,
    slurm_id: Optional[str] = None,
) -> list[tuple[str, str, str]]:
    """Scans a folder for NVML/RAPL log file pairs, matched by node identifier.

    Args:
        folder: Path to the flat folder containing all .log files.
        slurm_id: If given, only files whose name contains
            "slurmid_{slurm_id}" are considered -- use this when a
            folder contains multiple hours of data.

    Returns:
        list[tuple[str, str, str]]: (node_id, nvml_path, rapl_path)
        tuples, sorted by node_id.

    Raises:
        FileNotFoundError: If `folder` doesn't exist.
        ValueError: If any node has an NVML file but no matching RAPL
            file (or vice versa), if duplicate files are found for
            the same node without a `slurm_id` filter, or if no pairs
            are found at all.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"NLR folder not found: {folder}")

    nvml_files: dict[str, Path] = {}
    rapl_files: dict[str, Path] = {}

    for f in folder_path.iterdir():
        if f.suffix.lower() != ".log":
            continue

        if slurm_id is not None:
            m = _SLURM_ID_PATTERN.search(f.name)
            if not m or m.group(1) != str(slurm_id):
                continue

        node_id = _extract_node_id(f.name)
        if node_id is None:
            continue

        name_lower = f.name.lower()
        if "nvml" in name_lower:
            if node_id in nvml_files:
                raise ValueError(
                    f"Duplicate NVML file for node {node_id}:\n"
                    f"  existing: {nvml_files[node_id].name}\n"
                    f"  new:      {f.name}\n"
                    f"Hint: use the slurm_id parameter to filter by "
                    f"a specific hour."
                )
            nvml_files[node_id] = f
        elif "rapl" in name_lower:
            if node_id in rapl_files:
                raise ValueError(
                    f"Duplicate RAPL file for node {node_id}:\n"
                    f"  existing: {rapl_files[node_id].name}\n"
                    f"  new:      {f.name}\n"
                    f"Hint: use the slurm_id parameter to filter by "
                    f"a specific hour."
                )
            rapl_files[node_id] = f

    nvml_only = set(nvml_files) - set(rapl_files)
    rapl_only = set(rapl_files) - set(nvml_files)
    if nvml_only:
        raise ValueError(
            f"NVML files with no matching RAPL for nodes: {sorted(nvml_only)}"
        )
    if rapl_only:
        raise ValueError(
            f"RAPL files with no matching NVML for nodes: {sorted(rapl_only)}"
        )

    paired = [
        (node_id, str(nvml_files[node_id]), str(rapl_files[node_id]))
        for node_id in sorted(nvml_files.keys())
    ]

    if not paired:
        hint = f" with slurm_id='{slurm_id}'" if slurm_id else ""
        raise ValueError(
            f"No NVML/RAPL pairs found in {folder}{hint}. "
            f"Filenames must contain 'nvml' or 'rapl' and end with "
            f"'node_{{node_id}}.log' or 'wattameter_{{node_id}}.log'"
        )

    print(f"[discover_nlr_pairs] found {len(paired)} node pair(s)"
          + (f" for slurm_id={slurm_id}" if slurm_id else "")
          + f": {[p[0] for p in paired]}")

    return paired


def _parse_nvml_log(
    path: str,
    max_rows: Optional[int] = None,
) -> list[_RawGPURow]:
    """Parses one NVML log file into raw GPU rows.

    Args:
        path: Path to the NVML log file.
        max_rows: If given, stop after this many data rows.

    Returns:
        list[_RawGPURow]: One entry per valid data row. Rows with
        wattage over _GPU_LIMIT_W are dropped as hardware errors.

    Raises:
        RuntimeError: If a data row appears before the header line.
    """
    rows: list[_RawGPURow] = []
    header: list[str] = []
    with open(path) as fh:
        for line in fh:
            if max_rows is not None and len(rows) >= max_rows:
                break
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                if "reading-time" in line:
                    header = _nlr_parse_header(line)
                continue
            if not header:
                raise RuntimeError(f"data row before header in {path}")
            parts = line.split()
            if len(parts) < len(header):
                continue
            col = {name: parts[i] for i, name in enumerate(header)}
            try:
                gpu0_mw = float(col["gpu-0[mW]"])
                gpu1_mw = float(col["gpu-1[mW]"])
                gpu2_mw = float(col["gpu-2[mW]"])
                gpu3_mw = float(col["gpu-3[mW]"])
                gpu0_c  = float(col["gpu-0[C]"])
                gpu1_c  = float(col["gpu-1[C]"])
                gpu2_c  = float(col["gpu-2[C]"])
                gpu3_c  = float(col["gpu-3[C]"])
            except (KeyError, ValueError):
                continue
            if any(v * 1e-3 > _GPU_LIMIT_W
                   for v in [gpu0_mw, gpu1_mw, gpu2_mw, gpu3_mw]):
                continue
            rows.append(_RawGPURow(
                gpu0_mw=gpu0_mw, gpu1_mw=gpu1_mw,
                gpu2_mw=gpu2_mw, gpu3_mw=gpu3_mw,
                gpu0_c=gpu0_c,   gpu1_c=gpu1_c,
                gpu2_c=gpu2_c,   gpu3_c=gpu3_c,
            ))
    return rows


def _parse_rapl_log(
    path: str,
    max_rows: Optional[int] = None,
) -> list[_RawCPURow]:
    """Parses one RAPL log file into raw CPU rows, with energy counters unwrapped.

    Args:
        path: Path to the RAPL log file.
        max_rows: If given, stop after this many data rows.

    Returns:
        list[_RawCPURow]: One entry per valid data row, with each
        energy channel's full sequence already unwrapped (see
        _unwrap_uj_series()). Rows with wattage over _CPU_LIMIT_W are
        dropped as hardware errors.

    Raises:
        RuntimeError: If a data row appears before the header line.
    """
    rows: list[_RawCPURow] = []
    header: list[str] = []
    with open(path) as fh:
        for line in fh:
            if max_rows is not None and len(rows) >= max_rows:
                break
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                if "reading-time" in line:
                    header = _nlr_parse_header(line)
                continue
            if not header:
                raise RuntimeError(f"data row before header in {path}")
            parts = line.split()
            if len(parts) < len(header):
                continue
            col = {name: parts[i] for i, name in enumerate(header)}
            try:
                cpu0_uj      = float(col["cpu-0[uJ]"])
                cpu0_core_uj = float(col["cpu-0-core[uJ]"])
                cpu1_uj      = float(col["cpu-1[uJ]"])
                cpu1_core_uj = float(col["cpu-1-core[uJ]"])
                cpu0_w       = float(col["cpu-0[W]"])
                cpu0_core_w  = float(col["cpu-0-core[W]"])
                cpu1_w       = float(col["cpu-1[W]"])
                cpu1_core_w  = float(col["cpu-1-core[W]"])
            except (KeyError, ValueError):
                continue
            if any(v > _CPU_LIMIT_W for v in [cpu0_w, cpu1_w]):
                continue
            rows.append(_RawCPURow(
                cpu0_uj=cpu0_uj,           cpu0_core_uj=cpu0_core_uj,
                cpu1_uj=cpu1_uj,           cpu1_core_uj=cpu1_core_uj,
                cpu0_w=cpu0_w,             cpu0_core_w=cpu0_core_w,
                cpu1_w=cpu1_w,             cpu1_core_w=cpu1_core_w,
            ))

    if rows:
        unwrapped_cpu0_uj      = _unwrap_uj_series([r.cpu0_uj      for r in rows])
        unwrapped_cpu0_core_uj = _unwrap_uj_series([r.cpu0_core_uj for r in rows])
        unwrapped_cpu1_uj      = _unwrap_uj_series([r.cpu1_uj      for r in rows])
        unwrapped_cpu1_core_uj = _unwrap_uj_series([r.cpu1_core_uj for r in rows])
        for i, r in enumerate(rows):
            r.cpu0_uj      = unwrapped_cpu0_uj[i]
            r.cpu0_core_uj = unwrapped_cpu0_core_uj[i]
            r.cpu1_uj      = unwrapped_cpu1_uj[i]
            r.cpu1_core_uj = unwrapped_cpu1_core_uj[i]

    return rows


def _aggregate_gpu(
    rows: list[_RawGPURow],
    samples_per_window: int,
) -> list[AggregatedGPUWindow]:
    """Aggregates raw GPU rows into ENF-aligned windows by averaging.

    Args:
        rows: Raw GPU rows, in time order.
        samples_per_window: Number of raw samples per output window.

    Returns:
        list[AggregatedGPUWindow]: One window per group of
        `samples_per_window` raw rows.
    """
    windows = []
    for start in range(0, len(rows), samples_per_window):
        group = rows[start: start + samples_per_window]
        if not group:
            continue
        windows.append(AggregatedGPUWindow(
            index  = start // samples_per_window,
            gpu0_w = _nlr_mean([r.gpu0_mw * 1e-3 for r in group]),
            gpu1_w = _nlr_mean([r.gpu1_mw * 1e-3 for r in group]),
            gpu2_w = _nlr_mean([r.gpu2_mw * 1e-3 for r in group]),
            gpu3_w = _nlr_mean([r.gpu3_mw * 1e-3 for r in group]),
            gpu0_c = _nlr_mean([r.gpu0_c for r in group]),
            gpu1_c = _nlr_mean([r.gpu1_c for r in group]),
            gpu2_c = _nlr_mean([r.gpu2_c for r in group]),
            gpu3_c = _nlr_mean([r.gpu3_c for r in group]),
        ))
    return windows


def _aggregate_cpu(
    rows: list[_RawCPURow],
    samples_per_window: int,
) -> list[AggregatedCPUWindow]:
    """Aggregates raw CPU rows into ENF-aligned windows by averaging.

    Args:
        rows: Raw CPU rows, in time order.
        samples_per_window: Number of raw samples per output window.

    Returns:
        list[AggregatedCPUWindow]: One window per group of
        `samples_per_window` raw rows.
    """
    windows = []
    for start in range(0, len(rows), samples_per_window):
        group = rows[start: start + samples_per_window]
        if not group:
            continue
        windows.append(AggregatedCPUWindow(
            index        = start // samples_per_window,
            cpu0_uj      = _nlr_mean([r.cpu0_uj      for r in group]),
            cpu0_core_uj = _nlr_mean([r.cpu0_core_uj for r in group]),
            cpu1_uj      = _nlr_mean([r.cpu1_uj      for r in group]),
            cpu1_core_uj = _nlr_mean([r.cpu1_core_uj for r in group]),
            cpu0_w       = _nlr_mean([r.cpu0_w       for r in group]),
            cpu0_core_w  = _nlr_mean([r.cpu0_core_w  for r in group]),
            cpu1_w       = _nlr_mean([r.cpu1_w       for r in group]),
            cpu1_core_w  = _nlr_mean([r.cpu1_core_w  for r in group]),
        ))
    return windows


def load_nlr_multi(
    pairs: list[tuple[str, str, str]],
    nlr_sample_rate_hz: Optional[float] = None,
    enf_sample_rate_hz: float = 0.5,
    max_windows: int = _NLR_MAX_WINDOWS,
) -> dict[str, tuple[list[AggregatedGPUWindow], list[AggregatedCPUWindow]]]:
    """Loads and aggregates all node pairs returned by discover_nlr_pairs().

    Auto-detects the NLR sample rate from the first file, so the
    correct number of raw samples are averaged per ENF-aligned window.

    Args:
        pairs: (node_id, nvml_path, rapl_path) tuples.
        nlr_sample_rate_hz: Override auto-detection if given.
        enf_sample_rate_hz: ENF sample rate in Hz; sets the window
            duration.
        max_windows: Max windows to load per node.

    Returns:
        dict[str, tuple[list[AggregatedGPUWindow], list[AggregatedCPUWindow]]]:
        One entry per node ID.

    Raises:
        ValueError: If `pairs` is empty.
    """
    if not pairs:
        raise ValueError("No node pairs provided")

    if nlr_sample_rate_hz is None:
        nlr_sample_rate_hz = _detect_sample_rate_hz(pairs[0][1])

    enf_window_seconds = 1.0 / enf_sample_rate_hz
    samples_per_window = round(nlr_sample_rate_hz * enf_window_seconds)
    max_raw = max_windows * samples_per_window

    print(f"[load_nlr_multi] NLR={nlr_sample_rate_hz:.0f} Hz, "
          f"ENF={enf_sample_rate_hz} Hz -> "
          f"{samples_per_window} samples per window")

    node_windows: dict[str, tuple] = {}
    for node_id, nvml_path, rapl_path in pairs:
        print(f"[nlr_ingest] loading node {node_id}")
        raw_gpu = _parse_nvml_log(nvml_path, max_rows=max_raw)
        raw_cpu = _parse_rapl_log(rapl_path, max_rows=max_raw)
        gpu_windows = _aggregate_gpu(raw_gpu, samples_per_window)[:max_windows]
        cpu_windows = _aggregate_cpu(raw_cpu, samples_per_window)[:max_windows]
        print(f"  -> {len(gpu_windows)} GPU windows, "
              f"{len(cpu_windows)} CPU windows")
        node_windows[node_id] = (gpu_windows, cpu_windows)

    print(f"[nlr_ingest] loaded {len(node_windows)} node(s): "
          f"{list(node_windows.keys())}")
    return node_windows
def build_combined_records(
    enf: list[float],
    node_windows: dict[str, tuple[
        list[AggregatedGPUWindow], list[AggregatedCPUWindow]
    ]],
    pad_short_nlr: bool = True,
) -> list[dict]:
    """Merges ENF and all node windows into one flat dict per index.

    Never crashes on a length mismatch between ENF and NLR data --
    always produces a complete list of records.

    Args:
        enf: ENF frequency readings from load_enf().
        node_windows: As returned by load_nlr_multi().
        pad_short_nlr: If True (default), a node whose recording is
            shorter than the ENF file has its last real reading
            repeated to fill the gap. If False, the ENF list is
            trimmed to the shortest node's window count instead.

    Returns:
        list[dict]: One combined record per index, each containing
        "index", "FRQ", and every node's GPU/CPU columns, prefixed by
        node ID.
    """
    n_enf = len(enf)

    min_nlr = min(
        min(len(gpu), len(cpu))
        for gpu, cpu in node_windows.values()
    )

    if pad_short_nlr:
        padded_windows: dict[str, tuple] = {}
        for node_id, (gpu_windows, cpu_windows) in node_windows.items():
            gpu_padded = _pad_windows(gpu_windows, n_enf,
                                      label=f"{node_id} GPU")
            cpu_padded = _pad_windows(cpu_windows, n_enf,
                                      label=f"{node_id} CPU")
            padded_windows[node_id] = (gpu_padded, cpu_padded)
        target_length = n_enf
        node_windows = padded_windows
    else:
        if min_nlr < n_enf:
            print(f"[build_combined_records] trimming ENF from "
                  f"{n_enf} to {min_nlr} samples to match NLR length")
        enf = enf[:min_nlr]
        target_length = min_nlr

    records: list[dict] = []
    sorted_nodes = sorted(node_windows.keys())

    for i in range(target_length):
        record: dict = {"index": i, "FRQ": enf[i]}
        for node_id in sorted_nodes:
            gpu_windows, cpu_windows = node_windows[node_id]
            prefix = f"{node_id}_"
            record.update(gpu_windows[i].to_dict(prefix=prefix))
            record.update(cpu_windows[i].to_dict(prefix=prefix))
        records.append(record)

    return records


def write_combined_jsonl(records: list[dict], path: str) -> None:
    """Writes combined records to a JSON Lines file.

    Args:
        records: Records as produced by build_combined_records().
        path: Output file path. Parent directories are created if
            needed.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record))
            fh.write("\n")


def read_combined_jsonl(path: str):
    """Yields one combined record dict at a time from a JSONL file.

    Args:
        path: Path to a combined JSONL file.

    Yields:
        dict: One record per line.
    """
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


_ENVELOPE = {
    WorkloadClass.IDLE: (90, 130),
    WorkloadClass.LLM_INFERENCE: (250, 600),
    WorkloadClass.LLM_TRAINING: (650, 780),
    WorkloadClass.IMAGE_GENERATION: (300, 550),
}


def synthetic_power_profile(
    workload: WorkloadClass,
    node_id: str = "node_00",
    seconds: int = 120,
    hz: int = 5,
    seed: Optional[int] = None,
) -> list[PowerSample]:
    """Generates a synthetic power trace whose shape matches a workload class.

    For development use only -- not a research artifact.

    Args:
        workload: Workload class to shape the trace around.
        node_id: Node identifier to stamp on every sample.
        seconds: Duration of the generated trace.
        hz: Sample rate.
        seed: Optional random seed for reproducibility.

    Returns:
        list[PowerSample]: One sample per timestep.
    """
    rng = random.Random(seed)
    lo, hi = _ENVELOPE[workload]
    n = seconds * hz
    out: list[PowerSample] = []
    for i in range(n):
        t = i / hz
        if workload is WorkloadClass.LLM_TRAINING:
            base = hi - 30 + 30 * math.sin(t / 8)
        elif workload is WorkloadClass.LLM_INFERENCE:
            burst = hi if rng.random() < 0.18 else lo
            base = burst
        elif workload is WorkloadClass.IMAGE_GENERATION:
            base = lo + (hi - lo) * (0.5 + 0.5 * math.sin(t / 3))
        else:
            base = lo + 10 * rng.random()
        noise = rng.gauss(0, 8)
        out.append(
            PowerSample(
                timestamp=t,
                node_id=node_id,
                power_w=max(0.0, base + noise),
                workload_class=workload.value,
            )
        )
    return out


def synthetic_enf(
    seconds: int = 120,
    hz: int = 5,
    nominal: float = 60.0,
    seed: Optional[int] = None,
) -> list[float]:
    """Generates a synthetic ENF trace with a small wandering fluctuation around nominal.

    For development use only -- not a research artifact. The wander
    is the property real verification checks depend on; a replayed or
    fabricated trace will not share it.

    Args:
        seconds: Duration of the generated trace.
        hz: Sample rate.
        nominal: Nominal frequency to wander around, in Hz.
        seed: Optional random seed for reproducibility.

    Returns:
        list[float]: Generated frequency readings, in Hz.
    """
    rng = random.Random(seed)
    out: list[float] = []
    f = nominal
    for _ in range(seconds * hz):
        f += rng.gauss(0, 0.003)
        f += (nominal - f) * 0.02
        out.append(f)
    return out