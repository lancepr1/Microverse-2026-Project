"""
data_loaders.py: get power and ENF data into the contract types.

Two real datasets feed this project:
  - NLR GenAI Workload Power Profiles (data.nlr.gov/submissions/312, ~1 GB).
    5 to 10 Hz, per-node and whole-facility, real LLM and image-gen workloads.
  - The 2025 ENF measurements collected at AFRL Rome (from the lab / Dr. Qu).

Neither ships in the repo (see data/README.md). The real loaders below are
deliberately thin and marked TODO, because the exact column names live in each
dataset's own README and should be confirmed in week one rather than guessed.

Until the team has the real files, the synthetic_* generators produce
plausible-shaped traces so every lane can develop and run the smoke test on
day one. They are NOT a research artifact: swap in real data before any
result goes in the paper.

Pipeline (Leiva's ingestion path):
    load_enf()               ENF CSV -> list[float], 1800 readings at 0.5 Hz
    discover_nlr_pairs()     scans a flat folder, pairs NVML+RAPL by node name
                             works for any number of nodes (1 to 16+)
    load_nlr_multi()         loads all pairs, auto-detects sample rate,
                             prefixes columns with node name
    build_combined_records() merges ENF + all node windows into one record
                             per index. If NLR is shorter than ENF, pads
                             the last real NLR reading to fill the gap.
    write_combined_jsonl()   writes merged records to JSONL for Ethan
    read_combined_jsonl()    generator that yields one record at a time

Combined record header order (node-grouped, Option A):
    index, FRQ,
    {node_id}_gpu-0[W] .. {node_id}_gpu-3[W],
    {node_id}_gpu-0[C] .. {node_id}_gpu-3[C],
    {node_id}_cpu-0[uJ] .. {node_id}_cpu-1-core[uJ],
    {node_id}_cpu-0[W] .. {node_id}_cpu-1-core[W],
    (repeated for each node in sorted order)
"""
from __future__ import annotations

import csv
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .contracts import PowerSample, WorkloadClass


# --------------------------------------------------------------------------
# Real loaders. Fill these in against the dataset READMEs in week one.
# --------------------------------------------------------------------------

def load_nlr_profile(path: str) -> list[PowerSample]:
    """Load one NLR power profile into PowerSample records.

    TODO (week 1, Hendricks + Lance): confirm the real schema against the
    README at the top of the NLR zip and map its columns onto the four fields
    below. The current implementation assumes a CSV with a header containing
    timestamp, node_id, power_w, workload. Adjust the key names to match.
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
    """Load the 2025 AFRL ENF dataset into a flat list of Hz readings.

    Real file format:
      Row 0 col 0: "UTC: 2025-07-17 08:00:00"  (metadata, skipped)
      Row 0 col 1: "Duration: 1 Hour"           (metadata, skipped)
      Row 1+  col 0: integer index 0..1800
      Row 1+  col 1: frequency in Hz e.g. "59.2938797944005"

    Sample rate: 1800 samples / 3600 seconds = 0.5 Hz
    Returns a list of 1800 floats ordered by ascending index.
    ENF is grid-wide -- one file covers all nodes simultaneously.
    """
    values: list[float] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # skip metadata row (UTC timestamp + duration)
        for row in reader:
            if not row:
                continue
            try:
                values.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    return values


# --------------------------------------------------------------------------
# NLR wattmeter log loaders -- multi-node aware, variable sample rate
#
# File naming convention:
#   nvml_wattameter_emissions_parsed_slurmid_{id}_node_{node_id}.log
#   rapl_wattameter_emissions_parsed_slurmid_{id}_node_{node_id}.log
#
# Sample rates vary by workload:
#   training workloads  ->  5 Hz  (10 samples per 2-second ENF window)
#   inference workloads -> 10 Hz  (20 samples per 2-second ENF window)
#
# load_nlr_multi() auto-detects the sample rate from the first file
# and computes the correct samples_per_window automatically.
# --------------------------------------------------------------------------

_NLR_MAX_WINDOWS = 1800
_NLR_DATETIME_FORMAT = "%Y-%m-%d_%H:%M:%S.%f"

# Regex patterns for node ID and SLURM ID extraction
_NODE_ID_PATTERN_NEW = re.compile(r'node_([^.]+)\.log$', re.IGNORECASE)
_NODE_ID_PATTERN_OLD = re.compile(r'wattameter_([^.]+)\.log$', re.IGNORECASE)
_SLURM_ID_PATTERN   = re.compile(r'slurmid_(\d+)', re.IGNORECASE)

# Device power limits -- readings above these are hardware errors
_GPU_LIMIT_W = 800.0
_CPU_LIMIT_W = 800.0


@dataclass
class AggregatedGPUWindow:
    """
    One ENF-aligned window of GPU telemetry averaged across N raw samples.
    N = nlr_sample_rate_hz * enf_window_seconds (e.g. 10 at 5Hz, 20 at 10Hz).
    Column names match the original NVML log header exactly.
    index aligns with the ENF list index for the same time window.
    """
    index:  int
    gpu0_w: float   # gpu-0[W]  averaged, converted from mW
    gpu1_w: float   # gpu-1[W]
    gpu2_w: float   # gpu-2[W]
    gpu3_w: float   # gpu-3[W]
    gpu0_c: float   # gpu-0[C]  averaged temperature
    gpu1_c: float   # gpu-1[C]
    gpu2_c: float   # gpu-2[C]
    gpu3_c: float   # gpu-3[C]

    def to_dict(self, prefix: str = "") -> dict:
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
    """
    One ENF-aligned window of CPU telemetry averaged across N raw samples.
    Column names match the original RAPL log header exactly.
    index aligns with the ENF list index for the same time window.
    """
    index:        int
    cpu0_uj:      float   # cpu-0[uJ]
    cpu0_core_uj: float   # cpu-0-core[uJ]
    cpu1_uj:      float   # cpu-1[uJ]
    cpu1_core_uj: float   # cpu-1-core[uJ]
    cpu0_w:       float   # cpu-0[W]
    cpu0_core_w:  float   # cpu-0-core[W]
    cpu1_w:       float   # cpu-1[W]
    cpu1_core_w:  float   # cpu-1-core[W]

    def to_dict(self, prefix: str = "") -> dict:
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


# --- internal raw row containers ------------------------------------------

@dataclass
class _RawGPURow:
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
    cpu0_uj:      float
    cpu0_core_uj: float
    cpu1_uj:      float
    cpu1_core_uj: float
    cpu0_w:       float
    cpu0_core_w:  float
    cpu1_w:       float
    cpu1_core_w:  float


def _nlr_parse_header(line: str) -> list[str]:
    return line.lstrip("#").split()


def _nlr_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# RAPL energy counters are fixed-width hardware registers: true_total is
# stored as (true_total % _RAPL_WRAP_CEILING_UJ), so the raw readings
# periodically drop back near zero even though real energy use only ever
# increases. A plain modulo can't invert this -- the wrap count itself is
# lost -- so instead we watch for the characteristic near-ceiling drop and
# add the ceiling back on every time it happens, exactly the "phase
# unwrap" trick used for cyclic/angular data. Must run BEFORE aggregation:
# aggregating (averaging) raw wrapped values blends pre-wrap and post-wrap
# numbers together whenever a wrap lands mid-window, producing a
# meaningless window value and a bogus window-to-window "drop" that
# doesn't match the true wrap size.
_RAPL_WRAP_CEILING_UJ = 65_500_000_000
_RAPL_WRAP_TOLERANCE_UJ = 2_000_000_000


def _unwrap_uj_series(
    values: list[float],
    wrap_ceiling: float = _RAPL_WRAP_CEILING_UJ,
    wrap_tolerance: float = _RAPL_WRAP_TOLERANCE_UJ,
) -> list[float]:
    """
    Converts a raw RAPL energy counter sequence (periodically wraps back
    near zero) into a continuous, always-increasing sequence (true
    cumulative energy). Call this on each channel's full raw sample
    sequence before aggregating into ENF-aligned windows.

    Detects a wrap whenever a reading drops by roughly wrap_ceiling from
    the previous raw reading, and from that point on adds
    (wrap_count * wrap_ceiling) to every subsequent raw value. Handles
    any number of wraps across the file, in order.
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
    """
    Extracts the node identifier from a wattmeter log filename.

    Handles two naming conventions:
      New: nvml_wattameter_emissions_parsed_slurmid_10742842_node_x3105c0s41b0n0.log
             -> "x3105c0s41b0n0"
      Old: nvml_wattameter_x3115c0s33b0n0.log
             -> "x3115c0s33b0n0"
    """
    name = Path(filename).name
    m = _NODE_ID_PATTERN_NEW.search(name)
    if m:
        return m.group(1)
    m = _NODE_ID_PATTERN_OLD.search(name)
    if m:
        return m.group(1)
    return None


def _detect_sample_rate_hz(path: str, n_samples: int = 30) -> float:
    """
    Auto-detect NLR sample rate from the first n_samples timestamps.

    Reads only the first n_samples data rows so it stays fast on large
    files. Returns rate rounded to the nearest integer Hz.

    Training workloads  ->  5 Hz
    Inference workloads -> 10 Hz
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
    """
    Pads a window list to target_length by repeating the last window.

    If the NLR recording is shorter than the ENF file (e.g. a training
    run that lasted 17 minutes against a 1-hour ENF file), this fills
    the gap by holding the last real reading rather than crashing or
    producing empty records. The padded windows are flagged with a
    different index so downstream consumers can identify them if needed.

    If windows is empty, returns an empty list (cannot pad nothing).
    """
    if not windows or len(windows) >= target_length:
        return windows

    n_real = len(windows)
    n_pad  = target_length - n_real
    last   = windows[-1]

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
    """
    Scans a flat folder for NVML and RAPL log file pairs, matching them
    by the shared node identifier in their filenames.

    Parameters
    ----------
    folder : str
        Path to the flat folder containing all .log files.
    slurm_id : str, optional
        If provided, only files whose filename contains
        "slurmid_{slurm_id}" are considered. Use this when a folder
        contains multiple hours of data (each hour has a different
        SLURM job ID) and you want to ingest one specific hour.

        Example -- folder contains 2 hours for 16 nodes (64 files):
            pairs_h1 = discover_nlr_pairs("data/raw/", slurm_id="10742842")
            pairs_h2 = discover_nlr_pairs("data/raw/", slurm_id="10742844")

    Returns
    -------
    list of (node_id, nvml_path, rapl_path) tuples sorted by node_id.

    Raises ValueError if any node has NVML but no RAPL, or vice versa,
    or if duplicate nodes are found without a slurm_id filter.
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

    # Unwrap each uJ channel's full raw sequence BEFORE any aggregation
    # happens. Must be done here, not in _aggregate_cpu, so averaging
    # never sees a wrapped value in the first place.
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
    """
    Load and aggregate all node pairs returned by discover_nlr_pairs().

    Auto-detects the NLR sample rate from the first file so the correct
    number of raw samples are averaged per ENF-aligned window:
        training workloads  ->  5 Hz ->  10 samples per window
        inference workloads -> 10 Hz -> 20 samples per window

    Parameters
    ----------
    pairs : list of (node_id, nvml_path, rapl_path) tuples
    nlr_sample_rate_hz : float, optional
        Override auto-detection. Pass 5.0 or 10.0 explicitly if needed.
    enf_sample_rate_hz : float
        ENF sample rate in Hz (default 0.5). Sets the window duration.
    max_windows : int
        Max windows per node (default 1800 = 1 hour at 0.5 Hz).
        Windows are padded to this length in build_combined_records if
        the recording is shorter.
    """
    if not pairs:
        raise ValueError("No node pairs provided")

    # auto-detect sample rate from the first NVML file
    if nlr_sample_rate_hz is None:
        nlr_sample_rate_hz = _detect_sample_rate_hz(pairs[0][1])

    enf_window_seconds = 1.0 / enf_sample_rate_hz  # 2.0s at 0.5 Hz
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


# --------------------------------------------------------------------------
# Combined record builder
# --------------------------------------------------------------------------

def build_combined_records(
    enf: list[float],
    node_windows: dict[str, tuple[
        list[AggregatedGPUWindow], list[AggregatedCPUWindow]
    ]],
    pad_short_nlr: bool = True,
) -> list[dict]:
    """
    Merge ENF and all node windows into one flat dict per index.

    If the NLR recording is shorter than the ENF file (e.g. a training
    run that lasted 17 minutes against a 1-hour ENF file), the behaviour
    depends on pad_short_nlr:

        True (default): pad each short node's windows by repeating the
            last real reading until it matches the ENF length. The twin
            keeps displaying the last known hardware state rather than
            going blank, and the simulation continues uninterrupted.

        False: trim ENF to the shortest node's window count instead.
            Use this when you only want real data, no padding.

    Never crashes on a length mismatch -- always produces a complete
    list of records the same length as the ENF file (or the shortest
    NLR window count when pad_short_nlr=False).

    Parameters
    ----------
    enf : list[float]
        ENF frequency readings from load_enf().
    node_windows : dict
        As returned by load_nlr_multi().
    pad_short_nlr : bool
        Whether to pad short NLR data to match ENF length (default True).
    """
    n_enf = len(enf)

    # find shortest window count across all nodes
    min_nlr = min(
        min(len(gpu), len(cpu))
        for gpu, cpu in node_windows.values()
    )

    if pad_short_nlr:
        # pad each node's windows up to ENF length
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
        # trim ENF to shortest NLR
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
    """Write combined records to a JSON Lines file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record))
            fh.write("\n")


def read_combined_jsonl(path: str):
    """Generator that yields one combined record dict at a time."""
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# --------------------------------------------------------------------------
# Synthetic fallbacks. Shapes only, not measurements.
# --------------------------------------------------------------------------

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
    """A power trace whose *shape* matches the named workload class."""
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
    """A 60 Hz ENF trace with the small wandering fluctuation that makes
    ENF usable as a timestamp. A replay or fabricated trace will not share
    this wander, which is the property Leiva's verification exploits."""
    rng = random.Random(seed)
    out: list[float] = []
    f = nominal
    for _ in range(seconds * hz):
        f += rng.gauss(0, 0.003)
        f += (nominal - f) * 0.02
        out.append(f)
    return out