"""
data_loaders.py: get power and ENF data into the contract types.

Two real datasets feed this project:
  - NLR GenAI Workload Power Profiles (data.nlr.gov/submissions/312, ~1 GB).
    5 to 10 Hz, per-node and whole-facility, real LLM and image-gen workloads.
  - The 2025 ENF measurements collected at AFRL Rome (from the lab / Dr. Qu).

Neither ships in this repo (see data/README.md). The real loaders below are
deliberately thin and marked TODO, because the exact column names live in each
dataset's own README and should be confirmed in week one rather than guessed.

Until the team has the real files, the synthetic_* generators produce
plausible-shaped traces so every lane can develop and run the smoke test on
day one. They are NOT a research artifact: swap in the real data before any
result goes in the paper.

Pipeline (Leiva's ingestion path):
    load_enf()          ENF CSV -> list[float], 1800 readings at 0.5 Hz
    load_nlr()           NVML + RAPL .log -> aggregated 0.5 Hz windows
    build_combined_records()  merges ENF + GPU + CPU into one record per index
    write_combined_jsonl()    writes the merged records to JSONL, one line
                              per index, for Ethan's attack module to consume
"""
from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
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
    """
    values: list[float] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # skip metadata row (UTC timestamp + duration)
        for row in reader:
            if not row:
                continue
            try:
                values.append(float(row[1]))  # col 0 is index, col 1 is frequency
            except (ValueError, IndexError):
                continue
    return values


# --------------------------------------------------------------------------
# NLR wattmeter log loaders (Leiva)
# Reads NVML (.log) and RAPL (.log) files, aggregates from 10 Hz to 0.5 Hz
# to match the ENF sample rate. Output window index aligns with ENF index.
# --------------------------------------------------------------------------

# 10 Hz NLR / 0.5 Hz ENF = 20 raw samples per aggregated window
_NLR_SAMPLES_PER_WINDOW = 20

# Cap at 1800 windows to match one hour of ENF data (indices 0-1799)
_NLR_MAX_WINDOWS = 1800


@dataclass
class AggregatedGPUWindow:
    """
    One 2-second window of GPU telemetry averaged across 20 raw samples.
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

    def to_dict(self) -> dict:
        return {
            "gpu-0[W]": round(self.gpu0_w, 4),
            "gpu-1[W]": round(self.gpu1_w, 4),
            "gpu-2[W]": round(self.gpu2_w, 4),
            "gpu-3[W]": round(self.gpu3_w, 4),
            "gpu-0[C]": round(self.gpu0_c, 4),
            "gpu-1[C]": round(self.gpu1_c, 4),
            "gpu-2[C]": round(self.gpu2_c, 4),
            "gpu-3[C]": round(self.gpu3_c, 4),
        }


@dataclass
class AggregatedCPUWindow:
    """
    One 2-second window of CPU telemetry averaged across 20 raw samples.
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

    def to_dict(self) -> dict:
        return {
            "cpu-0[uJ]":      round(self.cpu0_uj, 4),
            "cpu-0-core[uJ]": round(self.cpu0_core_uj, 4),
            "cpu-1[uJ]":      round(self.cpu1_uj, 4),
            "cpu-1-core[uJ]": round(self.cpu1_core_uj, 4),
            "cpu-0[W]":       round(self.cpu0_w, 4),
            "cpu-0-core[W]":  round(self.cpu0_core_w, 4),
            "cpu-1[W]":       round(self.cpu1_w, 4),
            "cpu-1-core[W]":  round(self.cpu1_core_w, 4),
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


def _parse_nvml_log(
    path: str,
    max_rows: Optional[int] = None,
) -> list[_RawGPURow]:
    """Parse NVML wattmeter .log into raw GPU rows, optionally capped."""
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
            if any(v * 1e-3 > 800 for v in [gpu0_mw, gpu1_mw, gpu2_mw, gpu3_mw]):
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
    """Parse RAPL wattmeter .log into raw CPU rows, optionally capped."""
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
            if any(v > 800 for v in [cpu0_w, cpu1_w]):
                continue
            rows.append(_RawCPURow(
                cpu0_uj=cpu0_uj,           cpu0_core_uj=cpu0_core_uj,
                cpu1_uj=cpu1_uj,           cpu1_core_uj=cpu1_core_uj,
                cpu0_w=cpu0_w,             cpu0_core_w=cpu0_core_w,
                cpu1_w=cpu1_w,             cpu1_core_w=cpu1_core_w,
            ))
    return rows


def _aggregate_gpu(
    rows: list[_RawGPURow],
    samples_per_window: int = _NLR_SAMPLES_PER_WINDOW,
) -> list[AggregatedGPUWindow]:
    """Average groups of raw GPU rows. Converts mW to W during averaging."""
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
    samples_per_window: int = _NLR_SAMPLES_PER_WINDOW,
) -> list[AggregatedCPUWindow]:
    """Average groups of raw CPU rows. Both uJ and W columns averaged."""
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


def load_nlr(
    nvml_path: str,
    rapl_path: str,
    samples_per_window: int = _NLR_SAMPLES_PER_WINDOW,
    max_windows: int = _NLR_MAX_WINDOWS,
) -> tuple[list[AggregatedGPUWindow], list[AggregatedCPUWindow]]:
    """
    Load and aggregate both NLR wattmeter logs into 1800 windows each,
    aligned to ENF indices 0-1799.

    Reads only the first 36,000 raw rows from each file
    (1800 windows x 20 samples) and discards the rest.
    """
    max_raw = max_windows * samples_per_window

    raw_gpu = _parse_nvml_log(nvml_path, max_rows=max_raw)
    raw_cpu = _parse_rapl_log(rapl_path, max_rows=max_raw)

    gpu_windows = _aggregate_gpu(raw_gpu, samples_per_window)[:max_windows]
    cpu_windows = _aggregate_cpu(raw_cpu, samples_per_window)[:max_windows]

    return gpu_windows, cpu_windows


# --------------------------------------------------------------------------
# Combined record builder + JSONL writer
# Merges ENF + GPU + CPU into one record per index:
#   index, FRQ, gpu-0[W] .. gpu-3[C], cpu-0[uJ] .. cpu-1-core[W]
# This is the file handed to Ethan, and the file he hands back for
# verification.
# --------------------------------------------------------------------------

def build_combined_records(
    enf: list[float],
    gpu_windows: list[AggregatedGPUWindow],
    cpu_windows: list[AggregatedCPUWindow],
) -> list[dict]:
    """
    Merge ENF, GPU, and CPU data into one flat dict per index.

    Header order: index, FRQ, then all GPU columns, then all CPU columns.

    Requires all three inputs to have the same length (1800 entries each
    when using the default max_windows). Raises if they do not match,
    since silently truncating to the shortest list would misalign the
    ENF anchor from its corresponding power readings.
    """
    n_enf, n_gpu, n_cpu = len(enf), len(gpu_windows), len(cpu_windows)
    if not (n_enf == n_gpu == n_cpu):
        raise ValueError(
            f"length mismatch -- enf={n_enf}, gpu={n_gpu}, cpu={n_cpu}. "
            f"All three must be the same length before merging."
        )

    records: list[dict] = []
    for i in range(n_enf):
        record = {"index": i, "FRQ": enf[i]}
        record.update(gpu_windows[i].to_dict())
        record.update(cpu_windows[i].to_dict())
        records.append(record)

    return records


def write_combined_jsonl(records: list[dict], path: str) -> None:
    """
    Write combined records to a JSON Lines file -- one JSON object
    per line, no enclosing array. This lets downstream readers
    (Ethan's attack module, the dashboard, Blender) stream the file
    one record at a time without loading it all into memory.

    Usage:
        enf = load_enf(enf_path)
        gpu_windows, cpu_windows = load_nlr(nvml_path, rapl_path)
        records = build_combined_records(enf, gpu_windows, cpu_windows)
        write_combined_jsonl(records, "data/combined/run01.jsonl")
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record))
            fh.write("\n")


def read_combined_jsonl(path: str):
    """
    Generator that reads a combined JSONL file one line at a time.
    Use this on the receiving end (verification, dashboard, Blender)
    to stream records without loading the whole file into memory.

    Yields one dict per line in file order.
    """
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
    """A 60 Hz ENF trace with the small wandering fluctuation that makes ENF
    usable as a timestamp. A replay or fabricated trace will not share this
    wander, which is the property Leiva's verification exploits."""
    rng = random.Random(seed)
    out: list[float] = []
    f = nominal
    for _ in range(seconds * hz):
        f += rng.gauss(0, 0.003)
        f += (nominal - f) * 0.02
        out.append(f)
    return out