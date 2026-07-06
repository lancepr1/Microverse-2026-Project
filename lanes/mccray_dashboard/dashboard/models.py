"""
models.py — the dashboard's own data model.

This is the dashboard's equivalent of an external contracts module: every
other file (data_feed.py, ui/*.py) should depend on TelemetrySample rather
than on raw dict keys or JSON field names, so the on-disk format can change
in one place.

Field naming in the source JSONL (see data/run01.jsonl) follows the pattern
emitted by the rack's monitoring agent:
    "FRQ"             -> PDU-level AC line frequency, in Hz
    "gpu-<N>[W]"      -> GPU N instantaneous power draw, in watts
    "gpu-<N>[C]"      -> GPU N temperature, in degrees Celsius
    "cpu-<N>[uJ]"     -> CPU socket N cumulative package energy, in microjoules
    "cpu-<N>[W]"      -> CPU socket N instantaneous package power, in watts
    "cpu-<N>-core[uJ]"-> CPU socket N core-domain cumulative energy, in microjoules
    "cpu-<N>-core[W]" -> CPU socket N core-domain instantaneous power, in watts

GPU/CPU indices are discovered from whatever keys are present in a given
line rather than hardcoded, so a future run0N.jsonl with a different
component count parses without code changes.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field

_GPU_KEY_RE = re.compile(r"^gpu-(\d+)\[(W|C)\]$")
_CPU_KEY_RE = re.compile(r"^cpu-(\d+)(-core)?\[(uJ|W)\]$")


@dataclass
class TelemetrySample:
    """One polling interval of rack telemetry: PDU frequency plus per-GPU
    and per-CPU power/thermal/energy readings.

    GPU and CPU readings are keyed by component index (0, 1, 2, ...) so the
    UI layer can display either an aggregate (total power, max temp) or a
    per-component breakdown without this class knowing which the caller
    wants.
    """
    index: int
    frq_hz: float
    gpu_power_w: dict[int, float] = field(default_factory=dict)
    gpu_temp_c: dict[int, float] = field(default_factory=dict)
    cpu_power_w: dict[int, float] = field(default_factory=dict)
    cpu_energy_uj: dict[int, float] = field(default_factory=dict)
    cpu_core_power_w: dict[int, float] = field(default_factory=dict)
    cpu_core_energy_uj: dict[int, float] = field(default_factory=dict)

    @property
    def total_gpu_power_w(self) -> float:
        return sum(self.gpu_power_w.values())

    @property
    def total_cpu_power_w(self) -> float:
        return sum(self.cpu_power_w.values())

    @property
    def total_power_w(self) -> float:
        return self.total_gpu_power_w + self.total_cpu_power_w

    @property
    def average_gpu_temp_c(self) -> float | None:
        return statistics.mean(self.gpu_temp_c.values()) if self.gpu_temp_c else None

    @classmethod
    def from_dict(cls, d: dict) -> "TelemetrySample":
        sample = cls(index=int(d.get("index", 0)), frq_hz=float(d["FRQ"]))

        for key, value in d.items():
            gpu_match = _GPU_KEY_RE.match(key)
            if gpu_match:
                gpu_id, kind = int(gpu_match.group(1)), gpu_match.group(2)
                target = sample.gpu_power_w if kind == "W" else sample.gpu_temp_c
                target[gpu_id] = float(value)
                continue

            cpu_match = _CPU_KEY_RE.match(key)
            if cpu_match:
                cpu_id, is_core, kind = (
                    int(cpu_match.group(1)), cpu_match.group(2), cpu_match.group(3)
                )
                if is_core:
                    target = sample.cpu_core_energy_uj if kind == "uJ" else sample.cpu_core_power_w
                else:
                    target = sample.cpu_energy_uj if kind == "uJ" else sample.cpu_power_w
                target[cpu_id] = float(value)

        return sample

    @classmethod
    def from_json_line(cls, line: str) -> "TelemetrySample":
        return cls.from_dict(json.loads(line))
