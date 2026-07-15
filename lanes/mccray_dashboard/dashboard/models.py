"""
models.py — the dashboard's own data model.

This is the dashboard's equivalent of an external contracts module: every
other file (data_feed.py, ui/*.py) should depend on TelemetrySample rather
than on raw dict keys or JSON field names, so the on-disk format can change
in one place.

Field naming in the source JSONL (for_dashboard.jsonl, produced by
run_microverse.py's stage 3) follows the pattern emitted by the rack's
monitoring agent, with each node's columns prefixed by its raw node id
(e.g. "x3105c0s37b0n0_gpu-0[W]" -- see data_feed.py's list_node_ids() for
how that prefix is discovered):
    "FRQ"                       -> PDU-level AC line frequency, in Hz
    "<node_id>_gpu-<N>[W]"      -> GPU N instantaneous power draw, in watts
    "<node_id>_gpu-<N>[C]"      -> GPU N temperature, in degrees Celsius
    "<node_id>_cpu-<N>[uJ]"     -> CPU socket N cumulative package energy, in microjoules
    "<node_id>_cpu-<N>[W]"      -> CPU socket N instantaneous package power, in watts
    "<node_id>_cpu-<N>-core[uJ]"-> CPU socket N core-domain cumulative energy, in microjoules
    "<node_id>_cpu-<N>-core[W]" -> CPU socket N core-domain instantaneous power, in watts

GPU/CPU indices are discovered from whatever keys are present in a given
line rather than hardcoded, so a run with a different component count
parses without code changes.

CHANGED (2026-07, per Leiva's request): for_dashboard.jsonl also carries
Leiva's own verification status directly -- "ENF_status" (facility-wide,
shared across every node in a row) and "<node_id>_status" (per node) --
written by run_microverse.py's stage 3 as a numeric 0.0/0.5/1.0
(trusted/suspect/failed) encoding. Those two new fields below (status,
enf_status) are ALL verification status now comes from -- see
data_feed.py's own CHANGED comment for why the separate
runs/<run_id>/verification.jsonl + verification_feed.py path is now
obsolete.

REMOVED (2026-07, cleanup pass): from_json_line() and the node00..nodeNN
normalization requirement it implied are both gone -- see data_feed.py's
module docstring. from_dashboard_row() below already took node_id as a
plain parameter and never actually required that naming convention.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Optional

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

    # ADDED (2026-07): straight from for_dashboard.jsonl's own
    # "<node_id>_status" (this node's own NLR checks, worst-of) and
    # "ENF_status" (facility-wide, shared across every node in a row)
    # columns -- 0.0/0.5/1.0 (trusted/suspect/failed), or None if this
    # particular row/source doesn't carry them (e.g. data/run01.jsonl,
    # the single-node legacy demo file, never had these columns at all).
    status: Optional[float] = None
    enf_status: Optional[float] = None

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

        # ADDED (2026-07): "status" here is already-stripped -- see
        # from_dashboard_row() below, which turns "<node_id>_status" into
        # plain "status" via the same prefix-stripping it uses for every
        # other node-owned column. "ENF_status" is read as-is since it's
        # never node-prefixed to begin with.
        if "status" in d:
            sample.status = float(d["status"])
        if "ENF_status" in d:
            sample.enf_status = float(d["ENF_status"])

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
    def from_dashboard_row(cls, node_id: str, row: dict) -> "TelemetrySample":
        """Build one node's sample out of a for_dashboard.jsonl row, which
        packs every node into a single wide record with node-prefixed keys
        (e.g. "node00_gpu-0[W]") instead of one file per node. Strips the
        prefix and reuses from_dict()'s existing key parsing.

        CHANGED (2026-07): also carries "<node_id>_status" through (strips
        to plain "status", same prefix logic as every gpu/cpu column) and
        "ENF_status" (copied as-is -- it's shared/facility-wide, never
        node-prefixed, so the prefix-strip loop above would never pick it
        up on its own)."""
        prefix = f"{node_id}_"
        stripped = {
            key[len(prefix):]: value for key, value in row.items() if key.startswith(prefix)
        }
        stripped["index"] = row.get("index", 0)
        stripped["FRQ"] = row["FRQ"]
        if "ENF_status" in row:
            stripped["ENF_status"] = row["ENF_status"]
        return cls.from_dict(stripped)