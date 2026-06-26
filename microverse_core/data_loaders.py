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
"""
from __future__ import annotations

import csv
import math
import random
from typing import Iterator, Optional

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
    """Load an ENF time-series as a list of frequency readings near 60 Hz.

    TODO (week 1, Leiva): confirm the field name and sample rate against the
    2025 ENF dataset. This stub assumes a single-column CSV of Hz values.
    """
    values: list[float] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # skip the metadata row (UTC timestamp + duration)
        for row in reader:
            if not row:
                continue
            try:
                values.append(float(row[1]))  # column 1 is the frequency
            except (ValueError, IndexError):
                continue  # skip any malformed rows
    return values


# --------------------------------------------------------------------------
# Synthetic fallbacks. Shapes only, not measurements.
# --------------------------------------------------------------------------

# rough per-node draw envelopes in watts, by workload
_ENVELOPE = {
    WorkloadClass.IDLE: (90, 130),
    WorkloadClass.LLM_INFERENCE: (250, 600),    # spiky, bursty
    WorkloadClass.LLM_TRAINING: (650, 780),     # high, sustained
    WorkloadClass.IMAGE_GENERATION: (300, 550),  # periodic
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
            base = hi - 30 + 30 * math.sin(t / 8)        # steady high plateau
        elif workload is WorkloadClass.LLM_INFERENCE:
            burst = hi if rng.random() < 0.18 else lo     # random request spikes
            base = burst
        elif workload is WorkloadClass.IMAGE_GENERATION:
            base = lo + (hi - lo) * (0.5 + 0.5 * math.sin(t / 3))  # periodic
        else:
            base = lo + 10 * rng.random()                 # idle
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
        f += rng.gauss(0, 0.003)          # slow random walk
        f += (nominal - f) * 0.02         # gentle pull back toward 60 Hz
        out.append(f)
    return out
