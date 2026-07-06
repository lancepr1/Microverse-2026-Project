"""
data_feed.py — loads data/run01.jsonl and replays it in real time.

This module owns the only mutable state in the data pipeline: a cursor into
the recorded sample list, paced against wall-clock time so the dashboard
"streams" the recording at REPLAY_INTERVAL_S per sample, the same way it
would consume a live feed. There is exactly one rack in this version of the
dashboard (see RACK_ID); the state dict is keyed by rack so the UI layer
already has the right shape if a second rack's recording is added later.
"""
import os
import time

from models import TelemetrySample

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
RACK_ID = "Rack_01"
REPLAY_INTERVAL_S = 1.0  # one recorded sample is "emitted" per second of replay

_samples = []
_cursor = [0]
_t0 = [0.0]
_ready = [False]


def init_feed(run_file="run01.jsonl"):
    path = os.path.join(DATA_DIR, run_file)

    _samples.clear()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _samples.append(TelemetrySample.from_json_line(line))

    _cursor[0] = 0
    _t0[0] = time.time()
    _ready[0] = True


def poll() -> dict:
    """Return {} if no new sample is due yet, otherwise the latest
    TelemetrySample for each rack, flattened into a plain dict so the UI
    layer doesn't need to import models.py."""
    if not _ready[0] or _cursor[0] >= len(_samples):
        return {}

    elapsed = time.time() - _t0[0]
    target_index = int(elapsed / REPLAY_INTERVAL_S)
    if target_index < _cursor[0]:
        return {}

    next_index = min(target_index, len(_samples) - 1)
    sample = _samples[next_index]
    _cursor[0] = next_index + 1

    return {RACK_ID: _to_state(sample)}


def _to_state(sample: TelemetrySample) -> dict:
    return {
        "index": sample.index,
        "frq_hz": sample.frq_hz,
        "gpu_power_w": sample.gpu_power_w,
        "gpu_temp_c": sample.gpu_temp_c,
        "cpu_power_w": sample.cpu_power_w,
        "cpu_energy_uj": sample.cpu_energy_uj,
        "cpu_core_power_w": sample.cpu_core_power_w,
        "cpu_core_energy_uj": sample.cpu_core_energy_uj,
        "total_power_w": sample.total_power_w,
        "average_gpu_temp_c": sample.average_gpu_temp_c,
    }
