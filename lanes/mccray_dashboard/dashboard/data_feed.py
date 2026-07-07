"""
data_feed.py — loads data/run01.jsonl and replays it in real time.

This module owns the only mutable state in the data pipeline: a cursor into
the recorded sample list, paced against wall-clock time so the dashboard
"streams" the recording at REPLAY_INTERVAL_S per sample, the same way it
would consume a live feed. There is exactly one node in this version of the
dashboard; its id is derived from the run file being loaded (see
get_rack_id()) rather than hardcoded, since each run file represents one
node's recording and the same id has to match what
tools/generate_verification.py used as --run-id/--component-id when it
produced runs/<run_id>/verification.jsonl. The state dict is keyed by that
id so the UI layer already has the right shape if a second node's recording
is added later.
"""
import os
import time

from models import TelemetrySample
import verification_feed

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
REPLAY_INTERVAL_S = 1.0  # one recorded sample is "emitted" per second of replay

_samples = []
_cursor = [0]
_t0 = [0.0]
_ready = [False]
_rack_id = [None]


def init_feed(run_file="run01.jsonl"):
    path = os.path.join(DATA_DIR, run_file)
    run_id = os.path.splitext(run_file)[0]

    _samples.clear()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _samples.append(TelemetrySample.from_json_line(line))

    verification_feed.init_verifier(run_id)

    _cursor[0] = 0
    _t0[0] = time.time()
    _ready[0] = True
    _rack_id[0] = run_id


def get_rack_id() -> str:
    """The node id for the currently loaded run (e.g. run01.jsonl ->
    "run01"). None until init_feed() has run."""
    return _rack_id[0]


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
    verification = verification_feed.verify_sample(sample.index)
    _cursor[0] = next_index + 1

    return {get_rack_id(): _to_state(sample, verification)}


def _to_state(sample: TelemetrySample, verification: dict) -> dict:
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
        "status": verification["status"],
        "verification_score": verification["score"],
        "verification_reasons": verification["reasons"],
    }
