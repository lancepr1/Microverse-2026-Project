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

Below that single-node API (kept unchanged for the Analyst tab's chart feed
and the existing tests) is a second, parallel multi-node API --
init_multi_feed()/poll_all()/list_node_ids() -- that replays every node
recording in data/ concurrently, for the Operator tab's 16-node grid. All
node files share the same index/timing (see data/README.md), so a single
cursor paced against wall-clock time drives every node's replay position in
lockstep, the same pacing model as the single-node path above.
"""
import json
import os
import re
import time

from models import TelemetrySample
import verification_feed

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DASHBOARD_JSONL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "leiva_verification", "outputs", "for_dashboard.jsonl",
)
REPLAY_INTERVAL_S = 1.0  # one recorded sample is "emitted" per second of replay

# scripts/run_microverse.py's Component ID prompt defaults to "rack_00" and
# writes runs/<component_id>/verification.jsonl under that name -- this has
# to match whatever was actually entered at the prompt for verification
# status to show up. There's currently no field in DASHBOARD_JSONL itself
# that records which component_id produced it, so this is a convention, not
# a discovery.
DEFAULT_COMPONENT_ID = "rack_00"

_samples = []
_cursor = [0]
_t0 = [0.0]
_ready = [False]
_rack_id = [None]

_NODE_PREFIX_RE = re.compile(r"^(node\d+)_")


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


_multi_samples = {}
_multi_cursor = [0]
_multi_t0 = [0.0]
_multi_ready = [False]
_multi_max_len = [0]


def list_node_ids() -> list[str]:
    """Every node id ("node00".."node15") embedded as a column prefix in
    DASHBOARD_JSONL (e.g. "node00_gpu-0[W]" -> "node00"), sorted. Reads
    just the first line, since every row shares the same columns. This is
    the Rack 1-4 grouping source for the Operator tab: chunking this
    sorted list into groups of 4 gives Rack 1..4 (there is no rack field
    in the recording itself)."""
    with open(DASHBOARD_JSONL) as f:
        first_line = f.readline()
    row = json.loads(first_line)
    ids = {m.group(1) for key in row if (m := _NODE_PREFIX_RE.match(key))}
    return sorted(ids)


def node_display_label(node_id: str) -> str:
    """Human-friendly "Node 00".."Node 15" label for a node id. Node ids
    are already "node00".."node15" (the DASHBOARD_JSONL column prefix),
    so this just reformats the existing number."""
    return f"Node {node_id.removeprefix('node')}"


def init_multi_feed() -> None:
    """Loads DASHBOARD_JSONL once and slices it per node, so the Operator
    tab can replay all nodes' state in lockstep -- unlike init_feed(),
    which only ever replays one run file at a time."""
    _multi_samples.clear()
    rows = []
    with open(DASHBOARD_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    for node_id in list_node_ids():
        _multi_samples[node_id] = [
            TelemetrySample.from_dashboard_row(node_id, row) for row in rows
        ]
    verification_feed.init_verifier(DEFAULT_COMPONENT_ID)

    _multi_cursor[0] = 0
    _multi_t0[0] = time.time()
    _multi_max_len[0] = max((len(s) for s in _multi_samples.values()), default=0)
    _multi_ready[0] = bool(_multi_samples)


def poll_all() -> dict:
    """Return {} if no new sample is due yet, otherwise the latest state for
    every node, keyed by node id. Advances every node's cursor together,
    the same way poll() advances the single-node replay."""
    if not _multi_ready[0] or _multi_cursor[0] >= _multi_max_len[0]:
        return {}

    elapsed = time.time() - _multi_t0[0]
    target_index = int(elapsed / REPLAY_INTERVAL_S)
    if target_index < _multi_cursor[0]:
        return {}

    result = {}
    for node_id, samples in _multi_samples.items():
        next_index = min(target_index, len(samples) - 1)
        sample = samples[next_index]
        verification = verification_feed.verify_sample(
            sample.index, node_id=node_id, run_id=DEFAULT_COMPONENT_ID
        )
        result[node_id] = _to_state(sample, verification)

    _multi_cursor[0] = target_index + 1
    return result
