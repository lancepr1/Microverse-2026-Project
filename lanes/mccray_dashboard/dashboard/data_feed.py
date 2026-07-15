"""
data_feed.py — replays for_dashboard.jsonl in real time, one row per
REPLAY_INTERVAL_S, for every node in the current pipeline run.

This module owns the only mutable state in the data pipeline: a cursor
into the recorded row list, paced against wall-clock time so the dashboard
"streams" the recording the same way it would consume a live feed.

REMOVED (2026-07, cleanup pass): the single-node API that used to live
here (init_feed()/poll()/get_rack_id(), loading data/run01.jsonl) is gone.
Confirmed with Leiva that nothing should read run01.jsonl going forward,
and its only remaining caller was main.py's SQLite history logging
(history_store.py/record_sample()), which is also removed as part of this
same cleanup -- see main.py's own comment. If persistent logging of the
REAL multi-node stream is wanted later, that's a new feature to design,
not a restoration of this one.

REMOVED (2026-07, same pass): the node00..nodeNN normalization step.
list_node_ids() now discovers real node ids DIRECTLY from
for_dashboard.jsonl's own raw hostname column prefixes (e.g.
"x3105c0s37b0n0_gpu-0[W]") -- tools/normalize_node_ids.py, the rename
step in run_microverse.py's stage 4, and the runs/verification.jsonl
normalization it also used to do are no longer part of the pipeline.
models.py's TelemetrySample.from_dashboard_row() already took node_id as
a plain parameter and never actually required the node00 format -- that
requirement lived ENTIRELY in this file's old discovery regex, which is
the only thing that changed. Rack grouping/ordering is unaffected: the
old normalize() step assigned node00, node01, ... in sorted(raw_ids)
order, so "Rack 1" was always just "the first 4 raw hostnames,
alphabetically" -- reading raw hostnames directly and sorting them
produces the exact same grouping, just with real hostnames as labels
instead of node00.. ones.
"""
import json
import os
import re
import time

from models import TelemetrySample

DASHBOARD_JSONL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "leiva_verification", "outputs", "for_dashboard.jsonl",
)

REPLAY_INTERVAL_S = 2.0  # one recorded sample is "emitted" per 2 seconds of replay

# Node ids are discovered from gpu/cpu columns specifically -- the only
# columns guaranteed to appear exactly once per real node component, the
# same reliable signal tools/normalize_node_ids.py used to use for
# discovery (before that file existed at all). "<node_id>_status" columns
# share the same prefix but aren't used for DISCOVERY, same reasoning as
# before: nothing structurally ties a "_status" suffix to a real node the
# way "_gpu-N[...]" does. models.py's from_dashboard_row() then matches
# every column sharing whatever prefix comes out of here, status columns
# included -- no separate rename step needed for that anymore.
_NODE_COLUMN_RE = re.compile(r"^(.+?)_(?:gpu|cpu)-\d+(?:-core)?\[")

# Verification status ("good"/"suspect"/"warning") comes directly from
# for_dashboard.jsonl's own "ENF_status"/"<node_id>_status" columns (see
# models.py's TelemetrySample.status/enf_status) -- that's the whole reason
# those columns exist. "status" (per node) and "enf_status" are fully
# independent signals, never blended: a node's card must reflect ONLY that
# node's own NLR checks, and ENF status shows up ONLY on the header dot
# (see ui/layout.py's render_header_frq_dot_style()).
_STATUS_SCORE_TO_LABEL = {0.0: "good", 0.5: "suspect", 1.0: "warning"}


def _status_label(score: float | None) -> str:
    if score is None:
        return "--"
    return _STATUS_SCORE_TO_LABEL.get(score, "--")


def list_node_ids() -> list[str]:
    """Every real node id embedded as a raw hostname column prefix in
    DASHBOARD_JSONL (e.g. "x3105c0s37b0n0_gpu-0[W]" -> "x3105c0s37b0n0"),
    sorted. Reads just the first line, since every row shares the same
    columns. This is the Rack 1-4 grouping source for the Operator tab:
    chunking this sorted list into groups of 4 gives Rack 1..4 (there is
    no rack field in the recording itself)."""
    with open(DASHBOARD_JSONL) as f:
        first_line = f.readline()
    row = json.loads(first_line)
    ids = {m.group(1) for key in row if (m := _NODE_COLUMN_RE.match(key))}
    return sorted(ids)


def node_display_label(node_id: str) -> str:
    """Display label for a node id. REMOVED (2026-07): the old "Node
    00".."Node 15" reformatting -- that assumed the node00 naming
    convention, which no longer exists (see module docstring). Real node
    ids are raw hostnames (e.g. "x3105c0s37b0n0") and are shown as-is."""
    return node_id


_multi_samples = {}
_multi_cursor = [0]
_multi_t0 = [0.0]
_multi_ready = [False]
_multi_max_len = [0]


def init_multi_feed() -> None:
    """Loads DASHBOARD_JSONL once and slices it per node, so the Operator
    tab can replay all nodes' state in lockstep."""
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

    _multi_cursor[0] = 0
    _multi_t0[0] = time.time()
    _multi_max_len[0] = max((len(s) for s in _multi_samples.values()), default=0)
    _multi_ready[0] = bool(_multi_samples)


def _to_state(sample: TelemetrySample) -> dict:
    node_label = _status_label(sample.status)
    enf_label = _status_label(sample.enf_status)

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
        # This node's OWN NLR checks only (GPU/CPU wattage, temp, etc.) --
        # every node card/badge/border on the Operator and Analyst tabs is
        # driven by this field, and it must NEVER include ENF.
        "status": node_label,
        # ENF's own checks only, facility-wide -- every node in a given
        # tick carries the same value here since it's shared. This is
        # what ui/layout.py's render_header_frq_dot_style() reads to
        # drive the header dot. Never blended into "status" above.
        "enf_status": enf_label,
        # This node's own trust score only (not blended with ENF's) --
        # matches "status" above, same independence rule.
        "verification_score": sample.status,
        # for_dashboard.jsonl carries no per-check reason text, only the
        # discretized 0.0/0.5/1.0 status columns -- this is always [].
        "verification_reasons": [],
    }


def poll_all() -> dict:
    """Return {} if no new sample is due yet, otherwise the latest state for
    every node, keyed by node id, PLUS one synthetic "ENF" entry -- see the
    comment below."""
    if not _multi_ready[0] or _multi_cursor[0] >= _multi_max_len[0]:
        return {}

    elapsed = time.time() - _multi_t0[0]
    target_index = int(elapsed / REPLAY_INTERVAL_S)
    if target_index < _multi_cursor[0]:
        return {}

    result = {}
    enf_score = None
    for node_id, samples in _multi_samples.items():
        next_index = min(target_index, len(samples) - 1)
        sample = samples[next_index]
        result[node_id] = _to_state(sample)
        if enf_score is None and sample.enf_status is not None:
            enf_score = sample.enf_status

    # A synthetic "ENF" entry, separate from every real node_id -- "ENF"
    # can never collide with a real one (real ids come from gpu/cpu column
    # prefixes; nothing produces a bare "ENF" hostname). This exists ONLY
    # so alert_log.py's record_poll(state) -- which iterates every key in
    # this dict, not just real nodes -- picks up an ENF-only bad event as
    # its own alert episode, independent of any node's own status.
    #
    # CRITICAL: every UI element that renders PER-REAL-NODE data
    # (ui/operator.py's grid, ui/analyst.py's rack panels, the Anomaly
    # Log table) iterates list_node_ids() explicitly and will never see
    # this key -- only alert_log.py's blind state.items() iteration does,
    # which is exactly the point. ui/operator.py's render_summary_cards()
    # iterates list_node_ids() explicitly for the same reason -- worth
    # checking any NEW code that iterates poll_all()'s output directly.
    result["ENF"] = {
        "index": target_index,
        "status": _status_label(enf_score),
        "enf_status": _status_label(enf_score),
        "verification_score": enf_score,
        "verification_reasons": [],
        "total_power_w": None,
        "average_gpu_temp_c": None,
        "frq_hz": None,
    }

    _multi_cursor[0] = target_index + 1
    return result