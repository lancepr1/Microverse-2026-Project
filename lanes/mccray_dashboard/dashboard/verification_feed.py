"""
verification_feed.py — reads pre-computed VerificationResult records for
the current replay run and aggregates them into one rack-level status per
polled sample.

This module never imports the shared contracts package or Leiva's
verification code -- see the dashboard-folder coupling guard test in
tests/test_ui_imports.py, which checks the dashboard folder's ability to
run standalone with no sibling-lane code present. The actual verification
run happens offline, via
tools/generate_verification.py, which calls Leiva's AnchorExtractor/Verifier
and writes runs/<run_id>/verification.jsonl using the shared io_records
file-bus format:
    {"_type": "VerificationResult", "data": {timestamp, component_id,
     status, score, anchor_ref, reason}}
This module only ever reads that JSON back as plain dicts.

Leiva's own dashboard-facing label mapping (verification.py docstring):
    trusted -> "good", suspect -> "suspect", failed -> "warning"

init_verifier()/verify_sample() key their state by run_id. For the
multi-node Operator tab, one run_id's verification.jsonl covers every
node's results together (the whole rack is verified in a single pipeline
run, not one run per node) -- verify_sample()'s node_id argument is what
separates one node's results back out again. Passing no run_id to
verify_sample() falls back to whichever run_id was passed to
init_verifier() most recently, which keeps the original single-node
call pattern (data_feed.py's single-node poll(), and the existing tests)
working unchanged.
"""
import json
import os

_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_DASHBOARD_DIR, "..", "..", ".."))
_RUNS_DIR = os.path.join(_REPO_ROOT, "runs")

STATUS_LABELS = {"trusted": "good", "suspect": "suspect", "failed": "warning"}
_STATUS_RANK = {"good": 0, "suspect": 1, "warning": 2}

_by_run = {}
_current_run_id = [None]


def init_verifier(run_id: str) -> None:
    """Load runs/<run_id>/verification.jsonl into memory, grouped by sample
    index. Safe to call even if the file doesn't exist yet (e.g. the offline
    generator hasn't been run for this run_id) -- verify_sample() then
    falls back to "--" for every index in that run_id."""
    path = os.path.join(_RUNS_DIR, run_id, "verification.jsonl")
    by_index = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)["data"]
                index = int(float(record["timestamp"]))
                by_index.setdefault(index, []).append(record)

    _by_run[run_id] = by_index
    _current_run_id[0] = run_id


def verify_sample(index: int, node_id: str | None = None, run_id: str | None = None) -> dict:
    """Aggregate VerificationResult records for this sample index into one
    {"status", "score", "reasons"} dict. run_id defaults to the most recent
    init_verifier() call. Returns status "--" when no verification data is
    available for this index.

    node_id filters to just that node's results (matched by the node-id
    prefix embedded in component_id, e.g. "rack_00/node00_gpu-0[W]") plus
    the shared facility-wide "<run_id>/ENF" check that applies to every
    node -- this is for the multi-node case, where one run_id's
    verification.jsonl covers every node's results together rather than
    one file per node. Leaving node_id unset (the single-node poll()
    call pattern, and the pre-existing tests) keeps the old behavior:
    every record for that index counts, regardless of component_id."""
    run_id = run_id if run_id is not None else _current_run_id[0]
    records = _by_run.get(run_id, {}).get(index)
    if not records:
        return {"status": "--", "score": None, "reasons": []}

    if node_id is not None:
        records = [
            r for r in records
            if r["component_id"].split("/", 1)[-1] == "ENF"
            or r["component_id"].split("/", 1)[-1].startswith(f"{node_id}_")
        ]
        if not records:
            return {"status": "--", "score": None, "reasons": []}

    worst_label = "good"
    worst_score = 1.0
    reasons = []

    for record in records:
        label = STATUS_LABELS.get(record["status"], "good")
        if label != "good":
            suffix = record["component_id"].split("/", 1)[-1]
            reasons.append((suffix, record["reason"]))
        if _STATUS_RANK[label] > _STATUS_RANK[worst_label]:
            worst_label = label
            worst_score = record["score"]
        elif label == worst_label:
            worst_score = min(worst_score, record["score"])

    return {"status": worst_label, "score": round(worst_score, 4), "reasons": reasons}
