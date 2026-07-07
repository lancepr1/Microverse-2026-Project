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
"""
import json
import os

_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_DASHBOARD_DIR, "..", "..", ".."))
_RUNS_DIR = os.path.join(_REPO_ROOT, "runs")

STATUS_LABELS = {"trusted": "good", "suspect": "suspect", "failed": "warning"}
_STATUS_RANK = {"good": 0, "suspect": 1, "warning": 2}

_by_index = {}
_loaded = [False]


def init_verifier(run_id: str) -> None:
    """Load runs/<run_id>/verification.jsonl into memory, grouped by sample
    index. Safe to call even if the file doesn't exist yet (e.g. the offline
    generator hasn't been run for this run_id) -- verify_sample() then
    falls back to "--" for every index."""
    _by_index.clear()
    path = os.path.join(_RUNS_DIR, run_id, "verification.jsonl")
    _loaded[0] = os.path.exists(path)
    if not _loaded[0]:
        return

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)["data"]
            index = int(float(record["timestamp"]))
            _by_index.setdefault(index, []).append(record)


def verify_sample(index: int) -> dict:
    """Aggregate every VerificationResult recorded for this sample index
    into one rack-level {"status", "score", "reasons"} dict. Returns status
    "--" when no verification data is available for this index."""
    records = _by_index.get(index)
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
