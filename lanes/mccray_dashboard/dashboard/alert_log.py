"""
alert_log.py — session-scoped alert bookkeeping.

There is no dedicated alert-generation/classification module in this repo
yet, only Leiva's real-time verification status (good/suspect/warning) --
see verification_feed.py. An "alert" here means: a node's verification
status was not "good" on a polled sample. There is no existing classifier
that sorts alerts into rate_spike/deviation/proximity/divergence, so
alert_type stays "--" (unclassified) rather than guessing one; likewise
prediction is always "No prediction" until an RNN forecast exists to
compare against. "current anomaly score" reuses verification_feed's score
field per the dashboard spec's instruction to wire it to whatever score
field already exists -- note that field is actually a *trust* score
(higher = more trusted, e.g. 0.95 for a clean sample), not a literal
anomaly score (where higher would mean more anomalous), so read the numbers
with that in mind until a real anomaly-scoring model replaces it.

Purely in-memory and scoped to the current dashboard process -- "this
session" in the spec -- not persisted across restarts.
"""
import time

MAX_TIMELINE_ENTRIES = 500
_ALERT_STATUSES = ("suspect", "warning")

_timeline = []    # alert-entry dicts, oldest first
_node_stats = {}  # node_id -> {current_score, max_score, alert_count, last_alert_at, last_alert_type}


def reset() -> None:
    _timeline.clear()
    _node_stats.clear()


def record_poll(state: dict) -> None:
    """Call once per new multi-node poll tick (state = data_feed.poll_all()
    output) to update session stats and append any new alert entries."""
    now = time.time()
    for node_id, data in state.items():
        status = data.get("status", "--")
        score = data.get("verification_score")

        stats = _node_stats.setdefault(node_id, {
            "current_score": None, "max_score": None, "alert_count": 0,
            "last_alert_at": None, "last_alert_type": "--",
        })
        stats["current_score"] = score
        if score is not None:
            stats["max_score"] = (
                score if stats["max_score"] is None else max(stats["max_score"], score)
            )

        if status in _ALERT_STATUSES:
            stats["alert_count"] += 1
            stats["last_alert_at"] = now
            stats["last_alert_type"] = "--"

            _timeline.append({
                "timestamp": now,
                "node_id": node_id,
                "alert_type": "--",
                "score": score,
                "frq_hz": data.get("frq_hz"),
                "total_power_w": data.get("total_power_w"),
                "average_gpu_temp_c": data.get("average_gpu_temp_c"),
                "prediction": "No prediction",
            })
            del _timeline[:-MAX_TIMELINE_ENTRIES]


def get_timeline(limit: int = 100) -> list[dict]:
    """Newest first."""
    return list(reversed(_timeline[-limit:]))


def get_node_stats() -> dict:
    return dict(_node_stats)
