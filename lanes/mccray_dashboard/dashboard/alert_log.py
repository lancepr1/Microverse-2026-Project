"""
alert_log.py — session-scoped alert bookkeeping.

There is no dedicated alert-generation/classification module in this repo
yet, only Leiva's real-time verification status (good/suspect/warning) --
see verification_feed.py. An "alert" here means: a node's verification
status was not "good" on a polled sample, categorized by severity
("suspect" -> Warning, "warning" -> Alert) so the Alert log tab can group
and filter by it. Only severity is surfaced to the operator, not the
underlying verification score -- per spec, the raw score is noise they
don't want to see.

attack_vector IS real, not a guess: verification.py's individual checks
(_SequenceGuard, _DriftMonitor, _NLRRangeCheck, etc.) already write a
specific reason string for whatever they caught -- "REPLAY: timestamp...",
"DRIFT DETECTED: CUSUM=...", "FLAT SIGNATURE: zero variance...", and so
on -- and verification_feed.py's verify_sample() already threads that
through as state["verification_reasons"]. _classify_attack_vector() below
just recognizes which check's own wording fired and maps it to a
human-readable label (Replay/Drift/Rollback/Fabrication/Injection/
Localized anomaly). It is naming which detector tripped, not inferring
attacker intent -- there is still no true attack-type classifier here.

Alerts are grouped into episodes rather than kept as one row per polled
sample: a node entering "suspect" or "warning" opens an episode; further
consecutive samples at the same severity extend it (bumping sample_count
and refreshing the latest readings); a severity change or a return to
"good" closes it. This is what makes the Alert log actually useful for
spotting patterns -- a node stuck in Warning for 40 ticks reads as one row
("14:02:11 -> 14:03:35, 40 samples") instead of 40 identical ones burying
the signal.

Purely in-memory and scoped to the current dashboard process -- "this
session" in the spec -- not persisted across restarts.
"""
import time

MAX_EPISODES = 200
_SEVERITY = {"suspect": "Warning", "warning": "Alert"}

# Ordered by priority -- the first keyword found across a sample's reasons
# wins, roughly matching the order verification.py itself runs its checks
# in (sequence guard first, then range/continuity, drift, local anomaly).
_VECTOR_KEYWORDS = [
    ("REPLAY", "Replay"),
    ("OUT OF ORDER", "Replay"),
    ("MONOTONICITY VIOLATION", "Rollback"),
    ("DRIFT DETECTED", "Drift"),
    ("RAW VALUE TREND", "Drift"),
    ("FLAT SIGNATURE", "Fabrication"),
    ("EMPTY SIGNATURE", "Fabrication"),
    ("INVALID SIGNATURE", "Fabrication"),
    ("ENERGY SPIKE", "Injection"),
    ("OUT OF NOMINAL RANGE", "Injection"),
    ("OUT OF RANGE", "Injection"),
    ("DISCONTINUITY", "Injection"),
    ("LOCAL ANOMALY", "Localized anomaly"),
]
UNCLASSIFIED_VECTOR = "Unclassified"


def _classify_attack_vector(reasons: list) -> str:
    """Maps verification_feed.py's (channel, reason) tuples to a
    human-readable attack-vector label by recognizing the exact wording
    each verifier check already writes for what it caught -- see
    verification.py's _SequenceGuard/_DriftMonitor/_NLRRangeCheck/etc.
    Not a guess: this only names which detector fired."""
    for _channel, reason in reasons or []:
        for keyword, label in _VECTOR_KEYWORDS:
            if keyword in reason:
                return label
    return UNCLASSIFIED_VECTOR

_episodes = []        # closed episode dicts, no particular order
_open_episodes = {}   # node_id -> in-progress episode dict
_node_stats = {}      # node_id -> {alert_count, last_alert_at, last_severity}


def reset() -> None:
    _episodes.clear()
    _open_episodes.clear()
    _node_stats.clear()


def record_poll(state: dict) -> None:
    """Call once per new multi-node poll tick (state = data_feed.poll_all()
    output) to update session stats and advance/open/close alert episodes."""
    now = time.time()
    for node_id, data in state.items():
        status = data.get("status", "--")
        severity = _SEVERITY.get(status)

        stats = _node_stats.setdefault(node_id, {
            "alert_count": 0, "last_alert_at": None, "last_severity": "--",
        })

        if severity:
            stats["alert_count"] += 1
            stats["last_alert_at"] = now
            stats["last_severity"] = severity
            vector = _classify_attack_vector(data.get("verification_reasons"))
            _extend_or_open_episode(node_id, severity, now, data, vector)
        else:
            _close_episode(node_id)

    del _episodes[:-MAX_EPISODES]


def _extend_or_open_episode(node_id: str, severity: str, now: float,
                             data: dict, vector: str) -> None:
    episode = _open_episodes.get(node_id)
    if episode is not None and episode["severity"] != severity:
        _close_episode(node_id)
        episode = None

    if episode is None:
        episode = {
            "node_id": node_id,
            "severity": severity,
            "start_ts": now,
            "sample_count": 0,
            "attack_vector": UNCLASSIFIED_VECTOR,
        }
        _open_episodes[node_id] = episode

    # Keep the first real classification an episode gets -- what tripped
    # first is what triggered it -- but upgrade out of "Unclassified" if a
    # later sample in the same episode does resolve to a real vector.
    if episode["attack_vector"] == UNCLASSIFIED_VECTOR and vector != UNCLASSIFIED_VECTOR:
        episode["attack_vector"] = vector

    episode["end_ts"] = now
    episode["sample_count"] += 1
    episode["frq_hz"] = data.get("frq_hz")
    episode["total_power_w"] = data.get("total_power_w")
    episode["average_gpu_temp_c"] = data.get("average_gpu_temp_c")


def _close_episode(node_id: str) -> None:
    episode = _open_episodes.pop(node_id, None)
    if episode is not None:
        _episodes.append(episode)


def get_timeline(limit: int = 100, severity: str | None = None,
                  node_id: str | None = None) -> list[dict]:
    """Alert episodes, most recently active first. Still-open episodes (a
    node currently mid-alert) are included and tagged "ongoing": True.
    severity ("Warning"/"Alert") and node_id filter the result; both
    default to no filtering."""
    episodes = (
        [dict(e, ongoing=False) for e in _episodes]
        + [dict(e, ongoing=True) for e in _open_episodes.values()]
    )
    if severity:
        episodes = [e for e in episodes if e["severity"] == severity]
    if node_id:
        episodes = [e for e in episodes if e["node_id"] == node_id]
    episodes.sort(key=lambda e: e["end_ts"], reverse=True)
    return episodes[:limit]


def get_node_stats() -> dict:
    return dict(_node_stats)
