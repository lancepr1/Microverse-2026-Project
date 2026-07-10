"""
alert_grouping.py -- pure, stateless incident clustering for the Alert log
tab's Grouped view (see ui/grouped_alert_table.py for the renderer, and
ui/alert_log_tab.py for where this plugs in behind the Grouped/Flat
toggle).

Groups alert-episode rows -- the same dicts alert_log.get_timeline()
already returns (node_id, severity, start_ts, end_ts, sample_count,
total_power_w, average_gpu_temp_c, ongoing, prediction), plus a "rack" key
the caller injects -- into time-windowed Incidents, so a burst of
near-simultaneous alerts across nodes renders as one collapsible incident
instead of N nearly-identical rows.

No module-level state, no Dash imports: takes a list of alert dicts,
returns a list of Incident objects, never touches its input. The flat
timeline (ui/alert_log_tab.py's existing table) keeps reading the exact
same untouched rows regardless of which view mode is selected.
"""
from dataclasses import dataclass, field
from datetime import datetime

_SEVERITY_RANK = {"Warning": 0, "Alert": 1}


@dataclass
class Incident:
    incident_id: str
    severity: str
    start_ts: float
    end_ts: float
    node_count: int
    racks: list
    total_samples: int
    peak_power_w: float | None
    peak_gpu_temp_c: float | None
    ongoing: bool
    attack_vectors: list = field(default_factory=list)  # distinct, sorted, from child rows
    children: list = field(default_factory=list)  # original alert rows, UNMODIFIED


def _parse_hms(val) -> float | None:
    """Defensive fallback for "HH:MM:SS"-style timestamps -- our real
    schema uses numeric epoch seconds (start_ts/end_ts) and never needs
    this, but a row from a different source might."""
    if not isinstance(val, str):
        return None
    parts = val.split(":")
    if len(parts) != 3:
        return None
    try:
        h, m, s = (int(p) for p in parts)
    except ValueError:
        return None
    return float(h * 3600 + m * 60 + s)


def _resolve_ts(row: dict, ts_key: str, str_key: str) -> float | None:
    """Numeric epoch timestamp preferred (ts_key, e.g. "start_ts") --
    falls back to parsing ts_key as "HH:MM:SS", then to str_key (e.g.
    "start") as either numeric or "HH:MM:SS". Returns None, never raises,
    if nothing resolves -- callers treat that as a malformed row."""
    val = row.get(ts_key)
    if isinstance(val, (int, float)):
        return float(val)
    parsed = _parse_hms(val)
    if parsed is not None:
        return parsed

    val = row.get(str_key)
    if isinstance(val, (int, float)):
        return float(val)
    return _parse_hms(val)


def _severity_of(row: dict) -> str:
    return row.get("severity") or "--"


def _make_single_incident(row: dict, start: float | None, end: float | None,
                           index: int) -> Incident:
    """A row whose timestamp couldn't be resolved, or the seed of a new
    cluster, becomes its own one-child incident."""
    severity = _severity_of(row)
    label = datetime.fromtimestamp(start).strftime("%H:%M:%S") if start is not None else f"unknown-{index}"
    start_ts = start if start is not None else 0.0
    end_ts = end if end is not None else start_ts
    rack = row.get("rack")
    vector = row.get("attack_vector")

    return Incident(
        incident_id=f"INC-{label}-{severity.lower()}",
        severity=severity,
        start_ts=start_ts,
        end_ts=end_ts,
        node_count=1 if row.get("node_id") else 0,
        racks=[rack] if rack else [],
        total_samples=row.get("sample_count", 0) or 0,
        peak_power_w=row.get("total_power_w"),
        peak_gpu_temp_c=row.get("average_gpu_temp_c"),
        ongoing=bool(row.get("ongoing", False)),
        attack_vectors=[vector] if vector else [],
        children=[row],
    )


def _merge_into(incident: Incident, row: dict, start: float | None, end: float | None) -> None:
    """Folds one more alert row into an in-progress incident. Mutates the
    Incident being built (which this module owns), never the row itself."""
    if start is not None:
        incident.start_ts = min(incident.start_ts, start)
    if end is not None:
        incident.end_ts = max(incident.end_ts, end)

    severity = _severity_of(row)
    if _SEVERITY_RANK.get(severity, -1) > _SEVERITY_RANK.get(incident.severity, -1):
        incident.severity = severity  # severity escalation: Alert beats Warning

    node_id = row.get("node_id")
    if node_id and node_id not in {c.get("node_id") for c in incident.children}:
        incident.node_count += 1

    rack = row.get("rack")
    if rack and rack not in incident.racks:
        incident.racks.append(rack)

    vector = row.get("attack_vector")
    if vector and vector not in incident.attack_vectors:
        incident.attack_vectors.append(vector)

    incident.total_samples += row.get("sample_count", 0) or 0

    power = row.get("total_power_w")
    if isinstance(power, (int, float)):
        incident.peak_power_w = power if incident.peak_power_w is None else max(incident.peak_power_w, power)

    temp = row.get("average_gpu_temp_c")
    if isinstance(temp, (int, float)):
        incident.peak_gpu_temp_c = temp if incident.peak_gpu_temp_c is None else max(incident.peak_gpu_temp_c, temp)

    incident.ongoing = incident.ongoing or bool(row.get("ongoing", False))
    incident.children.append(row)


def group_alerts(alerts: list, window_seconds: float = 5.0,
                  group_keys: tuple = ("severity",)) -> list:
    """Greedy time-window clustering:

    1. Sort alerts by start time.
    2. Walk the sorted list; an alert joins the current incident if its
       start is within window_seconds of the incident's rolling end AND
       its group_keys (other than "severity") match; otherwise a new
       incident begins.
    3. Severity escalation: a severity mismatch within the window never
       blocks joining by itself -- it merges into one incident whose
       severity is the more severe of the two ("Alert" beats "Warning").
    4. Children are stored untouched (the original row objects) so the
       flat view can always be reconstructed from the same data.

    Rows with an unresolvable start timestamp become their own
    single-child incident rather than raising. Returns incidents newest
    (most recently active) first. Never mutates `alerts` or its rows.
    """
    if not alerts:
        return []

    resolved = []
    for i, row in enumerate(alerts):
        start = _resolve_ts(row, "start_ts", "start")
        end = _resolve_ts(row, "end_ts", "end")
        resolved.append((start, end if end is not None else start, row, i))

    good = sorted((r for r in resolved if r[0] is not None), key=lambda r: r[0])
    bad = [r for r in resolved if r[0] is None]

    incidents = []
    current = None

    for start, end, row, index in good:
        if current is None:
            current = _make_single_incident(row, start, end, index)
            continue

        gap = start - current.end_ts
        same_group = all(
            row.get(key) == current.children[0].get(key)
            for key in group_keys if key != "severity"
        )

        if gap <= window_seconds and same_group:
            _merge_into(current, row, start, end)
        else:
            incidents.append(current)
            current = _make_single_incident(row, start, end, index)

    if current is not None:
        incidents.append(current)

    for start, end, row, index in bad:
        incidents.append(_make_single_incident(row, start, end, index))

    incidents.sort(key=lambda inc: inc.end_ts, reverse=True)
    return incidents
