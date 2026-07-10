import copy

import alert_grouping


def _row(node_id, severity, start_ts, end_ts=None, rack="Rack 1", sample_count=1,
         total_power_w=100.0, average_gpu_temp_c=40.0, ongoing=False):
    return {
        "node_id": node_id,
        "severity": severity,
        "start_ts": start_ts,
        "end_ts": end_ts if end_ts is not None else start_ts,
        "sample_count": sample_count,
        "total_power_w": total_power_w,
        "average_gpu_temp_c": average_gpu_temp_c,
        "rack": rack,
        "ongoing": ongoing,
        "prediction": "No prediction",
    }


def test_burst_across_racks_collapses_to_one_incident():
    """10 warnings within 2s across 3 racks -> 1 incident, node_count == 10."""
    racks = ["Rack 1", "Rack 2", "Rack 3"]
    alerts = [
        _row(f"node{i:02d}", "Warning", start_ts=1000.0 + i * 0.2, rack=racks[i % 3])
        for i in range(10)
    ]
    incidents = alert_grouping.group_alerts(alerts, window_seconds=5.0)

    assert len(incidents) == 1
    assert incidents[0].node_count == 10
    assert set(incidents[0].racks) == set(racks)
    assert incidents[0].total_samples == 10


def test_severity_escalation_merges_warning_and_alert():
    """Warning burst + 1 Alert in-window -> merged, severity == 'Alert'."""
    alerts = [
        _row("node00", "Warning", start_ts=1000.0),
        _row("node01", "Warning", start_ts=1000.5),
        _row("node02", "Warning", start_ts=1001.0),
        _row("node03", "Alert", start_ts=1001.5),
    ]
    incidents = alert_grouping.group_alerts(alerts, window_seconds=5.0)

    assert len(incidents) == 1
    assert incidents[0].severity == "Alert"
    assert incidents[0].node_count == 4


def test_two_bursts_60s_apart_stay_separate():
    """Two bursts 60s apart -> 2 incidents."""
    alerts = [
        _row("node00", "Warning", start_ts=1000.0),
        _row("node01", "Warning", start_ts=1001.0),
        _row("node02", "Warning", start_ts=1061.0),
        _row("node03", "Warning", start_ts=1062.0),
    ]
    incidents = alert_grouping.group_alerts(alerts, window_seconds=5.0)

    assert len(incidents) == 2
    assert {inc.node_count for inc in incidents} == {2, 2}


def test_empty_list_returns_empty_list():
    assert alert_grouping.group_alerts([]) == []


def test_malformed_timestamp_becomes_its_own_incident_without_crashing():
    """A row with an unresolvable start timestamp gets its own incident
    instead of raising, and doesn't disturb grouping of the well-formed
    rows around it."""
    alerts = [
        _row("node00", "Warning", start_ts=1000.0),
        {"node_id": "node99", "severity": "Warning", "start_ts": "not-a-number",
         "end_ts": "also-bad", "sample_count": 1, "rack": "Rack 1"},
    ]
    incidents = alert_grouping.group_alerts(alerts, window_seconds=5.0)

    assert len(incidents) == 2
    assert sorted(inc.node_count for inc in incidents) == [1, 1]


def test_input_alerts_are_not_mutated():
    alerts = [
        _row("node00", "Warning", start_ts=1000.0),
        _row("node01", "Alert", start_ts=1000.5),
    ]
    before = copy.deepcopy(alerts)

    alert_grouping.group_alerts(alerts, window_seconds=5.0)

    assert alerts == before


def test_children_are_the_original_row_objects():
    """Children must be the literal original dicts (not copies with
    normalized keys) so the flat view can always be reconstructed."""
    row = _row("node00", "Warning", start_ts=1000.0)
    incidents = alert_grouping.group_alerts([row], window_seconds=5.0)

    assert incidents[0].children[0] is row


if __name__ == "__main__":
    test_burst_across_racks_collapses_to_one_incident()
    test_severity_escalation_merges_warning_and_alert()
    test_two_bursts_60s_apart_stay_separate()
    test_empty_list_returns_empty_list()
    test_malformed_timestamp_becomes_its_own_incident_without_crashing()
    test_input_alerts_are_not_mutated()
    test_children_are_the_original_row_objects()
    print("All alert_grouping tests passed.")
