"""
Import-only sanity checks: every dashboard module should import cleanly
with no external package on the path (these modules only define
functions/components at module scope, so import alone is a safe check —
no running Dash server required).
"""
import sys


def test_models_imports_with_no_external_path():
    assert "microverse_core" not in sys.modules
    import models
    assert hasattr(models, "TelemetrySample")


def test_data_feed_imports_and_exposes_poll():
    import data_feed
    assert callable(data_feed.init_feed)
    assert callable(data_feed.poll)


def test_ui_charts_imports():
    from ui.charts import update_node_history, build_rack_power_figure, get_node_history_lengths
    assert callable(update_node_history)
    assert callable(get_node_history_lengths)
    fig = build_rack_power_figure(["node-a", "node-b"])
    assert len(fig.data) == 2


def test_ui_controls_imports():
    from ui.controls import build_chart_controls, build_facility_control, DEFAULT_CONTROLS
    assert DEFAULT_CONTROLS["time_range"] == "10 Sec"
    assert DEFAULT_CONTROLS["facility"] == "Inference"
    assert build_chart_controls() is not None
    assert build_facility_control() is not None


def test_no_microverse_core_dependency_anywhere():
    """Guards against re-introducing the external repo coupling that made
    this dashboard non-transferable."""
    import os
    import re

    dashboard_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "dashboard")

    offenders = []
    for root, _, files in os.walk(dashboard_dir):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path) as f:
                content = f.read()
            if "microverse_core" in content or re.search(r"sys\.path\.insert", content):
                offenders.append(path)

    assert offenders == []
