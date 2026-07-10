"""
ui/grouped_alert_table.py -- renders alert_grouping.group_alerts() output
as a collapsible incident list for the Alert log tab's Grouped view (see
ui/alert_log_tab.py for where this is wired in behind the Grouped/Flat
toggle, and alert_grouping.py for the clustering logic itself).

Uses html.Details/html.Summary for expand/collapse -- zero-callback, so it
cannot conflict with any of the Alert log tab's existing pattern-matching
callbacks. A single-child incident (nothing to collapse) renders as a
plain row instead of a <details>.

Severity badge colors are replicated from ui/alert_log_tab.py's
_SEVERITY_BADGE_STYLE (same --status-warning-*/--status-alert-* CSS custom
properties -- not a new palette), rather than imported, since
alert_log_tab.py imports this module to build the grouped view and an
import the other way would be circular.
"""
from datetime import datetime

from dash import html

_SEVERITY_BADGE_STYLE = {
    "Warning": {
        "color": "var(--status-warning-text)",
        "backgroundColor": "var(--status-warning-bg)",
        "border": "1px solid var(--status-warning-border)",
    },
    "Alert": {
        "color": "var(--status-alert-text)",
        "backgroundColor": "var(--status-alert-bg)",
        "border": "1px solid var(--status-alert-border)",
    },
}


def render_grouped_alert_table(incidents: list) -> html.Div:
    if not incidents:
        return html.Div("No alerts recorded yet this session.", className="dimmed-block")

    return html.Div(className="grp-incident-list", children=[
        _render_incident(incident) for incident in incidents
    ])


def _fmt_time(ts) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S") if isinstance(ts, (int, float)) else "--"


def _badge(severity: str) -> html.Span:
    return html.Span(severity, className="node-card-badge",
                      style=_SEVERITY_BADGE_STYLE.get(severity, {}))


def _summary_cells(incident) -> list:
    time_range = f"{_fmt_time(incident.start_ts)} → " + (
        "ongoing" if incident.ongoing else _fmt_time(incident.end_ts))
    power = f"{incident.peak_power_w:.1f} W" if incident.peak_power_w is not None else "--"
    temp = f"{incident.peak_gpu_temp_c:.1f} °C" if incident.peak_gpu_temp_c is not None else "--"

    vectors = ", ".join(incident.attack_vectors) if incident.attack_vectors else "Unclassified"

    return [
        _badge(incident.severity),
        html.Span(f"{incident.node_count} node(s) · {len(incident.racks)} rack(s)",
                   className="grp-cell mono-value"),
        html.Span(vectors, className="grp-cell mono-value"),
        html.Span(time_range, className="grp-cell mono-value"),
        html.Span(power, className="grp-cell mono-value"),
        html.Span(temp, className="grp-cell mono-value"),
        html.Span(f"{incident.total_samples} samples", className="grp-cell mono-value"),
    ]


def _render_incident(incident) -> html.Div:
    if len(incident.children) <= 1:
        return html.Div(className="grp-incident-row", children=_summary_cells(incident))

    return html.Details(className="grp-incident", children=[
        html.Summary(className="grp-incident-summary", children=_summary_cells(incident)),
        html.Div(className="grp-incident-children", children=[
            _render_child_row(child) for child in incident.children
        ]),
    ])


def _render_child_row(child: dict) -> html.Div:
    end = child.get("end_ts")
    time_range = f"{_fmt_time(child.get('start_ts'))} → " + (
        "ongoing" if child.get("ongoing") else _fmt_time(end))
    power = child.get("total_power_w")
    temp = child.get("average_gpu_temp_c")

    return html.Div(className="grp-child-row", children=[
        html.Span(child.get("rack", "--"), className="grp-cell mono-value"),
        html.Span(child.get("node_id", "--"), className="grp-cell mono-value"),
        _badge(child.get("severity", "--")),
        html.Span(child.get("attack_vector", "Unclassified"), className="grp-cell mono-value"),
        html.Span(time_range, className="grp-cell mono-value"),
        html.Span(f"{power:.1f} W" if isinstance(power, (int, float)) else "--", className="grp-cell mono-value"),
        html.Span(f"{temp:.1f} °C" if isinstance(temp, (int, float)) else "--", className="grp-cell mono-value"),
    ])
