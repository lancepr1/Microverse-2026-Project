"""
ui/alert_log.py — the Alert log tab: a filterable alert-episode timeline
(Section 1) and a 16-node anomaly summary table (Section 2), both sourced
from alert_log.py's session bookkeeping. See that module's docstring for
what counts as an "alert" and how the attack-vector label is derived,
and for how consecutive same-node/same-severity samples collapse
into one "episode" row so a long-running alert doesn't bury the timeline
in duplicates.

Severity (All / Warning / Alert) and node filters are held in
alert-filter-store and applied in both render_alert_timeline() and
render_anomaly_log() -- narrowing either section to one node or severity is
what turns the tab from a firehose into something an operator can actually
read a pattern out of.

The single render callback below is driven by operator-state-store (the
same multi-node poll data ui/operator.py's Operator tab uses), not by
poll-interval directly -- data_feed.poll_all() advances a shared replay
cursor, so it must only ever be called once per tick (already done in
ui/operator.py's poll callback); this module just reacts to that store's
value changing instead of polling a second time.
"""
from datetime import datetime

from dash import html, dcc, callback, Input, Output, State, ALL, ctx, no_update

import alert_log
from alert_grouping import group_alerts
from data_feed import list_node_ids, node_display_label
from ui.operator import get_rack_groups
from ui.grouped_alert_table import render_grouped_alert_table

SEVERITY_FILTERS = ["All", "Warning", "Alert"]
DEFAULT_FILTERS = {"severity": "All", "node": "All"}
VIEW_MODES = ["Grouped", "Flat"]
DEFAULT_VIEW_MODE = "Grouped"

# Shares the same status hue tokens as ui/operator.py's node-card badges --
# Warning/Alert here are just the human-readable form of "suspect"/"warning".
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


def build_alert_log_tab() -> html.Div:
    return html.Div(children=[
        dcc.Store(id="alert-filter-store", data=dict(DEFAULT_FILTERS)),

        html.Div(className="controls-bar", children=[
            html.Span("Severity:", className="label"),
            html.Div(className="btn-group", children=[
                html.Button(
                    label,
                    id={"type": "alert-severity-filter-btn", "severity": label},
                    n_clicks=0,
                    className="btn" + (" btn-active" if label == "All" else ""),
                )
                for label in SEVERITY_FILTERS
            ]),
            html.Span("Node:", className="label", style={"marginLeft": "12px"}),
            dcc.Dropdown(
                id="alert-node-filter-dropdown",
                options=[{"label": "All nodes", "value": "All"}] + [
                    {"label": node_display_label(nid), "value": nid}
                    for nid in list_node_ids()
                ],
                value="All",
                clearable=False,
                searchable=False,
                style={"width": "160px", "fontSize": "12px"},
            ),
        ]),

        html.Div("Alert History Timeline", className="panel-title",
                  style={"marginTop": "10px"}),
        html.Hr(),
        dcc.RadioItems(
            id="grp-alert-view-mode",
            options=VIEW_MODES,
            value=DEFAULT_VIEW_MODE,
            inline=True,
            className="grp-view-toggle",
        ),
        # Existing flat table, untouched, just wrapped so the toggle can
        # hide/show it -- its id, data flow, and callback are unchanged.
        html.Div(id="grp-flat-timeline-wrapper",
                  style={"display": "none" if DEFAULT_VIEW_MODE == "Grouped" else "block"},
                  children=[
            html.Div(id="alert-timeline-container", className="alert-log-scroll",
                      children=render_alert_timeline()),
        ]),
        html.Div(id="grp-grouped-timeline-wrapper",
                  style={"display": "block" if DEFAULT_VIEW_MODE == "Grouped" else "none"},
                  children=[
            html.Div(id="grp-grouped-timeline-container", className="alert-log-scroll",
                      children=render_grouped_alert_table([])),
        ]),

        html.Div("Anomaly Log", className="panel-title",
                  style={"marginTop": "12px"}),
        html.Hr(),
        html.Div(id="anomaly-log-container", className="alert-log-scroll",
                  children=render_anomaly_log()),
    ])


def render_alert_timeline(severity: str = "All", node_id: str = "All") -> html.Div:
    rows = alert_log.get_timeline(
        severity=None if severity == "All" else severity,
        node_id=None if node_id == "All" else node_id,
    )
    if not rows:
        return html.Div("No alerts recorded yet this session.", className="dimmed-block")

    header = html.Tr([
        html.Th("Start"), html.Th("End"), html.Th("Rack"), html.Th("Node"),
        html.Th("Severity"), html.Th("Attack Vector"), html.Th("Samples"),
        html.Th("Power"), html.Th("GPU Temp"),
    ])
    body_rows = [
        html.Tr([
            html.Td(datetime.fromtimestamp(ep["start_ts"]).strftime("%H:%M:%S")),
            html.Td("ongoing" if ep["ongoing"]
                     else datetime.fromtimestamp(ep["end_ts"]).strftime("%H:%M:%S")),
            html.Td(_rack_label_for(ep["node_id"])),
            html.Td(ep["node_id"]),
            html.Td(ep["severity"], className="node-card-badge",
                     style=_SEVERITY_BADGE_STYLE.get(ep["severity"], {})),
            html.Td(ep.get("attack_vector", "Unclassified")),
            html.Td(str(ep["sample_count"])),
            html.Td(f"{ep['total_power_w']:.1f} W" if ep.get("total_power_w") is not None else "--"),
            html.Td(f"{ep['average_gpu_temp_c']:.1f} °C" if ep.get("average_gpu_temp_c") is not None else "--"),
        ])
        for ep in rows
    ]
    return html.Table([html.Thead(header), html.Tbody(body_rows)], className="alert-table")


def render_anomaly_log(severity: str = "All", node_id_filter: str = "All") -> html.Table:
    node_ids = list_node_ids()
    stats = alert_log.get_node_stats()

    if node_id_filter != "All":
        node_ids = [n for n in node_ids if n == node_id_filter]
    if severity != "All":
        node_ids = [n for n in node_ids if stats.get(n, {}).get("last_severity") == severity]

    header = html.Tr([
        html.Th("Node"), html.Th("Rack"), html.Th("Alert Count"),
        html.Th("Last Severity"), html.Th("Last Alert"),
    ])
    body_rows = []
    for node_id in node_ids:
        s = stats.get(node_id, {})
        last_at      = s.get("last_alert_at")
        last_severity = s.get("last_severity", "--")

        body_rows.append(html.Tr([
            html.Td(node_id),
            html.Td(_rack_label_for(node_id)),
            html.Td(str(s.get("alert_count", 0))),
            html.Td(last_severity, className="node-card-badge",
                     style=_SEVERITY_BADGE_STYLE.get(last_severity, {})),
            html.Td(datetime.fromtimestamp(last_at).strftime("%H:%M:%S") if last_at else "--"),
        ]))

    return html.Table([html.Thead(header), html.Tbody(body_rows)], className="alert-table")


def _rack_label_for(node_id: str) -> str:
    for rack_label, node_ids in get_rack_groups():
        if node_id in node_ids:
            return rack_label
    return "--"


@callback(
    Output("alert-filter-store", "data"),
    Input({"type": "alert-severity-filter-btn", "severity": ALL}, "n_clicks"),
    Input("alert-node-filter-dropdown", "value"),
    State("alert-filter-store", "data"),
    prevent_initial_call=True,
)
def _on_alert_filter_change(_n_clicks, node_value, store):
    store = dict(store or DEFAULT_FILTERS)
    triggered = ctx.triggered_id

    if isinstance(triggered, dict) and triggered.get("type") == "alert-severity-filter-btn":
        store["severity"] = triggered["severity"]
    elif triggered == "alert-node-filter-dropdown":
        store["node"] = node_value

    return store


@callback(
    *[Output({"type": "alert-severity-filter-btn", "severity": label}, "className")
      for label in SEVERITY_FILTERS],
    Input("alert-filter-store", "data"),
)
def _on_alert_severity_chip_style(store):
    active = (store or DEFAULT_FILTERS).get("severity", "All")
    return tuple(
        "btn" + (" btn-active" if label == active else "")
        for label in SEVERITY_FILTERS
    )


@callback(
    Output("alert-timeline-container", "children"),
    Output("anomaly-log-container", "children"),
    Input("operator-state-store", "data"),
    Input("alert-filter-store", "data"),
)
def _on_alert_state_change(state, filters):
    if state:
        alert_log.record_poll(state)
    elif not filters:
        return no_update, no_update

    filters = filters or DEFAULT_FILTERS
    severity, node = filters.get("severity", "All"), filters.get("node", "All")
    return (
        render_alert_timeline(severity, node),
        render_anomaly_log(severity, node),
    )


@callback(
    Output("grp-flat-timeline-wrapper", "style"),
    Output("grp-grouped-timeline-wrapper", "style"),
    Input("grp-alert-view-mode", "value"),
)
def _on_grp_view_mode_change(mode):
    flat_visible = mode == "Flat"
    return (
        {"display": "block" if flat_visible else "none"},
        {"display": "none" if flat_visible else "block"},
    )


@callback(
    Output("grp-grouped-timeline-container", "children"),
    # Fires off the flat table's own re-render, not off operator-state-store
    # directly -- alert_log.record_poll() must only run once per tick (see
    # _on_alert_state_change above), so this reads the same already-updated
    # session state instead of polling/recording a second time.
    Input("alert-timeline-container", "children"),
    Input("alert-filter-store", "data"),
)
def _on_grp_grouped_render(_flat_children, filters):
    filters = filters or DEFAULT_FILTERS
    severity, node = filters.get("severity", "All"), filters.get("node", "All")

    rows = alert_log.get_timeline(
        severity=None if severity == "All" else severity,
        node_id=None if node == "All" else node,
    )
    rows_with_rack = [dict(ep, rack=_rack_label_for(ep["node_id"])) for ep in rows]
    incidents = group_alerts(rows_with_rack)
    return render_grouped_alert_table(incidents)
