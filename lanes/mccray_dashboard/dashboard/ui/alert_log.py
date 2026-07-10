"""
ui/alert_log.py — the Alert log tab: a chronological alert timeline
(Section 1) and a 16-node anomaly summary table (Section 2), both sourced
from alert_log.py's session bookkeeping. See that module's docstring for
why alert_type and prediction are placeholders rather than live-computed
values.

The single render callback below is driven by operator-state-store (the
same multi-node poll data ui/operator.py's Operator tab uses), not by
poll-interval directly -- data_feed.poll_all() advances a shared replay
cursor, so it must only ever be called once per tick (already done in
ui/operator.py's poll callback); this module just reacts to that store's
value changing instead of polling a second time.
"""
from datetime import datetime

from dash import html, callback, Input, Output, no_update

import alert_log
from data_feed import list_node_ids
from ui.operator import get_rack_groups


def build_alert_log_tab() -> html.Div:
    return html.Div(children=[
        html.Div("Alert History Timeline", className="panel-title"),
        html.Hr(),
        html.Div(id="alert-timeline-container", className="alert-log-scroll",
                  children=render_alert_timeline()),

        html.Div("16-Node Anomaly Log", className="panel-title",
                  style={"marginTop": "12px"}),
        html.Hr(),
        html.Div(id="anomaly-log-container", className="alert-log-scroll",
                  children=render_anomaly_log()),
    ])


def render_alert_timeline() -> html.Div:
    rows = alert_log.get_timeline()
    if not rows:
        return html.Div("No alerts recorded yet this session.", className="dimmed-block")

    header = html.Tr([
        html.Th("Time"), html.Th("Rack"), html.Th("Node"), html.Th("Type"),
        html.Th("Score"), html.Th("Power"), html.Th("GPU Temp"),
        html.Th("Prediction"),
    ])
    body_rows = [
        html.Tr([
            html.Td(datetime.fromtimestamp(row["timestamp"]).strftime("%H:%M:%S")),
            html.Td(_rack_label_for(row["node_id"])),
            html.Td(row["node_id"]),
            html.Td(row["alert_type"]),
            html.Td(f"{row['score']:.2f}" if row["score"] is not None else "--"),
            html.Td(f"{row['total_power_w']:.1f} W" if row["total_power_w"] is not None else "--"),
            html.Td(f"{row['average_gpu_temp_c']:.1f} °C" if row["average_gpu_temp_c"] is not None else "--"),
            html.Td(row["prediction"]),
        ])
        for row in rows
    ]
    return html.Table([html.Thead(header), html.Tbody(body_rows)], className="alert-table")


def render_anomaly_log() -> html.Table:
    node_ids = list_node_ids()
    stats = alert_log.get_node_stats()

    header = html.Tr([
        html.Th("Node"), html.Th("Rack"), html.Th("Current Score"),
        html.Th("Max Score"), html.Th("Alert Count"), html.Th("Last Alert"),
    ])
    body_rows = []
    for node_id in node_ids:
        s = stats.get(node_id, {})
        current  = s.get("current_score")
        max_score = s.get("max_score")
        last_at  = s.get("last_alert_at")

        body_rows.append(html.Tr([
            html.Td(node_id),
            html.Td(_rack_label_for(node_id)),
            html.Td(f"{current:.2f}" if current is not None else "--"),
            html.Td(f"{max_score:.2f}" if max_score is not None else "--"),
            html.Td(str(s.get("alert_count", 0))),
            html.Td(datetime.fromtimestamp(last_at).strftime("%H:%M:%S") if last_at else "--"),
        ]))

    return html.Table([html.Thead(header), html.Tbody(body_rows)], className="alert-table")


def _rack_label_for(node_id: str) -> str:
    for rack_label, node_ids in get_rack_groups():
        if node_id in node_ids:
            return rack_label
    return "--"


@callback(
    Output("alert-timeline-container", "children"),
    Output("anomaly-log-container", "children"),
    Input("operator-state-store", "data"),
)
def _on_alert_state_change(state):
    if not state:
        return no_update, no_update
    alert_log.record_poll(state)
    return render_alert_timeline(), render_anomaly_log()
