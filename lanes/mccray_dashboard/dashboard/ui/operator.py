"""
ui/operator.py — the Operator tab: a summary bar, a 4-rack x 4-node grid,
and a detail panel for whichever node was last clicked.

Rack grouping is derived, not hardcoded: data_feed.list_node_ids() returns
every node hostname sorted, and get_rack_groups() chunks that sorted list
into groups of RACK_SIZE, labeled "Rack 1".."Rack 4" in that order. There is
no rack field in the recordings themselves yet (see data/README.md) --
this grouping is a placeholder subject to change once Leiva/the facility
team finalizes real rack assignments.

Node-card/rack-card border colors and the detail panel's status badge are
driven by Leiva's real-time verification status (good/suspect/warning),
the only live per-node state signal that exists in this repo today. The
node card's "forecast" badge reuses that same status as an interim stand-in
(good->Nominal, suspect->Warning, warning->Alert) -- there is no RNN
forecast model in this repo, so no "Breach ~Xs" countdown is fabricated.
The detail panel's 30-Second Forecast, Weakpoint RNN, and AI Explanation
sections are all labeled placeholders for the same reason: none of that
model inference exists yet, and building it is out of scope here.
"""
from dash import html, dcc, callback, Input, Output, ALL, ctx, no_update

from data_feed import list_node_ids, poll_all

RACK_SIZE = 4

COLOR_TEXT    = "rgb(30, 30, 30)"
COLOR_LABEL   = "rgb(90, 90, 90)"
COLOR_DEFAULT = "rgb(210, 210, 210)"
COLOR_NOMINAL = "rgb(0, 150, 70)"
COLOR_WARNING = "rgb(200, 150, 0)"
COLOR_ALERT   = "rgb(200, 50, 50)"

_STATUS_TO_LABEL  = {"good": "Nominal", "suspect": "Warning", "warning": "Alert"}
_STATUS_TO_BORDER = {"good": COLOR_NOMINAL, "suspect": COLOR_WARNING, "warning": COLOR_ALERT}
_STATUS_RANK      = {"good": 0, "suspect": 1, "warning": 2}


def get_rack_groups() -> list[tuple[str, list[str]]]:
    node_ids = list_node_ids()
    return [
        (f"Rack {i // RACK_SIZE + 1}", node_ids[i:i + RACK_SIZE])
        for i in range(0, len(node_ids), RACK_SIZE)
    ]


def build_operator_tab() -> html.Div:
    return html.Div(children=[
        dcc.Store(id="operator-state-store", data={}),
        dcc.Store(id="selected-node-store", data=None),

        html.Div(id="operator-summary-bar", className="row",
                  children=render_summary_cards({})),

        html.Div(className="operator-body", children=[
            html.Div(id="rack-grid-container", className="rack-grid",
                      children=render_rack_grid({}, None)),
            html.Div(id="operator-detail-container",
                      className="panel operator-detail-panel",
                      children=render_operator_detail(None, {})),
        ]),
    ])


def render_summary_cards(state: dict) -> list:
    total   = len(list_node_ids())
    nominal = sum(1 for d in state.values() if d.get("status") == "good")
    warning = sum(1 for d in state.values() if d.get("status") == "suspect")
    alert   = sum(1 for d in state.values() if d.get("status") == "warning")

    return [
        _summary_card("Total Nodes", str(total)),
        _summary_card("Nominal", str(nominal), COLOR_NOMINAL),
        _summary_card("Warning", str(warning), COLOR_WARNING),
        _summary_card("Alert", str(alert), COLOR_ALERT),
    ]


def _summary_card(label, value, color=COLOR_LABEL):
    return html.Div(className="card kpi-card", children=[
        html.Div(label, className="label"),
        html.Div(value, style={"color": color}),
    ])


def render_rack_grid(state: dict, selected_node: str | None) -> list:
    return [
        _render_rack_card(rack_label, node_ids, state, selected_node)
        for rack_label, node_ids in get_rack_groups()
    ]


def _render_rack_card(rack_label, node_ids, state, selected_node):
    node_states  = [state.get(nid, {}) for nid in node_ids]
    worst_status = _worst_status(d.get("status", "--") for d in node_states)
    total_power  = sum(d.get("total_power_w") or 0.0 for d in node_states)

    return html.Div(
        className="card rack-card",
        style={"borderColor": _STATUS_TO_BORDER.get(worst_status, COLOR_DEFAULT)},
        children=[
            html.Div(className="row", children=[
                html.Span(rack_label, style={"color": COLOR_TEXT, "fontWeight": "700"}),
                html.Span(f"{total_power:.1f} W", className="label",
                          style={"marginLeft": "auto"}),
            ]),
            html.Hr(),
            html.Div(className="node-grid", children=[
                _render_node_card(nid, state.get(nid, {}), nid == selected_node)
                for nid in node_ids
            ]),
        ],
    )


def _worst_status(statuses):
    ranked = [s for s in statuses if s in _STATUS_RANK]
    return max(ranked, key=lambda s: _STATUS_RANK[s]) if ranked else "--"


def _render_node_card(node_id, data, is_selected):
    status  = data.get("status", "--")
    power   = data.get("total_power_w")
    temp    = data.get("average_gpu_temp_c")

    power_text    = f"{power:.1f} W" if power is not None else "-- W"
    temp_text     = f"{temp:.1f} °C" if temp is not None else "-- °C"
    forecast_text = _STATUS_TO_LABEL.get(status, "--")
    border_color  = _STATUS_TO_BORDER.get(status, COLOR_DEFAULT)

    class_name = "node-card" + (" node-card-selected" if is_selected else "")

    return html.Button(
        id={"type": "node-card-btn", "node_id": node_id},
        n_clicks=0,
        className=class_name,
        style={"borderTopColor": border_color},
        children=[
            html.Div(node_id, className="node-card-id"),
            html.Div(power_text, className="label"),
            html.Div(temp_text, className="label"),
            html.Div(forecast_text, className="node-card-badge",
                      style={"color": border_color}),
        ],
    )


def render_operator_detail(selected_node: str | None, state: dict) -> html.Div:
    if not selected_node:
        return html.Div("Select a node to view details.", className="dimmed-block")

    data   = state.get(selected_node, {})
    status = data.get("status", "--")

    frq   = data.get("frq_hz")
    power = data.get("total_power_w")
    temp  = data.get("average_gpu_temp_c")

    frq_text   = f"{frq:.2f} Hz" if frq is not None else "-- Hz"
    power_text = f"{power:.1f} W" if power is not None else "-- W"
    temp_text  = f"{temp:.1f} °C" if temp is not None else "-- °C"

    return html.Div(children=[
        html.Div(selected_node, style={"color": COLOR_TEXT, "fontWeight": "700"}),
        html.Div(_rack_label_for(selected_node), className="label"),
        html.Hr(),

        _detail_row("Status", _STATUS_TO_LABEL.get(status, "--"),
                    _STATUS_TO_BORDER.get(status, COLOR_LABEL)),

        html.Div("Current Readings", className="section-title"),
        html.Hr(),
        _detail_row("Power Draw", power_text),
        _detail_row("GPU Temp", temp_text),
        _detail_row("Frequency", frq_text),

        html.Div("30-Second Forecast", className="section-title"),
        html.Hr(),
        _detail_row("Predicted Power", "--"),
        _detail_row("Predicted State", "--"),
        _anomaly_score_bar(None),
        html.Div("Pending model integration", className="dimmed-block"),

        html.Div("Weakpoint RNN", className="section-title"),
        html.Hr(),
        _detail_row("Next Event Type", "--"),
        _detail_row("Time to Next Event", "--"),

        html.Div("AI Explanation", className="section-title"),
        html.Hr(),
        html.Div("AI explanation — pending model integration",
                  className="dimmed-block ai-explanation-box"),
    ])


def _rack_label_for(node_id):
    for rack_label, node_ids in get_rack_groups():
        if node_id in node_ids:
            return rack_label
    return "--"


def _detail_row(label, value, color=COLOR_TEXT):
    return html.Div(className="row", children=[
        html.Span(f"{label}:", className="label"),
        html.Span(value, style={"color": color}),
    ])


def _anomaly_score_bar(score):
    """Placeholder fill bar -- no anomaly-scoring model exists yet, so this
    always renders empty/"--" until a later session wires it up. Thresholds
    match the spec (green <0.5, amber 0.5-1.0, red >1.0) for when it is."""
    pct, color, label = 0, COLOR_DEFAULT, "--"
    if score is not None:
        label = f"{score:.2f}"
        color = COLOR_NOMINAL if score < 0.5 else COLOR_WARNING if score <= 1.0 else COLOR_ALERT
        pct = min(score / 1.5, 1.0) * 100

    return html.Div(className="row", children=[
        html.Span("Anomaly Score:", className="label"),
        html.Div(className="score-bar-track", children=[
            html.Div(className="score-bar-fill", style={"width": f"{pct}%", "background": color}),
        ]),
        html.Span(label),
    ])


@callback(
    Output("operator-state-store", "data"),
    Input("poll-interval", "n_intervals"),
)
def _on_operator_poll(_n):
    state = poll_all()
    return state if state else no_update


@callback(
    Output("selected-node-store", "data"),
    Input({"type": "node-card-btn", "node_id": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def _on_node_click(_n_clicks):
    triggered = ctx.triggered_id
    if isinstance(triggered, dict) and triggered.get("type") == "node-card-btn":
        return triggered["node_id"]
    return no_update


@callback(
    Output("rack-grid-container", "children"),
    Output("operator-summary-bar", "children"),
    Output("operator-detail-container", "children"),
    Input("operator-state-store", "data"),
    Input("selected-node-store", "data"),
)
def _on_operator_render(state, selected_node):
    state = state or {}
    return (
        render_rack_grid(state, selected_node),
        render_summary_cards(state),
        render_operator_detail(selected_node, state),
    )
