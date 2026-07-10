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

The 16 node-card buttons and 4 rack cards are built ONCE, in
build_operator_tab(), and never replaced afterward. Earlier this module
rebuilt the whole grid's `children` on every poll tick, which re-created
every html.Button with a hardcoded n_clicks=0 -- Dash's pattern-matching
click Input then saw that as a real click event on every tick, repeatedly
overwriting whatever node you'd actually selected back to the first one in
the grid. Instead, each node/rack card's changing bits (power, temp,
badge, border color) have their own ids and are updated in place via
MATCH-pattern callbacks below, so the buttons themselves -- and their
client-side n_clicks/selection state -- are never touched by a data tick.
"""
from dash import html, dcc, callback, Input, Output, State, ALL, MATCH, ctx, no_update

from data_feed import list_node_ids, poll_all, node_display_label

RACK_SIZE = 4

COLOR_TEXT    = "#0f172a"
COLOR_LABEL   = "#64748b"
COLOR_DEFAULT = "#e2e8f0"
COLOR_NOMINAL = "#16a34a"
COLOR_WARNING = "#d97706"
COLOR_ALERT   = "#dc2626"

_STATUS_TO_LABEL  = {"good": "Nominal", "suspect": "Warning", "warning": "Alert"}
_STATUS_TO_BADGE  = {"good": "NOM", "suspect": "WRN", "warning": "ALR"}
_STATUS_TO_BORDER = {"good": COLOR_NOMINAL, "suspect": COLOR_WARNING, "warning": COLOR_ALERT}
_STATUS_RANK      = {"good": 0, "suspect": 1, "warning": 2}

# Badge background/border/text share one status hue (move 4); only the
# alert badge blinks -- motion is reserved for what needs eyes.
_STATUS_BADGE_STYLE = {
    "good": {
        "color": "var(--status-nominal-text)",
        "backgroundColor": "var(--status-nominal-bg)",
        "border": "1px solid var(--status-nominal-border)",
    },
    "suspect": {
        "color": "var(--status-warning-text)",
        "backgroundColor": "var(--status-warning-bg)",
        "border": "1px solid var(--status-warning-border)",
    },
    "warning": {
        "color": "var(--status-alert-text)",
        "backgroundColor": "var(--status-alert-bg)",
        "border": "1px solid var(--status-alert-border)",
        "animation": "badge-blink 1.2s ease-in-out infinite",
    },
}
_DEFAULT_BADGE_STYLE = {
    "color": "var(--status-default-text)",
    "backgroundColor": "var(--status-default-bg)",
    "border": "1px solid var(--status-default-border)",
}


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
            html.Div(className="rack-grid", children=[
                _build_rack_card(rack_label, node_ids)
                for rack_label, node_ids in get_rack_groups()
            ]),
            html.Div(className="panel operator-detail-panel", children=[
                html.Div("Node Inspector", className="section-title"),
                html.Hr(),
                html.Div(id="operator-detail-container",
                          children=render_operator_detail(None, {})),
            ]),
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
        html.Div(value, className="stat-value", style={"color": color}),
    ])


def _build_rack_card(rack_label, node_ids):
    return html.Div(
        id={"type": "rack-card", "rack": rack_label},
        className="card rack-card",
        style={"borderColor": COLOR_DEFAULT},
        children=[
            html.Div(className="row", children=[
                html.Span(rack_label, style={"color": COLOR_TEXT, "fontWeight": "700"}),
                html.Span("-- W", id={"type": "rack-total-power", "rack": rack_label},
                          className="mono-value", style={"marginLeft": "auto"}),
            ]),
            html.Hr(),
            html.Div(className="node-grid", children=[
                _build_node_card(nid) for nid in node_ids
            ]),
        ],
    )


def _build_node_card(node_id):
    return html.Button(
        id={"type": "node-card-btn", "node_id": node_id},
        n_clicks=0,
        className="node-card",
        style={"borderLeftColor": COLOR_DEFAULT},
        children=[
            html.Div(node_display_label(node_id), className="node-card-id"),
            html.Div("-- W", id={"type": "node-card-power", "node_id": node_id},
                      className="mono-value"),
            html.Div("-- °C", id={"type": "node-card-temp", "node_id": node_id},
                      className="mono-value"),
            html.Div("--", id={"type": "node-card-badge", "node_id": node_id},
                      className="node-card-badge"),
        ],
    )


def _worst_status(statuses):
    ranked = [s for s in statuses if s in _STATUS_RANK]
    return max(ranked, key=lambda s: _STATUS_RANK[s]) if ranked else "--"


def render_operator_detail(selected_node: str | None, state: dict) -> html.Div:
    if not selected_node:
        return html.Div("Select a node to view details.", className="dimmed-block")

    data   = state.get(selected_node, {})
    status = data.get("status", "--")

    power = data.get("total_power_w")
    temp  = data.get("average_gpu_temp_c")

    power_text = f"{power:.1f} W" if power is not None else "-- W"
    temp_text  = f"{temp:.1f} °C" if temp is not None else "-- °C"

    return html.Div(children=[
        html.Div(node_display_label(selected_node), style={"color": COLOR_TEXT, "fontWeight": "700"}),
        html.Div(_rack_label_for(selected_node), className="label"),
        html.Hr(),

        _detail_row("Status", _STATUS_TO_LABEL.get(status, "--"),
                    _STATUS_TO_BORDER.get(status, COLOR_LABEL)),

        html.Div("Current Readings", className="section-title"),
        html.Hr(),
        _detail_row("Power Draw", power_text),
        _detail_row("GPU Temp", temp_text),

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

        html.Div("AI Explanation", className="section-title ai-label"),
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
        html.Span(value, className="mono-value", style={"color": color}),
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
        html.Span(label, className="mono-value"),
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
    Output({"type": "node-card-power", "node_id": MATCH}, "children"),
    Output({"type": "node-card-temp", "node_id": MATCH}, "children"),
    Output({"type": "node-card-badge", "node_id": MATCH}, "children"),
    Output({"type": "node-card-badge", "node_id": MATCH}, "style"),
    Output({"type": "node-card-btn", "node_id": MATCH}, "style"),
    Input("operator-state-store", "data"),
    State({"type": "node-card-btn", "node_id": MATCH}, "id"),
)
def _on_node_card_data_update(state, btn_id):
    data = (state or {}).get(btn_id["node_id"], {})

    status = data.get("status", "--")
    power  = data.get("total_power_w")
    temp   = data.get("average_gpu_temp_c")

    power_text    = f"{power:.1f} W" if power is not None else "-- W"
    temp_text     = f"{temp:.1f} °C" if temp is not None else "-- °C"
    forecast_text = _STATUS_TO_BADGE.get(status, "--")
    border_color  = _STATUS_TO_BORDER.get(status, COLOR_DEFAULT)
    badge_style   = _STATUS_BADGE_STYLE.get(status, _DEFAULT_BADGE_STYLE)

    return (
        power_text,
        temp_text,
        forecast_text,
        badge_style,
        {"borderLeftColor": border_color},
    )


@callback(
    Output({"type": "node-card-btn", "node_id": MATCH}, "className"),
    Input("selected-node-store", "data"),
    State({"type": "node-card-btn", "node_id": MATCH}, "id"),
)
def _on_node_selection_change(selected_node, btn_id):
    is_selected = btn_id["node_id"] == selected_node
    return "node-card" + (" node-card-selected" if is_selected else "")


@callback(
    Output({"type": "rack-card", "rack": MATCH}, "style"),
    Output({"type": "rack-total-power", "rack": MATCH}, "children"),
    Input("operator-state-store", "data"),
    State({"type": "rack-card", "rack": MATCH}, "id"),
)
def _on_rack_card_data_update(state, rack_id):
    node_ids = dict(get_rack_groups()).get(rack_id["rack"], [])
    state = state or {}
    node_states  = [state.get(nid, {}) for nid in node_ids]
    worst_status = _worst_status(d.get("status", "--") for d in node_states)
    total_power  = sum(d.get("total_power_w") or 0.0 for d in node_states)

    return (
        {"borderColor": _STATUS_TO_BORDER.get(worst_status, COLOR_DEFAULT)},
        f"{total_power:.1f} W",
    )


@callback(
    Output("operator-summary-bar", "children"),
    Output("operator-detail-container", "children"),
    Input("operator-state-store", "data"),
    Input("selected-node-store", "data"),
)
def _on_operator_render(state, selected_node):
    state = state or {}
    return (
        render_summary_cards(state),
        render_operator_detail(selected_node, state),
    )
