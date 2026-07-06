"""
ui/controls.py — the toolbar above the three-column body: chart time-range
buttons, the live/pause toggle, and the facility selector.

Dash callbacks are stateless per invocation, unlike DPG's button callbacks
which mutated module-level lists directly. The selected time range,
live/pause state, and facility now live in a dcc.Store so the browser can
tell the server which button was clicked; the callback below still routes
those clicks into ui.charts' module-level state (_current_range,
_auto_follow), which stays the single source of truth for the chart
callback exactly like before.
"""
from dash import html, dcc, callback, Input, Output, State, ALL, ctx, no_update

from ui import charts

TIME_RANGES = ["10 Sec", "45 Sec", "90 Sec", "1 Hour", "6 Hours", "24 Hours", "All"]
FACILITIES  = ["Colocation", "Inference"]

DEFAULT_CONTROLS = {
    "time_range": "10 Sec",
    "auto_follow": True,
    "facility": "Inference",
}


def build_controls() -> html.Div:
    return html.Div(className="controls-bar", children=[
        dcc.Store(id="controls-store", data=dict(DEFAULT_CONTROLS)),

        html.Span("Time Range:", className="label"),
        html.Div(className="btn-group", children=[
            html.Button(
                label,
                id={"type": "time-range-btn", "label": label},
                n_clicks=0,
                className="btn"
            )
            for label in TIME_RANGES
        ]),

        html.Button("Live", id="live-toggle-btn", n_clicks=0, className="btn"),

        html.Span("Facility:", className="label"),
        html.Div(className="btn-group", children=[
            html.Button(
                label,
                id={"type": "facility-btn", "label": label},
                n_clicks=0,
                className="btn"
            )
            for label in FACILITIES
        ]),
    ])


@callback(
    Output("controls-store", "data"),
    Output("live-toggle-btn", "children"),
    Input({"type": "time-range-btn", "label": ALL}, "n_clicks"),
    Input("live-toggle-btn", "n_clicks"),
    Input({"type": "facility-btn", "label": ALL}, "n_clicks"),
    State("controls-store", "data"),
    prevent_initial_call=True,
)
def _on_control_click(_time_clicks, _live_clicks, _facility_clicks, store):
    store = dict(store or DEFAULT_CONTROLS)
    triggered = ctx.triggered_id

    if triggered == "live-toggle-btn":
        store["auto_follow"] = not store.get("auto_follow", True)
    elif isinstance(triggered, dict) and triggered.get("type") == "time-range-btn":
        store["time_range"] = triggered["label"]
        store["auto_follow"] = True
    elif isinstance(triggered, dict) and triggered.get("type") == "facility-btn":
        store["facility"] = triggered["label"]

    charts.set_time_range(store["time_range"])
    charts.set_auto_follow(store["auto_follow"])

    return store, ("Live" if store["auto_follow"] else "Paused")


@callback(
    Output("chart-pause-btn", "children"),
    Input("chart-pause-store", "data"),
    prevent_initial_call=True,
)
def _on_chart_pause(data):
    action = (data or {}).get("action")
    if action == "pause":
        charts.set_paused(True)
        return "▶"
    if action == "resume":
        charts.set_paused(False)
        return "⏸"
    return no_update
