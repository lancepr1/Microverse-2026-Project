"""
ui/controls.py — chart time-range buttons and the live/pause toggle.

Dash callbacks are stateless per invocation, unlike DPG's button callbacks
which mutated module-level lists directly. The selected time range and
live/pause state now live in a dcc.Store so the browser can tell the server
which button was clicked; the callback below still routes those clicks into
ui.charts' module-level state (_current_range, _auto_follow), which stays
the single source of truth for the chart callback exactly like before.
"""
from dash import html, dcc, callback, Input, Output, State, ALL, ctx

from ui import charts

TIME_RANGES = ["10 Sec", "45 Sec", "90 Sec", "1 Hour", "6 Hours", "24 Hours", "All"]

DEFAULT_CONTROLS = {
    "time_range": "10 Sec",
    "auto_follow": True,
}


def build_chart_controls() -> html.Div:
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

        html.Button(
            "Follow: On", id="live-toggle-btn", n_clicks=0, className="btn",
            title="X-axis auto-follow. Off = the window stops sliding so you "
                  "can pan/zoom freely; data keeps flowing. To freeze the "
                  "charts entirely, use ⏸ Pause.",
        ),
    ])


@callback(
    Output("controls-store", "data"),
    Output("live-toggle-btn", "children"),
    Input({"type": "time-range-btn", "label": ALL}, "n_clicks"),
    Input("live-toggle-btn", "n_clicks"),
    State("controls-store", "data"),
    prevent_initial_call=True,
)
def _on_control_click(_time_clicks, _live_clicks, store):
    store = dict(store or DEFAULT_CONTROLS)
    triggered = ctx.triggered_id

    if triggered == "live-toggle-btn":
        store["auto_follow"] = not store.get("auto_follow", True)
    elif isinstance(triggered, dict) and triggered.get("type") == "time-range-btn":
        store["time_range"] = triggered["label"]
        store["auto_follow"] = True

    charts.set_time_range(store["time_range"])
    charts.set_auto_follow(store["auto_follow"])

    return store, ("Follow: On" if store["auto_follow"] else "Follow: Off")


# Note: "Follow: On/Off" above toggles auto_follow -- whether the x-axis
# keeps sliding with the selected Time Range window, or sits unwindowed so
# Plotly's own pan/zoom sticks. That's a different axis from
# ui/analyst.py's "⏸ Pause"/"▶ Resume" button, which freezes the whole
# Analyst tab (buffer growth and all eight figures) regardless of the
# Time Range or auto_follow state. It used to read "Live"/"Paused", which
# made it look like a second pause button next to ⏸ Pause -- it never
# paused data, only the sliding window.
