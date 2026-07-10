"""
main.py — Dash entry point.

Builds the layout, starts the telemetry replay, and wires the single
polling callback that drives the header FRQ text once per tick. ui.controls,
ui.blender_feed, ui.tabs, ui.operator, ui.analyst, and ui.alert_log
register their own smaller callbacks as a side effect of being imported.
"""
from dash import Dash, Output, Input, no_update

from ui.layout import build_layout, render_header_frq
import ui.controls   # noqa: F401 -- registers the controls-store callback
import ui.blender_feed  # noqa: F401 -- registers the Blender image callback
import ui.tabs          # noqa: F401 -- registers the tab-switch callbacks
import ui.operator      # noqa: F401 -- registers the Operator tab callbacks
import ui.analyst       # noqa: F401 -- registers the Analyst tab rack chart callback
import ui.alert_log     # noqa: F401 -- registers the Alert log tab callback
from data_feed import init_feed, init_multi_feed, poll, get_rack_id
from history_store import init_history_db, record_sample

app = Dash(__name__, title="AFRL Microverse — Data Center Dashboard")
init_feed()
init_multi_feed()
init_history_db()
app.layout = build_layout()


@app.callback(
    Output("header-frq-hz", "children"),
    Input("poll-interval", "n_intervals"),
)
def on_tick(_n):
    state = poll()
    if state:
        record_sample(get_rack_id(), state[get_rack_id()])

    return render_header_frq(state) if state else no_update


if __name__ == "__main__":
    app.run(debug=True)
