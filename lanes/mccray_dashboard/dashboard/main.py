"""
main.py — Dash entry point.

Builds the layout, starts the telemetry replay, and wires the header's
polling callback (FRQ text + ENF status dot). ui.controls, ui.blender_feed,
ui.tabs, ui.operator, ui.analyst, and ui.alert_log_tab register their own
smaller callbacks as a side effect of being imported.

REMOVED (2026-07, cleanup pass): the single-node data_feed.py feed
(init_feed()/poll()/get_rack_id()) and its SQLite history logging
(history_store.py/record_sample()) are both gone. Confirmed with Leiva
that nothing should read data/run01.jsonl going forward, and that was the
only thing feeding this logging -- removing the dead code means the
SQLite logging feature is gone too, not just the demo file it was reading.
If persistent logging of the REAL multi-node stream is wanted later,
that's a new feature to design, not a restoration of this one.
"""
from dash import Dash, Output, Input, no_update

from ui.layout import (
    build_layout,
    render_header_frq_multi,
    render_header_frq_dot_style,
    render_header_frq_bubble_style,
)
import ui.controls   # noqa: F401 -- registers the controls-store callback
import ui.blender_feed  # noqa: F401 -- registers the Blender image callback
import ui.tabs          # noqa: F401 -- registers the tab-switch callbacks
import ui.operator      # noqa: F401 -- registers the Operator tab callbacks
import ui.analyst       # noqa: F401 -- registers the Analyst tab rack chart callback
import ui.alert_log_tab # noqa: F401 -- registers the Alert log tab callback
from data_feed import init_multi_feed

app = Dash(__name__, title="Data Center Dashboard")
init_multi_feed()
app.layout = build_layout()


# The header's FRQ text, ENF status dot, and the whole frq-readout
# bubble around them all read operator-state-store directly -- NOT
# poll_all() itself, since ui/operator.py's own callback already owns
# that (poll_all() advances a shared replay cursor and must only be
# called once per tick). This callback just reads the store's
# already-current value, the same pattern ui/alert_log_tab.py and
# ui/analyst.py already use.
#
# ADDED (2026-07, follow-up): a third Output, "header-frq-bubble".style
# -- the dot alone wasn't prominent enough; the whole pill/bubble around
# the FRQ reading now gets the same ENF-driven coloring.
@app.callback(
    Output("header-frq-hz", "children"),
    Output("header-frq-dot", "style"),
    Output("header-frq-bubble", "style"),
    Input("operator-state-store", "data"),
)
def on_state_change(state):
    if not state:
        return no_update, no_update, no_update

    return (
        render_header_frq_multi(state),
        render_header_frq_dot_style(state),
        render_header_frq_bubble_style(state),
    )


if __name__ == "__main__":
    app.run(debug=True)