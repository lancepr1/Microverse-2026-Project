"""
main.py — Dash entry point.

Builds the layout, starts the telemetry replay, and wires the single
polling callback that drives both charts, the rack card, the detail panel,
and the header/KPI text once per tick. ui.controls, ui.blender_feed, and
ui.rack_history register their own smaller callbacks as a side effect of
being imported.
"""
from dash import Dash, Output, Input, no_update

from ui.layout import build_layout, render_header_and_kpis
from ui.charts import update_charts, get_figures
from ui.rack_cards import render_rack_card, render_detail_panel
import ui.controls      # noqa: F401 -- registers the controls-store callback
import ui.blender_feed  # noqa: F401 -- registers the Blender image callback
import ui.rack_history  # noqa: F401 -- registers the History toggle callback
import ui.tabs          # noqa: F401 -- registers the tab-switch callbacks
from data_feed import init_feed, poll, get_rack_id
from history_store import init_history_db, record_sample

app = Dash(__name__, title="AFRL Microverse — Data Center Dashboard")
init_feed()
init_history_db()
app.layout = build_layout()


@app.callback(
    Output("frq-graph", "figure"),
    Output("power-graph", "figure"),
    Output("rack-card-container", "children"),
    Output("detail-panel-container", "children"),
    Output("header-frq-hz", "children"),
    Output("kpi-frq", "children"),
    Output("kpi-power", "children"),
    Output("kpi-temp", "children"),
    Input("poll-interval", "n_intervals"),
)
def on_tick(_n):
    state = poll()
    if state:
        update_charts(state)
        record_sample(get_rack_id(), state[get_rack_id()])

    frq_fig, power_fig = get_figures()

    if state:
        rack_card = render_rack_card(state)
        detail_panel = render_detail_panel(state)
        header_frq, kpi_frq, kpi_power, kpi_temp = render_header_and_kpis(state)
    else:
        rack_card = detail_panel = no_update
        header_frq = kpi_frq = kpi_power = kpi_temp = no_update

    return (
        frq_fig, power_fig, rack_card, detail_panel,
        header_frq, kpi_frq, kpi_power, kpi_temp,
    )


if __name__ == "__main__":
    app.run(debug=True)
