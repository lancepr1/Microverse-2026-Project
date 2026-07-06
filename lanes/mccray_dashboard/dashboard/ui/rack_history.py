"""
ui/rack_history.py — a "look back" table for the rack card, sourced from
history_store.py's SQLite log. This is a snapshot view, not another live
stream: it queries once when the History panel is opened, not on every
poll tick.
"""
from datetime import datetime

from dash import html, callback, Input, Output, State

import history_store

RACK_ID = "Rack_01"

COLOR_LABEL = "rgb(90, 90, 90)"
COLOR_TEXT  = "rgb(30, 30, 30)"


def render_history_table(rows: list[dict]) -> html.Div:
    if not rows:
        return html.Div("No history recorded yet.", className="dimmed-block")

    header = html.Tr([
        html.Th("Time"), html.Th("FRQ"), html.Th("Power"),
        html.Th("Average GPU Temp"), html.Th("Status"),
    ])
    body_rows = [
        html.Tr([
            html.Td(datetime.fromtimestamp(row["captured_at"]).strftime("%H:%M:%S")),
            html.Td(f"{row['frq_hz']:.2f} Hz" if row["frq_hz"] is not None else "--"),
            html.Td(f"{row['total_power_w']:.1f} W" if row["total_power_w"] is not None else "--"),
            html.Td(f"{row['average_gpu_temp_c']:.1f} °C" if row["average_gpu_temp_c"] is not None else "--"),
            html.Td(str(row["status"])),
        ])
        for row in rows
    ]

    return html.Div(className="card", children=[
        html.Table([html.Thead(header), html.Tbody(body_rows)],
                   style={"color": COLOR_LABEL, "width": "100%"}),
    ])


@callback(
    Output("rack-history-open-store", "data"),
    Output("rack-history-container", "children"),
    Output("rack-history-btn", "children"),
    Input("rack-history-btn", "n_clicks"),
    State("rack-history-open-store", "data"),
    prevent_initial_call=True,
)
def _on_history_toggle(_n_clicks, is_open):
    is_open = not is_open
    if not is_open:
        return is_open, None, "History"

    rows = history_store.get_recent(RACK_ID)
    return is_open, render_history_table(rows), "Hide History"
