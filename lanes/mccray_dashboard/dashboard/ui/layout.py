"""
ui/layout.py — builds the Dash page layout: header, tab bar (Operator /
Analyst / Alert log / Digital Twin, see ui/tabs.py), and the four tabs'
content, all mounted at once (tabs only toggle CSS display -- see
ui/tabs.py's docstring for why).

Below the tabs, the original single-node rack card / KPI panel / detail
panel (ui/rack_cards.py, driven by data_feed's single-node get_rack_id()
API) is left in place, still always visible regardless of which tab is
selected -- it predates the Operator tab's 16-node grid and wasn't
reassigned a home by the tab spec this was built against.
"""
from dash import html, dcc

from data_feed import get_rack_id
from ui.rack_cards import render_rack_card, render_detail_panel
from ui.charts import build_figures, FRQ_HEIGHT, POWER_HEIGHT
from ui.blender_feed import render_blender_feed
from ui.controls import build_chart_controls, build_facility_control
from ui.tabs import build_tab_bar, tab_content_style
from ui.operator import build_operator_tab
from ui.alert_log import build_alert_log_tab

POLL_INTERVAL_MS = 250

FRQ_HZ_DEFAULT  = "-- Hz"
POWER_W_DEFAULT = "-- W"
TEMP_C_DEFAULT  = "-- °C"

COLOR_LABEL = "rgb(90, 90, 90)"
COLOR_GREEN = "rgb(0, 150, 70)"


def build_layout() -> html.Div:
    rack_id = get_rack_id()
    frq_fig, power_fig = build_figures()

    return html.Div(className="app", children=[
        dcc.Interval(id="poll-interval", interval=POLL_INTERVAL_MS),

        html.Div(className="header", children=[
            html.Span("AFRL Microverse — Data Center Dashboard",
                      className="header-title"),
            html.Span("FRQ:", className="label"),
            html.Span(FRQ_HZ_DEFAULT, id="header-frq-hz",
                      style={"color": COLOR_GREEN}),
        ]),
        html.Hr(),

        build_tab_bar(),
        build_facility_control(),

        html.Div(className="tab-content-panels", children=[
            html.Div(
                build_operator_tab(),
                id="tab-content-operator",
                style=tab_content_style("operator"),
            ),
            html.Div(
                _build_analyst_tab(rack_id, frq_fig, power_fig),
                id="tab-content-analyst",
                style=tab_content_style("analyst"),
            ),
            html.Div(
                render_blender_feed(),
                id="tab-content-digital-twin",
                style=tab_content_style("digital-twin"),
            ),
            html.Div(
                build_alert_log_tab(),
                id="tab-content-alert-log",
                style=tab_content_style("alert-log"),
            ),
        ]),

        html.Div(className="body-columns", children=[

            # LEFT — Rack status
            html.Div(className="panel left-panel", children=[
                html.Div("Rack Status", className="panel-title"),
                html.Hr(),
                html.Div(id="rack-card-container",
                         children=render_rack_card({})),
                dcc.Store(id="rack-history-open-store", data=False),
                html.Button("History", id="rack-history-btn",
                             className="btn"),
                html.Div(id="rack-history-container"),
            ]),

            # RIGHT — KPI + detail
            html.Div(className="panel right-panel", children=[
                html.Div("Facility Metrics", className="panel-title"),
                html.Hr(),
                _kpi_card("Live FRQ", FRQ_HZ_DEFAULT, "kpi-frq"),
                _kpi_card("Total Power Draw", POWER_W_DEFAULT, "kpi-power"),
                _kpi_card("Average GPU Temp", TEMP_C_DEFAULT, "kpi-temp"),

                html.Div("Rack Inspection", className="panel-title"),
                html.Hr(),
                html.Div(id="detail-panel-container", className="detail-frame",
                         children=render_detail_panel({})),
            ]),
        ]),
    ])


def _build_analyst_tab(rack_id, frq_fig, power_fig) -> html.Div:
    return html.Div(children=[
        html.Div(
            "Signal analysis and historical data — select a node or "
            "time range to drill down",
            className="dimmed-block",
        ),
        build_chart_controls(),
        html.Div(className="panel center-panel", children=[
            html.Div("Real-Time Signal Monitoring",
                      className="panel-title"),
            html.Hr(),
            dcc.Store(id="chart-pause-store"),
            html.Div(className="chart-frame", children=[
                html.Button("⏸", id="chart-pause-btn",
                            className="chart-pause-btn"),
                html.Div(f"PDU Frequency — {rack_id}",
                          className="chart-label"),
                dcc.Graph(id="frq-graph", figure=frq_fig,
                          style={"height": f"{FRQ_HEIGHT}px"},
                          config={"displayModeBar": False}),
                html.Div(f"Power Consumption — {rack_id}",
                          className="chart-label"),
                dcc.Graph(id="power-graph", figure=power_fig,
                          style={"height": f"{POWER_HEIGHT}px"},
                          config={"displayModeBar": False}),
            ]),
        ]),
    ])


def _kpi_card(label, value, value_id):
    return html.Div(className="card kpi-card", children=[
        html.Div(label, className="label"),
        html.Div(value, id=value_id, style={"color": COLOR_LABEL}),
    ])


def render_header_and_kpis(state: dict):
    """Returns (header_frq, kpi_frq, kpi_power, kpi_temp) text for the
    Output tuple in main.py's polling callback. Assumes state[get_rack_id()]
    exists — callers should only invoke this when poll() returned data."""
    data = state[get_rack_id()]

    frq         = data.get("frq_hz")
    total_power = data.get("total_power_w")
    avg_temp    = data.get("average_gpu_temp_c")

    frq_text   = f"{frq:.2f} Hz" if frq is not None else FRQ_HZ_DEFAULT
    power_text = f"{total_power:.1f} W" if total_power is not None else POWER_W_DEFAULT
    temp_text  = f"{avg_temp:.1f} °C" if avg_temp is not None else TEMP_C_DEFAULT

    return frq_text, frq_text, power_text, temp_text
