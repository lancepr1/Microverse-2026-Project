"""
ui/layout.py — builds the Dash page layout: header, tab bar (Operator /
Analyst / Alert log / Digital Twin, see ui/tabs.py), and the four tabs'
content, all mounted at once (tabs only toggle CSS display -- see
ui/tabs.py's docstring for why).
"""
from dash import html, dcc

from data_feed import get_rack_id
from ui.charts import build_figures, FRQ_HEIGHT, POWER_HEIGHT
from ui.blender_feed import render_blender_feed
from ui.controls import build_chart_controls, build_facility_control
from ui.tabs import build_tab_bar, tab_content_style
from ui.operator import build_operator_tab
from ui.alert_log import build_alert_log_tab

POLL_INTERVAL_MS = 250

FRQ_HZ_DEFAULT = "-- Hz"

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


def render_header_frq(state: dict) -> str:
    """Header FRQ text for main.py's polling callback. Assumes
    state[get_rack_id()] exists -- callers should only invoke this when
    poll() returned data."""
    data = state[get_rack_id()]
    frq = data.get("frq_hz")
    return f"{frq:.2f} Hz" if frq is not None else FRQ_HZ_DEFAULT
