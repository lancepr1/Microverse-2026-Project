"""
ui/layout.py — builds the Dash page layout: header, tab bar (Operator /
Analyst / Alert log / Digital Twin, see ui/tabs.py), and the four tabs'
content, all mounted at once (tabs only toggle CSS display -- see
ui/tabs.py's docstring for why).
"""
from dash import html, dcc

from data_feed import get_rack_id
from ui.blender_feed import render_blender_feed
from ui.tabs import build_tab_bar, tab_content_style
from ui.operator import build_operator_tab
from ui.analyst import build_analyst_tab
from ui.alert_log import build_alert_log_tab

POLL_INTERVAL_MS = 250

FRQ_HZ_DEFAULT = "-- Hz"


def build_layout() -> html.Div:
    return html.Div(className="app", children=[
        dcc.Interval(id="poll-interval", interval=POLL_INTERVAL_MS),

        html.Div(className="header", children=[
            html.Span("AFRL Microverse — Data Center Dashboard",
                      className="header-title"),
            html.Span("FRQ:", className="label"),
            html.Div(className="frq-readout", children=[
                html.Span(className="live-pulse-dot"),
                html.Span(FRQ_HZ_DEFAULT, id="header-frq-hz"),
            ]),
        ]),
        html.Hr(),

        build_tab_bar(),

        html.Div(className="tab-content-panels", children=[
            html.Div(
                build_operator_tab(),
                id="tab-content-operator",
                style=tab_content_style("operator"),
            ),
            html.Div(
                build_analyst_tab(),
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


def render_header_frq(state: dict) -> str:
    """Header FRQ text for main.py's polling callback. Assumes
    state[get_rack_id()] exists -- callers should only invoke this when
    poll() returned data."""
    data = state[get_rack_id()]
    frq = data.get("frq_hz")
    return f"{frq:.2f} Hz" if frq is not None else FRQ_HZ_DEFAULT
