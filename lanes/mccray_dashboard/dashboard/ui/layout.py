"""
ui/layout.py — builds the Dash page layout: header, tab bar (Operator /
Analyst / Alert log / Digital Twin, see ui/tabs.py), and the four tabs'
content, all mounted at once (tabs only toggle CSS display -- see
ui/tabs.py's docstring for why).
"""
from dash import html, dcc

from ui.blender_feed import render_blender_feed
from ui.tabs import build_tab_bar, tab_content_style
from ui.operator import build_operator_tab
from ui.analyst import build_analyst_tab
from ui.alert_log_tab import build_alert_log_tab

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
                # CHANGED (2026-07): added id -- this dot previously had no
                # id at all, so nothing could ever target it with a style
                # Output. It's meant to sit right where an ENF status
                # indicator would be looked for (next to the FRQ readout),
                # but no callback anywhere in the dashboard ever wrote to
                # it -- it was a static, decorative "live" pulse only,
                # which is why it visibly never changed color even when
                # ENF_status genuinely went bad. See main.py's own CHANGED
                # comment for the callback that now drives it, and
                # render_header_frq_dot_style() below for the color logic.
                html.Span(className="live-pulse-dot", id="header-frq-dot"),
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


# CHANGED (2026-07): reads operator-state-store (poll_all()'s multi-node
# output, the real for_dashboard.jsonl feed) -- REMOVED (cleanup pass,
# same date): a legacy render_header_frq() used to live here, reading a
# single-node poll()/get_rack_id() feed off data/run01.jsonl. That whole
# path is gone now (see data_feed.py's module docstring) and nothing ever
# called this function's replacement anyway, so it's simply deleted
# rather than left broken.
def render_header_frq_multi(state: dict) -> str:
    """Header FRQ text sourced from operator-state-store (poll_all()'s
    multi-node output, i.e. the real for_dashboard.jsonl feed) instead of
    the old single-node demo feed. FRQ is a single facility-wide PDU
    reading -- every node in a given for_dashboard.jsonl row shares the
    same "FRQ" value (see models.py's TelemetrySample.frq_hz) -- so any
    one node's reading is representative of the whole row; this just takes
    the first one present rather than needing to pick a "canonical" node.

    CHANGED (2026-07, per Leiva's request): was f"{frq:.2f} Hz" -- rounded
    to 4 decimal places instead of 2, to make it possible to actually
    cross-check this reading against the raw terminal/file data, which
    carries more precision than 2 decimals shows."""
    if not state:
        return FRQ_HZ_DEFAULT
    for data in state.values():
        frq = data.get("frq_hz")
        if frq is not None:
            return f"{frq:.4f} Hz"
    return FRQ_HZ_DEFAULT


# ADDED (2026-07): fixes a real gap, not just a cosmetic tweak -- the
# "live-pulse-dot" next to the FRQ readout had NO callback anywhere
# targeting it (see build_layout()'s own CHANGED comment on that element).
# It looked like an ENF status indicator (right next to the ENF-derived
# FRQ number) but was purely decorative, hardcoded green via CSS, and
# could never turn yellow/red no matter what ENF_status said. This gives
# it real status-driven color, using data_feed.py's now-distinct
# "enf_status" field (see that module's own 2026-07 comment) rather than
# the blended per-node "status" -- this dot should reflect ENF specifically,
# not any one node's own NLR checks.
_ENF_DOT_STATUS_STYLE = {
    "good": {
        "backgroundColor": "var(--status-nominal-bg)",
        "border": "1px solid var(--status-nominal-border)",
    },
    "suspect": {
        "backgroundColor": "var(--status-warning-bg)",
        "border": "1px solid var(--status-warning-border)",
    },
    "warning": {
        "backgroundColor": "var(--status-alert-bg)",
        "border": "1px solid var(--status-alert-border)",
    },
}
_ENF_DOT_DEFAULT_STYLE = {
    "backgroundColor": "var(--status-default-bg)",
    "border": "1px solid var(--status-default-border)",
}


def render_header_frq_dot_style(state: dict) -> dict:
    """Inline style for the header's ENF status dot, sourced from
    operator-state-store the same way render_header_frq_multi() is.
    Reads "enf_status" specifically (not the blended "status" field) --
    every node in a given tick carries the same enf_status value since
    it's facility-wide, so the first one present is representative.
    Returned as a plain style dict, not a full replacement -- Dash merges
    this into the element's existing className-driven styling (e.g. the
    pulse animation), it doesn't remove it."""
    if not state:
        return _ENF_DOT_DEFAULT_STYLE
    for data in state.values():
        enf = data.get("enf_status")
        if enf and enf != "--":
            return _ENF_DOT_STATUS_STYLE.get(enf, _ENF_DOT_DEFAULT_STYLE)
    return _ENF_DOT_DEFAULT_STYLE