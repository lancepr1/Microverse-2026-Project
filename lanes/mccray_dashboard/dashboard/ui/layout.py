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
            # CHANGED (2026-07, per Leiva's request): label text only --
            # was "FRQ:". Internal ids (header-frq-hz, header-frq-dot,
            # header-frq-bubble) and the underlying data field name
            # (frq_hz) are deliberately left as-is -- renaming those too
            # would mean touching main.py's Output targets and every
            # reference to frq_hz throughout data_feed.py/models.py for a
            # purely cosmetic label change with no functional benefit.
            html.Span("ENF:", className="label"),
            # ADDED (2026-07): id -- the whole pill/bubble now gets
            # status-driven styling (see render_header_frq_bubble_style()
            # below), not just the small dot inside it.
            html.Div(className="frq-readout", id="header-frq-bubble", children=[
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
                # CHANGED (2026-07, per Leiva's request): pinned to a
                # fixed black -- previously this span's text color was
                # implicitly whatever the "frq-readout" className's CSS
                # made it (green), which never changed even as the
                # bubble's own background/border started turning
                # yellow/red -- confusing, since it looked like the text
                # itself was claiming "still fine" while the bubble
                # around it disagreed. An explicit style here always wins
                # over inherited/className color, and is intentionally
                # NOT wired to any callback Output -- it's meant to never
                # change, unlike the bubble/dot above it.
                html.Span(FRQ_HZ_DEFAULT, id="header-frq-hz", style={"color": "#000000"}),
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
#
# CHANGED (2026-07, follow-up): the dot alone wasn't visible/prominent
# enough -- the whole frq-readout pill/bubble now gets the same
# status-driven coloring too (render_header_frq_bubble_style() below),
# via the shared _current_enf_label() lookup so both stay in sync off the
# exact same source rather than two separate iterations that could drift.
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

# Same status hue tokens as the dot, applied to the whole bubble
# container instead -- background + border, matching the pattern
# ui/operator.py's node-card badges already use (_STATUS_BADGE_STYLE),
# just reused here for the header bubble. No "color" key -- CHANGED
# (2026-07): the FRQ text span (header-frq-hz, in build_layout() above)
# is now pinned to a fixed black via its own explicit style, which always
# wins over anything set here, so a "color" entry in this dict would be
# dead/misleading -- removed rather than left in place unused.
_ENF_BUBBLE_STATUS_STYLE = {
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
_ENF_BUBBLE_DEFAULT_STYLE = {
    "backgroundColor": "var(--status-default-bg)",
    "border": "1px solid var(--status-default-border)",
}


def _current_enf_label(state: dict) -> str:
    """The current ENF status label ("good"/"suspect"/"warning"/"--"),
    read from operator-state-store the same way render_header_frq_multi()
    reads frq_hz -- "enf_status" specifically (not the blended "status"
    field), and every node in a given tick carries the same value since
    it's facility-wide, so the first one present is representative.
    Shared by both render_header_frq_dot_style() and
    render_header_frq_bubble_style() so they can never disagree with each
    other -- one lookup, two style dicts applied to it."""
    if not state:
        return "--"
    for data in state.values():
        enf = data.get("enf_status")
        if enf and enf != "--":
            return enf
    return "--"


def render_header_frq_dot_style(state: dict) -> dict:
    """Inline style for the header's ENF status dot. Returned as a plain
    style dict, not a full replacement -- Dash merges this into the
    element's existing className-driven styling (e.g. the pulse
    animation), it doesn't remove it."""
    return _ENF_DOT_STATUS_STYLE.get(_current_enf_label(state), _ENF_DOT_DEFAULT_STYLE)


def render_header_frq_bubble_style(state: dict) -> dict:
    """Inline style for the WHOLE frq-readout pill/bubble (not just the
    dot inside it) -- background/border/text color all reflect ENF
    status. Same source as render_header_frq_dot_style() via
    _current_enf_label(), so the dot and the bubble around it never show
    disagreeing colors."""
    return _ENF_BUBBLE_STATUS_STYLE.get(_current_enf_label(state), _ENF_BUBBLE_DEFAULT_STYLE)