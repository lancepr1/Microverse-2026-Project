"""
ui/charts.py — per-node rolling history buffers and the rack-panel power/
temperature figures built from them for the Analyst tab.

Replaces the old single-run PDU Frequency/Power Consumption charts (which
plotted whichever one legacy run data_feed.get_rack_id() had loaded) with
rack-centric multi-node figures fed by data_feed.poll_all() via
operator-state-store (see ui/analyst.py, which owns the layout + the
callback that drives these builders once per tick).

TIME_RANGE_CONFIG/_current_range/_auto_follow/_paused are unchanged from
the single-run version -- the same Time Range selector and pause/resume
toggle still drive every chart, just against the new per-node buffers
instead of the old single-series deques.
"""
import time
from datetime import datetime
from collections import deque

import plotly.graph_objects as go

from data_feed import node_display_label

COLOR_LABEL     = "#64748b"
COLOR_TEXT      = "#0f172a"
COLOR_PLOT_BG   = "#ffffff"
COLOR_BORDER    = "#e8eaed"
COLOR_GRID      = "#f1f5f9"
COLOR_LEGEND_BG = "rgba(255, 255, 255, 0.9)"

# Fixed color-by-position-in-rack mapping: node index 0-3 within a rack
# always gets the same hue, on every rack panel and on both its power and
# temp chart, so identity reads consistently across the whole tab. These
# are the dataviz skill's validated categorical slots 1-4 (light-mode
# worst-adjacent CVD ΔE 24.2, in their documented fixed order) -- not a new
# palette, just reusing that skill's default categorical ramp.
NODE_COLORS = [
    "rgb(42, 120, 214)",   # slot 1 -- blue
    "rgb(27, 175, 122)",   # slot 2 -- aqua
    "rgb(237, 161, 0)",    # slot 3 -- yellow
    "rgb(0, 131, 0)",      # slot 4 -- green
]

TIME_RANGE_CONFIG = {
    "10 Sec":   {"window": 10,    "tick_step": 2,     "fmt": "%H:%M:%S"},
    "45 Sec":   {"window": 45,    "tick_step": 9,     "fmt": "%H:%M:%S"},
    "90 Sec":   {"window": 90,    "tick_step": 15,    "fmt": "%H:%M:%S"},
    "1 Hour":   {"window": 3600,  "tick_step": 600,   "fmt": "%H:%M"},
    "6 Hours":  {"window": 21600, "tick_step": 3600,  "fmt": "%H:%M"},
    "24 Hours": {"window": 86400, "tick_step": 14400, "fmt": "%H:%M"},
    "All":      {"window": None,  "tick_step": None,  "fmt": "%H:%M:%S"},
}
ALL_TICK_COUNT = 7

_t0            = [time.time()]
_current_range = ["10 Sec"]
_auto_follow   = [True]
_paused        = [False]


def set_time_range(label: str):
    if label not in TIME_RANGE_CONFIG:
        return
    _current_range[0] = label
    _auto_follow[0] = True


def set_auto_follow(enabled: bool):
    _auto_follow[0] = enabled


def get_auto_follow():
    return _auto_follow[0]


def set_paused(enabled: bool):
    _paused[0] = enabled


def get_paused():
    return _paused[0]


# --- Per-node rack-centric history (Analyst tab rack panels) ---------------
#
# One rolling buffer per node, holding both metrics the rack panels chart
# (total power, average GPU temp) against a shared timestamp axis, since
# both come from the same data_feed.poll_all() sample per node per tick.
#
# NODE_BUFFER_MAXLEN is sized for a 1-hour "All"-adjacent window at a
# 2-second sampling cadence (1800 points), per spec -- data_feed.py's
# actual replay pacing (REPLAY_INTERVAL_S) is currently faster than that,
# so at today's replay speed the buffer holds less than an hour of
# wall-clock time; the depth is intentionally pinned to the 2-second-cadence
# target rather than derived from the replay's current speed.
NODE_BUFFER_MAXLEN = 1800

_node_history = {}  # node_id -> {"t": deque, "power": deque, "temp": deque}


def _node_buffer(node_id: str) -> dict:
    if node_id not in _node_history:
        _node_history[node_id] = {
            "t":     deque(maxlen=NODE_BUFFER_MAXLEN),
            "power": deque(maxlen=NODE_BUFFER_MAXLEN),
            "temp":  deque(maxlen=NODE_BUFFER_MAXLEN),
        }
    return _node_history[node_id]


def update_node_history(state: dict) -> None:
    """Appends one point per node to the rolling per-node buffers above,
    from data_feed.poll_all() output routed in via operator-state-store
    (see ui/operator.py) -- this must never call poll_all() itself, since
    poll_all() advances a shared replay cursor and can only be called once
    per tick. Paused stops accumulation, so a frozen display doesn't
    silently keep buffering behind it."""
    if _paused[0]:
        return
    if not state:
        return

    now = time.time() - _t0[0]
    for node_id, data in state.items():
        buf = _node_buffer(node_id)
        buf["t"].append(now)
        buf["power"].append(data.get("total_power_w"))
        buf["temp"].append(data.get("average_gpu_temp_c"))


def get_node_history_lengths() -> dict:
    """Debug/verification helper: current buffer length per node (the
    three deques for a given node are always the same length, so "t" is
    representative of all of them)."""
    return {node_id: len(buf["t"]) for node_id, buf in _node_history.items()}


def _current_window():
    cfg    = TIME_RANGE_CONFIG[_current_range[0]]
    window = cfg["window"]
    now_t  = time.time() - _t0[0]

    if window is None:
        return 0, max(now_t, 1), cfg
    x_max = max(now_t, window)
    return x_max - window, x_max, cfg


def _windowed(tx, ty, x_min, x_max):
    if not (_auto_follow[0] and tx):
        return tx, ty
    pairs = [(x, y) for x, y in zip(tx, ty) if x_min <= x <= x_max]
    return ([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else (tx, ty)


DIMMED_OPACITY = 0.15


def build_rack_power_figure(node_ids: list[str], isolated_node: str | None = None) -> go.Figure:
    """One line per node in node_ids, in NODE_COLORS order (position in
    rack -> color), windowed to the current Time Range selection.
    isolated_node, if set, is drawn at full opacity while every other node
    in this figure is dimmed (see ui/analyst.py's click-to-isolate
    callback, which tracks isolated_node per rack)."""
    return _build_multi_line_figure(node_ids, "power", y_title="Watts", isolated_node=isolated_node)


def build_rack_temp_figure(node_ids: list[str], isolated_node: str | None = None) -> go.Figure:
    """Same shape as build_rack_power_figure, reading the "temp" series so
    a node's power and temp lines share NODE_COLORS' position-based color
    -- and the same isolated_node, so isolating a node dims its
    counterpart line on both charts together."""
    return _build_multi_line_figure(node_ids, "temp", y_title="°C", isolated_node=isolated_node)


def _build_multi_line_figure(
    node_ids: list[str], metric: str, y_title: str, isolated_node: str | None = None,
) -> go.Figure:
    x_min, x_max, cfg = _current_window()

    fig = go.Figure()
    for i, node_id in enumerate(node_ids):
        buf = _node_buffer(node_id)
        tx, ty = list(buf["t"]), list(buf[metric])
        tx, ty = _windowed(tx, ty, x_min, x_max)
        opacity = 1.0 if isolated_node in (None, node_id) else DIMMED_OPACITY
        fig.add_trace(go.Scatter(
            x=tx, y=ty, mode="lines", name=node_display_label(node_id), opacity=opacity,
            line=dict(color=NODE_COLORS[i % len(NODE_COLORS)], width=1.5),
        ))

    x_range = [x_min, x_max] if _auto_follow[0] else None
    ticks = _build_ticks(x_min, x_max, cfg) if _auto_follow[0] else None
    _apply_layout(fig, x_range, ticks, y_title=y_title)
    return fig


def _apply_layout(fig, x_range, ticks, y_title):
    xaxis = dict(gridcolor=COLOR_GRID, linecolor=COLOR_BORDER, title="")
    if x_range is not None:
        xaxis["range"] = x_range
    if ticks:
        xaxis["tickmode"] = "array"
        xaxis["tickvals"] = [pos for _, pos in ticks]
        xaxis["ticktext"] = [label for label, _ in ticks]

    fig.update_layout(
        uirevision="rack-charts",
        margin=dict(l=50, r=10, t=10, b=30),
        plot_bgcolor=COLOR_PLOT_BG,
        paper_bgcolor=COLOR_PLOT_BG,
        font=dict(color=COLOR_LABEL),
        legend=dict(
            bgcolor=COLOR_LEGEND_BG, bordercolor=COLOR_BORDER,
            borderwidth=1, font=dict(color=COLOR_TEXT)
        ),
        xaxis=xaxis,
        yaxis=dict(
            gridcolor=COLOR_GRID, linecolor=COLOR_BORDER,
            title=y_title, autorange=True
        ),
    )


def _build_ticks(x_min, x_max, cfg):
    fmt       = cfg["fmt"]
    tick_step = cfg["tick_step"]
    t0_real   = _t0[0]

    if tick_step is not None:
        start = (int(x_min / tick_step) + 1) * tick_step
        positions = []
        t = start
        while t <= x_max:
            positions.append(t)
            t += tick_step
        if not positions:
            positions = [x_min, x_max]
    else:
        span = x_max - x_min
        if span <= 0:
            return None
        step = span / (ALL_TICK_COUNT - 1)
        positions = [x_min + i * step for i in range(ALL_TICK_COUNT)]

    return [
        (datetime.fromtimestamp(t0_real + pos).strftime(fmt), pos)
        for pos in positions
    ]
