"""
ui/charts.py — two time-series plots: PDU line frequency (Hz) and total
rack power draw (W), both built straight from data_feed.poll() output.

FRQ_NOMINAL_HZ is a reference line only (US grid nominal, 60 Hz) — it is
not a pass/fail threshold. Anomaly/verification thresholds are out of
scope for this module; see README for what's intentionally left open.

get_figures() is called once per tick by the main polling callback in
main.py; it rebuilds both figures from the buffered deques below, windowed
to whatever time range is currently selected. update_charts() only
appends to those buffers — it does not build figures itself, so the
windowed x-axis keeps sliding on every tick even between new samples.
"""
import time
from datetime import datetime
from collections import deque

import plotly.graph_objects as go

from data_feed import get_rack_id

MAX_POINTS      = 90_000
SAMPLE_INTERVAL = 1.0
FRQ_NOMINAL_HZ  = 60.0
FRQ_HEIGHT      = 200
POWER_HEIGHT    = 150

COLOR_LINE      = "rgb(0, 150, 70)"
COLOR_THRESHOLD = "rgb(180, 150, 0)"
COLOR_LABEL     = "rgb(90, 90, 90)"
COLOR_TEXT      = "rgb(30, 30, 30)"
COLOR_PLOT_BG   = "rgb(255, 255, 255)"
COLOR_BORDER    = "rgb(210, 210, 210)"
COLOR_GRID      = "rgb(220, 220, 220)"
COLOR_LEGEND_BG = "rgba(255, 255, 255, 0.9)"
COLOR_CHART_BUTTON = ""

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

frq_x   = deque(maxlen=MAX_POINTS)
frq_y   = deque(maxlen=MAX_POINTS)
power_x = deque(maxlen=MAX_POINTS)
power_y = deque(maxlen=MAX_POINTS)

_t0             = [time.time()]
_last_plot_time = [0.0]
_accum_frq      = []
_accum_power    = []
_current_range  = ["10 Sec"]
_auto_follow    = [True]
_paused         = [False]
_frozen_figures = [None, None]


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
    """Freeze/unfreeze the chart display only. Does not affect the rack
    card, KPIs, or detail panel, which are driven separately from the same
    poll() output in main.py."""
    if enabled and not _paused[0]:
        _frozen_figures[0], _frozen_figures[1] = _compute_figures()
    elif not enabled and _paused[0]:
        _frozen_figures[0] = None
        _frozen_figures[1] = None
    _paused[0] = enabled


def get_paused():
    return _paused[0]


def update_charts(state: dict):
    if _paused[0]:
        return
    if not state:
        return

    data = state.get(get_rack_id())
    if not data:
        return

    try:
        _accum_frq.append(float(data.get("frq_hz", FRQ_NOMINAL_HZ)))
    except (ValueError, TypeError):
        pass
    try:
        _accum_power.append(float(data.get("total_power_w", 0.0)))
    except (ValueError, TypeError):
        pass

    now = time.time()
    if now - _last_plot_time[0] < SAMPLE_INTERVAL:
        return
    _last_plot_time[0] = now

    t = now - _t0[0]

    if _accum_frq:
        avg = sum(_accum_frq) / len(_accum_frq)
        frq_x.append(t)
        frq_y.append(avg)
        _accum_frq.clear()

    if _accum_power:
        avg = sum(_accum_power) / len(_accum_power)
        power_x.append(t)
        power_y.append(avg)
        _accum_power.clear()


def reset_charts():
    frq_x.clear()
    frq_y.clear()
    power_x.clear()
    power_y.clear()
    _accum_frq.clear()
    _accum_power.clear()
    _t0[0] = time.time()
    _last_plot_time[0] = 0.0


def build_figures():
    """Initial empty figures for the layout, before any state has been
    polled."""
    return get_figures()


def get_figures():
    if _paused[0] and _frozen_figures[0] is not None:
        return tuple(_frozen_figures)
    return _compute_figures()


def _compute_figures():
    cfg    = TIME_RANGE_CONFIG[_current_range[0]]
    window = cfg["window"]
    now_t  = time.time() - _t0[0]

    if window is None:
        x_min, x_max = 0, max(now_t, 1)
    else:
        x_max = max(now_t, window)
        x_min = x_max - window

    fx, fy = list(frq_x), list(frq_y)
    px, py = list(power_x), list(power_y)

    x_range = None
    ticks = None
    if _auto_follow[0]:
        if fx:
            pairs = [(x, y) for x, y in zip(fx, fy) if x_min <= x <= x_max]
            if pairs:
                fx, fy = [p[0] for p in pairs], [p[1] for p in pairs]
        if px:
            pairs = [(x, y) for x, y in zip(px, py) if x_min <= x <= x_max]
            if pairs:
                px, py = [p[0] for p in pairs], [p[1] for p in pairs]

        x_range = [x_min, x_max]
        ticks = _build_ticks(x_min, x_max, cfg)

    frq_fig   = _build_frq_figure(fx, fy, x_range, ticks, now_t)
    power_fig = _build_power_figure(px, py, x_range, ticks)
    return frq_fig, power_fig


def _build_frq_figure(fx, fy, x_range, ticks, now_t):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fx, y=fy, mode="lines", name=get_rack_id(),
        line=dict(color=COLOR_LINE, width=1.5)
    ))
    nominal_x_max = max(now_t, 1)
    fig.add_trace(go.Scatter(
        x=[0, nominal_x_max], y=[FRQ_NOMINAL_HZ, FRQ_NOMINAL_HZ],
        mode="lines", name="Nominal (60 Hz)",
        line=dict(color=COLOR_THRESHOLD, width=1, dash="dash")
    ))
    _apply_layout(fig, x_range, ticks, y_title="Hz")
    return fig


def _build_power_figure(px, py, x_range, ticks):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=px, y=py, mode="lines", name=get_rack_id(),
        line=dict(color=COLOR_LINE, width=1.5)
    ))
    _apply_layout(fig, x_range, ticks, y_title="Watts")
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
