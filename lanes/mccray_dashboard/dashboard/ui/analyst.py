"""
ui/analyst.py — the Analyst tab: four rack panels in a 2x2 grid, one per
rack, each with a power chart and a smaller GPU-temperature chart below it
(one line per node in that rack, on both). Replaces the old single-run PDU
Frequency/Power Consumption charts entirely.

Reuses ui.operator.get_rack_groups() for the rack/node grouping -- the same
placeholder Rack 1-4 chunking the Operator tab's grid uses -- rather than
redefining it. This module only reads that function; it does not modify
ui/operator.py. Node buffering and figure-building live in ui/charts.py
(NODE_BUFFER_MAXLEN, update_node_history(), build_rack_power_figure(),
build_rack_temp_figure(), the shared TIME_RANGE_CONFIG/pause state the Time
Range selector already drove); this module is the layout plus the callback
that drives the eight graphs.

A node's power and temp lines share a color because both figures index
ui.charts.NODE_COLORS by the same node_ids list, in the same order, for a
given rack -- position-in-rack determines color, not the metric.

Rack graph ids are concrete ("rack-1-power-graph".."rack-4-power-graph",
likewise "-temp-graph"), not ALL-pattern-matched, since there are always
exactly 4 racks -- this avoids relying on Dash's pattern-matching Output
ordering guarantees for what is a small, fixed set.

Two separate callbacks drive the eight graphs -- deliberately not one,
see _on_chart_click's docstring for why combining them broke live updates
in a real browser:
  - _on_analyst_tick: operator-state-store fires roughly once per real
    replay sample (see data_feed.poll_all()'s pacing) -- this is the only
    trigger allowed to call update_node_history(), since poll_all()'s
    cursor already only advances once per sample and double-appending on
    other ticks would duplicate points in the buffer. poll-interval fires
    every 250ms regardless of whether new data arrived -- figures are
    still rebuilt on every one of these ticks (just without touching the
    buffer), so the auto-follow window keeps sliding smoothly with
    wall-clock time and a Time Range/pause click takes effect immediately.
  - _on_chart_click: each graph's clickData (clicking a line) toggles that
    node's isolation for its rack. Isolation state is tracked server-side
    (_isolated_node) and baked into every figure build via
    build_rack_*_figure's isolated_node param, rather than left as a
    client-side-only opacity tweak, because _on_analyst_tick's 250ms
    live-redraw would otherwise overwrite a client-only effect on the very
    next tick before an analyst could see it hold.

Deliberately NOT wired to restyleData (legend-entry clicks): Plotly can
emit that event as a side effect of the server pushing a new figure, not
just from a genuine legend click -- wiring it up made _on_chart_click fire
on nearly every live redraw too, and since both callbacks wrote the same
outputs (via allow_duplicate), whichever one's response landed last won,
causing exactly the flicker/stale-data bug this replaced. Clicking a
line's clickData doesn't have this problem (it only fires on a real
plotly_click). Legend entries still work, just via Plotly's own default
click-to-hide-that-trace behavior rather than the isolate/dim treatment.
"""
from dash import html, dcc, callback, Input, Output, ctx

from ui.operator import get_rack_groups
from ui import charts
from ui.charts import update_node_history, build_rack_power_figure, build_rack_temp_figure
from ui.controls import build_chart_controls
from data_feed import node_display_label


RACK_POWER_HEIGHT = 220
RACK_TEMP_HEIGHT  = 130

_RACK_GROUPS = get_rack_groups()
_RACK_SLUGS  = [label.lower().replace(" ", "-") for label, _ in _RACK_GROUPS]
_RACK_BY_SLUG = {slug: node_ids for slug, (_, node_ids) in zip(_RACK_SLUGS, _RACK_GROUPS)}

_last_state      = [{}]   # most recent operator-state-store value, for the pause snapshot
_frozen_snapshot = [None]  # None (live) or a frozen return tuple for the tick callback
_isolated_node   = {}      # rack slug -> node_id currently isolated in that rack's charts, or absent


def build_analyst_tab() -> html.Div:
    return html.Div(children=[
        html.Div(
            "Per-rack signal analysis — one line per node. Click a line "
            "to isolate a node.",
            className="dimmed-block",
        ),
        html.Div(className="row", children=[
            build_chart_controls(),
            html.Button("⏸ Pause", id="rack-chart-pause-btn", className="btn"),
        ]),
        html.Div(className="rack-chart-grid", children=[
            _build_rack_panel(rack_label, node_ids)
            for rack_label, node_ids in _RACK_GROUPS
        ]),
    ])


def _build_rack_panel(rack_label: str, node_ids: list[str]) -> html.Div:
    slug = rack_label.lower().replace(" ", "-")
    return html.Div(className="panel rack-chart-panel", children=[
        html.Div(className="row", children=[
            html.Span(rack_label, className="panel-title"),
            html.Span("-- W", id=f"{slug}-total-power", className="mono-value",
                      style={"marginLeft": "auto"}),
        ]),
        html.Hr(),
        html.Div(f"Power — {', '.join(node_display_label(n) for n in node_ids)}", className="chart-label"),
        dcc.Graph(id=f"{slug}-power-graph",
                  figure=build_rack_power_figure(node_ids),
                  style={"height": f"{RACK_POWER_HEIGHT}px"},
                  config={"displayModeBar": False}),
        html.Div("GPU Temperature", className="chart-label"),
        dcc.Graph(id=f"{slug}-temp-graph",
                  figure=build_rack_temp_figure(node_ids),
                  style={"height": f"{RACK_TEMP_HEIGHT}px"},
                  config={"displayModeBar": False}),
    ])


def _format_total_power(state: dict, node_ids: list[str]) -> str:
    state = state or {}
    total = sum((state.get(nid, {}) or {}).get("total_power_w") or 0.0 for nid in node_ids)
    return f"{total:.1f} W"


def _build_all_figures(state: dict) -> tuple:
    power_figs = [
        build_rack_power_figure(node_ids, _isolated_node.get(slug))
        for slug, node_ids in _RACK_BY_SLUG.items()
    ]
    temp_figs = [
        build_rack_temp_figure(node_ids, _isolated_node.get(slug))
        for slug, node_ids in _RACK_BY_SLUG.items()
    ]
    totals = [_format_total_power(state, node_ids) for _, node_ids in _RACK_GROUPS]
    return (*power_figs, *temp_figs, *totals)


def _rack_slug_from_graph_id(graph_id: str) -> str:
    return graph_id.replace("-power-graph", "").replace("-temp-graph", "")


def _extract_trace_index(payload):
    """clickData shape: {"points": [{"curveNumber": N, ...}, ...]}.
    restyleData (legend click) shape: [{...visibility change...}, [N, ...]]."""
    if not payload:
        return None
    if isinstance(payload, dict) and payload.get("points"):
        return payload["points"][0].get("curveNumber")
    if isinstance(payload, list) and len(payload) >= 2 and payload[1]:
        return payload[1][0]
    return None


def _handle_chart_click(graph_id: str, payload) -> None:
    """Pure logic, deliberately kept independent of Dash's ctx global (which
    only works inside a real callback invocation) so it can be unit tested
    directly with a plain payload dict/list."""
    slug = _rack_slug_from_graph_id(graph_id)
    node_ids = _RACK_BY_SLUG.get(slug)
    if not node_ids:
        return

    trace_index = _extract_trace_index(payload)
    if trace_index is None or trace_index >= len(node_ids):
        return

    clicked_node = node_ids[trace_index]
    if _isolated_node.get(slug) == clicked_node:
        _isolated_node.pop(slug, None)  # clicking the isolated node again resets
    else:
        _isolated_node[slug] = clicked_node


@callback(
    *[Output(f"{slug}-power-graph", "figure") for slug in _RACK_SLUGS],
    *[Output(f"{slug}-temp-graph", "figure") for slug in _RACK_SLUGS],
    *[Output(f"{slug}-total-power", "children") for slug in _RACK_SLUGS],
    Input("operator-state-store", "data"),
    Input("poll-interval", "n_intervals"),
)
def _on_analyst_tick(state, _n):
    trigger = ctx.triggered_id

    if trigger == "operator-state-store":
        update_node_history(state or {})
        _last_state[0] = state or {}

    if _frozen_snapshot[0] is not None:
        return _frozen_snapshot[0]

    return _build_all_figures(_last_state[0])


@callback(
    *[Output(f"{slug}-power-graph", "figure", allow_duplicate=True) for slug in _RACK_SLUGS],
    *[Output(f"{slug}-temp-graph", "figure", allow_duplicate=True) for slug in _RACK_SLUGS],
    *[Output(f"{slug}-total-power", "children", allow_duplicate=True) for slug in _RACK_SLUGS],
    *[Input(f"{slug}-power-graph", "clickData") for slug in _RACK_SLUGS],
    *[Input(f"{slug}-temp-graph", "clickData") for slug in _RACK_SLUGS],
    prevent_initial_call=True,
)
def _on_chart_click(*_click_data):
    """Split out from _on_analyst_tick above: that callback both wrote
    each graph's figure and read click/restyle data FROM those same
    graphs, a self-referential wiring that let a real browser's own
    figure-replacement redraws feed back into this callback's inputs and
    quietly stall the poll-interval-driven live redraw after the first
    render (not reproducible through direct, non-browser callback
    simulation, which is why testing this in isolation looked fine). This
    callback now only ever fires on a genuine line click, via
    allow_duplicate, and never touches poll-interval/operator-state-store
    at all. See the module docstring for why restyleData (legend clicks)
    isn't wired here either."""
    trigger = ctx.triggered_id
    if isinstance(trigger, str) and (
        trigger.endswith("-power-graph") or trigger.endswith("-temp-graph")
    ):
        _handle_chart_click(trigger, ctx.triggered[0]["value"])

    if _frozen_snapshot[0] is not None:
        return _frozen_snapshot[0]

    return _build_all_figures(_last_state[0])


@callback(
    Output("rack-chart-pause-btn", "children"),
    Input("rack-chart-pause-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _on_pause_toggle(_n_clicks):
    now_paused = not charts.get_paused()
    charts.set_paused(now_paused)

    if now_paused:
        _frozen_snapshot[0] = _build_all_figures(_last_state[0])
        return "▶ Resume"

    _frozen_snapshot[0] = None
    return "⏸ Pause"
