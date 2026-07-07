"""
ui/rack_cards.py — the rack status card (left panel) and its detail
breakdown (right panel "Rack Inspection" section).

Displays the raw telemetry fields produced by data_feed.poll() -- PDU
frequency, total power, per-GPU/per-CPU power and temperature -- plus the
verification status/reasons that data_feed.poll() now merges in from
verification_feed.verify_sample() (Leiva's Verifier, one call per replayed
sample). "good"/"suspect"/"warning" are Leiva's own documented
presentation-layer labels for TRUSTED/SUSPECT/FAILED.
"""
from dash import html

from data_feed import get_rack_id

COLOR_LABEL  = "rgb(90, 90, 90)"
COLOR_DIMMED = "rgb(140, 140, 140)"
COLOR_TEXT   = "rgb(30, 30, 30)"
COLOR_GOOD   = "rgb(0, 150, 70)"
COLOR_SUSPECT = "rgb(200, 150, 0)"
COLOR_WARNING = "rgb(200, 50, 50)"

FRQ_HZ_DEFAULT  = "-- Hz"
POWER_W_DEFAULT = "-- W"
TEMP_C_DEFAULT  = "-- °C"

_STATUS_COLORS = {
    "good":    COLOR_GOOD,
    "suspect": COLOR_SUSPECT,
    "warning": COLOR_WARNING,
}


def status_color(status):
    """Maps verification_feed's status labels (good/suspect/warning) to a
    display color; unknown/placeholder values (e.g. "--") fall through to
    COLOR_LABEL."""
    return _STATUS_COLORS.get(status, COLOR_LABEL)


def render_rack_card(state: dict) -> html.Div:
    rack_id = get_rack_id()
    data = state.get(rack_id, {})

    frq         = data.get("frq_hz")
    total_power = data.get("total_power_w")
    avg_temp    = data.get("average_gpu_temp_c")
    status      = data.get("status", "--")

    frq_text   = f"{frq:.2f} Hz" if frq is not None else FRQ_HZ_DEFAULT
    power_text = f"{total_power:.1f} W" if total_power is not None else POWER_W_DEFAULT
    temp_text  = f"{avg_temp:.1f} °C" if avg_temp is not None else TEMP_C_DEFAULT

    return html.Div(className="card", children=[
        html.Div(className="row", children=[
            html.Span(str(rack_id), style={"color": COLOR_TEXT}),
            html.Span(str(status), style={"color": status_color(status), "marginLeft": "auto"}),
        ]),
        html.Hr(),
        html.Div(className="row", children=[
            html.Span("FRQ:", className="label"),
            html.Span(frq_text),
            html.Span("PWR:", className="label"),
            html.Span(power_text),
        ]),
        html.Div(className="row", children=[
            html.Span("Average GPU Temp", className="label"),
            html.Span(temp_text),
        ]),
    ])


def render_detail_panel(state: dict) -> html.Div:
    rack_id = get_rack_id()
    data = state.get(rack_id, {})

    frq         = data.get("frq_hz")
    total_power = data.get("total_power_w")
    avg_temp    = data.get("average_gpu_temp_c")
    gpu_power   = data.get("gpu_power_w", {})
    gpu_temp    = data.get("gpu_temp_c", {})
    cpu_power   = data.get("cpu_power_w", {})
    status      = data.get("status", "--")
    reasons     = data.get("verification_reasons", [])

    frq_text   = f"{frq:.2f} Hz" if frq is not None else FRQ_HZ_DEFAULT
    power_text = f"{total_power:.1f} W" if total_power is not None else POWER_W_DEFAULT
    temp_text  = f"{avg_temp:.1f} °C" if avg_temp is not None else TEMP_C_DEFAULT
    reasons_text = (
        "; ".join(f"{component}: {reason}" for component, reason in reasons)
        if reasons else "All channels nominal"
    )

    gpu_lines = [
        f"GPU {gpu_id}: {gpu_power.get(gpu_id, 0.0):.1f} W, "
        f"{gpu_temp.get(gpu_id, 0.0):.1f} °C"
        for gpu_id in sorted(gpu_power)
    ]
    cpu_lines = [
        f"CPU {cpu_id}: {cpu_power.get(cpu_id, 0.0):.1f} W"
        for cpu_id in sorted(cpu_power)
    ]

    return html.Div(children=[
        html.Div(str(rack_id), style={"color": COLOR_TEXT}),
        html.Hr(),

        _detail_row("Status", str(status), status_color(status)),
        _detail_row("Verification", reasons_text),
        _detail_row("FRQ", frq_text),
        _detail_row("Total Power", power_text),
        _detail_row("Average GPU Temp", temp_text),

        html.Div("GPU Breakdown", className="section-title"),
        html.Hr(),
        html.Div(
            "\n".join(gpu_lines) if gpu_lines else "Awaiting data...",
            className="dimmed-block",
        ),

        html.Div("CPU Breakdown", className="section-title"),
        html.Hr(),
        html.Div(
            "\n".join(cpu_lines) if cpu_lines else "Awaiting data...",
            className="dimmed-block",
        ),
    ])


def _detail_row(label, value, color=COLOR_TEXT):
    return html.Div(className="row", children=[
        html.Span(f"{label}:", className="label"),
        html.Span(value, style={"color": color}),
    ])
