"""
ui/rack_cards.py — the rack status card (left panel) and its detail
breakdown (right panel "Rack Inspection" section).

Only the raw telemetry fields produced by data_feed.poll() are displayed
here: PDU frequency, total power, and per-GPU/per-CPU power and
temperature. There is no health/verification status yet — `status` is read
from the state dict if present and shown as-is, but nothing in this repo
sets it. That field is a placeholder for a verification/alerting module to
populate later; until then the card simply shows "--".
"""
from dash import html

RACK_ID = "Rack_01"

COLOR_LABEL  = "rgb(90, 90, 90)"
COLOR_DIMMED = "rgb(140, 140, 140)"
COLOR_TEXT   = "rgb(30, 30, 30)"

FRQ_HZ_DEFAULT  = "-- Hz"
POWER_W_DEFAULT = "-- W"
TEMP_C_DEFAULT  = "-- °C"


def status_color(status):
    """No status values are produced anywhere in this repo yet; this always
    falls through to COLOR_LABEL until a verification module starts setting
    state[RACK_ID]["status"]."""
    return {}.get(status, COLOR_LABEL)


def render_rack_card(state: dict) -> html.Div:
    data = state.get(RACK_ID, {})

    frq         = data.get("frq_hz")
    total_power = data.get("total_power_w")
    avg_temp    = data.get("average_gpu_temp_c")
    status      = data.get("status", "--")

    frq_text   = f"{frq:.2f} Hz" if frq is not None else FRQ_HZ_DEFAULT
    power_text = f"{total_power:.1f} W" if total_power is not None else POWER_W_DEFAULT
    temp_text  = f"{avg_temp:.1f} °C" if avg_temp is not None else TEMP_C_DEFAULT

    return html.Div(className="card", children=[
        html.Div(className="row", children=[
            html.Span(RACK_ID, style={"color": COLOR_TEXT}),
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
    data = state.get(RACK_ID, {})

    frq         = data.get("frq_hz")
    total_power = data.get("total_power_w")
    avg_temp    = data.get("average_gpu_temp_c")
    gpu_power   = data.get("gpu_power_w", {})
    gpu_temp    = data.get("gpu_temp_c", {})
    cpu_power   = data.get("cpu_power_w", {})
    status      = data.get("status", "--")

    frq_text   = f"{frq:.2f} Hz" if frq is not None else FRQ_HZ_DEFAULT
    power_text = f"{total_power:.1f} W" if total_power is not None else POWER_W_DEFAULT
    temp_text  = f"{avg_temp:.1f} °C" if avg_temp is not None else TEMP_C_DEFAULT

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
        html.Div(RACK_ID, style={"color": COLOR_TEXT}),
        html.Hr(),

        _detail_row("Status", str(status), status_color(status)),
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
