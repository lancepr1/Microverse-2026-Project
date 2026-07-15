"""
ui/blender_feed.py — the "Live Digital Twin" panel: polls a PNG written by
an external Blender process and displays it if present.

_get_latest_frame() is the entire integration surface with Blender. It
currently mtime-polls a file on disk; when Blender starts pushing frames
over a socket instead, only this function's body changes — the callback
and the html.Img component below stay the same. This has no dependency on
data/run01.jsonl or any external package; it simply keeps showing
"DT loading..." if the file never appears, so it works standalone with no
Blender process running at all.
"""
import base64
import os
import tempfile

from dash import html, dcc, callback, Input, Output, no_update

# CHANGED (2026-07): was a hardcoded "/tmp/..." -- /tmp doesn't exist on
# Windows, which would have silently broken this panel entirely for the
# new Windows teammate (mtime lookup just always fails -> permanently
# stuck on "DT loading..."). tempfile.gettempdir() resolves to the
# correct OS temp directory on Windows/Mac/Linux. MUST match
# main_run.py's own _capture_viewport() default exactly -- two separate
# processes (Blender and this dashboard) on the same machine, only
# working because tempfile.gettempdir() is deterministic per-OS, not
# because the two files coordinate directly.
IMG_PATH    = os.path.join(tempfile.gettempdir(), "blender_viewport.png")
FEED_WIDTH  = 300
FEED_HEIGHT = 720

_last_mtime = [0.0]


def render_blender_feed() -> html.Div:
    return html.Div(className="blender-feed", children=[
        dcc.Interval(id="blender-interval", interval=500),
        html.Div("Live Digital Twin", className="label"),
        html.Img(id="blender-img", className="blender-img",
                 style={"display": "none"}),
        html.Div("DT loading...", id="blender-loading-text",
                  className="dimmed-block"),
    ])


def _get_latest_frame() -> bytes | None:
    """Returns the latest PNG bytes if the file has changed since the last
    call, else None (missing or unchanged)."""
    try:
        mtime = os.path.getmtime(IMG_PATH)
    except OSError:
        return None
    if mtime == _last_mtime[0]:
        return None
    _last_mtime[0] = mtime
    try:
        with open(IMG_PATH, "rb") as f:
            return f.read()
    except OSError:
        return None


@callback(
    Output("blender-img", "src"),
    Output("blender-img", "style"),
    Output("blender-loading-text", "style"),
    Input("blender-interval", "n_intervals"),
    prevent_initial_call=True,
)
def _on_blender_tick(_n):
    frame = _get_latest_frame()
    if frame is None:
        return no_update, no_update, no_update

    encoded = base64.b64encode(frame).decode("ascii")
    src = f"data:image/png;base64,{encoded}"
    return src, {"display": "block"}, {"display": "none"}
