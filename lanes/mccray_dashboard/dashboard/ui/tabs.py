"""
ui/tabs.py — the top-level tab bar (Operator / Analyst / Alert log /
Digital Twin) and the show/hide logic for each tab's content panel.

Follows the same click-pattern ui/controls.py already uses for its button
groups: a row of html.Button, a dcc.Store holding the selected value, and a
callback keyed off ctx.triggered_id that updates the store. A second
callback maps the store value to per-panel {"display": "block"/"none"}
styles, so every tab's content stays permanently mounted in the DOM (just
hidden when its tab isn't active) instead of being torn down and rebuilt on
every switch. That keeps every existing component id main.py's polling
callback targets (frq-graph, rack-card-container, etc.) present at all
times, so nothing about that callback has to change as content moves into
tabs in later steps.
"""
from dash import html, dcc, callback, Input, Output, State, ALL, ctx

TABS = [
    ("operator", "Operator"),
    ("analyst", "Analyst"),
    ("alert-log", "Alert log"),
    ("digital-twin", "Digital Twin"),
]
DEFAULT_TAB = "operator"

_CONTENT_IDS = [f"tab-content-{tab_id}" for tab_id, _ in TABS]


def build_tab_bar() -> html.Div:
    return html.Div(className="tab-bar", children=[
        dcc.Store(id="active-tab-store", data=DEFAULT_TAB),
        html.Div(className="btn-group", children=[
            html.Button(
                label,
                id={"type": "tab-btn", "tab": tab_id},
                n_clicks=0,
                className="btn tab-btn" + (" btn-active" if tab_id == DEFAULT_TAB else ""),
            )
            for tab_id, label in TABS
        ]),
    ])


def tab_content_style(tab_id: str) -> dict:
    """Initial inline style for a tab's content wrapper at layout build
    time, before any click has happened."""
    return {"display": "block"} if tab_id == DEFAULT_TAB else {"display": "none"}


@callback(
    Output("active-tab-store", "data"),
    Input({"type": "tab-btn", "tab": ALL}, "n_clicks"),
    State("active-tab-store", "data"),
    prevent_initial_call=True,
)
def _on_tab_click(_n_clicks, current):
    triggered = ctx.triggered_id
    if isinstance(triggered, dict) and triggered.get("type") == "tab-btn":
        return triggered["tab"]
    return current


@callback(
    *[Output(f"tab-content-{tab_id}", "style") for tab_id, _ in TABS],
    *[Output({"type": "tab-btn", "tab": tab_id}, "className") for tab_id, _ in TABS],
    Input("active-tab-store", "data"),
)
def _on_tab_switch(active_tab):
    styles = [
        {"display": "block"} if tab_id == active_tab else {"display": "none"}
        for tab_id, _ in TABS
    ]
    class_names = [
        "btn tab-btn" + (" btn-active" if tab_id == active_tab else "")
        for tab_id, _ in TABS
    ]
    return (*styles, *class_names)
