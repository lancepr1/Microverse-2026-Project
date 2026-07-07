// Dash auto-serves any .js dropped in assets/, same as style.css.
//
// Dash's html.Button only exposes single-click n_clicks, not a
// double-click event -- and a real double-click also fires two ordinary
// clicks first. So chart-pause-btn's single/double click are told apart
// here with a debounce timer, and routed into chart-pause-store, which
// ui/controls.py's _on_chart_pause callback reads.
(function () {
  var DBLCLICK_MS = 300;

  document.addEventListener("click", function (e) {
    if (e.target.id !== "chart-pause-btn") return;
    if (window._chartPauseTimer) return; // part of a double click in progress
    window._chartPauseTimer = setTimeout(function () {
      window._chartPauseTimer = null;
      window.dash_clientside.set_props("chart-pause-store", {
        data: { action: "pause", t: Date.now() },
      });
    }, DBLCLICK_MS);
  });

  document.addEventListener("dblclick", function (e) {
    if (e.target.id !== "chart-pause-btn") return;
    clearTimeout(window._chartPauseTimer);
    window._chartPauseTimer = null;
    window.dash_clientside.set_props("chart-pause-store", {
      data: { action: "resume", t: Date.now() },
    });
  });
})();
