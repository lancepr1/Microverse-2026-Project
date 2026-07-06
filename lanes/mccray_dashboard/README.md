# McCray: operator dashboard and integration

## What it does
The operator-facing screen: rack/PDU status cards, live FRQ/power charts, a
facility KPI panel, a local history log, and a Blender viewport panel. Runs
standalone today, replaying recorded telemetry (`data/run01.jsonl`); the
`status` field each card reads is a placeholder for Leiva's
`VerificationResult` — nothing in this lane computes it yet (see "Known gaps").

## How to run
```
pip install -r requirements.txt
cd lanes/mccray_dashboard/dashboard
python main.py
```
Open `http://127.0.0.1:8050`. `pytest lanes/mccray_dashboard/tests/` runs the
unit tests (model parsing, replay pacing, history round-trip, import sanity).

## Who to ask
Lance (day-to-day, integration), Dr. Qu (is the interface compatible with the
verification output format).

## Week-1 deliverable
Rack/PDU state cards with live power draw and FRQ, done. Workload class,
verification indicators, and the time-ordered alert log are not built yet —
they depend on `StateVariable`/`VerificationResult` records this lane doesn't
produce or consume yet.

## Known gaps
Nothing here generates attacks, detects anomalies, or computes a trust score —
that's Leiva's and Marchisano's lanes. `status` is read from the polled state
dict and displayed as-is, defaulting to `"--"`; wiring it to
`microverse_core.io_records.read_records(run_id, "verification")` and
`VerificationStatus` is the next step once that output format is final.
