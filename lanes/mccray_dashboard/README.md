# McCray: operator dashboard and integration

## What it does
The operator-facing screen: per-component twin state, verification status, and
attack alerts. Also owns a chunk of the integration plumbing. Your prior
Python + Tkinter + SQLite monitoring project is close to the shape needed.

## How to run
Read twin state via `blender_bridge.get_state` / `list_state`. Read
verification via `io_records.read_records(run_id, "verification")`. Render
status as green/yellow/red from `VerificationStatus`, and an alert log ordered
by severity. Use `metrics` for the summary numbers.

## Who to ask
Lance (day-to-day, integration), Dr. Qu (is the interface compatible with the
verification output format).

## Week-1 deliverable
UI mockup showing rack/PDU state cards with workload class and live power draw,
verification indicators per anchor, and a time-ordered alert log.
