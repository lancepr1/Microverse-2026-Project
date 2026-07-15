# McCray: operator dashboard and integration

## What it does
The operator-facing screen: rack/PDU status cards, live FRQ/power charts, a
facility KPI panel, a local history log, and a Blender viewport panel. Runs
standalone today, replaying recorded telemetry (`data/run01.jsonl`); the
`status` field each card reads is a placeholder for Leiva's
`VerificationResult` — nothing in this lane computes it yet (see "Known gaps").

## How to run

### Data locations (one-time setup)
The pipeline now reads data from fixed paths automatically — no more typing
paths at the prompts. Place your files here first:
- NLR raw datasets: `~/Projects/00_raw_datasets/`
- ENF data: `~/Projects/ENF-ML (CNN+MAMBA)/Data/`
- Digital twin `.blend`: `~/Projects/`

### Full pipeline → dashboard (cold start)
1. `cd` to the repo root.
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt` (only if imports fail)
4. `python scripts/run_microverse.py`
5. Answer the prompts (data paths are auto-detected from the locations above):
   - Dataset: `training_llama2_70b_lora`
   - Nodes: `16node`
   - SLURM job ID: any (e.g. option 1)
   - Recording device: `Dev1`
   - Hour: any (e.g. `Hr01`)
   - Component ID: press enter (default `rack_00`)
6. `attack.py` launches automatically. Answer its prompts:
   - Difficulty: 1 (Easy)
   - Attack Engine: pick an UNUSED number (check
     `lanes/marchisano_attacks/outputs/` for `attack_easy_N_check.jsonl` —
     reusing N crashes the run)
   - Target Metric Category: 6 (Everything)
   - If Injection engine: Sub-Type 1 (Bias), value 10.0 (or enter for default)
7. Outputs land in `lanes/leiva_verification/outputs/` (`for_scoreboard.jsonl`,
   `for_dashboard.jsonl`, `for_digital_twin.jsonl`). Node IDs in
   `for_dashboard.jsonl` are auto-normalized — no manual step.
8. `cd lanes/mccray_dashboard/dashboard`
9. `python main.py`
10. Open `http://127.0.0.1:8050/`
11. If the page errors or shows "Total Nodes: 0", a stale process holds the
    port: `pkill -9 -f main.py`, then repeat step 9.

### Dashboard only (replay mode, no pipeline run)
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
