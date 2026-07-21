# Microverse-2026-Project

A research project building an ENF (Electric Network Frequency) + NLR
(node-level resource: GPU/CPU wattage, temperature, energy) verification
system for a simulated datacenter digital twin.

---

## Goal

Modern datacenters increasingly need to trust the telemetry they're acting
on — is this GPU wattage reading real, or has it been tampered with,
replayed, or spoofed? This project explores whether **physical signals that
are hard for an attacker to fake** can be used to verify that the rest of a
datacenter's telemetry is trustworthy, inspired by two real research
papers:

- **ANCHOR-Grid** (Hatami, Qu, Chen et al., *Sensors* 2025)
- **SAVE** (Qu, Chen, Blasch, *Future Internet* 2025)

Both papers use ENF — the tiny, constantly-fluctuating deviations in a power
grid's AC frequency around 60 Hz — as a physical "anchor": something a
remote attacker can't simply overwrite, because it's tied to the real
electrical grid a facility is plugged into. This project extends that idea
to node-level resource telemetry (GPU/CPU power draw, temperature, energy
counters) and builds an end-to-end pipeline that:

1. Ingests real recorded ENF and GPU/CPU telemetry data
2. Simulates a realistic tampering attack against it (replay, injection,
   drift, extreme value substitution)
3. Runs a verification system that flags which telemetry it does and
   doesn't trust, per node and per physical channel
4. Displays the results live on both a web dashboard and a 3D digital
   twin of the simulated datacenter

The goal isn't a production security product — it's a testbed for
answering a concrete question: **how well can physical-signal-based
verification actually catch realistic tampering, and where does it
genuinely fail?** Both the wins and the honest failures below are part of
that answer.

---

## Known Limitations

These are real, measured gaps — found by testing against real attack data,
not guessed. Reported here the same way they'd be reported in a results
section, not hidden.

- **Constant-offset / slow-drift ENF attacks are a genuine, mathematically
  proven blind spot.** The ENF alternative-correlation check (and every
  other confidence-based ENF check) relies on Pearson correlation, which is
  mathematically invariant to any positive-scale affine transform:
  `corr(x, a·x + b) = 1.0` for any `a > 0`. Confirmed directly against a
  real attack scenario that applied a slow, smooth drift to ENF (growing to
  about −0.04 Hz over ~1260 windows): overall recall was only **47.5%**,
  because the drift was gradual enough that window-to-window confidence
  barely moved (94% of the attack window stayed above the hard-fail
  threshold). This is not a tuning problem — no amount of threshold
  adjustment fixes an invariance that's true by definition of the math
  being used. A genuinely different signal (e.g. checking window-to-window
  *variance*, not just shape correlation) would be needed to close this.
- **Coordinated, "replay everything on a node together" NLR attacks**
  remain a real, confirmed weak spot. `_CrossSiblingConsistencyCheck` gets
  100% recall against a single GPU being replayed on its own, but if an
  attacker replays *every* GPU on a node together, sibling ratios stay
  internally consistent and recall drops to 2–46%.
- **CPU wattage sibling-consistency detection is meaningfully weaker**
  (3–33% recall) than GPU (100%) — fewer siblings per node (2 vs. 4) and
  less tightly synchronized workload.
- **The replay/consecutive-value check's parameters are reasoned, not yet
  fully validated.** `REPLAY_MATCH_LENGTH` and `REPLAY_LOOKBACK_WINDOW` are
  set from `attack.py`'s documented ~90-sample replay cycle, not yet
  confirmed against a real, labeled Replay-engine attack file.
- **Startup-ramp calibration comes from a small sample.** The GPU
  wattage-ramp thresholds (target range, deadline, sustain window) are
  calibrated from only 4 real runs, all the same workload type
  (LLaMA-2-70B LoRA training) — may not generalize to a meaningfully
  different workload's startup profile.
- **GPU temperature has no startup-ramp check yet** — a real signal exists
  in the data (idle ~40–41°C rising to steady-state ~61–69°C), but no
  target range has been calibrated for it, unlike wattage.
- **The dashboard's status display is discretized, not continuous.** By
  design, `for_dashboard.jsonl` only carries the compressed 0.0/0.5/1.0
  trust score per node — not the underlying continuous confidence score or
  per-check reason text. One direct consequence: the Alert Log's "Attack
  Vector" column will always show "Unclassified," since it has no reason
  text to classify from.
- **The CNN+Mamba classifier** explored early in the project (ENF-specific)
  was deprioritized in favor of the cheaper threshold/rule-based checks
  that make up the current system, given realistic time and data-volume
  constraints. It has known caveats (possible data leakage, a missing
  dependency) and was never validated against real ground truth.
- **The full pipeline → dashboard → digital-twin launch chain** has not
  yet been confirmed working end-to-end from inside a real Blender session
  by this lane — `main_run.py`'s own logic has been validated separately,
  but the live hand-off hasn't been watched happen for real yet.

---

## Breakdown of Work

Five lanes, working in parallel against a shared set of interface
contracts (`microverse_core/contracts.py`) so no one lane blocks another.

| Lane | Owner | Responsibility |
|---|---|---|
| **Verification** | Leiva | `lanes/leiva_verification/` — `anchor.py` (ENF signature/confidence extraction), `verification.py` (all 17 detection checks + the Verifier), `test_verifier.py` (regression suite), and `scripts/run_microverse.py` (the full pipeline orchestrator) |
| **Attack injection** | Ethan Marchisano | `lanes/marchisano_attacks/attack.py` — simulates realistic tampering (Replay, Injection, Extreme, Drift engines) against clean telemetry, with ground-truth labels kept separate for scoring |
| **State management** | Hendricks | Twin state variables and the Blender integration layer (`blender_bridge.py`) inside `microverse_core/` |
| **Digital twin** | Baron | The Blender scene itself and `main_run.py` — plays back verified telemetry live, coloring nodes/ENF green/yellow/red by trust status |
| **Dashboard** | McCray | `lanes/mccray_dashboard/` — the live web dashboard (Operator, Analyst, Alert Log, and Digital Twin tabs) |
| **Integration layer** | shared (Lance / Hendricks) | `microverse_core/` — the contracts every lane builds against, the file-bus (`io_records.py`), data loaders, and scoring metrics (`metrics.py`) |

**Core governing principles**, agreed on across every lane:

1. No detection check gets access to data a simulated attacker doesn't also
   have access to.
2. Nothing gets added to the compressed dashboard-facing JSONL purely to
   help detection — only data the digital twin/dashboard actually needs
   belongs there.
3. Every threshold is calibrated against real data, never assumed.
4. Honest, weak, or negative results get reported as clearly as strong
   ones — several ideas were tried and explicitly abandoned after real
   testing rather than kept for looking good on paper.

---

## Running the Program

### 1. Where to put your files

All local data lives in one shared, repo-relative location — **not**
inside your home directory, and **not** committed to git (see
`.gitignore` — it's excluded for space reasons):

```
<repo root>/data/rawdata/
├── 00_raw_datasets/              <- your raw NLR (GPU/CPU) datasets
│   ├── training_llama2_70b_lora/
│   ├── training_stable_diffusion/
│   └── ...
├── ENF-ML (CNN+MAMBA)/
│   └── Data/                     <- DevN_ENF_HrNN.csv files
└── YourDigitalTwinScene.blend    <- the .blend file, directly in this folder
```

This exact same folder works on Windows, Mac, and Linux — nothing to edit
per person, per machine.

### 2. One-time environment setup

```
cd <repo root>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Make sure Blender is installed and its folder is on your system `PATH` —
the pipeline launches it as plain `blender`, relying on `PATH` to find it.

### 3. Run the pipeline

```
python scripts/run_microverse.py
```

### 4. What to expect — the prompts, in order

**Data ingestion (this script's own prompts):**

1. **Which dataset to ingest?** — a menu built from whatever folders are
   actually present under `data/rawdata/00_raw_datasets/`
   (`training_*`, `inference_*`).
2. **How many nodes?** — options differ per dataset (not a fixed list);
   whatever node-count folders actually exist under the dataset you chose.
3. **Which SLURM job ID?** — only asked for training datasets. Discovered
   from the actual filenames present, since one node folder can contain
   more than one training run.
4. **Which recording device / which hour?** — for the ENF data, from
   whatever `DevN_ENF_HrNN.csv` files are present.
5. **Component ID** `[rack_00]` — a label for which simulated rack this run
   represents. Just press Enter for the default unless you're specifically
   simulating more than one rack.

**Attack injection (`attack.py`'s own prompts — this script hands off the
terminal to it directly):**

6. **Detection Evaluation Difficulty** — Easy / Medium / Hard.
7. **Attack Engine** — Extreme Injection, Replay, Injection (bias/scale/
   absolute), or Drift. If you pick a difficulty/engine combo whose output
   filename already exists (Easy mode uses fixed filenames), the run will
   fail — pick one you haven't used yet, or check
   `lanes/marchisano_attacks/outputs/` for what's already there.
8. **Target Metric Category** — which telemetry gets attacked: FRQ (ENF)
   only, GPU metrics, CPU metrics, or Everything.
9. *(If you picked Injection)* — Injection sub-type (Bias/Scale/Absolute)
   and a value, or press Enter for the default.

After that, the pipeline runs the rest automatically — no more prompts.

### 5. Opening the dashboard

Once the pipeline reaches the dashboard-launch stage, it starts
automatically in the background. Open a browser and go to:

```
http://127.0.0.1:8050/
```

Four tabs:

- **Operator** — a live grid of every node, colored by trust status, with
  a click-through detail panel showing every GPU and CPU's own readings
  individually.
- **Analyst** — per-rack power/temperature charts over time.
- **Alert Log** — a timeline of every trust-status episode, per node and
  for ENF specifically.
- **Digital Twin** — a live mirror of the Blender viewport.

If the dashboard shows "Total Nodes: 0" or looks broken, it's almost
always a stale process from a previous run still holding the port —
`pkill -9 -f main.py`, then re-run.

### 6. The digital twin

Blender launches after the dashboard and plays back the same verified data
live, coloring each node/GPU/CPU and the ENF grid anchor green (trusted),
yellow (suspect), or red (failed). If more than one `.blend` file exists in
`data/rawdata/`, you'll be asked which one to use; with exactly one
present, it's picked automatically.

**Closing the Blender window ends the run** — the dashboard is
automatically shut down at that point too, so both processes stop cleanly
together.

### Troubleshooting

- **`ModuleNotFoundError: No module named 'dash'`** — the dashboard's
  dependencies aren't installed in your active environment. Re-run step 2
  above with your `.venv` activated.
- **Dashboard shows stale/blended data from a previous run** — clear
  `runs/<component_id>/verification.jsonl` if it exists, or just re-run
  the full pipeline; recent fixes make this self-cleaning on every run.
- **Nothing prints to the console during a run** — this is intentional.
  Routine step-by-step status output was turned off by default once the
  dashboard could show results live; genuine warnings and errors still
  print. If you want the old verbose output back for debugging, set
  `VERBOSE = True` near the top of `scripts/run_microverse.py`.
