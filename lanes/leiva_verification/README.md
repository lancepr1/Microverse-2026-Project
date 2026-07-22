# Leiva: sensor anchors and verification logic

## What it does

Extracts a physical anchor from ENF (Electric Network Frequency) and
runs it, alongside GPU/CPU telemetry, through a verification engine
that flags tampered data. Produces `AnchorRecord` and
`VerificationResult` records consumed by the rest of the pipeline
(digital twin, dashboard, scoring).

## Files in this lane

| File | Purpose |
|---|---|
| `verification.py` | Core detection engine -- `Verifier` and all 17 ENF/NLR checks. |
| `test_verifier.py` | Regression suite for `verification.py`. Run this after any change. |
| `anchor.py` | `AnchorExtractor` -- builds the ENF signature/confidence anchor. |

This lane previously also included `calibrate_thresholds.py` (a
threshold-recalibration utility), `verify_file.py` (a manual
single-file verification utility), and `diagnose_enf_glitch.py` (a
diagnostic tool built for one specific, already-resolved ENF
false-positive investigation). All three have been removed as this
project wound down -- their design history is still preserved in
`.readme/` for reference (`calibrate_thresholds.md`,
`verify_file.md`, `diagnose_enf_glitch.md`) in case any of them are
useful as a starting point for someone picking this project back up.

## Documentation convention

Code comments in this lane are minimal by design -- every function and
class carries a standard docstring (what it does, arguments, return
value), and nothing else. The design history, calibration numbers,
real bugs found and fixed, and honest limitations behind every
threshold and check live in `.readme/`, one file per source file
(e.g. `verification.py`'s history is in `.readme/verification.md`).
Read the relevant `.readme/*.md` file before changing a threshold or
a check's behavior -- the reasoning for why a value is what it is is
there, not in the code.

## How to run

```
python lanes/leiva_verification/test_verifier.py
```

should print `69/69 passed`. For a full pipeline run, see the
repo-root `README.md` and `scripts/run_microverse.py`.

## Project status

This lane is functionally complete and in its final, minimal form:
`verification.py`, `test_verifier.py`, and `anchor.py` are the real,
ongoing deliverable and should be kept as-is going forward. The
supporting utilities that existed alongside them during development
have been removed; see `.readme/` for their preserved history if
they're ever needed again.

## Who to ask

Dr. Qu (methodology, ANCHOR-Grid), Lance (repo and integration).