"""
tools/generate_verification.py — offline step that runs Leiva's
AnchorExtractor + Verifier over a recorded telemetry file and writes the
results to runs/<run_id>/verification.jsonl via microverse_core.io_records,
the file-bus format contracts.py defines for VerificationResult:
    runs/<run_id>/verification.jsonl   VerificationResult   (Leiva)

This script is intentionally NOT part of the dashboard package -- the
dashboard/ folder must stay importable with no sibling-lane code present
(see tests/test_ui_imports.py::test_no_microverse_core_dependency_anywhere).
It's a one-off/offline generation step: run it manually whenever the input
recording changes, then the dashboard's own verification_feed.py just reads
the plain JSON it produces.

    python lanes/mccray_dashboard/tools/generate_verification.py \
        --input lanes/mccray_dashboard/data/run01.jsonl --run-id run01
"""
import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEIVA_DIR = _REPO_ROOT / "lanes" / "leiva_verification"

for _path in (_REPO_ROOT, _LEIVA_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from microverse_core.io_records import write_records  # noqa: E402
from anchor import AnchorExtractor                     # noqa: E402
from verification import Verifier                      # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Run Leiva's Verifier over a recorded telemetry file and "
                    "write runs/<run_id>/verification.jsonl for the dashboard to read."
    )
    parser.add_argument("--input", required=True, help="Path to the combined JSONL recording")
    parser.add_argument("--run-id", required=True, help="e.g. run01 -- becomes runs/<run_id>/")
    parser.add_argument("--component-id", default=None,
                         help="Node id these results are attributed to; must match the "
                              "dashboard's run_id (derived from the run file name, e.g. "
                              "run01.jsonl -> run01) for component_id to line up. Defaults "
                              "to --run-id.")
    args = parser.parse_args()
    component_id = args.component_id or args.run_id

    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    enf_list = [r["FRQ"] for r in records]
    extractor = AnchorExtractor(enf=enf_list, sample_rate_hz=0.5)
    verifier = Verifier(component_id=component_id, warmup_windows=10, check_nlr=True)

    all_results = []
    for record in records:
        record = dict(record)
        record["timestamp"] = float(record["index"])
        anchor = extractor.extract(record["timestamp"])
        all_results.extend(verifier.verify(record, anchor))

    path = write_records(args.run_id, "verification", all_results)
    print(f"Wrote {len(all_results)} VerificationResult records to {path}")


if __name__ == "__main__":
    main()
