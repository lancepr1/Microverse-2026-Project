"""
verify_file.py
---------------
Runs an ALREADY-tampered (or already-clean) combined JSONL file straight
through AnchorExtractor + Verifier, with no attack injection. Use this
when the input file has already been altered upstream -- by Ethan's
attack module, or by you manually editing a file -- and you just want
to see what your verification system catches.

This is the counterpart to test_attack_detection.py, which is for when
you want to inject a synthetic attack yourself into a clean file. If the
file is already in whatever state you want to test, use this script
instead.

Feed it the WHOLE combined record, all nodes at once -- one Verifier
instance is enough. Verifier.verify() attributes every result down to
the exact column name (which already contains the node ID, e.g.
"x3105c0s37b0n0_gpu-0[W]"), and never stops early, so per-node/
per-component attribution falls out naturally with no manual slicing.

Run from the repo root:
    python lanes/leiva_verification/verify_file.py \
        --input "/path/to/your/file.jsonl" \
        --component-id rack_00
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from microverse_core.data_loaders import read_combined_jsonl

from anchor import AnchorExtractor
from verification import Verifier


def main():
    parser = argparse.ArgumentParser(description="Verify a combined JSONL file with no attack injection")
    parser.add_argument("--input", required=True, help="Path to the JSONL file to verify")
    parser.add_argument("--component-id", default="rack_00")
    parser.add_argument(
        "--show-all", action="store_true",
        help="Print every component's result every window, not just non-TRUSTED ones"
    )
    parser.add_argument(
        "--output",
        help="Write the complete SUSPECT/FAILED report (plus summary) to this file, "
             "in addition to printing to console. No truncation -- every flagged "
             "line included, useful for reviewing everything before importing "
             "downstream (e.g. into Blender)."
    )
    args = parser.parse_args()

    out_lines = []

    def emit(text=""):
        print(text)
        if args.output:
            out_lines.append(text)

    print(f"Loading records from {args.input} ...")
    records = list(read_combined_jsonl(args.input))
    print(f"  -> {len(records)} records loaded\n")

    enf_list = [r["FRQ"] for r in records]
    extractor = AnchorExtractor(enf=enf_list, sample_rate_hz=0.5)
    verifier = Verifier(component_id=args.component_id, warmup_windows=10, check_nlr=True)

    # Per-component counters -- component_id already carries node + channel
    component_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"trusted": 0, "suspect": 0, "failed": 0})
    total_counts = {"trusted": 0, "suspect": 0, "failed": 0}

    for record in records:
        record = dict(record)
        # FIXED (2026-07): was record["timestamp"] = float(record["index"]),
        # which passes the raw record index directly as if it were already
        # in seconds. AnchorExtractor expects real elapsed seconds (its own
        # docstring: "t=2.0 -> index 1" at 0.5 Hz) -- passing raw index
        # silently caused int(timestamp * sample_rate_hz) to under-advance,
        # visiting each real window twice and never reaching roughly the
        # back half of the file as a window center at all. Confirmed via
        # a direct precision/recall test against ground-truth attack
        # labels: this bug does NOT affect ENFNominalRangeCheck (operates
        # on raw FRQ directly, no windowing), but DOES affect every
        # confidence-based check (continuity, drift).
        record["timestamp"] = float(record["index"]) / 0.5
        anchor = extractor.extract(record["timestamp"])

        # verify() now returns a LIST -- one merged ENF result plus one
        # result per NLR/GPU-temp channel present, every call, always.
        results = verifier.verify(record, anchor)

        for result in results:
            component_counts[result.component_id][result.status] += 1
            total_counts[result.status] += 1

            if args.show_all or result.status != "trusted":
                emit(f"  [{record['index']:5d}] {result.component_id:40s} "
                     f"{result.status.upper():8s} score={result.score:.3f}  {result.reason}")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    emit(f"\n{'='*70}")
    emit("OVERALL SUMMARY (across all components, all windows)")
    emit(f"{'='*70}")
    emit(f"  TRUSTED : {total_counts['trusted']}")
    emit(f"  SUSPECT : {total_counts['suspect']}")
    emit(f"  FAILED  : {total_counts['failed']}")

    flagged = {
        cid: counts for cid, counts in component_counts.items()
        if counts["failed"] or counts["suspect"]
    }
    if flagged:
        emit(f"\n  Components with at least one non-TRUSTED result:")
        for cid, counts in sorted(flagged.items()):
            emit(f"    {cid:40s} trusted={counts['trusted']:5d}  "
                 f"suspect={counts['suspect']:4d}  failed={counts['failed']:4d}")
    else:
        emit("\n  Every component was TRUSTED for the entire file.")
    emit(f"{'='*70}\n")

    if args.output:
        Path(args.output).write_text("\n".join(out_lines) + "\n")
        print(f"Full flagged report written to {args.output}")


if __name__ == "__main__":
    main()