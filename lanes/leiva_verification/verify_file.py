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
    args = parser.parse_args()

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
        record["timestamp"] = float(record["index"])
        anchor = extractor.extract(record["timestamp"])

        # verify() now returns a LIST -- one merged ENF result plus one
        # result per NLR/GPU-temp channel present, every call, always.
        results = verifier.verify(record, anchor)

        for result in results:
            component_counts[result.component_id][result.status] += 1
            total_counts[result.status] += 1

            if args.show_all or result.status != "trusted":
                print(f"  [{record['index']:5d}] {result.component_id:40s} "
                      f"{result.status.upper():8s} score={result.score:.3f}  {result.reason}")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY (across all components, all windows)")
    print(f"{'='*70}")
    print(f"  TRUSTED : {total_counts['trusted']}")
    print(f"  SUSPECT : {total_counts['suspect']}")
    print(f"  FAILED  : {total_counts['failed']}")

    flagged = {
        cid: counts for cid, counts in component_counts.items()
        if counts["failed"] or counts["suspect"]
    }
    if flagged:
        print(f"\n  Components with at least one non-TRUSTED result:")
        for cid, counts in sorted(flagged.items()):
            print(f"    {cid:40s} trusted={counts['trusted']:5d}  "
                  f"suspect={counts['suspect']:4d}  failed={counts['failed']:4d}")
    else:
        print("\n  Every component was TRUSTED for the entire file.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()