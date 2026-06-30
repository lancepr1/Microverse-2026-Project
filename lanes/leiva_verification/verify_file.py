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

Run from the repo root:
    python lanes/leiva_verification/verify_file.py \
        --input "/path/to/your/file.jsonl" \
        --component-id rack_00
"""

import argparse
import sys
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
        help="Print every record's result, not just non-TRUSTED ones"
    )
    args = parser.parse_args()

    print(f"Loading records from {args.input} ...")
    records = list(read_combined_jsonl(args.input))
    print(f"  -> {len(records)} records loaded\n")

    enf_list = [r["FRQ"] for r in records]
    extractor = AnchorExtractor(enf=enf_list, sample_rate_hz=0.5)
    verifier = Verifier(component_id=args.component_id, warmup_windows=10, check_nlr=True)

    results = []
    for record in records:
        record = dict(record)
        record["timestamp"] = float(record["index"])
        anchor = extractor.extract(record["timestamp"])
        result = verifier.verify(record, anchor)
        results.append(result)

        if args.show_all or result.status != "trusted":
            print(f"  [{record['index']:5d}] {result.status.upper():8s} "
                  f"score={result.score:.3f}  {result.reason}")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    counts = {"trusted": 0, "suspect": 0, "failed": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Total records : {len(results)}")
    print(f"  TRUSTED       : {counts.get('trusted', 0)}")
    print(f"  SUSPECT       : {counts.get('suspect', 0)}")
    print(f"  FAILED        : {counts.get('failed', 0)}")

    failed_indices = [r.timestamp for r in results if r.status == "failed"]
    if failed_indices:
        print(f"\n  Flagged (FAILED) at timestamps/indices: {[int(t) for t in failed_indices]}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()