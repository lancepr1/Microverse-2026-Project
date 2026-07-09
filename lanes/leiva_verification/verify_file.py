"""
verify_file.py
---------------
Runs an ALREADY-tampered (or already-clean) combined JSONL file straight
through AnchorExtractor + Verifier, with no attack injection. Use this
when the input file has already been altered upstream -- by Ethan's
attack module, or by you manually editing a file -- and you just want
to see what your verification system catches.

OUTPUT (2026-07 redesign): writes a JSONL file identical in shape to
the input -- every original field untouched -- with ONE new field
added: "status", a numeric trust score:
    0.0  = TRUSTED  (nothing flagged this window)
    0.5  = SUSPECT  (soft flag, worst result across all components this window)
    1.0  = FAILED   (hard flag, worst result across all components this window)
This is the file meant to go to the scoreboard, alongside the known-
attack ground truth file, to score how well the verifier did. It is
NOT a modification of the data Baron receives -- that stays whatever
file Ethan's attack module produced; this is a separate, derived file.
Defaults to writing anchor_verified.jsonl in the current directory --
override with --output if you want a different name/location.

No console output except errors -- runs silently on success. If you
want to inspect results interactively, read the output file back in
rather than expecting anything printed here.

Run from the repo root:
    python lanes/leiva_verification/verify_file.py \\
        --input "/path/to/your/file.jsonl" \\
        --component-id rack_00
    # writes ./anchor_verified.jsonl by default
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from microverse_core.data_loaders import read_combined_jsonl

from anchor import AnchorExtractor
from verification import Verifier

# Worst-of ranking (higher = worse) and the numeric score the scoreboard expects.
STATUS_RANK  = {"trusted": 0, "suspect": 1, "failed": 2}
STATUS_SCORE = {"trusted": 0.0, "suspect": 0.5, "failed": 1.0}


def main():
    parser = argparse.ArgumentParser(description="Verify a combined JSONL file, produce a scoreboard-ready output")
    parser.add_argument("--input", required=True, help="Path to the JSONL file to verify")
    parser.add_argument("--component-id", default="rack_00")
    parser.add_argument(
        "--output", default="anchor_verified.jsonl",
        help="Path to write the scoreboard-ready JSONL: a copy of the input file "
             "with one added numeric 'status' field (0.0=trusted, 0.5=suspect, "
             "1.0=failed -- worst result across every component checked that "
             "window). Defaults to anchor_verified.jsonl in the current directory."
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: file not found: {args.input}")
        return

    records = list(read_combined_jsonl(args.input))

    enf_list = [r["FRQ"] for r in records]
    extractor = AnchorExtractor(enf=enf_list, sample_rate_hz=0.5)
    verifier = Verifier(component_id=args.component_id, warmup_windows=10, check_nlr=True)

    # Per-component counters -- component_id already carries node + channel
    component_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"trusted": 0, "suspect": 0, "failed": 0})
    total_counts = {"trusted": 0, "suspect": 0, "failed": 0}

    out_fh = open(args.output, "w")

    for record in records:
        scoreboard_record = dict(record)
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

        # verify() returns a LIST -- one merged ENF result plus one result
        # per NLR/GPU-temp channel present, every call, always.
        results = verifier.verify(record, anchor)

        # Worst-of across every component checked this window -- one
        # simple numeric field, matching the annotated-output design
        # already used elsewhere. No per-component detail in this file
        # by design; that's what console output (below) is for.
        worst = "trusted"
        for result in results:
            if STATUS_RANK[result.status] > STATUS_RANK[worst]:
                worst = result.status
        scoreboard_record["status"] = STATUS_SCORE[worst]

        for result in results:
            component_counts[result.component_id][result.status] += 1
            total_counts[result.status] += 1

        out_fh.write(json.dumps(scoreboard_record) + "\n")

    out_fh.close()

    # component_counts / total_counts are still tracked above in case
    # you want to inspect them interactively or re-add reporting later --
    # just no longer printed, per request to remove console output.


if __name__ == "__main__":
    main()