"""
diagnose_uj_violations.py
--------------------------
Inspects the RAW (pre-aggregation) RAPL samples around a flagged
NLRMonotonicityCheck / calibrate_thresholds.py violation, to check
whether the RAPL hardware wraparound (~65.5B uJ) is landing in the
MIDDLE of an aggregation window rather than at a window boundary.

Why this matters: _aggregate_cpu() in data_loaders.py takes a plain
mean of raw uJ samples per window. That's correct for an instantaneous
measurement (power, W) but wrong for a cumulative energy counter --
if the wrap happens mid-window, the window's mean is a blend of
pre-wrap and post-wrap values, producing a distorted number and a
window-to-window "drop" that doesn't match the clean ~65.5B ceiling.
This script shows the raw samples directly so you can see it happen.

Usage -- one violation:
    python diagnose_uj_violations.py \
        --nlr-folder "/home/brandon/Desktop/00_raw_datasets/training_llama2_70b_lora/16node/" \
        --slurm-id 10742842 \
        --violation "x3106c0s5b0n0:cpu-1[uJ]:1"

Usage -- multiple violations in one run:
    python diagnose_uj_violations.py \
        --nlr-folder "..." --slurm-id 10742842 \
        --violation "x3106c0s5b0n0:cpu-1[uJ]:1" \
        --violation "x3106c0s25b0n0:cpu-1[uJ]:10" \
        --violation "x3106c0s25b0n0:cpu-1[uJ]:11" \
        --violation "x3108c0s5b0n0:cpu-0[uJ]:35" \
        --violation "x3108c0s5b0n0:cpu-0[uJ]:36"

The node_id:channel:window_index format matches what
calibrate_thresholds.py prints in its violation list.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from microverse_core.data_loaders import (
    discover_nlr_pairs,
    _detect_sample_rate_hz,
    _parse_rapl_log,
)

CHANNEL_ATTR = {
    "cpu-0[uJ]": "cpu0_uj",
    "cpu-1[uJ]": "cpu1_uj",
    "cpu-0-core[uJ]": "cpu0_core_uj",
    "cpu-1-core[uJ]": "cpu1_core_uj",
}

WRAP_CEILING_UJ = 65_500_000_000
WRAP_TOLERANCE_UJ = 2_000_000_000


def parse_violation(spec: str):
    node_id, channel, window_idx = spec.split(":")
    return node_id, channel, int(window_idx)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect raw RAPL samples around a flagged uJ monotonicity violation"
    )
    parser.add_argument("--nlr-folder", required=True)
    parser.add_argument("--slurm-id", default=None)
    parser.add_argument(
        "--violation", action="append", required=True,
        help="node_id:channel:window_index, e.g. x3106c0s5b0n0:cpu-1[uJ]:1 -- repeatable"
    )
    parser.add_argument(
        "--enf-sample-rate", type=float, default=0.5,
        help="Must match what pipeline_test.py used (default 0.5 Hz)"
    )
    args = parser.parse_args()

    pairs = discover_nlr_pairs(args.nlr_folder, slurm_id=args.slurm_id)
    rapl_by_node = {node_id: rapl for node_id, _nvml, rapl in pairs}

    for spec in args.violation:
        node_id, channel, window_idx = parse_violation(spec)
        attr = CHANNEL_ATTR.get(channel)
        if attr is None:
            print(f"Unknown channel '{channel}', skipping. Known: {list(CHANNEL_ATTR)}")
            continue
        if node_id not in rapl_by_node:
            print(f"Node '{node_id}' not found in {args.nlr_folder}, skipping "
                  f"(available: {sorted(rapl_by_node)[:5]}...)")
            continue

        rapl_path = rapl_by_node[node_id]
        nlr_hz = _detect_sample_rate_hz(rapl_path)
        samples_per_window = round(nlr_hz * (1.0 / args.enf_sample_rate))

        rows = _parse_rapl_log(rapl_path)

        lo = max(0, (window_idx - 1) * samples_per_window)
        hi = (window_idx + 2) * samples_per_window
        chunk = rows[lo:hi]

        print(f"\n{'='*70}")
        print(f"{node_id}  {channel}  window {window_idx}  "
              f"(samples_per_window={samples_per_window}, raw rows {lo}:{hi})")
        print(f"{'='*70}")

        prev_val = None
        for raw_i, row in enumerate(chunk, start=lo):
            val = getattr(row, attr)
            win = raw_i // samples_per_window
            at_boundary = (raw_i % samples_per_window == 0)
            marker = ""
            if prev_val is not None and val < prev_val:
                drop = prev_val - val
                is_full_wrap = abs(drop - WRAP_CEILING_UJ) < WRAP_TOLERANCE_UJ
                kind = "full wrap" if is_full_wrap else "PARTIAL -- mid-window!"
                marker = f"  <-- raw-sample DROP {drop:,.1f} uJ ({kind})"
            boundary_marker = "  | window boundary" if at_boundary else ""
            print(f"  raw[{raw_i:5d}] win={win:4d} {attr}={val:,.1f}{boundary_marker}{marker}")
            prev_val = val

    print(f"\n{'='*70}")
    print("READ THIS: if a raw-sample DROP appears on a line that is NOT")
    print("marked 'window boundary', the RAPL counter wrapped in the middle")
    print("of an aggregation window. _aggregate_cpu() takes a plain mean of")
    print("raw uJ values per window, so that window's average blends")
    print("pre-wrap and post-wrap values -- producing a distorted number and")
    print("a window-to-window 'drop' that won't match the clean ~65.5B")
    print("ceiling. That's the likely source of the 58.7B / 19.5B / 45.7B")
    print("drops calibrate_thresholds.py flagged, instead of one clean wrap.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()