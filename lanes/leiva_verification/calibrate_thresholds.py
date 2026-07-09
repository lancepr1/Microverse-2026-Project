"""
calibrate_thresholds.py
------------------------
Runs a known-clean combined JSONL file through the anchor extractor
and records every intermediate value produced on honest data. From
those observations it computes recommended threshold values that
minimize false positives while preserving detection sensitivity.

The principle: your thresholds should sit in the gap between what
honest data looks like and what tampered data looks like. If you set
them based on guesses rather than real observations, you either get
too many false positives (thresholds too tight) or miss real attacks
(thresholds too loose). This script eliminates the guessing.

NLR channels are discovered dynamically via verification._find_nlr_keys()
rather than hardcoded, since every column in the combined JSONL is
node-prefixed (e.g. "x3105c0s37b0n0_gpu-0[W]"), even in a 1-node file.
Hardcoding bare names like "gpu-0[W]" silently matches nothing and
falls back to whatever placeholder default is coded below it -- this
version discovers real keys so it either computes real numbers or
tells you plainly that it found none.

Run from the repo root against a known-clean combined JSONL file:
    python lanes/leiva_verification/calibrate_thresholds.py \
        --input "data/combined/run01.jsonl" \
        --false-positive-rate 0.01

--false-positive-rate (default 0.01 = 1%) controls how tight the
thresholds are. A lower rate means fewer false positives but slightly
less sensitivity to subtle attacks. 0.01 is a reasonable starting
point; tighten to 0.005 once you have real attack data to confirm
the gap between honest and tampered distributions is wide enough.

Output: prints recommended values to paste directly into the
threshold block at the top of verification.py
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from microverse_core.data_loaders import read_combined_jsonl

from anchor import AnchorExtractor
from verification import _find_nlr_keys

# Known RAPL hardware wraparound -- used only to separate expected
# counter wraps from real monotonicity violations in the sanity check.
WRAP_CEILING_UJ = 65_500_000_000
WRAP_TOLERANCE_UJ = 2_000_000_000


def percentile(values: list[float], p: float) -> float:
    """p-th percentile via linear interpolation."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    idx = (p / 100) * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate verification thresholds from clean data"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a known-clean combined JSONL file"
    )
    parser.add_argument(
        "--false-positive-rate", type=float, default=0.01,
        help="Target false positive rate (default 0.01 = 1%%)"
    )
    args = parser.parse_args()

    print(f"Loading clean records from {args.input} ...")
    records = list(read_combined_jsonl(args.input))
    print(f"  -> {len(records)} records loaded")
    print(f"  -> target false positive rate: {args.false_positive_rate:.1%}\n")

    if not records:
        print("ERROR: no records loaded")
        return

    # ----------------------------------------------------------------
    # Pass 1: collect raw FRQ values, NLR channels (dynamic, node-aware),
    # and GPU/CPU step sizes + uJ wraparound sanity check
    # ----------------------------------------------------------------
    frq_values = [r["FRQ"] for r in records]
    frq_deviations = [abs(f - 60.0) for f in frq_values]

    channels = _find_nlr_keys(records[0])
    print(f"  -> discovered {len(channels['gpu_power'])} GPU power channels, "
          f"{len(channels['cpu_power'])} CPU power channels, "
          f"{len(channels['cpu_uj'])} CPU energy channels\n")

    gpu_steps = []
    cpu_steps = []
    cpu_uj_violations = []
    prev = None
    for i, r in enumerate(records):
        if prev is not None:
            for ch in channels["gpu_power"]:
                v, p = r.get(ch), prev.get(ch)
                if v is not None and p is not None:
                    gpu_steps.append(abs(v - p))
            for ch in channels["cpu_power"]:
                v, p = r.get(ch), prev.get(ch)
                if v is not None and p is not None:
                    cpu_steps.append(abs(v - p))
            for ch in channels["cpu_uj"]:
                v, p = r.get(ch), prev.get(ch)
                if v is not None and p is not None and v < p:
                    drop = p - v
                    if abs(drop - WRAP_CEILING_UJ) >= WRAP_TOLERANCE_UJ:
                        cpu_uj_violations.append((i, ch, drop))
        prev = r

    if not gpu_steps and channels["gpu_power"]:
        print("WARNING: GPU channels were found but no consecutive-window "
              "steps could be computed -- check the file has >1 record.\n")
    if not channels["gpu_power"]:
        print("WARNING: no GPU power channels discovered at all -- "
              "GPU_MAX_STEP_W below is an uncalibrated fallback, not a "
              "real measurement.\n")
    if not channels["cpu_power"]:
        print("WARNING: no CPU power channels discovered at all -- "
              "CPU_MAX_STEP_W below is an uncalibrated fallback, not a "
              "real measurement.\n")

    # ----------------------------------------------------------------
    # Pass 2: run through AnchorExtractor and collect confidence values
    # ----------------------------------------------------------------
    enf_list = frq_values
    extractor = AnchorExtractor(enf=enf_list, sample_rate_hz=0.5)
    confidence_values = []

    # FIXED (2026-07): was ts = float(record["index"]), which passes the raw
    # record index directly as if it were already in seconds. AnchorExtractor
    # expects real elapsed seconds (its own docstring: "t=2.0 -> index 1" at
    # 0.5 Hz) -- passing raw index silently caused int(timestamp *
    # sample_rate_hz) to under-advance, visiting each real window twice and
    # never reaching roughly the back half of the file as a window center at
    # all. Same bug found and fixed in verify_file.py -- every threshold
    # calibrated before this fix was computed against the wrong windowing.
    for i, record in enumerate(records):
        ts = float(record["index"]) / 0.5
        anchor = extractor.extract(ts)
        confidence_values.append(anchor.confidence)

    # skip index 0 since it always returns 1.0 (no previous window)
    confidence_values = confidence_values[1:]

    # ----------------------------------------------------------------
    # Compute recommended thresholds
    # ----------------------------------------------------------------
    fpr = args.false_positive_rate
    upper_pct = (1.0 - fpr) * 100  # e.g. 99.0 for 1% FPR

    # ENF nominal range: set tolerance so (1 - fpr) of honest readings pass
    tolerance_hz = round(percentile(frq_deviations, upper_pct) * 1.1, 4)

    # Confidence thresholds:
    # CONFIDENCE_SUSPECT (hard failure below this) = percentile that catches fpr
    # CONFIDENCE_TRUSTED (soft suspect below this) = a wider gap above it --
    # P1/P2 (fpr vs 2x fpr) leaves almost no SUSPECT band and is fragile
    # against the known ENF glitch spikes contaminating the low tail, so
    # TRUSTED is pinned to the median instead of a nearby percentile.
    conf_suspect = round(percentile(confidence_values, fpr * 100), 4)
    conf_trusted = round(percentile(confidence_values, 50), 4)

    # CUSUM baseline: mean of (1 - confidence) on honest data
    cusum_baseline = round(1.0 - statistics.mean(confidence_values), 4)

    # NLR step thresholds: P99 of honest step sizes + 20% margin
    gpu_max_step = round(percentile(gpu_steps, upper_pct) * 1.2, 2) if gpu_steps else 470.0
    cpu_max_step = round(percentile(cpu_steps, upper_pct) * 1.2, 2) if cpu_steps else 16.0

    # ----------------------------------------------------------------
    # Print analysis
    # ----------------------------------------------------------------
    print("=" * 60)
    print("RAW FREQUENCY (Hz)")
    print("=" * 60)
    print(f"  mean                  : {statistics.mean(frq_values):.4f} Hz")
    print(f"  std                   : {statistics.stdev(frq_values):.4f} Hz")
    print(f"  min / max             : {min(frq_values):.4f} / {max(frq_values):.4f} Hz")
    print(f"  mean deviation        : {statistics.mean(frq_deviations):.4f} Hz")
    print(f"  P{upper_pct:.0f} deviation      : {percentile(frq_deviations, upper_pct):.4f} Hz")

    print(f"\n{'=' * 60}")
    print("WINDOW CONFIDENCE (Pearson correlation with previous window)")
    print("=" * 60)
    print(f"  samples               : {len(confidence_values)}")
    print(f"  mean                  : {statistics.mean(confidence_values):.4f}")
    print(f"  std                   : {statistics.stdev(confidence_values):.4f}")
    print(f"  min                   : {min(confidence_values):.4f}")
    print(f"  P{fpr*100:.1f}                  : {percentile(confidence_values, fpr*100):.4f}")
    print(f"  P50 (median)          : {percentile(confidence_values, 50):.4f}")
    print()
    print("  False positive rate at candidate thresholds:")
    for t in [0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.30, 0.10, 0.0]:
        below = sum(1 for c in confidence_values if c < t)
        pct = 100 * below / len(confidence_values)
        bar = "#" * int(pct / 2)
        print(f"    below {t:.2f}: {below:4d} windows ({pct:5.1f}% FPR)  {bar}")

    if gpu_steps:
        print(f"\n{'=' * 60}")
        print("GPU POWER STEP SIZES (W between consecutive windows)")
        print("=" * 60)
        print(f"  mean                  : {statistics.mean(gpu_steps):.2f} W")
        print(f"  std                   : {statistics.stdev(gpu_steps):.2f} W")
        print(f"  max                   : {max(gpu_steps):.2f} W")
        print(f"  P{upper_pct:.0f}               : {percentile(gpu_steps, upper_pct):.2f} W")

    if cpu_steps:
        print(f"\n{'=' * 60}")
        print("CPU POWER STEP SIZES (W between consecutive windows)")
        print("=" * 60)
        print(f"  mean                  : {statistics.mean(cpu_steps):.2f} W")
        print(f"  std                   : {statistics.stdev(cpu_steps):.2f} W")
        print(f"  max                   : {max(cpu_steps):.2f} W")
        print(f"  P{upper_pct:.0f}               : {percentile(cpu_steps, upper_pct):.2f} W")

    print(f"\n{'=' * 60}")
    print("CPU uJ MONOTONICITY SANITY CHECK (should be 0 on clean data)")
    print("=" * 60)
    print(f"  unexplained violations : {len(cpu_uj_violations)}"
          f"{'  <-- check parsing, not expected on clean data' if cpu_uj_violations else ''}")
    for i, ch, drop in cpu_uj_violations[:5]:
        print(f"    record {i}: {ch} dropped {drop:.1f} uJ")

    # ----------------------------------------------------------------
    # Recommended thresholds
    # ----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"RECOMMENDED THRESHOLDS (target FPR = {fpr:.1%})")
    print("Copy these into the threshold block at the top of verification.py")
    print("=" * 60)
    print()
    print(f"  # ENF nominal range -- raw frequency check")
    print(f"  NOMINAL_HZ            = 60.0")
    print(f"  NOMINAL_TOLERANCE_HZ  = {tolerance_hz}")
    print()
    print(f"  # Confidence thresholds")
    print(f"  CONFIDENCE_TRUSTED    = {conf_trusted}   "
          f"# below this -> SUSPECT")
    print(f"  CONFIDENCE_SUSPECT    = {conf_suspect}   "
          f"# below this -> FAILED (hard discontinuity)")
    print()
    print(f"  # CUSUM drift detection")
    print(f"  CUSUM_BASELINE        = {cusum_baseline}   "
          f"# mean(1 - confidence) on clean data")
    print()
    print(f"  # NLR step thresholds")
    print(f"  GPU_MAX_STEP_W        = {gpu_max_step}"
          f"{'   # FALLBACK -- no channels found, not calibrated' if not gpu_steps else ''}")
    print(f"  CPU_MAX_STEP_W        = {cpu_max_step}"
          f"{'   # FALLBACK -- no channels found, not calibrated' if not cpu_steps else ''}")
    print()
    print(f"{'=' * 60}")
    print()
    print("NOTE: CONFIDENCE_SUSPECT controls your hard-failure boundary.")
    print("Setting it too low means subtle injection attacks producing a")
    print("sharp confidence drop may not reach FAILED status.")
    print("After updating verification.py, re-run test_attack_detection.py")
    print("to confirm all three attack types are still caught as FAILED.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()