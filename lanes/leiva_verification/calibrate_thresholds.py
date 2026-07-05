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
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from microverse_core.data_loaders import read_combined_jsonl
from anchor import AnchorExtractor


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
    # Pass 1: collect raw FRQ values and GPU/CPU step sizes
    # ----------------------------------------------------------------
    frq_values = [r["FRQ"] for r in records]
    frq_deviations = [abs(f - 60.0) for f in frq_values]

    gpu_channels = ["gpu-0[W]", "gpu-1[W]", "gpu-2[W]", "gpu-3[W]"]
    cpu_channels = ["cpu-0[W]", "cpu-1[W]"]
    uj_channels  = ["cpu-0[uJ]", "cpu-1[uJ]"]

    gpu_steps = []
    cpu_steps = []
    prev = None
    for r in records:
        if prev is not None:
            for ch in gpu_channels:
                v, p = r.get(ch), prev.get(ch)
                if v is not None and p is not None:
                    gpu_steps.append(abs(v - p))
            for ch in cpu_channels:
                v, p = r.get(ch), prev.get(ch)
                if v is not None and p is not None:
                    cpu_steps.append(abs(v - p))
        prev = r

    # ----------------------------------------------------------------
    # Pass 2: run through AnchorExtractor and collect confidence values
    # ----------------------------------------------------------------
    enf_list = frq_values
    extractor = AnchorExtractor(enf=enf_list, sample_rate_hz=0.5)
    confidence_values = []

    for i, record in enumerate(records):
        ts = float(record["index"])
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
    # CONFIDENCE_TRUSTED (soft suspect below this) = slightly higher
    conf_suspect = round(percentile(confidence_values, fpr * 100), 4)
    conf_trusted = round(percentile(confidence_values, fpr * 2 * 100), 4)

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
    print(f"  P{fpr*200:.1f}                  : {percentile(confidence_values, fpr*200):.4f}")
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
    print(f"  GPU_MAX_STEP_W        = {gpu_max_step}")
    print(f"  CPU_MAX_STEP_W        = {cpu_max_step}")
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