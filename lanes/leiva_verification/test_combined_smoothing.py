"""
test_combined_smoothing.py
-----------------------------
Standalone validation suite for combined_smoothing.py + bandpass_smooth.py.
Runs the same battery of tests used to validate this approach during
development: clean-data confidence, sustained-attack detection (correct
pipeline order -- smooth once, then attack the smoothed stream, never
re-smooth downstream), a quick-splice sweep comparing single-window vs
local CUSUM detection, and a false-positive check on clean data.

Needs to sit in the same folder as anchor.py, combined_smoothing.py,
and bandpass_smooth.py (lanes/leiva_verification/).

Usage:
    python test_combined_smoothing.py --path /path/to/your/Dev1_ENF_Hr01.csv
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from microverse_core.data_loaders import load_enf
from anchor import AnchorExtractor
from combined_smoothing import combined_smooth, LocalCUSUMDetector


def confidence_stats(series, sample_rate_hz=0.5):
    extractor = AnchorExtractor(enf=series, sample_rate_hz=sample_rate_hz)
    confs = []
    for i in range(len(series)):
        ts = float(i) / sample_rate_hz
        confs.append(extractor.extract(ts).confidence)
    return confs[1:]


def inject_replay(series, idx, replay_from, length):
    s = list(series)
    s[idx:idx + length] = series[replay_from:replay_from + length]
    return s


def main():
    parser = argparse.ArgumentParser(description="Validate the combined smoothing pipeline against real ENF data")
    parser.add_argument("--path", required=True, help="Path to a real ENF CSV file")
    parser.add_argument("--lowpass-cutoff", type=float, default=0.02)
    parser.add_argument("--single-window-threshold", type=float, default=0.85)
    args = parser.parse_args()

    if not Path(args.path).exists():
        print(f"ERROR: file not found: {args.path}")
        return

    print(f"Loading {args.path} ...")
    raw = load_enf(args.path)
    print(f"  -> {len(raw)} samples\n")

    if len(raw) < 950:
        print("WARNING: file has fewer than 950 samples -- the splice test uses "
              "index 900, this may not work correctly on a short file.\n")

    print("Applying combined smoothing pipeline (Hampel outlier correction + Butterworth lowpass)...")
    smoothed = combined_smooth(raw, lowpass_cutoff_hz=args.lowpass_cutoff)
    print("  -> done\n")

    # ------------------------------------------------------------
    # TEST 1: clean-data confidence, raw vs smoothed
    # ------------------------------------------------------------
    print("=" * 70)
    print("TEST 1: Clean-data confidence (raw vs smoothed)")
    print("=" * 70)
    raw_confs = confidence_stats(raw)
    smoothed_confs = confidence_stats(smoothed)
    print(f"  RAW:      mean={statistics.mean(raw_confs):.4f}  median={statistics.median(raw_confs):.4f}")
    print(f"  SMOOTHED: mean={statistics.mean(smoothed_confs):.4f}  median={statistics.median(smoothed_confs):.4f}")
    baseline = statistics.mean([1 - c for c in smoothed_confs])
    print(f"  Calibrated local-CUSUM baseline (mean 1-confidence on smoothed clean data): {baseline:.4f}")

    # ------------------------------------------------------------
    # TEST 2: sustained attack -- CORRECT pipeline order (smooth once,
    # then attack the already-smoothed stream, never re-smooth)
    # ------------------------------------------------------------
    print()
    print("=" * 70)
    print("TEST 2: Sustained attack (20 samples, FRQ forced to 0.0)")
    print("=" * 70)
    attacked = list(smoothed)
    attack_start, attack_len = 20, 20
    for i in range(attack_start, attack_start + attack_len):
        attacked[i] = 0.0
    extractor = AnchorExtractor(enf=attacked, sample_rate_hz=0.5)
    detector = LocalCUSUMDetector(window_size=10, baseline=baseline, cusum_threshold=2.0)
    sw_hits, cusum_hits = 0, 0
    for i in range(len(attacked)):
        ts = float(i) / 0.5
        c = extractor.extract(ts).confidence
        flagged = detector.record(c)
        if attack_start <= i < attack_start + attack_len:
            if c < args.single_window_threshold:
                sw_hits += 1
            if flagged:
                cusum_hits += 1
    print(f"  single-window (threshold={args.single_window_threshold}): {sw_hits}/{attack_len} windows caught")
    print(f"  local CUSUM detector:                     {cusum_hits}/{attack_len} windows caught")

    # ------------------------------------------------------------
    # TEST 3: quick-splice sweep
    # ------------------------------------------------------------
    print()
    print("=" * 70)
    print("TEST 3: Quick-splice sweep (single-window vs local CUSUM)")
    print("=" * 70)
    print(f"  {'length':>8s} {'seconds':>8s} {'single-window':>14s} {'local CUSUM':>12s}")
    for length in [1, 2, 4, 6, 10, 14, 22]:
        spliced = inject_replay(smoothed, 900, 100, length=length)
        extractor = AnchorExtractor(enf=spliced, sample_rate_hz=0.5)
        detector = LocalCUSUMDetector(window_size=10, baseline=baseline, cusum_threshold=2.0)
        sw_hits, cusum_hits = 0, 0
        for i in range(len(spliced)):
            ts = float(i) / 0.5
            c = extractor.extract(ts).confidence
            flagged = detector.record(c)
            if 900 <= i < 900 + length:
                if c < args.single_window_threshold:
                    sw_hits += 1
                if flagged:
                    cusum_hits += 1
        print(f"  {length:8d} {length*2:8d} {sw_hits:6d}/{length:<7d} {cusum_hits:6d}/{length}")

    # ------------------------------------------------------------
    # TEST 4: false positives on completely clean data
    # ------------------------------------------------------------
    print()
    print("=" * 70)
    print("TEST 4: False positive rate on completely clean smoothed data")
    print("=" * 70)
    extractor = AnchorExtractor(enf=smoothed, sample_rate_hz=0.5)
    detector = LocalCUSUMDetector(window_size=10, baseline=baseline, cusum_threshold=2.0)
    n_flagged = 0
    for i in range(len(smoothed)):
        ts = float(i) / 0.5
        c = extractor.extract(ts).confidence
        if detector.record(c):
            n_flagged += 1
    print(f"  Local CUSUM false positives: {n_flagged}/{len(smoothed)} ({100*n_flagged/len(smoothed):.2f}%)")

    print()
    print("=" * 70)
    print("Reference numbers from this approach's original validation (Dev1_ENF_Hr01.csv):")
    print("  Clean smoothed confidence: mean~0.978, median~0.990")
    print("  Sustained attack: single-window 9/20, local CUSUM 20/20")
    print("  Splice sweep: local CUSUM caught 100% at every tested length (2-44s)")
    print("  False positives: 0.00% on clean data")
    print("If your numbers look meaningfully different, that's real information")
    print("about how this generalizes to different files -- worth flagging, not")
    print("just re-running until it matches.")
    print("=" * 70)


if __name__ == "__main__":
    main()