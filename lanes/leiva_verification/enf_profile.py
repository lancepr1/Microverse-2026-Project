"""
enf_profile.py
--------------
Statistical profile of the ENF dataset to determine detection thresholds.

Run this against your real ENF file before finalizing any threshold values
in leiva_verification/verification.py. The output tells you exactly what
honest ENF looks like for this specific recording so your thresholds are
grounded in real data rather than guesses.

Usage:
    python enf_profile.py --path data/enf/your_enf_file.csv

Output:
    Prints a full statistical summary and recommended threshold values
    you can copy directly into verification.py.
"""

import argparse
import csv
import math
import statistics
from pathlib import Path


# ---------------------------------------------------------------------------
# ENF loader (mirrors the updated load_enf in data_loaders.py)
# ---------------------------------------------------------------------------

def load_enf(path: str) -> list[float]:
    """Load the real ENF file -- skips metadata row, reads column 1."""
    values = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # skip metadata row (UTC timestamp + duration)
        for row in reader:
            if not row:
                continue
            try:
                values.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    return values


# ---------------------------------------------------------------------------
# Statistical calculations (standard library only)
# ---------------------------------------------------------------------------

def step_sizes(values: list[float]) -> list[float]:
    """Absolute difference between each consecutive pair of readings."""
    return [abs(values[i+1] - values[i]) for i in range(len(values) - 1)]


def pearson(a: list[float], b: list[float]) -> float:
    """Pearson correlation between two equal-length lists."""
    n = min(len(a), len(b))
    if n < 2:
        return 1.0
    mean_a = statistics.mean(a[:n])
    mean_b = statistics.mean(b[:n])
    num = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    den_a = math.sqrt(sum((a[i] - mean_a) ** 2 for i in range(n)))
    den_b = math.sqrt(sum((b[i] - mean_b) ** 2 for i in range(n)))
    if den_a == 0 or den_b == 0:
        return 0.0
    return max(-1.0, min(1.0, num / (den_a * den_b)))


def normalize(window: list[float]) -> list[float]:
    """Min-max normalize to [0, 1] matching AnchorExtractor._normalize."""
    lo, hi = min(window), max(window)
    span = hi - lo
    if span == 0:
        return [0.5] * len(window)
    return [(v - lo) / span for v in window]


def window_correlations(
    values: list[float],
    window_radius: int = 5,
) -> list[float]:
    """
    Compute Pearson correlation between every consecutive pair of
    normalized windows, exactly as AnchorExtractor does at runtime.
    Returns a list of correlation values across the full recording.
    """
    correlations = []
    prev_window = None
    window_size = 2 * window_radius + 1

    for i in range(len(values)):
        lo = max(0, i - window_radius)
        hi = min(len(values), i + window_radius + 1)
        # pad edges by repeating nearest value
        window = []
        for j in range(i - window_radius, i + window_radius + 1):
            clamped = max(0, min(j, len(values) - 1))
            window.append(values[clamped])

        norm = normalize(window)

        if prev_window is not None:
            correlations.append(pearson(norm, prev_window))
        prev_window = norm

    return correlations


def autocorrelation(values: list[float], max_lag: int = 10) -> list[float]:
    """
    Autocorrelation at lags 1 through max_lag.
    High autocorrelation at short lags confirms the slow random walk.
    """
    mean = statistics.mean(values)
    variance = sum((v - mean) ** 2 for v in values)
    if variance == 0:
        return [0.0] * max_lag

    acf = []
    n = len(values)
    for lag in range(1, max_lag + 1):
        cov = sum(
            (values[i] - mean) * (values[i + lag] - mean)
            for i in range(n - lag)
        )
        acf.append(cov / variance)
    return acf


def percentile(values: list[float], p: float) -> float:
    """
    Compute the p-th percentile (0-100) of a list.
    Uses linear interpolation matching numpy's default behavior.
    """
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    idx = (p / 100) * (n - 1)
    lo_idx = int(idx)
    hi_idx = min(lo_idx + 1, n - 1)
    frac = idx - lo_idx
    return sorted_vals[lo_idx] * (1 - frac) + sorted_vals[hi_idx] * frac


# ---------------------------------------------------------------------------
# Main profiling routine
# ---------------------------------------------------------------------------

def profile(path: str, window_radius: int = 5, sample_rate_hz: float = 0.5):

    print(f"\n{'='*60}")
    print(f"ENF STATISTICAL PROFILE")
    print(f"{'='*60}")
    print(f"File            : {path}")
    print(f"Sample rate     : {sample_rate_hz} Hz "
          f"({1/sample_rate_hz:.1f}s between readings)")
    print(f"Window radius   : {window_radius} samples "
          f"({window_radius/sample_rate_hz:.0f}s each side)")

    # load
    values = load_enf(path)
    if not values:
        print("ERROR: no values loaded -- check file path and format")
        return

    duration_s = len(values) / sample_rate_hz
    print(f"Samples loaded  : {len(values)}")
    print(f"Duration        : {duration_s:.0f}s "
          f"({duration_s/3600:.2f} hours)")

    # ----------------------------------------------------------------
    # Section 1: Raw frequency statistics
    # ----------------------------------------------------------------
    mean_f   = statistics.mean(values)
    std_f    = statistics.stdev(values)
    min_f    = min(values)
    max_f    = max(values)
    range_f  = max_f - min_f

    print(f"\n{'─'*60}")
    print("RAW FREQUENCY (Hz)")
    print(f"{'─'*60}")
    print(f"  Mean           : {mean_f:.6f} Hz")
    print(f"  Std deviation  : {std_f:.6f} Hz")
    print(f"  Min            : {min_f:.6f} Hz")
    print(f"  Max            : {max_f:.6f} Hz")
    print(f"  Range          : {range_f:.6f} Hz")
    print(f"  P5             : {percentile(values, 5):.6f} Hz")
    print(f"  P95            : {percentile(values, 95):.6f} Hz")

    deviation_from_60 = [abs(v - 60.0) for v in values]
    print(f"\n  Deviation from 60.0 Hz:")
    print(f"    Mean         : {statistics.mean(deviation_from_60):.6f} Hz")
    print(f"    Max          : {max(deviation_from_60):.6f} Hz")
    print(f"    P99          : {percentile(deviation_from_60, 99):.6f} Hz")

    # ----------------------------------------------------------------
    # Section 2: Step sizes (continuity check calibration)
    # ----------------------------------------------------------------
    steps = step_sizes(values)
    mean_step = statistics.mean(steps)
    std_step  = statistics.stdev(steps)
    max_step  = max(steps)
    p99_step  = percentile(steps, 99)

    print(f"\n{'─'*60}")
    print("STEP SIZES BETWEEN CONSECUTIVE SAMPLES (Hz)")
    print(f"{'─'*60}")
    print(f"  Mean step      : {mean_step:.6f} Hz")
    print(f"  Std deviation  : {std_step:.6f} Hz")
    print(f"  Max step       : {max_step:.6f} Hz")
    print(f"  P95 step       : {percentile(steps, 95):.6f} Hz")
    print(f"  P99 step       : {p99_step:.6f} Hz")

    # ----------------------------------------------------------------
    # Section 3: Window correlations (confidence calibration)
    # ----------------------------------------------------------------
    print(f"\n{'─'*60}")
    print("WINDOW CORRELATIONS (confidence calibration)")
    print(f"{'─'*60}")
    print(f"  Computing correlations across {len(values)} windows...")

    corrs = window_correlations(values, window_radius=window_radius)
    mean_corr = statistics.mean(corrs)
    std_corr  = statistics.stdev(corrs)
    min_corr  = min(corrs)
    p1_corr   = percentile(corrs, 1)
    p5_corr   = percentile(corrs, 5)

    print(f"  Mean           : {mean_corr:.4f}")
    print(f"  Std deviation  : {std_corr:.4f}")
    print(f"  Min            : {min_corr:.4f}")
    print(f"  P1             : {p1_corr:.4f}")
    print(f"  P5             : {p5_corr:.4f}")

    # how many windows fall below candidate thresholds
    for threshold in [0.90, 0.85, 0.80, 0.70, 0.60, 0.50]:
        below = sum(1 for c in corrs if c < threshold)
        pct   = 100 * below / len(corrs)
        print(f"  Below {threshold:.2f}      : {below:4d} windows "
              f"({pct:.1f}% false positive rate)")

    # ----------------------------------------------------------------
    # Section 4: Autocorrelation (confirms slow random walk)
    # ----------------------------------------------------------------
    acf = autocorrelation(values, max_lag=5)
    print(f"\n{'─'*60}")
    print("AUTOCORRELATION (confirms slow random walk)")
    print(f"{'─'*60}")
    for lag, val in enumerate(acf, start=1):
        bar = '#' * int(abs(val) * 20)
        print(f"  Lag {lag}  : {val:+.4f}  {bar}")

    # ----------------------------------------------------------------
    # Section 5: Recommended thresholds
    # ----------------------------------------------------------------
    # Range check: use P99 deviation from 60 Hz + 20% margin
    recommended_tolerance = round(
        percentile(deviation_from_60, 99) * 1.2, 4
    )

    # Continuity: use P99 step size + 20% margin
    recommended_max_step = round(p99_step * 1.2, 4)

    # Confidence thresholds: mean - N*std, never below 0
    recommended_trusted  = round(max(0.0, mean_corr - 1.0 * std_corr), 4)
    recommended_suspect  = round(max(0.0, mean_corr - 2.5 * std_corr), 4)

    # CUSUM baseline: mean of (1 - correlation)
    mean_deviation_proxy = round(1.0 - mean_corr, 4)

    print(f"\n{'─'*60}")
    print("RECOMMENDED THRESHOLDS FOR verification.py")
    print(f"{'─'*60}")
    print(f"  # ENF range tolerance (Hz)")
    print(f"  TOLERANCE_HZ      = {recommended_tolerance}")
    print()
    print(f"  # Continuity check -- max plausible step (Hz)")
    print(f"  MAX_STEP_HZ       = {recommended_max_step}")
    print()
    print(f"  # Confidence thresholds")
    print(f"  CONFIDENCE_TRUSTED = {recommended_trusted}  "
          f"# mean - 1*std  ({1.0 - recommended_trusted:.1%} false positive rate)")
    print(f"  CONFIDENCE_SUSPECT = {recommended_suspect}  "
          f"# mean - 2.5*std (hard failure below this)")
    print()
    print(f"  # CUSUM baseline deviation proxy")
    print(f"  CUSUM_BASELINE    = {mean_deviation_proxy}  "
          f"# 1 - mean_correlation")
    print()
    print(f"{'─'*60}")
    print(f"Copy these values into the threshold block at the top of")
    print(f"leiva_verification/verification.py")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Statistical profile of ENF dataset for threshold calibration"
    )
    parser.add_argument(
        "--path", required=True,
        help="Path to ENF CSV file"
    )
    parser.add_argument(
        "--window-radius", type=int, default=5,
        help="Window radius for correlation computation (default 5, "
             "must match AnchorExtractor window_radius)"
    )
    parser.add_argument(
        "--sample-rate", type=float, default=0.5,
        help="ENF sample rate in Hz (default 0.5 for real AFRL dataset)"
    )
    args = parser.parse_args()

    if not Path(args.path).exists():
        print(f"ERROR: file not found: {args.path}")
        return

    profile(
        path=args.path,
        window_radius=args.window_radius,
        sample_rate_hz=args.sample_rate,
    )


if __name__ == "__main__":
    main()