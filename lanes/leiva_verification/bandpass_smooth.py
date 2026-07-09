"""
bandpass_smooth.py
--------------------
PROTOTYPE -- testing whether a proper digital filter (applied to the
already-extracted ENF value sequence) can smooth the data the way the
original documented pipeline's 10th-order Butterworth bandpass filter
did on the raw audio, before frequency estimation.

IMPORTANT DISTINCTION from clean_enf():
  clean_enf() (already built, already shipped) does POINT-WISE
  rejection -- flags individual bad readings, replaces them via
  interpolation between good neighbors. At a tight tolerance (e.g.
  +/-0.02 Hz), this would flag 80-99% of real files (measured
  directly), meaning it would fabricate almost the entire signal.

  This module instead applies an actual LOWPASS FILTER to the
  deviation-from-nominal series. Every output point is a weighted
  combination of real input points -- nothing is discarded or
  replaced with guesswork. This is the standard meaning of "smooth
  the data with a Butterworth filter," and does not require throwing
  away the majority of the real signal the way a tight point-wise
  gate would.

Two functions are provided so both interpretations can be tested and
compared directly:
  lowpass_filter_enf()   -- Option B: proper digital filter (recommended)
  gate_and_interpolate()  -- Option A: point-wise range gate (for comparison,
                              mirrors what clean_enf()'s physical-range
                              check already does, just parameterized to
                              any bandwidth)
"""

from __future__ import annotations

from scipy.signal import butter, filtfilt


def lowpass_filter_enf(
    values: list[float],
    sample_rate_hz: float = 0.5,
    cutoff_hz: float = 0.01,
    order: int = 10,
    nominal: float = 60.0,
) -> list[float]:
    """
    Applies a zero-phase Butterworth lowpass filter to the
    deviation-from-nominal series (v - nominal), then adds nominal
    back. Filtering the deviation rather than the raw values avoids
    any edge/transient artifacts from the large constant 60.0 offset.

    cutoff_hz here is the cutoff of the LOWPASS FILTER APPLIED TO THE
    TIME SERIES ITSELF -- units are "cycles per second of the ENF
    reading sequence" (how fast the ENF value is allowed to change),
    NOT "Hz of allowable deviation from 60" the way clean_enf()'s
    physical_floor/physical_ceiling are. These are conceptually
    different knobs -- see module docstring.

    order=10 matches the order used in the original documented
    pipeline (10th-order Butterworth), applied here to the extracted
    value sequence rather than the raw audio it was originally
    designed for.

    Uses filtfilt (forward-backward filtering) for zero phase shift --
    a plain forward-only filter would introduce a time lag, which
    would be actively harmful here since AnchorExtractor's window
    comparisons depend on precise timing alignment.
    """
    nyquist = sample_rate_hz / 2.0
    normalized_cutoff = min(cutoff_hz / nyquist, 0.99)  # keep below Nyquist

    deviation = [v - nominal for v in values]
    b, a = butter(order, normalized_cutoff, btype='low')
    smoothed_deviation = filtfilt(b, a, deviation)

    return [d + nominal for d in smoothed_deviation]


def gate_and_interpolate(
    values: list[float],
    band_hz: float = 0.2,
    nominal: float = 60.0,
) -> tuple[list[float], float]:
    """
    Option A, for direct comparison: point-wise range gate. Any
    reading outside [nominal - band_hz, nominal + band_hz] is treated
    as bad and replaced via linear interpolation between the nearest
    good neighbors (same mechanism as clean_enf()'s physical-range
    check, parameterized to any bandwidth for this comparison).

    Returns (result, fraction_replaced) -- the fraction is reported
    explicitly since at tight bandwidths this can be the majority of
    the file, which matters a great deal for whether this is a
    reasonable thing to do at all.
    """
    n = len(values)
    bad = [abs(v - nominal) > band_hz for v in values]
    fraction_replaced = sum(bad) / n if n else 0.0

    result = list(values)
    bad_idx = [i for i, b in enumerate(bad) if b]
    for i in bad_idx:
        lo = i - 1
        while lo >= 0 and bad[lo]:
            lo -= 1
        hi = i + 1
        while hi < n and bad[hi]:
            hi += 1
        if lo >= 0 and hi < n:
            frac = (i - lo) / (hi - lo)
            result[i] = values[lo] + frac * (values[hi] - values[lo])
        elif lo >= 0:
            result[i] = values[lo]
        elif hi < n:
            result[i] = values[hi]

    return result, fraction_replaced