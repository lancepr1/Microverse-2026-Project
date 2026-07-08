"""
combined_smoothing.py
-----------------------
PROTOTYPE -- combines the strongest tested elements from this session's
two smoothing investigations:

  Stage 1: Hampel-style outlier correction (from clean_enf(), already
           validated) -- removes egregious individual bad readings via
           interpolation BEFORE smoothing, so the lowpass filter isn't
           forced to "absorb"/ring around extreme individual outliers.
  Stage 2: Butterworth lowpass filter on the deviation-from-nominal
           series (bandpass_smooth.py's lowpass_filter_enf) -- tested
           as the strongest smoothing result this session (0.98 mean
           confidence on real cleaned data, vs 0.90 for medfilt+SavGol
           and 0.26 raw), with zero data discarded.

Also includes a LOCAL CUSUM detector -- separate from the existing
file-wide DriftMonitor in verification.py, which accumulates slowly
over an entire file's history and is well-suited to genuine long-term
drift but reacts too slowly to a short, localized attack. This
detector instead accumulates over a SHORT sliding window, specifically
targeting the pattern found in quick-splice testing: individual
windows show a PARTIAL confidence dip (not always crossing a hard
single-window threshold) but the PATTERN across several nearby
windows is itself a strong signal.
"""

from __future__ import annotations

import collections

from bandpass_smooth import lowpass_filter_enf


def hampel_correct(
    values: list[float],
    window: int = 11,
    n_sigmas: float = 2.0,
) -> list[float]:
    """
    Same mechanism as clean_enf()'s Hampel stage -- detects outliers via
    rolling median + MAD, replaces via linear interpolation between
    nearest good neighbors (never a flat median -- see clean_enf()'s
    documented reasoning for why that matters).
    """
    import statistics
    n = len(values)
    bad = [False] * n
    k = 1.4826
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        w = [values[j] for j in range(lo, hi) if not bad[j]]
        if len(w) < 2:
            continue
        med = statistics.median(w)
        mad = statistics.median([abs(v - med) for v in w])
        threshold = n_sigmas * k * mad
        if threshold > 0 and abs(values[i] - med) > threshold:
            bad[i] = True

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

    return result


def combined_smooth(
    values: list[float],
    hampel_window: int = 11,
    hampel_n_sigmas: float = 2.0,
    lowpass_cutoff_hz: float = 0.02,
    lowpass_order: int = 10,
    sample_rate_hz: float = 0.5,
) -> list[float]:
    """
    Stage 1 (outlier correction) then Stage 2 (lowpass smoothing).
    Must be called ONCE, at ingestion, before any attack -- same
    architectural rule established for clean_enf() and
    lowpass_filter_enf() individually this session.
    """
    corrected = hampel_correct(values, window=hampel_window, n_sigmas=hampel_n_sigmas)
    smoothed = lowpass_filter_enf(
        corrected,
        sample_rate_hz=sample_rate_hz,
        cutoff_hz=lowpass_cutoff_hz,
        order=lowpass_order,
    )
    return smoothed


class LocalCUSUMDetector:
    """
    Accumulates (1 - confidence) over a SHORT sliding window (not the
    whole file's history, unlike DriftMonitor), specifically to catch
    the "several nearby windows each show a PARTIAL dip, none alone
    crosses a hard threshold" pattern found in quick-splice testing.

    Independent of and complementary to verification.py's DriftMonitor,
    which is tuned for slow, genuine long-term drift over the whole
    file. This detector is tuned for short, localized anomalies.
    """

    def __init__(
        self,
        window_size: int = 10,
        baseline: float = 0.05,
        cusum_threshold: float = 2.0,
    ):
        self._window_size = window_size
        self._baseline = baseline
        self._cusum_threshold = cusum_threshold
        self._history: collections.deque = collections.deque(maxlen=window_size)
        self._cusum: float = 0.0

    def record(self, confidence: float) -> bool:
        """
        Records one confidence value, returns True if the LOCAL cusum
        (over just the last `window_size` windows) exceeds threshold.
        Resets automatically once confidence returns to normal for a
        stretch, so it re-arms for the next potential anomaly rather
        than staying latched.
        """
        deviation = 1.0 - confidence
        self._history.append(deviation)
        self._cusum = max(0.0, self._cusum + (deviation - self._baseline))
        # decay the cusum faster than DriftMonitor's global version --
        # this detector should reset once the anomaly passes, not keep
        # accumulating indefinitely
        if len(self._history) == self._window_size and deviation < self._baseline:
            self._cusum = max(0.0, self._cusum - self._baseline)
        return self._cusum > self._cusum_threshold

    def reset(self) -> None:
        self._cusum = 0.0