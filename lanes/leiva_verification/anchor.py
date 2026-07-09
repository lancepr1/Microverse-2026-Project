"""
anchor.py
---------
AnchorExtractor: builds one AnchorRecord per time step from an ENF stream.

Replaces fake_anchor in scripts/smoke_test.py.

The AnchorRecord produced here carries three things per the task spec:
  timestamp  : elapsed seconds matching the PowerSample being processed
  signature  : normalized ENF window -- the shape fingerprint
  confidence : Pearson correlation with the previous window -- smoothness
               metric that the Verifier uses to detect discontinuities

Why this works as a verification anchor:
  Real ENF follows a slow random walk with a restoring force toward 60 Hz.
  Consecutive windows of honest ENF are highly correlated (above 0.85).
  A replayed or fabricated window will not share the same random walk
  history, causing confidence to drop sharply. The Verifier catches this.
"""

from __future__ import annotations

import math
import statistics
from typing import Optional

from microverse_core.contracts import AnchorRecord, AnchorType


class AnchorExtractor:
    """
    Reads an ENF list and produces one AnchorRecord per call to extract().

    Parameters
    ----------
    enf : list[float]
        Full ENF time series from data_loaders.load_enf().
        1800 floats for the real AFRL dataset (1 hour at 0.5 Hz).
    sample_rate_hz : float
        ENF samples per second. Real dataset: 0.5 Hz.
        Synthetic fallback: whatever hz was passed to synthetic_enf().
    window_radius : int
        Samples on each side of the centre index.
        Default 5 gives 11 samples = 22 seconds at 0.5 Hz.
        Tune after inspecting real data.
    source : str
        Embedded in AnchorRecord.source for traceability.
    """

    # RETUNED (2026-07) from 5 to 9. Tested a full sweep 5-170 against both
    # a known-smooth synthetic reference and the cleanest verified real ENF
    # segment available: widening the window reduces small-sample
    # correlation noise (real gains on both signals), but ALSO makes a
    # sustained constant-value attack progressively harder to detect, since
    # a long enough anomaly becomes self-correlated with its own
    # window-shifted copy. Found a sharp cliff: confidence during a tested
    # 20-sample sustained attack stayed at exactly 0.0 (fully caught) for
    # radius 5-9, then jumped to 0.62+ at radius=11 (effectively blind).
    # 9 is the widest value with ZERO measured cost against that attack.
    # CAVEAT: this safety margin is tied to that attack's specific 20-sample
    # duration (window size stays just under attack length). NOT verified
    # safe against a shorter sustained attack -- re-test if one becomes
    # available.
    DEFAULT_WINDOW_RADIUS = 9
    DEFAULT_SOURCE = "enf_dataset"

    def __init__(
        self,
        enf: list[float],
        sample_rate_hz: float = 0.5,
        window_radius: int = DEFAULT_WINDOW_RADIUS,
        source: str = DEFAULT_SOURCE,
    ):
        if not enf:
            raise ValueError(
                "ENF list is empty -- check data_loaders.load_enf() returned data"
            )
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if window_radius < 1:
            raise ValueError("window_radius must be at least 1")

        self._enf = enf
        self._sample_rate_hz = sample_rate_hz
        self._window_radius = window_radius
        self._source = source
        self._prev_window: Optional[list[float]] = None

    # --- getters/setters -------------------------------------------------------

    @property
    def sample_rate_hz(self) -> float:
        return self._sample_rate_hz

    @property
    def window_radius(self) -> int:
        return self._window_radius

    @window_radius.setter
    def window_radius(self, value: int) -> None:
        if value < 1:
            raise ValueError("window_radius must be at least 1")
        self._window_radius = value

    @property
    def enf_length(self) -> int:
        return len(self._enf)

    # --- public API ------------------------------------------------------------

    def extract(self, timestamp: float) -> AnchorRecord:
        """
        Build one AnchorRecord for the given elapsed timestamp.

        Parameters
        ----------
        timestamp : float
            Elapsed seconds from start of run.
            Must match the timestamp on the PowerSample being verified.

        Returns
        -------
        AnchorRecord
            All fields match microverse_core.contracts exactly.
        """
        window = self._slice_window(timestamp)
        normalized = self._normalize(window)
        confidence = self._compute_confidence(normalized)

        self._prev_window = normalized

        return AnchorRecord(
            timestamp=timestamp,
            anchor_type=AnchorType.ENF.value,
            signature=normalized,
            confidence=confidence,
            source=self._source,
        )

    # --- private helpers -------------------------------------------------------

    def _timestamp_to_index(self, timestamp: float) -> int:
        """
        Convert elapsed seconds to ENF list index.

        At 0.5 Hz: t=0.0 -> index 0
                   t=2.0 -> index 1
                   t=10.0 -> index 5

        Clamps to valid range so edge timestamps never crash.
        """
        idx = int(timestamp * self._sample_rate_hz)
        return max(0, min(idx, len(self._enf) - 1))

    def _slice_window(self, timestamp: float) -> list[float]:
        """
        Extract a fixed-size window of ENF values centred on timestamp.
        Pads by repeating the nearest edge value when the window extends
        beyond the list boundaries so every window is the same length.
        """
        centre = self._timestamp_to_index(timestamp)
        lo = centre - self._window_radius
        hi = centre + self._window_radius + 1

        window = []
        for i in range(lo, hi):
            clamped = max(0, min(i, len(self._enf) - 1))
            window.append(self._enf[clamped])

        return window

    def _normalize(self, window: list[float]) -> list[float]:
        """
        Min-max normalize the window to [0, 1].

        Makes the signature represent the shape of the ENF fluctuation
        rather than its absolute level. A replay or fabricated window
        may have a similar mean but will not share the same shape.

        A flat window returns all 0.5s and gets flagged by ENFRangeCheck.
        """
        lo = min(window)
        hi = max(window)
        span = hi - lo
        if span == 0:
            return [0.5] * len(window)
        return [(v - lo) / span for v in window]

    def _compute_confidence(self, current: list[float]) -> float:
        """
        Pearson correlation between the current normalized window and
        the previous one. Returns 1.0 on the first call.

        High confidence means ENF is evolving smoothly as expected for
        a real power grid. A sudden drop signals tampering.
        """
        if self._prev_window is None:
            return 1.0
        return _pearson(current, self._prev_window)


def _pearson(a: list[float], b: list[float]) -> float:
    """
    Pearson correlation coefficient. Standard library only, no numpy.
    Returns 0.0 if variance is zero. Clamps result to [-1, 1].
    """
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