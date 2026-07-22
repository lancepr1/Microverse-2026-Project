"""Physical-signal anchor extraction from Electric Network Frequency data.

Provides AnchorExtractor, which converts a raw ENF time series into
AnchorRecord objects carrying a normalized signature and a confidence
score, consumed by the verification pipeline.
"""

from __future__ import annotations

import math
import statistics
from typing import Optional

from microverse_core.contracts import AnchorRecord, AnchorType


class AnchorExtractor:
    """Extracts one AnchorRecord per timestamp from an ENF time series.

    Args:
        enf: Full ENF time series, one value per sample (e.g. 1800
            floats for a 1-hour recording at 0.5 Hz).
        sample_rate_hz: Samples per second in `enf`.
        window_radius: Number of samples on each side of the center
            index to include in each extracted window.
        source: Value stored in ``AnchorRecord.source``.

    Raises:
        ValueError: If `enf` is empty, `sample_rate_hz` is not
            positive, or `window_radius` is less than 1.
    """

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

    @property
    def sample_rate_hz(self) -> float:
        """float: Samples per second in the underlying ENF series."""
        return self._sample_rate_hz

    @property
    def window_radius(self) -> int:
        """int: Samples on each side of the center index per window."""
        return self._window_radius

    @window_radius.setter
    def window_radius(self, value: int) -> None:
        if value < 1:
            raise ValueError("window_radius must be at least 1")
        self._window_radius = value

    @property
    def enf_length(self) -> int:
        """int: Total number of samples in the underlying ENF series."""
        return len(self._enf)

    def extract(self, timestamp: float) -> AnchorRecord:
        """Builds one AnchorRecord for the given elapsed timestamp.

        Args:
            timestamp: Elapsed seconds from the start of the run. Must
                match the timestamp on the record being verified.

        Returns:
            AnchorRecord: Normalized signature and confidence score
            for the window centered on `timestamp`.
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

    def _timestamp_to_index(self, timestamp: float) -> int:
        """Converts elapsed seconds to an ENF list index, clamped to range.

        Args:
            timestamp: Elapsed seconds from the start of the run.

        Returns:
            int: Index into the ENF list, clamped to [0, len(enf) - 1].
        """
        idx = int(timestamp * self._sample_rate_hz)
        return max(0, min(idx, len(self._enf) - 1))

    def _slice_window(self, timestamp: float) -> list[float]:
        """Extracts a fixed-size ENF window centered on `timestamp`.

        Args:
            timestamp: Elapsed seconds from the start of the run.

        Returns:
            list[float]: Window of length ``2 * window_radius + 1``,
            padded by repeating the nearest edge value where the
            window extends past the list boundaries.
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
        """Min-max normalizes a window to [0, 1].

        Args:
            window: Raw ENF values.

        Returns:
            list[float]: Normalized values. A flat input window
            returns all 0.5s.
        """
        lo = min(window)
        hi = max(window)
        span = hi - lo
        if span == 0:
            return [0.5] * len(window)
        return [(v - lo) / span for v in window]

    def _compute_confidence(self, current: list[float]) -> float:
        """Computes Pearson correlation against the previous window.

        Args:
            current: Normalized ENF window for the current call.

        Returns:
            float: Correlation in [-1, 1], or 1.0 on the first call
            (no previous window to compare against).
        """
        if self._prev_window is None:
            return 1.0
        return _pearson(current, self._prev_window)


def _pearson(a: list[float], b: list[float]) -> float:
    """Computes the Pearson correlation coefficient of two sequences.

    Standard library only, no numpy dependency.

    Args:
        a: First sequence.
        b: Second sequence.

    Returns:
        float: Correlation in [-1, 1]. Returns 0.0 if either input
        has zero variance, or 1.0 if fewer than 2 samples overlap.
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