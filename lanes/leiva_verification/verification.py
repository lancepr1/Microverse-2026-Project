"""Runs ENF and NLR detection checks against every component in a record.

Verifier.verify() runs every check, every call, against every
component present, and returns a full list of per-component results.
Nothing stops early -- the goal is coverage and attribution (test
everything, mark exactly what and where) rather than gating whether
data keeps moving downstream.

ENF checks (one shared grid signal, one merged result per record):
    1. _SequenceGuard: rejects duplicate/out-of-order timestamps.
    2. _ENFNominalRangeCheck: raw-frequency plausibility.
    3. _ENFRangeCheck: normalized-signature plausibility.
    4. _ENFContinuityCheck: hard confidence threshold.
    5. _DriftMonitor: whole-file CUSUM on confidence.
    6. _LocalCUSUMDetector: short-horizon CUSUM on confidence.
    7. _RawDriftCheck: sustained directional trend on the raw value.
    8. _ENFAlternativeCorrelationCheck: windowed correlation against
       an independently-measured second ENF stream.

NLR checks (multi-node aware, one result per physical channel):
    9. _NLRRangeCheck: GPU/CPU wattage plausibility.
    10. _NLRContinuityCheck: GPU/CPU wattage step-size plausibility.
    11. _NLRMonotonicityCheck: CPU energy counter direction and rate.
    12. _GPUTempRangeCheck: GPU temperature plausibility.
    13. _GPUTempContinuityCheck: GPU temperature step-size plausibility.
    14. _CrossSiblingConsistencyCheck: same-node channel agreement.
    15. _NLRSustainedDeviationCheck: sticky discontinuity with recovery.
    16. _NLRStartupRampCheck: confirms a channel ever reaches a
        plausible operating level after startup.
    17. _NLRReplayCheck: detects repeated historical subsequences.

    Plus a cross-cutting pass, _apply_synchronized_event_correlation,
    which can downgrade a pure step-size failure from FAILED to
    SUSPECT when enough other nodes show the identical step in the
    same window (a real synchronized system event, not tampering).

Status mapping (dashboard-facing label in parentheses):
    TRUSTED: passes cleanly ("good")
    SUSPECT: ENF confidence in the soft zone, no hard check failed
        ("suspect")
    FAILED: a hard check failed definitively ("warning")

See .readme/verification.md for full calibration history, measured
values, and known limitations behind every threshold below.
"""

from __future__ import annotations

import collections
import math
import re
import statistics
from typing import Optional

from microverse_core.contracts import (
    AnchorRecord,
    VerificationResult,
    VerificationStatus,
)

CONFIDENCE_TRUSTED = 0.93
CONFIDENCE_SUSPECT = 0.85

NOMINAL_HZ = 60.0
NOMINAL_TOLERANCE_HZ = 2.0

CUSUM_THRESHOLD = 5.0
CUSUM_BASELINE = 0.03
CUSUM_HISTORY = 60

LOCAL_CUSUM_WINDOW_SIZE = 10
LOCAL_CUSUM_THRESHOLD = 2.0

RAW_DRIFT_WINDOW_SIZE = 30
RAW_DRIFT_THRESHOLD = 0.6

ENF_ALT_NOISE_STD = 0.0001
ENF_ALT_WINDOW_SIZE = 40
ENF_ALT_CORRELATION_THRESHOLD = 0.999
ENF_ALT_MIN_VARIANCE = 0.000005

GPU_POWER_CEILING_W = 800.0
CPU_POWER_CEILING_W = 800.0
GPU_MAX_STEP_W = 400.0
CPU_MAX_STEP_W = 16.0

CPU_UJ_WRAP_CEILING = 65_500_000_000
CPU_UJ_WRAP_TOLERANCE = 2_000_000_000

GPU_TEMP_FLOOR_C = 0.0
GPU_TEMP_CEILING_C = 95.0
GPU_TEMP_MAX_STEP_C = 8.0

GPU_POWER_RECOVERY_BAND_W = 150.0
GPU_TEMP_RECOVERY_BAND_C = 6.0

NLR_STARTUP_RAMP_WINDOWS = 150

GPU_STARTUP_MIN_W = 400.0
GPU_STARTUP_MAX_W = 700.0
GPU_STARTUP_DEADLINE_WINDOWS = 200

GPU_SUSTAIN_SAMPLES = 5

REPLAY_MATCH_LENGTH = 4
REPLAY_LOOKBACK_WINDOW = 100
REPLAY_MIN_MAGNITUDE_W = 5.0

_ENF_WINDOW_SECONDS = 2.0
CPU_UJ_MAX_STEP_UJ = CPU_POWER_CEILING_W * _ENF_WINDOW_SECONDS * 1_000_000 * 1.5

SYNC_EVENT_MIN_NODES = 4
SYNC_EVENT_MIN_FRACTION = 0.5
SYNC_EVENT_MAX_STEP_MULTIPLE = 1.2

GPU_POWER_SIBLING_RATIO_MIN = 0.85
GPU_POWER_SIBLING_RATIO_MAX = 1.10
GPU_TEMP_SIBLING_RATIO_MIN = 0.90
GPU_TEMP_SIBLING_RATIO_MAX = 1.15
CPU_POWER_SIBLING_RATIO_MIN = 0.75
CPU_POWER_SIBLING_RATIO_MAX = 1.40

SCORE_TRUSTED = 0.95
SCORE_SUSPECT = 0.50
SCORE_FAILED_HARD = 0.05
SCORE_FAILED_DRIFT = 0.10
def _find_nlr_keys(record: dict) -> dict[str, list[str]]:
    """Groups a record's keys by NLR channel type.

    Args:
        record: A combined record dict. Works for any node count --
            keys are matched by suffix/substring, never hardcoded to
            a specific node prefix.

    Returns:
        dict[str, list[str]]: Keys ``gpu_power``, ``cpu_power``,
        ``cpu_uj``, and ``gpu_temp``, each mapping to the list of
        matching column names found in `record`.
    """
    gpu_power, cpu_power, cpu_uj, gpu_temp = [], [], [], []

    for key in record:
        if not isinstance(key, str):
            continue
        key_lower = key.lower()

        if key.endswith("[W]"):
            if "gpu-" in key_lower:
                gpu_power.append(key)
            elif "cpu-" in key_lower:
                cpu_power.append(key)

        elif key.endswith("[uJ]"):
            if "cpu-" in key_lower:
                cpu_uj.append(key)

        elif key.endswith("[C]"):
            if "gpu-" in key_lower:
                gpu_temp.append(key)

    return {
        "gpu_power": gpu_power,
        "cpu_power": cpu_power,
        "cpu_uj":    cpu_uj,
        "gpu_temp":  gpu_temp,
    }


def _worse(a: str, b: str) -> str:
    """Returns whichever status is worse, ranked TRUSTED < SUSPECT < FAILED.

    Args:
        a: First status value.
        b: Second status value.

    Returns:
        str: The worse of the two statuses.
    """
    rank = {
        VerificationStatus.TRUSTED.value: 0,
        VerificationStatus.SUSPECT.value: 1,
        VerificationStatus.FAILED.value:  2,
    }
    return a if rank[a] >= rank[b] else b


class _SequenceGuard:
    """Rejects duplicate or out-of-order timestamps.

    Args:
        strict_ordering: If True, also rejects any timestamp that is
            not strictly greater than the last one seen.
    """

    def __init__(self, strict_ordering: bool = True):
        self._seen: set = set()
        self._last: float = -1.0
        self._strict = strict_ordering

    def check(self, timestamp: float) -> tuple[bool, str]:
        """Checks one timestamp against everything seen so far.

        Args:
            timestamp: The timestamp to check.

        Returns:
            tuple[bool, str]: (passed, reason).
        """
        if timestamp in self._seen:
            return False, f"REPLAY: timestamp {timestamp:.3f} already processed"
        if self._strict and self._last >= 0 and timestamp <= self._last:
            return False, (
                f"OUT OF ORDER: expected timestamp > {self._last:.3f}, "
                f"got {timestamp:.3f}"
            )
        self._seen.add(timestamp)
        self._last = timestamp
        return True, "ok"


class _ENFNominalRangeCheck:
    """Confirms the raw frequency value is within a physically plausible range.

    Runs on the raw measurement rather than the normalized anchor
    signature, since normalization can absorb an extreme value as a
    new local max/min without the resulting shape looking abnormal.
    """

    def check(self, raw_frequency_hz: float) -> tuple[bool, str]:
        """Checks one raw frequency reading.

        Args:
            raw_frequency_hz: Raw ENF value in Hz.

        Returns:
            tuple[bool, str]: (passed, reason).
        """
        deviation = abs(raw_frequency_hz - NOMINAL_HZ)
        if deviation > NOMINAL_TOLERANCE_HZ:
            return False, (
                f"OUT OF NOMINAL RANGE: raw frequency {raw_frequency_hz:.4f} Hz "
                f"deviates {deviation:.4f} Hz from {NOMINAL_HZ} Hz nominal, "
                f"exceeds tolerance {NOMINAL_TOLERANCE_HZ} Hz"
            )
        return True, "ok"


class _ENFRangeCheck:
    """Confirms the normalized anchor signature looks like real ENF.

    Flags flat signatures (zero variance) and values outside [0, 1].
    """

    def check(self, signature: list) -> tuple[bool, str]:
        """Checks one normalized signature.

        Args:
            signature: Normalized ENF window from AnchorRecord.signature.

        Returns:
            tuple[bool, str]: (passed, reason).
        """
        if not signature:
            return False, "EMPTY SIGNATURE: anchor has no values"
        if any(v < 0.0 or v > 1.0 for v in signature):
            return False, "INVALID SIGNATURE: values outside normalized [0,1] range"
        unique = set(round(v, 8) for v in signature)
        if len(unique) == 1:
            return False, (
                "FLAT SIGNATURE: zero variance -- "
                "real ENF always fluctuates, this looks fabricated"
            )
        return True, "ok"


class _ENFContinuityCheck:
    """Flags a sudden drop in window-to-window confidence.

    Uses AnchorRecord.confidence as the continuity signal. A
    discontinuous change is physically implausible on a real grid and
    signals injection or fabrication.
    """

    def check(self, confidence: float) -> tuple[bool, str]:
        """Checks one confidence value.

        Args:
            confidence: Pearson correlation with the previous window.

        Returns:
            tuple[bool, str]: (passed, reason).
        """
        if confidence < CONFIDENCE_SUSPECT:
            return False, (
                f"DISCONTINUITY: confidence {confidence:.4f} below hard "
                f"threshold {CONFIDENCE_SUSPECT} -- "
                f"ENF jumped abruptly, physically impossible on real grid"
            )
        return True, "ok"


class _DriftMonitor:
    """One-sided CUSUM on (1 - confidence), accumulated over the whole file.

    Catches slow drift attacks that keep each individual window within
    per-window thresholds but accumulate a directional bias over time.
    """

    def __init__(self):
        self._cusum: float = 0.0
        self._history: collections.deque = collections.deque(maxlen=CUSUM_HISTORY)
        self._baseline: float = CUSUM_BASELINE
        self._n: int = 0

    @property
    def cusum(self) -> float:
        """float: Current accumulated CUSUM value."""
        return self._cusum

    @property
    def baseline(self) -> float:
        """float: Current baseline deviation, updated by calibrate()."""
        return self._baseline

    @property
    def sample_count(self) -> int:
        """int: Number of samples recorded since the last reset."""
        return self._n

    def record(self, confidence: float) -> None:
        """Records one confidence value and updates the CUSUM.

        Args:
            confidence: Pearson correlation with the previous window.
        """
        deviation = 1.0 - confidence
        self._history.append(deviation)
        self._n += 1
        self._cusum = max(0.0, self._cusum + (deviation - self._baseline))

    def is_drifting(self) -> bool:
        """Returns:
            bool: True if the accumulated CUSUM exceeds CUSUM_THRESHOLD.
        """
        return self._cusum > CUSUM_THRESHOLD

    def calibrate(self) -> None:
        """Sets the baseline from recorded history after a clean warmup period."""
        if self._history:
            self._baseline = statistics.mean(self._history)
            self._cusum = 0.0

    def reset(self) -> None:
        """Resets the accumulated CUSUM to zero."""
        self._cusum = 0.0


class _LocalCUSUMDetector:
    """Short-horizon CUSUM on (1 - confidence), tuned for localized anomalies.

    Unlike _DriftMonitor, which accumulates over an entire file's
    history and is tuned for slow, genuine long-term drift, this
    accumulates over a short sliding window to catch short sustained
    anomalies where no single window's confidence alone crosses
    CONFIDENCE_SUSPECT.

    Includes a fast-recovery mechanism: after `recovery_windows`
    consecutive windows with confidence back above
    `recovery_threshold`, the accumulator resets fully rather than
    waiting on its own slow linear decay.

    Args:
        window_size: Size of the sliding window.
        baseline: Expected deviation on clean data.
        cusum_threshold: Accumulated value above which this fires.
        recovery_threshold: Confidence level counted as "recovered".
        recovery_windows: Consecutive recovered windows required
            before a full reset.
    """

    def __init__(
        self,
        window_size: int = LOCAL_CUSUM_WINDOW_SIZE,
        baseline: float = CUSUM_BASELINE,
        cusum_threshold: float = LOCAL_CUSUM_THRESHOLD,
        recovery_threshold: float = CONFIDENCE_SUSPECT,
        recovery_windows: int = 20,
    ):
        self._window_size = window_size
        self._baseline = baseline
        self._cusum_threshold = cusum_threshold
        self._recovery_threshold = recovery_threshold
        self._recovery_windows = recovery_windows
        self._history: collections.deque = collections.deque(maxlen=window_size)
        self._cusum: float = 0.0
        self._consecutive_good: int = 0

    @property
    def cusum(self) -> float:
        """float: Current accumulated CUSUM value."""
        return self._cusum

    def record(self, confidence: float) -> bool:
        """Records one confidence value and updates the CUSUM.

        Args:
            confidence: Pearson correlation with the previous window.

        Returns:
            bool: True if the accumulated value now exceeds the
            configured threshold.
        """
        deviation = 1.0 - confidence
        self._history.append(deviation)
        self._cusum = max(0.0, self._cusum + (deviation - self._baseline))
        if len(self._history) == self._window_size and deviation < self._baseline:
            self._cusum = max(0.0, self._cusum - self._baseline)

        if confidence >= self._recovery_threshold:
            self._consecutive_good += 1
            if self._consecutive_good >= self._recovery_windows:
                self._cusum = 0.0
        else:
            self._consecutive_good = 0

        return self._cusum > self._cusum_threshold

    def calibrate(self, baseline: float) -> None:
        """Sets the baseline explicitly, e.g. sharing a value calibrated elsewhere.

        Args:
            baseline: New baseline deviation to use going forward.
        """
        self._baseline = baseline
        self._cusum = 0.0
        self._consecutive_good = 0

    def reset(self) -> None:
        """Resets the accumulated CUSUM and recovery counter to zero."""
        self._cusum = 0.0
        self._consecutive_good = 0


_TRUSTED = VerificationStatus.TRUSTED.value
_FAILED = VerificationStatus.FAILED.value


class _RawDriftCheck:
    """Detects sustained directional drift in the raw ENF value.

    Independent of confidence entirely. Splits a rolling window of
    raw values into two halves and compares their means -- a real
    random walk around a stable nominal frequency should show a
    roughly-zero difference between an early and late half of any
    given window, while a sustained directional ramp produces a
    clear, growing difference instead.

    Args:
        window_size: Number of samples in the rolling window.
        drift_threshold: Maximum plausible half-to-half mean
            difference, in Hz.
    """

    def __init__(
        self,
        window_size: int = RAW_DRIFT_WINDOW_SIZE,
        drift_threshold: float = RAW_DRIFT_THRESHOLD,
    ):
        self._window_size = window_size
        self._drift_threshold = drift_threshold
        self._history: collections.deque = collections.deque(maxlen=window_size)

    def check(self, raw_freq: float) -> tuple[bool, str]:
        """Checks one raw frequency reading.

        Args:
            raw_freq: Raw ENF value in Hz.

        Returns:
            tuple[bool, str]: (passed, reason). Always passes until
            the rolling window fills.
        """
        self._history.append(raw_freq)
        if len(self._history) < self._window_size:
            return True, "ok"
        half = self._window_size // 2
        window = list(self._history)
        first_half_mean = statistics.mean(window[:half])
        second_half_mean = statistics.mean(window[half:])
        trend = second_half_mean - first_half_mean
        if abs(trend) > self._drift_threshold:
            return False, (
                f"RAW VALUE TREND: sustained {trend:+.4f} Hz drift within "
                f"a {self._window_size}-window span (early-half mean "
                f"{first_half_mean:.4f}, late-half mean {second_half_mean:.4f}) "
                f"-- real ENF doesn't sustain a directional trend this "
                f"large this consistently"
            )
        return True, "ok"


class _ENFAlternativeCorrelationCheck:
    """Compares the observed ENF stream against an independent reference via correlation.

    ENF only. Simulates two independently-measured sensors: maintains
    a rolling window of the last `window_size` observed values and
    the aligned window from `alternative`, recomputing Pearson
    correlation each time the window is full. Skips windows where
    either signal is nearly flat (see `min_variance`).

    Known limitation: mathematically blind to a constant or
    slowly-varying additive offset, since Pearson correlation is
    invariant to such a shift. See .readme/verification.md for full
    validated performance figures.

    Args:
        alternative: Independently-noised reference ENF stream,
            indexed the same way as the primary stream.
        window_size: Number of samples per correlation window.
        threshold: Minimum correlation to pass.
        min_variance: Minimum window variance required for the
            correlation computation to be considered meaningful.
    """

    def __init__(
        self,
        alternative: list,
        window_size: int = ENF_ALT_WINDOW_SIZE,
        threshold: float = ENF_ALT_CORRELATION_THRESHOLD,
        min_variance: float = ENF_ALT_MIN_VARIANCE,
    ):
        self._alternative = alternative
        self._window_size = window_size
        self._threshold = threshold
        self._min_variance = min_variance
        self._recent_observed: collections.deque = collections.deque(maxlen=window_size)
        self._recent_alt: collections.deque = collections.deque(maxlen=window_size)

    def check(self, index: int, observed_freq: float) -> tuple[bool, str]:
        """Checks one observed frequency reading against the reference stream.

        Args:
            index: Position into the reference stream corresponding
                to this observation.
            observed_freq: Observed (possibly tampered) raw frequency.

        Returns:
            tuple[bool, str]: (passed, reason). Always passes until
            the rolling window fills or if `index` is out of range.
        """
        if index < 0 or index >= len(self._alternative):
            return True, "ok"

        self._recent_observed.append(observed_freq)
        self._recent_alt.append(self._alternative[index])

        if len(self._recent_observed) < self._window_size:
            return True, "ok"

        observed_window = list(self._recent_observed)
        alt_window = list(self._recent_alt)

        var_observed = statistics.variance(observed_window)
        var_alt = statistics.variance(alt_window)
        if var_observed < self._min_variance or var_alt < self._min_variance:
            return True, "ok"

        mean_o = statistics.mean(observed_window)
        mean_a = statistics.mean(alt_window)
        cov = sum(
            (observed_window[i] - mean_o) * (alt_window[i] - mean_a)
            for i in range(self._window_size)
        )
        denom = (var_observed * var_alt) ** 0.5 * (self._window_size - 1)
        if denom == 0:
            return True, "ok"
        correlation = cov / denom

        if correlation < self._threshold:
            return False, (
                f"CORRELATION MISMATCH: observed ENF over the last "
                f"{self._window_size} windows correlates at "
                f"{correlation:.3f} with the independently-measured "
                f"reference -- below the {self._threshold} threshold "
                f"real, untampered sensor pairs stay above"
            )
        return True, "ok"
def _extract_node_id_from_key(key: str) -> str:
    """Extracts the node ID prefix from a full column name.

    Args:
        key: Full column name, e.g. "x3102c0s25b0n0_gpu-0[W]".

    Returns:
        str: The node ID prefix, e.g. "x3102c0s25b0n0", or None if
        neither "_gpu-" nor "_cpu-" appears in `key`.
    """
    if "_gpu-" in key:
        return key.split("_gpu-")[0]
    if "_cpu-" in key:
        return key.split("_cpu-")[0]
    return None


class _CrossSiblingConsistencyCheck:
    """Compares each NLR channel against the median of its siblings on the same node.

    Uses only channels already present in the compressed record every
    consumer receives -- no held-back or hidden reference data. GPUs
    (and, more weakly, CPUs) on the same node running the same
    distributed job track each other closely; a targeted, single-
    channel replay that leaves siblings untouched breaks that
    agreement even when the replayed value is plausible in isolation.

    Compares only within a single record and a single node -- never
    across nodes, never against history. "-core[W]" channels are
    excluded (see .readme/verification.md for why).
    """

    def check(self, record: dict) -> dict[str, tuple[str, str]]:
        """Checks every eligible channel in one record against its siblings.

        Args:
            record: A combined record dict.

        Returns:
            dict[str, tuple[str, str]]: One (status, reason) entry
            per channel that could be checked (has at least one
            sibling on the same node).
        """
        channels = _find_nlr_keys(record)
        results: dict[str, tuple[str, str]] = {}

        typed_groups = (
            (channels["gpu_power"], GPU_POWER_SIBLING_RATIO_MIN, GPU_POWER_SIBLING_RATIO_MAX, "W"),
            (channels["gpu_temp"],  GPU_TEMP_SIBLING_RATIO_MIN,  GPU_TEMP_SIBLING_RATIO_MAX,  "C"),
            (channels["cpu_power"], CPU_POWER_SIBLING_RATIO_MIN, CPU_POWER_SIBLING_RATIO_MAX, "W"),
        )

        for keys, ratio_min, ratio_max, unit in typed_groups:
            keys = [k for k in keys if "-core" not in k]

            by_node: dict[str, list[str]] = collections.defaultdict(list)
            for key in keys:
                node_id = _extract_node_id_from_key(key)
                if node_id is not None:
                    by_node[node_id].append(key)

            for node_id, node_keys in by_node.items():
                if len(node_keys) < 2:
                    continue
                for key in node_keys:
                    val = record.get(key)
                    if val is None:
                        continue
                    sibling_vals = [
                        record.get(k) for k in node_keys if k != key
                    ]
                    sibling_vals = [v for v in sibling_vals if v is not None]
                    if not sibling_vals:
                        continue
                    sibling_ref = statistics.median(sibling_vals)
                    if sibling_ref <= 1:
                        continue
                    ratio = val / sibling_ref
                    if not (ratio_min <= ratio <= ratio_max):
                        results[key] = (_FAILED, (
                            f"SIBLING MISMATCH: {key}={val:.2f}{unit} vs "
                            f"sibling median {sibling_ref:.2f}{unit} on the "
                            f"same node (ratio={ratio:.2f}, expected "
                            f"[{ratio_min}, {ratio_max}]) -- same-node "
                            f"channels of this type normally track each "
                            f"other closely"
                        ))

        return results

    def reset(self) -> None:
        """No-op. This check is fully stateless -- every record is checked independently."""
        pass


class _NLRRangeCheck:
    """Confirms GPU and CPU power readings are physically plausible."""

    def check(self, record: dict) -> dict[str, tuple[str, str]]:
        """Checks every GPU/CPU power channel in one record.

        Args:
            record: A combined record dict.

        Returns:
            dict[str, tuple[str, str]]: One (status, reason) entry
            per power channel present.
        """
        channels = _find_nlr_keys(record)
        results: dict[str, tuple[str, str]] = {}

        for key in channels["gpu_power"]:
            val = record.get(key)
            if val is None:
                continue
            if val < 0:
                results[key] = (_FAILED, f"OUT OF RANGE: {key}={val:.2f}W is negative")
            elif val > GPU_POWER_CEILING_W:
                results[key] = (_FAILED, (
                    f"OUT OF RANGE: {key}={val:.2f}W exceeds hardware "
                    f"ceiling {GPU_POWER_CEILING_W}W"
                ))
            else:
                results[key] = (_TRUSTED, "ok")

        for key in channels["cpu_power"]:
            val = record.get(key)
            if val is None:
                continue
            if val < 0:
                results[key] = (_FAILED, f"OUT OF RANGE: {key}={val:.2f}W is negative")
            elif val > CPU_POWER_CEILING_W:
                results[key] = (_FAILED, (
                    f"OUT OF RANGE: {key}={val:.2f}W exceeds hardware "
                    f"ceiling {CPU_POWER_CEILING_W}W"
                ))
            else:
                results[key] = (_TRUSTED, "ok")

        return results


class _NLRContinuityCheck:
    """Confirms GPU and CPU power does not jump implausibly between consecutive windows.

    State is kept per full column name, so nodes never collide and
    every channel is tracked independently.
    """

    def __init__(self):
        self._prev: dict[str, float] = {}

    def check(self, record: dict) -> dict[str, tuple[str, str]]:
        """Checks every GPU/CPU power channel's step size in one record.

        Args:
            record: A combined record dict.

        Returns:
            dict[str, tuple[str, str]]: One (status, reason) entry
            per power channel present.
        """
        channels = _find_nlr_keys(record)
        results: dict[str, tuple[str, str]] = {}

        for key in channels["gpu_power"]:
            val = record.get(key)
            if val is None:
                continue
            prev = self._prev.get(key)
            if prev is not None and abs(val - prev) > GPU_MAX_STEP_W:
                step = abs(val - prev)
                results[key] = (_FAILED, (
                    f"DISCONTINUITY: {key} stepped {step:.2f}W between "
                    f"windows, exceeds max plausible step {GPU_MAX_STEP_W}W"
                ))
            else:
                results[key] = (_TRUSTED, "ok")
            self._prev[key] = val

        for key in channels["cpu_power"]:
            val = record.get(key)
            if val is None:
                continue
            prev = self._prev.get(key)
            if prev is not None and abs(val - prev) > CPU_MAX_STEP_W:
                step = abs(val - prev)
                results[key] = (_FAILED, (
                    f"DISCONTINUITY: {key} stepped {step:.2f}W between "
                    f"windows, exceeds max plausible step {CPU_MAX_STEP_W}W"
                ))
            else:
                results[key] = (_TRUSTED, "ok")
            self._prev[key] = val

        return results

    def reset(self) -> None:
        """Clears all tracked previous values."""
        self._prev.clear()


class _NLRSustainedDeviationCheck:
    """Sticky discontinuity check for GPU wattage and temperature.

    Closes a gap _NLRContinuityCheck leaves open: that check compares
    each value only to its immediately preceding window, so it goes
    silent the instant a tampered value stabilizes. Here, once a step
    exceeds the continuity threshold, the channel stays FAILED on
    every subsequent window until either of two recovery conditions
    is met: the value returns within a recovery band of its pre-jump
    reference, or it holds GPU_SUSTAIN_SAMPLES consecutive steps at a
    new, different stable level. See .readme/verification.md for why
    both conditions exist.

    Inactive during the documented startup-ramp window (see
    NLR_STARTUP_RAMP_WINDOWS) -- a legitimate ramp passes through
    several different stable levels, so "must return to the original
    reference" is the wrong test there.

    Does not apply to CPU wattage or CPU energy.
    """

    def __init__(self):
        self._prev: dict[str, float] = {}
        self._reference: dict[str, float] = {}
        self._stuck: dict[str, bool] = {}
        self._recovery_step_run: dict[str, int] = {}
        self._window_count = 0

    def check(self, record: dict) -> dict[str, tuple[str, str]]:
        """Checks every GPU wattage/temperature channel in one record.

        Args:
            record: A combined record dict.

        Returns:
            dict[str, tuple[str, str]]: One (status, reason) entry
            per channel, once past the startup-ramp window.
        """
        self._window_count += 1
        channels = _find_nlr_keys(record)
        results: dict[str, tuple[str, str]] = {}

        if self._window_count <= NLR_STARTUP_RAMP_WINDOWS:
            for keys, _, _, _ in (
                (channels["gpu_power"], None, None, None),
                (channels["gpu_temp"], None, None, None),
            ):
                for key in keys:
                    val = record.get(key)
                    if val is not None:
                        self._prev[key] = val
            return results

        typed_groups = (
            (channels["gpu_power"], GPU_MAX_STEP_W, GPU_POWER_RECOVERY_BAND_W, "W"),
            (channels["gpu_temp"],  GPU_TEMP_MAX_STEP_C, GPU_TEMP_RECOVERY_BAND_C, "C"),
        )

        for keys, step_threshold, recovery_band, unit in typed_groups:
            for key in keys:
                val = record.get(key)
                if val is None:
                    continue

                prev = self._prev.get(key)
                if prev is None:
                    self._prev[key] = val
                    continue

                if self._stuck.get(key):
                    ref = self._reference[key]
                    step = abs(val - prev)

                    if step <= step_threshold:
                        self._recovery_step_run[key] = self._recovery_step_run.get(key, 0) + 1
                    else:
                        self._recovery_step_run[key] = 0

                    recovered_to_reference = abs(val - ref) <= recovery_band
                    recovered_to_new_plateau = (
                        self._recovery_step_run.get(key, 0) >= GPU_SUSTAIN_SAMPLES
                    )

                    if recovered_to_reference or recovered_to_new_plateau:
                        self._stuck[key] = False
                        self._recovery_step_run[key] = 0
                        results[key] = (_TRUSTED, "ok")
                    else:
                        results[key] = (_FAILED, (
                            f"SUSTAINED DEVIATION: {key}={val:.2f}{unit} has "
                            f"not recovered -- still {abs(val - ref):.2f}{unit} "
                            f"away from the {ref:.2f}{unit} it held before the "
                            f"original jump (recovery band: "
                            f"+/-{recovery_band}{unit}), and has not yet held "
                            f"{GPU_SUSTAIN_SAMPLES} consecutive stable steps at "
                            f"a new level either "
                            f"({self._recovery_step_run.get(key, 0)}/"
                            f"{GPU_SUSTAIN_SAMPLES} so far)"
                        ))
                else:
                    step = abs(val - prev)
                    if step > step_threshold:
                        self._stuck[key] = True
                        self._reference[key] = prev
                        self._recovery_step_run[key] = 0
                        results[key] = (_FAILED, (
                            f"SUSTAINED DEVIATION: {key} jumped "
                            f"{step:.2f}{unit} to {val:.2f}{unit} -- will stay "
                            f"FAILED until it returns within "
                            f"+/-{recovery_band}{unit} of {prev:.2f}{unit}, or "
                            f"holds {GPU_SUSTAIN_SAMPLES} consecutive stable "
                            f"steps at a new level"
                        ))

                self._prev[key] = val

        return results

    def reset(self) -> None:
        """Clears all per-channel tracking state."""
        self._prev.clear()
        self._reference.clear()
        self._stuck.clear()
        self._recovery_step_run.clear()
        self._window_count = 0


class _NLRStartupRampCheck:
    """Confirms a GPU wattage channel ever reaches a plausible operating level after startup.

    Closes a gap distinct from _NLRSustainedDeviationCheck: a channel
    stuck at idle-level wattage for an entire run passes every other
    NLR check cleanly, since idle values are individually "in range"
    and never step hard enough to trip continuity or sustained-
    deviation.

    Never flags before GPU_STARTUP_DEADLINE_WINDOWS -- the real
    startup transition is non-monotonic, bouncing between idle and
    steady-state repeatedly before locking in. Only judges whether a
    sustained run in range was ever achieved by the deadline. Once a
    channel achieves GPU_SUSTAIN_SAMPLES consecutive samples inside
    [GPU_STARTUP_MIN_W, GPU_STARTUP_MAX_W], it is marked stabilized
    permanently. Past the deadline, a channel that has never achieved
    that sustained run is FAILED and stays FAILED until it does.
    """

    def __init__(self):
        self._window_count = 0
        self._stabilized: dict[str, bool] = {}
        self._sustain_run: dict[str, int] = {}

    def check(self, record: dict) -> dict[str, tuple[str, str]]:
        """Checks every GPU wattage channel in one record.

        Args:
            record: A combined record dict.

        Returns:
            dict[str, tuple[str, str]]: One (status, reason) entry
            per GPU wattage channel present.
        """
        self._window_count += 1
        channels = _find_nlr_keys(record)
        results: dict[str, tuple[str, str]] = {}

        for key in channels["gpu_power"]:
            val = record.get(key)
            if val is None:
                continue

            if self._stabilized.get(key):
                results[key] = (_TRUSTED, "ok")
                continue

            in_range = GPU_STARTUP_MIN_W <= val <= GPU_STARTUP_MAX_W
            self._sustain_run[key] = (self._sustain_run.get(key, 0) + 1) if in_range else 0

            if self._sustain_run[key] >= GPU_SUSTAIN_SAMPLES:
                self._stabilized[key] = True
                results[key] = (_TRUSTED, "ok")
            elif self._window_count <= GPU_STARTUP_DEADLINE_WINDOWS:
                results[key] = (_TRUSTED, "ok")
            else:
                results[key] = (_FAILED, (
                    f"STARTUP RAMP: {key}={val:.2f}W has never sustained "
                    f"{GPU_SUSTAIN_SAMPLES} consecutive samples "
                    f"inside the expected operating range "
                    f"[{GPU_STARTUP_MIN_W:.0f},{GPU_STARTUP_MAX_W:.0f}]W by "
                    f"window {self._window_count} (deadline "
                    f"{GPU_STARTUP_DEADLINE_WINDOWS}) -- will clear once it "
                    f"does"
                ))

        return results

    def reset(self) -> None:
        """Clears all per-channel tracking state."""
        self._window_count = 0
        self._stabilized.clear()
        self._sustain_run.clear()


class _NLRReplayCheck:
    """Detects a short subsequence of consecutive values recurring in a channel's history.

    Applies to every NLR channel (GPU wattage/temp, CPU wattage/
    energy). Excludes near-zero wattage readings (below
    REPLAY_MIN_MAGNITUDE_W) and degenerate flat matches (all matched
    values identical) -- see .readme/verification.md for why both
    exclusions exist.

    Recovery: clears once GPU_SUSTAIN_SAMPLES consecutive windows pass
    with no new repeated subsequence found.
    """

    def __init__(self):
        self._history: dict[str, collections.deque] = {}
        self._stuck: dict[str, bool] = {}
        self._clean_run: dict[str, int] = {}

    def _buffer(self, key: str) -> collections.deque:
        """Returns the rolling history buffer for one channel, creating it if needed.

        Args:
            key: Full column name.

        Returns:
            collections.deque: The channel's history buffer, bounded
            to REPLAY_LOOKBACK_WINDOW entries.
        """
        if key not in self._history:
            self._history[key] = collections.deque(maxlen=REPLAY_LOOKBACK_WINDOW)
        return self._history[key]

    def check(self, record: dict) -> dict[str, tuple[str, str]]:
        """Checks every NLR channel in one record for repeated subsequences.

        Args:
            record: A combined record dict.

        Returns:
            dict[str, tuple[str, str]]: One (status, reason) entry
            per channel currently flagged or newly matched. Silent
            (no entry) for a channel that is calm and has no match.
        """
        channels = _find_nlr_keys(record)
        results: dict[str, tuple[str, str]] = {}

        all_keys = (
            channels["gpu_power"] + channels["gpu_temp"]
            + channels["cpu_power"] + channels["cpu_uj"]
        )
        low_magnitude_wattage_keys = set(channels["gpu_power"] + channels["cpu_power"])

        for key in all_keys:
            val = record.get(key)
            if val is None:
                continue

            if key in low_magnitude_wattage_keys and abs(val) < REPLAY_MIN_MAGNITUDE_W:
                continue

            buf = self._buffer(key)
            buf.append(round(val, 4))

            matched = False
            tail = None
            if len(buf) >= 2 * REPLAY_MATCH_LENGTH:
                snapshot = list(buf)
                tail = snapshot[-REPLAY_MATCH_LENGTH:]
                if len(set(tail)) > 1:
                    history = snapshot[:-REPLAY_MATCH_LENGTH]
                    for start in range(len(history) - REPLAY_MATCH_LENGTH + 1):
                        if history[start:start + REPLAY_MATCH_LENGTH] == tail:
                            matched = True
                            break

            if matched:
                self._stuck[key] = True
                self._clean_run[key] = 0
                results[key] = (_FAILED, (
                    f"REPLAY: {key} -- the last {REPLAY_MATCH_LENGTH} "
                    f"readings ({tail}) exactly match an earlier run "
                    f"within the last {REPLAY_LOOKBACK_WINDOW} samples -- "
                    f"real sensor data does not repeat itself like this"
                ))
            elif self._stuck.get(key):
                self._clean_run[key] = self._clean_run.get(key, 0) + 1
                if self._clean_run[key] >= GPU_SUSTAIN_SAMPLES:
                    self._stuck[key] = False
                    results[key] = (_TRUSTED, "ok")
                else:
                    results[key] = (_FAILED, (
                        f"REPLAY: {key} -- no new repeat found this "
                        f"window, but has not yet held "
                        f"{GPU_SUSTAIN_SAMPLES} consecutive clean windows "
                        f"to clear ({self._clean_run[key]}/"
                        f"{GPU_SUSTAIN_SAMPLES})"
                    ))

        return results

    def reset(self) -> None:
        """Clears all per-channel history and tracking state."""
        self._history.clear()
        self._stuck.clear()
        self._clean_run.clear()
class _NLRMonotonicityCheck:
    """Confirms CPU energy counters behave like a real hardware energy counter.

    Checks both directions independently. Decreases are compared
    against a running maximum (not just the immediately preceding
    value), since a cumulative counter should never drop below the
    highest value legitimately observed, except via the known
    hardware wraparound. Increases are compared against the
    immediately preceding actual reading, to catch an implausibly
    large single-window jump.
    """

    def __init__(self):
        self._prev: dict[str, float] = {}
        self._running_max: dict[str, float] = {}

    def check(self, record: dict) -> dict[str, tuple[str, str]]:
        """Checks every CPU energy channel in one record.

        Args:
            record: A combined record dict.

        Returns:
            dict[str, tuple[str, str]]: One (status, reason) entry
            per CPU energy channel present.
        """
        channels = _find_nlr_keys(record)
        results: dict[str, tuple[str, str]] = {}

        for key in channels["cpu_uj"]:
            val = record.get(key)
            if val is None:
                continue
            prev = self._prev.get(key)
            running_max = self._running_max.get(key)

            if prev is None:
                results[key] = (_TRUSTED, "ok")
                self._prev[key] = val
                self._running_max[key] = val
                continue

            if val < running_max:
                drop_from_prev = prev - val
                is_expected_wrap = (
                    drop_from_prev > 0
                    and abs(drop_from_prev - CPU_UJ_WRAP_CEILING) < CPU_UJ_WRAP_TOLERANCE
                )
                if is_expected_wrap:
                    results[key] = (_TRUSTED, "ok")
                    self._running_max[key] = val
                else:
                    results[key] = (_FAILED, (
                        f"MONOTONICITY VIOLATION: {key}={val:.1f} uJ is "
                        f"{running_max - val:.1f} uJ below the highest "
                        f"legitimately observed value ({running_max:.1f} uJ) "
                        f"-- not consistent with known hardware wrap "
                        f"(~{CPU_UJ_WRAP_CEILING:.0f} uJ). Stays flagged "
                        f"until a reading naturally exceeds this high-water "
                        f"mark again."
                    ))
                self._prev[key] = val
            else:
                step = val - prev
                if step > CPU_UJ_MAX_STEP_UJ:
                    results[key] = (_FAILED, (
                        f"ENERGY SPIKE: {key} increased by {step:.1f} uJ "
                        f"in one window, exceeds plausibility ceiling "
                        f"{CPU_UJ_MAX_STEP_UJ:.1f} uJ (PLACEHOLDER threshold)"
                    ))
                else:
                    results[key] = (_TRUSTED, "ok")
                self._prev[key] = val
                self._running_max[key] = val

        return results

    def reset(self) -> None:
        """Clears all per-channel tracking state."""
        self._prev.clear()
        self._running_max.clear()


class _GPUTempRangeCheck:
    """Confirms GPU temperature readings are physically plausible.

    Thresholds are placeholders, not calibrated against real data.
    """

    def check(self, record: dict) -> dict[str, tuple[str, str]]:
        """Checks every GPU temperature channel in one record.

        Args:
            record: A combined record dict.

        Returns:
            dict[str, tuple[str, str]]: One (status, reason) entry
            per GPU temperature channel present.
        """
        channels = _find_nlr_keys(record)
        results: dict[str, tuple[str, str]] = {}

        for key in channels["gpu_temp"]:
            val = record.get(key)
            if val is None:
                continue
            if val <= GPU_TEMP_FLOOR_C or val > GPU_TEMP_CEILING_C:
                results[key] = (_FAILED, (
                    f"OUT OF RANGE: {key}={val:.1f}C outside plausible "
                    f"[{GPU_TEMP_FLOOR_C}, {GPU_TEMP_CEILING_C}]C "
                    f"(PLACEHOLDER threshold)"
                ))
            else:
                results[key] = (_TRUSTED, "ok")

        return results


class _GPUTempContinuityCheck:
    """Confirms GPU temperature does not jump implausibly between consecutive windows.

    Threshold is a placeholder, not calibrated against real data.
    """

    def __init__(self):
        self._prev: dict[str, float] = {}

    def check(self, record: dict) -> dict[str, tuple[str, str]]:
        """Checks every GPU temperature channel's step size in one record.

        Args:
            record: A combined record dict.

        Returns:
            dict[str, tuple[str, str]]: One (status, reason) entry
            per GPU temperature channel present.
        """
        channels = _find_nlr_keys(record)
        results: dict[str, tuple[str, str]] = {}

        for key in channels["gpu_temp"]:
            val = record.get(key)
            if val is None:
                continue
            prev = self._prev.get(key)
            if prev is not None and abs(val - prev) > GPU_TEMP_MAX_STEP_C:
                step = abs(val - prev)
                results[key] = (_FAILED, (
                    f"DISCONTINUITY: {key} stepped {step:.1f}C between "
                    f"windows, exceeds max plausible step {GPU_TEMP_MAX_STEP_C}C "
                    f"(PLACEHOLDER threshold)"
                ))
            else:
                results[key] = (_TRUSTED, "ok")
            self._prev[key] = val

        return results

    def reset(self) -> None:
        """Clears all tracked previous values."""
        self._prev.clear()


def _merge_key_results(*result_dicts: dict[str, tuple[str, str]]) -> dict[str, tuple[str, str]]:
    """Merges per-key (status, reason) dicts from multiple checks into one.

    Args:
        *result_dicts: Any number of per-key result dicts.

    Returns:
        dict[str, tuple[str, str]]: One entry per key. If a key
        appears in more than one input dict, the merged entry keeps
        the worst status and concatenates every non-"ok" reason.
    """
    merged: dict[str, tuple[str, str]] = {}
    for result_dict in result_dicts:
        for key, (status, reason) in result_dict.items():
            if key not in merged:
                merged[key] = (status, reason)
            else:
                prev_status, prev_reason = merged[key]
                new_status = _worse(prev_status, status)
                reasons = [r for r in (prev_reason, reason) if r != "ok"]
                new_reason = " | ".join(reasons) if reasons else "ok"
                merged[key] = (new_status, new_reason)
    return merged


def _bare_channel_name(key: str) -> str:
    """Strips the node-id prefix from a full column name.

    Args:
        key: Full column name, e.g. "x3105c0s37b0n0_cpu-0[W]".

    Returns:
        str: The bare physical channel name, e.g. "cpu-0[W]".
    """
    for marker in ("_gpu-", "_cpu-"):
        idx = key.find(marker)
        if idx != -1:
            return key[idx + 1:]
    return key


def _is_pure_discontinuity(reason: str) -> bool:
    """Checks whether DISCONTINUITY is the sole problem named in a reason string.

    Args:
        reason: A merged reason string.

    Returns:
        bool: True only if DISCONTINUITY appears and no disqualifying
        violation (out-of-range, monotonicity, energy spike) also
        appears.
    """
    if "DISCONTINUITY" not in reason:
        return False
    disqualifying = ("OUT OF RANGE", "MONOTONICITY VIOLATION", "ENERGY SPIKE")
    return not any(d in reason for d in disqualifying)


_DISCONTINUITY_STEP_RE = re.compile(
    r"stepped ([\d.]+)\S* between windows, exceeds max plausible step ([\d.]+)"
)


def _is_within_corroboration_ceiling(
    reason: str, max_multiple: float = SYNC_EVENT_MAX_STEP_MULTIPLE
) -> bool:
    """Checks whether a discontinuity's step size is within the corroboration ceiling.

    Args:
        reason: A DISCONTINUITY reason string.
        max_multiple: Maximum allowed multiple of the threshold that
            was crossed.

    Returns:
        bool: True if the step is within `max_multiple` times the
        threshold it crossed, or if `reason` can't be parsed (fails
        open).
    """
    match = _DISCONTINUITY_STEP_RE.search(reason)
    if not match:
        return True
    step, threshold = float(match.group(1)), float(match.group(2))
    if threshold <= 0:
        return True
    return step <= threshold * max_multiple


def _apply_synchronized_event_correlation(
    merged: dict[str, tuple[str, str]],
    max_min_nodes: int = SYNC_EVENT_MIN_NODES,
    min_fraction: float = SYNC_EVENT_MIN_FRACTION,
) -> dict[str, tuple[str, str]]:
    """Downgrades pure step-size failures corroborated by many other nodes.

    Independent hardware does not coincidentally step together. If a
    large step shows up on the same physical channel across many
    different nodes in the same window, that is itself evidence of a
    real synchronized system event (checkpoint, sync barrier, job
    startup) rather than tampering.

    Only pure discontinuity failures are eligible (see
    _is_pure_discontinuity); corroboration never rescues a channel
    that is also out of range or violating monotonicity. Also
    requires the step stay within SYNC_EVENT_MAX_STEP_MULTIPLE of the
    threshold it crossed (see _is_within_corroboration_ceiling).

    The absolute-node requirement scales with how many nodes actually
    have this channel present this window:
    ``effective_min_nodes = max(2, min(max_min_nodes, total_nodes))``.

    Args:
        merged: Merged per-key (status, reason) results.
        max_min_nodes: Ceiling on the absolute-node requirement.
        min_fraction: Minimum fraction of present nodes that must
            agree.

    Returns:
        dict[str, tuple[str, str]]: A copy of `merged` with eligible
        entries downgraded from FAILED to SUSPECT, annotated with how
        many nodes corroborated the step.
    """
    by_channel: dict[str, list[str]] = collections.defaultdict(list)
    for key in merged:
        by_channel[_bare_channel_name(key)].append(key)

    result = dict(merged)

    for bare, keys in by_channel.items():
        total_nodes = len(keys)
        failed_keys = [
            k for k in keys
            if merged[k][0] == _FAILED
            and _is_pure_discontinuity(merged[k][1])
            and _is_within_corroboration_ceiling(merged[k][1])
        ]
        if not failed_keys:
            continue

        count = len(failed_keys)
        fraction = count / total_nodes if total_nodes else 0.0
        effective_min_nodes = max(2, min(max_min_nodes, total_nodes))

        if count >= effective_min_nodes and fraction >= min_fraction:
            for k in failed_keys:
                orig_status, orig_reason = merged[k]
                result[k] = (
                    VerificationStatus.SUSPECT.value,
                    f"{orig_reason} | SYNCHRONIZED EVENT: corroborated by "
                    f"{count}/{total_nodes} nodes stepping on {bare} in the "
                    f"same window (required >={effective_min_nodes} of "
                    f"{total_nodes} present) -- likely a real system-wide "
                    f"event (checkpoint/sync/startup), not tampering "
                    f"(PLACEHOLDER thresholds, not yet validated against "
                    f"coordinated multi-node attack scenarios)"
                )

    return result


class Verifier:
    """Runs every ENF and NLR check for every component, every call.

    Nothing stops early -- verify() always returns a result for the
    ENF anchor plus one result per NLR channel present in the record.
    Fully multi-node aware: NLR checks discover channel names
    dynamically from record keys.

    Args:
        component_id: Identifies what's being verified, e.g.
            "rack_00/x3105c0s37b0n0". Individual NLR results extend
            this with the channel name.
        warmup_windows: Clean windows before the drift baseline is
            calibrated.
        strict_ordering: Enforce strictly increasing timestamps.
        check_nlr: Whether to run the NLR/GPU-temp checks. Set False
            for ENF-only testing.
        enf_alternative: Independently-noised second ENF stream, ENF
            only. Must come from data held before any attack
            injection touched it. If not provided, the alternative-
            correlation check simply doesn't run.
    """

    def __init__(
        self,
        component_id: str,
        warmup_windows: int = 10,
        strict_ordering: bool = True,
        check_nlr: bool = True,
        enf_alternative: list = None,
    ):
        self._component_id = component_id
        self._warmup_windows = warmup_windows
        self._check_nlr = check_nlr

        self._sequence_guard      = _SequenceGuard(strict_ordering)
        self._nominal_range_check = _ENFNominalRangeCheck()
        self._range_check         = _ENFRangeCheck()
        self._continuity_check    = _ENFContinuityCheck()
        self._drift_monitor       = _DriftMonitor()
        self._local_cusum         = _LocalCUSUMDetector()
        self._raw_drift_check     = _RawDriftCheck()
        self._alt_correlation_check = _ENFAlternativeCorrelationCheck(enf_alternative) if enf_alternative is not None else None

        self._nlr_range_check        = _NLRRangeCheck()
        self._nlr_continuity_check   = _NLRContinuityCheck()
        self._nlr_monotonicity_check = _NLRMonotonicityCheck()
        self._cross_sibling_check = _CrossSiblingConsistencyCheck()
        self._sustained_deviation_check = _NLRSustainedDeviationCheck()
        self._startup_ramp_check        = _NLRStartupRampCheck()
        self._replay_check              = _NLRReplayCheck()
        self._gpu_temp_range_check      = _GPUTempRangeCheck()
        self._gpu_temp_continuity_check = _GPUTempContinuityCheck()

        self._windows_processed: int = 0

    @property
    def component_id(self) -> str:
        """str: Identifier for what this Verifier instance is checking."""
        return self._component_id

    @property
    def windows_processed(self) -> int:
        """int: Number of windows processed so far."""
        return self._windows_processed

    def verify(self, sample, anchor: AnchorRecord) -> list[VerificationResult]:
        """Verifies one sample against its anchor.

        Always runs every check and always returns a full list of
        results: one merged result for the ENF anchor, plus one
        result per NLR/GPU-temp channel present in `sample` (when
        `check_nlr=True`).

        Args:
            sample: Combined record (or per-node sub-record), either
                a dict or an object with a `.timestamp` attribute.
            anchor: ENF anchor from AnchorExtractor.extract() for the
                same timestamp.

        Returns:
            list[VerificationResult]: One entry for the ENF anchor,
            plus one entry per NLR channel checked.
        """
        if hasattr(sample, "timestamp"):
            ts = sample.timestamp
        elif isinstance(sample, dict):
            ts = float(sample.get("timestamp", sample.get("index", 0)))
        else:
            ts = 0.0

        results: list[VerificationResult] = []

        enf_status = VerificationStatus.TRUSTED.value
        enf_reasons: list[str] = []

        passed, reason = self._sequence_guard.check(ts)
        if not passed:
            enf_status = _worse(enf_status, VerificationStatus.FAILED.value)
            enf_reasons.append(reason)

        raw_freq = None
        if isinstance(sample, dict):
            raw_freq = sample.get("FRQ", sample.get("frequency_hz"))
        else:
            raw_freq = getattr(sample, "FRQ", getattr(sample, "frequency_hz", None))

        if raw_freq is not None:
            passed, reason = self._nominal_range_check.check(raw_freq)
            if not passed:
                enf_status = _worse(enf_status, VerificationStatus.FAILED.value)
                enf_reasons.append(reason)

            passed, reason = self._raw_drift_check.check(raw_freq)
            if not passed:
                enf_status = _worse(enf_status, VerificationStatus.FAILED.value)
                enf_reasons.append(reason)

            if self._alt_correlation_check is not None:
                index = None
                if isinstance(sample, dict):
                    index = sample.get("index")
                else:
                    index = getattr(sample, "index", None)
                if index is not None:
                    passed, reason = self._alt_correlation_check.check(index, raw_freq)
                    if not passed:
                        enf_status = _worse(enf_status, VerificationStatus.FAILED.value)
                        enf_reasons.append(reason)

        passed, reason = self._range_check.check(anchor.signature)
        if not passed:
            enf_status = _worse(enf_status, VerificationStatus.FAILED.value)
            enf_reasons.append(reason)

        passed, reason = self._continuity_check.check(anchor.confidence)
        if not passed:
            enf_status = _worse(enf_status, VerificationStatus.FAILED.value)
            enf_reasons.append(reason)

        self._drift_monitor.record(anchor.confidence)
        self._windows_processed += 1
        if self._windows_processed == self._warmup_windows:
            self._drift_monitor.calibrate()

        if self._drift_monitor.is_drifting():
            enf_status = _worse(enf_status, VerificationStatus.FAILED.value)
            enf_reasons.append(
                f"DRIFT DETECTED: CUSUM={self._drift_monitor.cusum:.3f} "
                f"exceeded threshold={CUSUM_THRESHOLD} "
                f"over {self._drift_monitor.sample_count} windows"
            )
            self._drift_monitor.reset()

        if self._local_cusum.record(anchor.confidence):
            enf_status = _worse(enf_status, VerificationStatus.FAILED.value)
            enf_reasons.append(
                f"LOCAL ANOMALY: CUSUM={self._local_cusum.cusum:.3f} "
                f"exceeded threshold={LOCAL_CUSUM_THRESHOLD} "
                f"within a {LOCAL_CUSUM_WINDOW_SIZE}-window span"
            )

        if enf_status == VerificationStatus.TRUSTED.value and anchor.confidence < CONFIDENCE_TRUSTED:
            enf_status = VerificationStatus.SUSPECT.value
            enf_reasons.append(
                f"confidence {anchor.confidence:.4f} below normal threshold "
                f"{CONFIDENCE_TRUSTED} -- monitoring"
            )

        enf_score = {
            VerificationStatus.TRUSTED.value: SCORE_TRUSTED,
            VerificationStatus.SUSPECT.value: anchor.confidence,
            VerificationStatus.FAILED.value:  SCORE_FAILED_HARD,
        }[enf_status]

        results.append(VerificationResult(
            timestamp=ts,
            component_id=f"{self._component_id}/ENF",
            status=enf_status,
            score=round(enf_score, 4),
            anchor_ref=anchor.timestamp,
            reason=" | ".join(enf_reasons) if enf_reasons else "ok",
        ))

        if self._check_nlr:
            record_dict = (
                sample if isinstance(sample, dict)
                else getattr(sample, "__dict__", {})
            )

            merged = _merge_key_results(
                self._nlr_range_check.check(record_dict),
                self._nlr_continuity_check.check(record_dict),
                self._nlr_monotonicity_check.check(record_dict),
                self._cross_sibling_check.check(record_dict),
                self._sustained_deviation_check.check(record_dict),
                self._startup_ramp_check.check(record_dict),
                self._replay_check.check(record_dict),
                self._gpu_temp_range_check.check(record_dict),
                self._gpu_temp_continuity_check.check(record_dict),
            )

            merged = _apply_synchronized_event_correlation(merged)

            for key, (status, reason) in merged.items():
                if status == VerificationStatus.TRUSTED.value:
                    score = SCORE_TRUSTED
                elif status == VerificationStatus.SUSPECT.value:
                    score = SCORE_SUSPECT
                else:
                    score = SCORE_FAILED_HARD
                results.append(VerificationResult(
                    timestamp=ts,
                    component_id=f"{self._component_id}/{key}",
                    status=status,
                    score=round(score, 4),
                    anchor_ref=anchor.timestamp,
                    reason=reason,
                ))

        return results