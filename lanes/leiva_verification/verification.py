"""
verification.py
---------------
Verifier: runs sequential ENF and NLR checks and produces VerificationResult.

Replaces fake_verify in scripts/smoke_test.py.

Checks in order -- stops at the first failure:
  ENF side:
    1. SequenceGuard          replay attack defense
    2. ENFNominalRangeCheck   injection defense on the RAW frequency value
                               (catches single-sample spikes that normalization
                               would otherwise absorb as a new local max/min)
    3. ENFRangeCheck          injection defense on the normalized signature
                               (flat or out-of-[0,1] shapes)
    4. ENFContinuityCheck     discontinuity defense (confidence hard threshold)
    5. DriftMonitor           slow drift defense (CUSUM across many windows)
  NLR side (multi-node aware):
    6. NLRRangeCheck          injection defense (impossible GPU/CPU wattage)
                               -- works for any number of nodes, discovers
                               channels dynamically from record keys
    7. NLRContinuityCheck     discontinuity defense (sudden power jumps)
                               -- tracks per-channel state, node-prefixed
                               column names are unique keys so no collision
    8. NLRMonotonicityCheck   tamper defense (cpu uJ counter must only
                               increase, except for known hardware wraps)
                               -- discovers all cpu uJ columns dynamically

Status mapping (per metrics.py -- only FAILED counts as a detection):
  TRUSTED  all checks pass, confidence >= CONFIDENCE_TRUSTED
  SUSPECT  all checks pass, confidence in [CONFIDENCE_SUSPECT, TRUSTED)
  FAILED   any check fails definitively

Thresholds at the top of this file are starting estimates calibrated
against the two real NLR files profiled so far. Run calibrate_thresholds.py
and nlr_profile.py against your full clean dataset and update these before
relying on detection results for the paper.
"""

from __future__ import annotations

import collections
import math
import statistics
from typing import Optional

from microverse_core.contracts import (
    AnchorRecord,
    VerificationResult,
    VerificationStatus,
)

# ---------------------------------------------------------------------------
# Tunable thresholds -- ENF side
# TODO: calibrate against real ENF data once the glitch/smoothing question
# is resolved. Run calibrate_thresholds.py against a clean combined JSONL
# and paste the output here.
# ---------------------------------------------------------------------------

CONFIDENCE_TRUSTED = 0.70
CONFIDENCE_SUSPECT = 0.50

# Raw frequency plausibility -- checked BEFORE normalization since
# normalization absorbs extreme values into the signature shape.
# Default ±2.0 Hz is generous; tighten once smoothing is confirmed.
NOMINAL_HZ = 60.0
NOMINAL_TOLERANCE_HZ = 2.0

CUSUM_THRESHOLD  = 5.0
CUSUM_BASELINE   = 0.30
CUSUM_HISTORY    = 60

# ---------------------------------------------------------------------------
# Tunable thresholds -- NLR side
# TODO: run nlr_profile.py across all five workload modes and update.
# These placeholders come from the P99 step sizes observed in the two
# real files profiled so far.
# ---------------------------------------------------------------------------

GPU_POWER_CEILING_W = 800.0
CPU_POWER_CEILING_W = 800.0
GPU_MAX_STEP_W      = 470.0
CPU_MAX_STEP_W      = 16.0

# RAPL energy counter hardware wraparound
CPU_UJ_WRAP_CEILING   = 65_500_000_000
CPU_UJ_WRAP_TOLERANCE =  2_000_000_000

# Score values
SCORE_TRUSTED      = 0.95
SCORE_FAILED_HARD  = 0.05
SCORE_FAILED_DRIFT = 0.10


# ---------------------------------------------------------------------------
# Helper: discover NLR channel keys dynamically from a record dict
# ---------------------------------------------------------------------------

def _find_nlr_keys(record: dict) -> dict[str, list[str]]:
    """
    Scans a combined record and groups its keys by channel type.

    Works for any node count -- a 1-node record has 16 NLR keys,
    a 16-node record has 256 NLR keys. The checks never hardcode
    channel names or node prefixes; they call this function instead.

    Returns a dict with four lists:
      gpu_power : all keys matching *_gpu-N[W]  (wattage, any node)
      cpu_power : all keys matching *_cpu-N[W]  (wattage, any node)
      cpu_uj    : all keys matching *_cpu-N[uJ] (energy, any node)
      gpu_temp  : all keys matching *_gpu-N[C]  (temperature, any node)
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


# ---------------------------------------------------------------------------
# Internal check classes -- ENF side
# ---------------------------------------------------------------------------

class _SequenceGuard:
    """
    Rejects duplicate or out-of-order timestamps.
    Replay attacks resend an old timestamp -- caught here before any
    signal processing runs so we waste no computation on them.
    """

    def __init__(self, strict_ordering: bool = True):
        self._seen: set = set()
        self._last: float = -1.0
        self._strict = strict_ordering

    def check(self, timestamp: float) -> tuple[bool, str]:
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
    """
    Confirms the RAW frequency value is physically plausible before
    normalization. Normalization absorbs extreme values as the new local
    max/min without the resulting shape looking abnormal, so this check
    must run on the actual measurement, not the normalized signature.
    """

    def check(self, raw_frequency_hz: float) -> tuple[bool, str]:
        deviation = abs(raw_frequency_hz - NOMINAL_HZ)
        if deviation > NOMINAL_TOLERANCE_HZ:
            return False, (
                f"OUT OF NOMINAL RANGE: raw frequency {raw_frequency_hz:.4f} Hz "
                f"deviates {deviation:.4f} Hz from {NOMINAL_HZ} Hz nominal, "
                f"exceeds tolerance {NOMINAL_TOLERANCE_HZ} Hz"
            )
        return True, "ok"


class _ENFRangeCheck:
    """
    Confirms the normalized anchor signature looks like real ENF.
    Catches flat signatures (zero variance) and out-of-[0,1] values.
    """

    def check(self, signature: list) -> tuple[bool, str]:
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
    """
    Uses anchor.confidence as the continuity signal.
    A sudden drop below the hard threshold means the ENF changed
    discontinuously between windows, which is physically impossible
    on a real grid and signals injection or fabrication.
    """

    def check(self, confidence: float) -> tuple[bool, str]:
        if confidence < CONFIDENCE_SUSPECT:
            return False, (
                f"DISCONTINUITY: confidence {confidence:.4f} below hard "
                f"threshold {CONFIDENCE_SUSPECT} -- "
                f"ENF jumped abruptly, physically impossible on real grid"
            )
        return True, "ok"


class _DriftMonitor:
    """
    One-sided CUSUM on (1 - confidence) as a deviation proxy.
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
        return self._cusum

    @property
    def sample_count(self) -> int:
        return self._n

    def record(self, confidence: float) -> None:
        deviation = 1.0 - confidence
        self._history.append(deviation)
        self._n += 1
        self._cusum = max(0.0, self._cusum + (deviation - self._baseline))

    def is_drifting(self) -> bool:
        return self._cusum > CUSUM_THRESHOLD

    def calibrate(self) -> None:
        """Set baseline from current history after a clean warmup period."""
        if self._history:
            self._baseline = statistics.mean(self._history)
            self._cusum = 0.0

    def reset(self) -> None:
        self._cusum = 0.0


# ---------------------------------------------------------------------------
# Internal check classes -- NLR side (multi-node aware)
# ---------------------------------------------------------------------------

class _NLRRangeCheck:
    """
    Confirms GPU and CPU power readings are physically plausible.

    Discovers channel keys dynamically from the record so it works
    for any number of nodes without any hardcoded column names.
    A single-node record has 4 GPU wattage keys; a 16-node record
    has 64. Both are checked with identical logic.
    """

    def check(self, record: dict) -> tuple[bool, str]:
        channels = _find_nlr_keys(record)

        for key in channels["gpu_power"]:
            val = record.get(key)
            if val is None:
                continue
            if val < 0:
                return False, f"OUT OF RANGE: {key}={val:.2f}W is negative"
            if val > GPU_POWER_CEILING_W:
                return False, (
                    f"OUT OF RANGE: {key}={val:.2f}W exceeds hardware "
                    f"ceiling {GPU_POWER_CEILING_W}W"
                )

        for key in channels["cpu_power"]:
            val = record.get(key)
            if val is None:
                continue
            if val < 0:
                return False, f"OUT OF RANGE: {key}={val:.2f}W is negative"
            if val > CPU_POWER_CEILING_W:
                return False, (
                    f"OUT OF RANGE: {key}={val:.2f}W exceeds hardware "
                    f"ceiling {CPU_POWER_CEILING_W}W"
                )

        return True, "ok"


class _NLRContinuityCheck:
    """
    Confirms GPU and CPU power does not jump implausibly between
    consecutive aggregated windows.

    Maintains previous-value state keyed by the full column name
    (e.g. "x3105c0s41b0n0_gpu-0[W]"). Since each node's columns
    have a unique prefix, there is no collision between nodes and
    no changes are needed to support multi-node configurations --
    the dict naturally tracks all nodes independently.
    """

    def __init__(self):
        self._prev: dict[str, float] = {}

    def check(self, record: dict) -> tuple[bool, str]:
        channels = _find_nlr_keys(record)

        for key in channels["gpu_power"]:
            val = record.get(key)
            if val is None:
                continue
            prev = self._prev.get(key)
            if prev is not None:
                step = abs(val - prev)
                if step > GPU_MAX_STEP_W:
                    self._prev[key] = val
                    return False, (
                        f"DISCONTINUITY: {key} stepped {step:.2f}W between "
                        f"windows, exceeds max plausible step {GPU_MAX_STEP_W}W"
                    )
            self._prev[key] = val

        for key in channels["cpu_power"]:
            val = record.get(key)
            if val is None:
                continue
            prev = self._prev.get(key)
            if prev is not None:
                step = abs(val - prev)
                if step > CPU_MAX_STEP_W:
                    self._prev[key] = val
                    return False, (
                        f"DISCONTINUITY: {key} stepped {step:.2f}W between "
                        f"windows, exceeds max plausible step {CPU_MAX_STEP_W}W"
                    )
            self._prev[key] = val

        return True, "ok"

    def reset(self) -> None:
        self._prev.clear()


class _NLRMonotonicityCheck:
    """
    Confirms CPU energy counters (uJ) only increase, except for known
    RAPL hardware wraparounds (~65.5 billion uJ drop).

    Like _NLRContinuityCheck, state is keyed by the full prefixed
    column name so all nodes are tracked independently with no
    code changes required as node count varies.
    """

    def __init__(self):
        self._prev: dict[str, float] = {}

    def check(self, record: dict) -> tuple[bool, str]:
        channels = _find_nlr_keys(record)

        for key in channels["cpu_uj"]:
            val = record.get(key)
            if val is None:
                continue
            prev = self._prev.get(key)
            if prev is not None and val < prev:
                drop = prev - val
                is_expected_wrap = (
                    abs(drop - CPU_UJ_WRAP_CEILING) < CPU_UJ_WRAP_TOLERANCE
                )
                if not is_expected_wrap:
                    self._prev[key] = val
                    return False, (
                        f"MONOTONICITY VIOLATION: {key} decreased by "
                        f"{drop:.1f} uJ -- not consistent with known hardware "
                        f"wrap (~{CPU_UJ_WRAP_CEILING:.0f} uJ). "
                        f"Counter should only increase or wrap."
                    )
            self._prev[key] = val

        return True, "ok"

    def reset(self) -> None:
        self._prev.clear()


# ---------------------------------------------------------------------------
# Public Verifier
# ---------------------------------------------------------------------------

class Verifier:
    """
    Runs all ENF and NLR checks for every (sample, anchor) pair.

    Fully multi-node aware: the NLR checks discover channel names
    dynamically from the record so they work identically for 1 node
    or 16 nodes with no configuration required.

    Parameters
    ----------
    component_id : str
        Twin component being verified e.g. "rack_00".
        Must match Hendricks' StateVariable.source_object and
        Marchisano's AttackEvent.target_component.
    warmup_windows : int
        Clean windows before drift baseline is calibrated.
        Keep below the earliest possible attack start.
    strict_ordering : bool
        Enforce strictly increasing timestamps.
    check_nlr : bool
        Whether to run the NLR checks. Default True.
        Set False for ENF-only testing.
    """

    def __init__(
        self,
        component_id: str,
        warmup_windows: int = 10,
        strict_ordering: bool = True,
        check_nlr: bool = True,
    ):
        self._component_id = component_id
        self._warmup_windows = warmup_windows
        self._check_nlr = check_nlr

        # ENF checks
        self._sequence_guard      = _SequenceGuard(strict_ordering)
        self._nominal_range_check = _ENFNominalRangeCheck()
        self._range_check         = _ENFRangeCheck()
        self._continuity_check    = _ENFContinuityCheck()
        self._drift_monitor       = _DriftMonitor()

        # NLR checks -- multi-node aware, no configuration needed
        self._nlr_range_check        = _NLRRangeCheck()
        self._nlr_continuity_check   = _NLRContinuityCheck()
        self._nlr_monotonicity_check = _NLRMonotonicityCheck()

        self._windows_processed: int = 0

    @property
    def component_id(self) -> str:
        return self._component_id

    @property
    def windows_processed(self) -> int:
        return self._windows_processed

    def verify(self, sample, anchor: AnchorRecord) -> VerificationResult:
        """
        Verify one sample against its anchor.

        Parameters
        ----------
        sample : dict or object with .timestamp
            Combined record from read_combined_jsonl() (post-Ethan),
            or a PowerSample / StateVariable object.
            For dict input: must have "index" or "timestamp" key.
            NLR checks use whatever node-prefixed channel keys are
            present -- no configuration required for multi-node.
        anchor : AnchorRecord
            ENF anchor from AnchorExtractor.extract() for same timestamp.
        """
        # extract timestamp
        if hasattr(sample, "timestamp"):
            ts = sample.timestamp
        elif isinstance(sample, dict):
            ts = float(sample.get("timestamp", sample.get("index", 0)))
        else:
            ts = 0.0

        # ----------------------------------------------------------------
        # Check 1: Sequence guard -- replay defense
        # ----------------------------------------------------------------
        passed, reason = self._sequence_guard.check(ts)
        if not passed:
            return self._make_result(
                ts, anchor, VerificationStatus.FAILED,
                SCORE_FAILED_HARD, reason
            )

        # ----------------------------------------------------------------
        # Check 2: ENF nominal range -- raw frequency injection defense
        # ----------------------------------------------------------------
        raw_freq = None
        if isinstance(sample, dict):
            raw_freq = sample.get("FRQ", sample.get("frequency_hz"))
        else:
            raw_freq = getattr(sample, "FRQ",
                       getattr(sample, "frequency_hz", None))

        if raw_freq is not None:
            passed, reason = self._nominal_range_check.check(raw_freq)
            if not passed:
                return self._make_result(
                    ts, anchor, VerificationStatus.FAILED,
                    SCORE_FAILED_HARD, reason
                )

        # ----------------------------------------------------------------
        # Check 3: ENF signature range -- normalized shape check
        # ----------------------------------------------------------------
        passed, reason = self._range_check.check(anchor.signature)
        if not passed:
            return self._make_result(
                ts, anchor, VerificationStatus.FAILED,
                SCORE_FAILED_HARD, reason
            )

        # ----------------------------------------------------------------
        # Check 4: ENF continuity -- confidence hard threshold
        # ----------------------------------------------------------------
        passed, reason = self._continuity_check.check(anchor.confidence)
        if not passed:
            return self._make_result(
                ts, anchor, VerificationStatus.FAILED,
                anchor.confidence, reason
            )

        # ----------------------------------------------------------------
        # Checks 6-8: NLR checks -- multi-node, dynamic channel discovery
        # ----------------------------------------------------------------
        if self._check_nlr:
            record_dict = (
                sample if isinstance(sample, dict)
                else getattr(sample, "__dict__", {})
            )

            passed, reason = self._nlr_range_check.check(record_dict)
            if not passed:
                return self._make_result(
                    ts, anchor, VerificationStatus.FAILED,
                    SCORE_FAILED_HARD, reason
                )

            passed, reason = self._nlr_continuity_check.check(record_dict)
            if not passed:
                return self._make_result(
                    ts, anchor, VerificationStatus.FAILED,
                    SCORE_FAILED_HARD, reason
                )

            passed, reason = self._nlr_monotonicity_check.check(record_dict)
            if not passed:
                return self._make_result(
                    ts, anchor, VerificationStatus.FAILED,
                    SCORE_FAILED_HARD, reason
                )

        # ----------------------------------------------------------------
        # Check 5: Drift monitor -- slow drift via CUSUM
        # ----------------------------------------------------------------
        self._drift_monitor.record(anchor.confidence)
        self._windows_processed += 1

        if self._windows_processed == self._warmup_windows:
            self._drift_monitor.calibrate()

        if self._drift_monitor.is_drifting():
            drift_score = max(
                0.0,
                SCORE_FAILED_DRIFT *
                (1.0 - self._drift_monitor.cusum / CUSUM_THRESHOLD)
            )
            reason = (
                f"DRIFT DETECTED: CUSUM={self._drift_monitor.cusum:.3f} "
                f"exceeded threshold={CUSUM_THRESHOLD} "
                f"over {self._drift_monitor.sample_count} windows"
            )
            self._drift_monitor.reset()
            return self._make_result(
                ts, anchor, VerificationStatus.FAILED,
                drift_score, reason
            )

        # ----------------------------------------------------------------
        # All checks passed
        # ----------------------------------------------------------------
        if anchor.confidence >= CONFIDENCE_TRUSTED:
            return self._make_result(
                ts, anchor, VerificationStatus.TRUSTED,
                anchor.confidence, "ok"
            )
        else:
            return self._make_result(
                ts, anchor, VerificationStatus.SUSPECT,
                anchor.confidence,
                f"confidence {anchor.confidence:.4f} below "
                f"normal threshold {CONFIDENCE_TRUSTED} -- monitoring"
            )

    def _make_result(
        self,
        timestamp: float,
        anchor: AnchorRecord,
        status: VerificationStatus,
        score: float,
        reason: str,
    ) -> VerificationResult:
        return VerificationResult(
            timestamp=timestamp,
            component_id=self._component_id,
            status=status.value,
            score=round(score, 4),
            anchor_ref=anchor.timestamp,
            reason=reason,
        )