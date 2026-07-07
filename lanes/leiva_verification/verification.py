"""
verification.py
---------------
Verifier: runs EVERY ENF and NLR check against EVERY component, every
call, and returns a full list of per-component results. Nothing stops
early. The job of this module is coverage and attribution -- test if
something is wrong, mark exactly what and where, and pass all of that
information along -- not to gate whether data keeps moving.

  ENF side (one shared grid signal -- one merged result per record):
    1. SequenceGuard          replay attack defense
    2. ENFNominalRangeCheck   injection defense on the RAW frequency value
                               (catches single-sample spikes that normalization
                               would otherwise absorb as a new local max/min)
    3. ENFRangeCheck          injection defense on the normalized signature
                               (flat or out-of-[0,1] shapes)
    4. ENFContinuityCheck     discontinuity defense (confidence hard threshold)
    5. DriftMonitor           slow drift defense (CUSUM across many windows)
    All five always run. If more than one fires in the same window, the
    merged ENF result reports the worst status and every reason that fired,
    not just the first.

  NLR side (multi-node aware -- one result per physical channel):
    6. NLRRangeCheck           injection defense (impossible GPU/CPU wattage)
    7. NLRContinuityCheck      discontinuity defense (sudden power jumps)
    8. NLRMonotonicityCheck    tamper defense on CPU energy counters --
                               checks BOTH directions: an illegitimate
                               decrease (rollback, hiding real energy use)
                               and an implausibly large increase (energy
                               spike) that a decrease-only check would
                               never catch.
    9. GPUTempRangeCheck       NEW -- GPU temperature physical plausibility.
                               Was previously discovered by _find_nlr_keys()
                               but never actually checked by anything.
   10. GPUTempContinuityCheck  NEW -- GPU temperature step-size plausibility.
    Every channel present in the record gets its own result, every call --
    a bad gpu-0[W] reading does not stop cpu-1[uJ] from being checked, and
    if BOTH a range and a step-size problem hit the same channel in the
    same window, they merge into one result naming both.

Status mapping (per metrics.py -- only FAILED counts as a detection):
  TRUSTED  passes cleanly (dashboard-facing label: "good")
  SUSPECT  ENF confidence in the soft zone, no hard check failed
           (dashboard-facing label: "suspect")
  FAILED   a hard check failed definitively
           (dashboard-facing label: "warning")
These three enum values (TRUSTED/SUSPECT/FAILED) are unchanged from
before and metrics.py should keep working against them as-is -- the
good/suspect/warning wording is a presentation-layer relabeling for
the dashboard, not a change to what's stored or computed here.

Thresholds at the top of this file are starting estimates. Two blocks
are marked PLACEHOLDER below (GPU temperature, CPU energy upward-spike)
because nothing has ever profiled real data for them -- treat any flag
from those specific checks as provisional until they're calibrated the
same way GPU_MAX_STEP_W etc. were.
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

# ---------------------------------------------------------------------------
# PLACEHOLDER thresholds -- NEW checks, NOT calibrated against real data.
# GPU temperature was never checked by anything before this version.
# CPU energy was only ever checked for illegitimate DECREASES; an
# implausibly large but still-increasing jump passed unnoticed.
# ---------------------------------------------------------------------------

GPU_TEMP_FLOOR_C    = 0.0    # a GPU under any load at/below 0C is a stuck/fabricated sensor
GPU_TEMP_CEILING_C  = 95.0   # datacenter GPUs throttle/shutdown in the 83-95C range
GPU_TEMP_MAX_STEP_C = 15.0   # placeholder max plausible swing per ENF window

# Derived (not arbitrary): CPU_POWER_CEILING_W for one ENF window,
# converted to uJ, with a 1.5x safety margin.
_ENF_WINDOW_SECONDS = 2.0
CPU_UJ_MAX_STEP_UJ = CPU_POWER_CEILING_W * _ENF_WINDOW_SECONDS * 1_000_000 * 1.5

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


def _worse(a: str, b: str) -> str:
    """Rank: TRUSTED < SUSPECT < FAILED. Returns whichever is worse."""
    rank = {
        VerificationStatus.TRUSTED.value: 0,
        VerificationStatus.SUSPECT.value: 1,
        VerificationStatus.FAILED.value:  2,
    }
    return a if rank[a] >= rank[b] else b


# ---------------------------------------------------------------------------
# Internal check classes -- ENF side
# One shared signal -- these check the WHOLE record, not per-key, and
# every one of them always runs regardless of what the others found.
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
#
# Each .check() method below returns a dict keyed by the FULL column
# name, mapping to (status, reason) for EVERY matching key found in the
# record -- never just the first bad one. Callers merge these per-key
# dicts across multiple check classes into one final result per channel.
# ---------------------------------------------------------------------------

_TRUSTED = VerificationStatus.TRUSTED.value
_FAILED  = VerificationStatus.FAILED.value


class _NLRRangeCheck:
    """
    Confirms GPU and CPU power readings are physically plausible.
    Every gpu_power/cpu_power key gets its own entry, always.
    """

    def check(self, record: dict) -> dict[str, tuple[str, str]]:
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
    """
    Confirms GPU and CPU power does not jump implausibly between
    consecutive aggregated windows. Every key gets its own entry,
    always -- state is kept per full column name so nodes never
    collide and every channel is tracked independently.
    """

    def __init__(self):
        self._prev: dict[str, float] = {}

    def check(self, record: dict) -> dict[str, tuple[str, str]]:
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
        self._prev.clear()


class _NLRMonotonicityCheck:
    """
    Confirms CPU energy counters (uJ) behave like a real hardware
    energy counter in BOTH directions:
      - may only decrease via a known RAPL wraparound (~65.5B uJ drop)
        -- any other decrease is a rollback attack, hiding real energy use
      - may not increase by more than CPU_UJ_MAX_STEP_UJ in one window
        (PLACEHOLDER, derived from CPU_POWER_CEILING_W -- not yet
        profiled against real data) -- an implausibly large increase
        that stayed "monotonic" would previously have passed unnoticed
    """

    def __init__(self):
        self._prev: dict[str, float] = {}

    def check(self, record: dict) -> dict[str, tuple[str, str]]:
        channels = _find_nlr_keys(record)
        results: dict[str, tuple[str, str]] = {}

        for key in channels["cpu_uj"]:
            val = record.get(key)
            if val is None:
                continue
            prev = self._prev.get(key)

            if prev is None:
                results[key] = (_TRUSTED, "ok")
            elif val < prev:
                drop = prev - val
                is_expected_wrap = abs(drop - CPU_UJ_WRAP_CEILING) < CPU_UJ_WRAP_TOLERANCE
                if is_expected_wrap:
                    results[key] = (_TRUSTED, "ok")
                else:
                    results[key] = (_FAILED, (
                        f"MONOTONICITY VIOLATION: {key} decreased by "
                        f"{drop:.1f} uJ -- not consistent with known hardware "
                        f"wrap (~{CPU_UJ_WRAP_CEILING:.0f} uJ). "
                        f"Counter should only increase or wrap."
                    ))
            else:
                increase = val - prev
                if increase > CPU_UJ_MAX_STEP_UJ:
                    results[key] = (_FAILED, (
                        f"ENERGY SPIKE: {key} increased by {increase:.1f} uJ "
                        f"in one window, exceeds plausibility ceiling "
                        f"{CPU_UJ_MAX_STEP_UJ:.1f} uJ (PLACEHOLDER threshold)"
                    ))
                else:
                    results[key] = (_TRUSTED, "ok")

            self._prev[key] = val

        return results

    def reset(self) -> None:
        self._prev.clear()


class _GPUTempRangeCheck:
    """
    NEW. Confirms GPU temperature readings are physically plausible.
    gpu_temp keys were already discovered by _find_nlr_keys() but
    nothing ever checked them before this class existed -- an attacker
    could set GPU temperature to any value with zero detection.

    PLACEHOLDER thresholds -- not calibrated against real data.
    """

    def check(self, record: dict) -> dict[str, tuple[str, str]]:
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
    """
    NEW. Confirms GPU temperature does not jump implausibly between
    consecutive windows -- thermal mass gives real GPUs inertia; an
    instant multi-degree swing is a strong tamper signal.

    PLACEHOLDER threshold -- not calibrated against real data.
    """

    def __init__(self):
        self._prev: dict[str, float] = {}

    def check(self, record: dict) -> dict[str, tuple[str, str]]:
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
        self._prev.clear()


def _merge_key_results(*result_dicts: dict[str, tuple[str, str]]) -> dict[str, tuple[str, str]]:
    """
    Merges per-key (status, reason) dicts from multiple check classes
    into one final dict per key. If a channel is flagged by more than
    one check in the same window (e.g. both a range AND a step-size
    problem), the merged entry keeps the WORST status and concatenates
    every non-"ok" reason -- nothing gets silently dropped in a merge.
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


# ---------------------------------------------------------------------------
# Public Verifier
# ---------------------------------------------------------------------------

class Verifier:
    """
    Runs EVERY ENF and NLR check for EVERY component, every call.
    Nothing stops early -- verify() always returns a result for the
    ENF anchor plus one result per NLR channel present in the record.

    Fully multi-node aware: the NLR checks discover channel names
    dynamically from record keys, so an instance constructed for one
    node's own sub-record only ever reports on that node's channels.

    Parameters
    ----------
    component_id : str
        Identifies what's being verified, e.g. "rack_00/x3105c0s37b0n0".
        Individual NLR results extend this with the channel name, e.g.
        "rack_00/x3105c0s37b0n0/gpu-0[W]".
    warmup_windows : int
        Clean windows before drift baseline is calibrated.
    strict_ordering : bool
        Enforce strictly increasing timestamps.
    check_nlr : bool
        Whether to run the NLR/GPU-temp checks. Default True.
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
        self._gpu_temp_range_check      = _GPUTempRangeCheck()
        self._gpu_temp_continuity_check = _GPUTempContinuityCheck()

        self._windows_processed: int = 0

    @property
    def component_id(self) -> str:
        return self._component_id

    @property
    def windows_processed(self) -> int:
        return self._windows_processed

    def verify(self, sample, anchor: AnchorRecord) -> list[VerificationResult]:
        """
        Verify one sample against its anchor. Always runs every check
        and always returns a full list of results -- one merged result
        for the ENF anchor, plus one result per NLR/GPU-temp channel
        present in `sample` (when check_nlr=True).

        Parameters
        ----------
        sample : dict or object with .timestamp
            Combined record (or per-node sub-record) from
            read_combined_jsonl(), post-Ethan.
        anchor : AnchorRecord
            ENF anchor from AnchorExtractor.extract() for same timestamp.
        """
        if hasattr(sample, "timestamp"):
            ts = sample.timestamp
        elif isinstance(sample, dict):
            ts = float(sample.get("timestamp", sample.get("index", 0)))
        else:
            ts = 0.0

        results: list[VerificationResult] = []

        # ------------------------------------------------------------
        # ENF side: run every check unconditionally, merge into ONE
        # result for the shared anchor component.
        # ------------------------------------------------------------
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

        passed, reason = self._range_check.check(anchor.signature)
        if not passed:
            enf_status = _worse(enf_status, VerificationStatus.FAILED.value)
            enf_reasons.append(reason)

        passed, reason = self._continuity_check.check(anchor.confidence)
        if not passed:
            enf_status = _worse(enf_status, VerificationStatus.FAILED.value)
            enf_reasons.append(reason)

        # Drift monitor always records -- independent of every other
        # check's outcome, ENF-side or NLR-side.
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

        # Soft confidence tier only applies if nothing hard already failed
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

        # ------------------------------------------------------------
        # NLR side: run every check unconditionally, merge per-channel,
        # emit one result per channel present -- always, good or bad.
        # ------------------------------------------------------------
        if self._check_nlr:
            record_dict = (
                sample if isinstance(sample, dict)
                else getattr(sample, "__dict__", {})
            )

            merged = _merge_key_results(
                self._nlr_range_check.check(record_dict),
                self._nlr_continuity_check.check(record_dict),
                self._nlr_monotonicity_check.check(record_dict),
                self._gpu_temp_range_check.check(record_dict),
                self._gpu_temp_continuity_check.check(record_dict),
            )

            for key, (status, reason) in merged.items():
                score = SCORE_TRUSTED if status == VerificationStatus.TRUSTED.value else SCORE_FAILED_HARD
                results.append(VerificationResult(
                    timestamp=ts,
                    component_id=f"{self._component_id}/{key}",
                    status=status,
                    score=round(score, 4),
                    anchor_ref=anchor.timestamp,
                    reason=reason,
                ))

        return results