"""
verification.py
---------------
Verifier: runs sequential ENF and NLR checks and produces VerificationResult.

Replaces fake_verify in scripts/smoke_test.py.

Checks in order -- stops at the first failure:
  ENF side (existing):
    1. SequenceGuard          replay attack defense
    2. ENFNominalRangeCheck   injection defense on the RAW frequency
                               value (catches single-sample spikes that
                               normalization would otherwise absorb)
    3. ENFRangeCheck          injection defense on the normalized
                               signature (flat or out-of-[0,1] shapes)
    4. ENFContinuityCheck     discontinuity defense (confidence threshold)
    5. DriftMonitor           slow drift defense (CUSUM across windows)
  NLR side (new):
    6. NLRRangeCheck          injection defense (impossible GPU/CPU wattage)
    7. NLRContinuityCheck     discontinuity defense (sudden power/temp jumps)
    8. NLRMonotonicityCheck   tamper defense (cpu uJ counter must only
                               increase, except for known hardware wraps)

Status mapping (per metrics.py -- only FAILED counts as a detection):
  TRUSTED  all checks pass, confidence >= CONFIDENCE_TRUSTED
  SUSPECT  all checks pass, confidence in [CONFIDENCE_SUSPECT, TRUSTED)
  FAILED   any check fails definitively

Thresholds at the top of this file are starting estimates. Run
enf_profile.py and nlr_profile.py against real clean data and update
these before relying on detection results. As of the last team check-in,
the real ENF dataset showed isolated single-sample capture glitches --
confirm with the team whether those get smoothed upstream in
data_loaders.py before tightening CONFIDENCE_TRUSTED/SUSPECT further.
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
# TODO: calibrate against real ENF data once the glitch-handling question
# is resolved with the team (see module docstring above).
# ---------------------------------------------------------------------------

CONFIDENCE_TRUSTED = 0.70    # above this -> TRUSTED
CONFIDENCE_SUSPECT = 0.50    # above this but below TRUSTED -> SUSPECT
                              # below this -> FAILED (discontinuity)

# Raw frequency plausibility -- checked BEFORE normalization, since
# normalization can absorb an extreme raw value into the signature's
# shape without it ever looking abnormal post-normalization. This is
# the raw-value equivalent of GPU_POWER_CEILING_W below: a hard physical
# bound on what the actual measurement can be, not what its shape looks
# like relative to its own window.
# TODO: this generous default (±2.0 Hz) is a placeholder until the team
# resolves the ENF glitch/smoothing question -- ANCHOR-Grid's real grid
# data stays within ±0.02-0.05 Hz, but the real dataset profiled so far
# shows isolated spikes loosely related to capture artifacts. Tighten
# this once that's resolved.
NOMINAL_HZ = 60.0
NOMINAL_TOLERANCE_HZ = 2.0

CUSUM_THRESHOLD  = 5.0       # accumulated deviation before DRIFT alert
CUSUM_BASELINE   = 0.30      # expected mean (1 - confidence) for honest ENF
                              # calibrated automatically after warmup
CUSUM_HISTORY    = 60        # rolling window size in samples

# ---------------------------------------------------------------------------
# Tunable thresholds -- NLR side
# TODO: calibrate against nlr_profile.py output once you've profiled
# clean data across all five workload modes (offline / online finite /
# online rate / training llama2 lora / training stable diffusion).
# These defaults are conservative placeholders based on the two real
# files already profiled.
# ---------------------------------------------------------------------------

# Hard power ceilings -- matches the 800W hardware-error cutoff already
# used in data_loaders.py during ingestion. A value above this is not
# physically possible for this hardware.
GPU_POWER_CEILING_W = 800.0
CPU_POWER_CEILING_W = 800.0

# Max plausible change in watts between two consecutive aggregated
# windows (each window = 2 seconds of real time). Calibrate from
# nlr_profile.py's GPU_MAX_STEP_W / CPU_MAX_STEP_W recommendation --
# these placeholders come from the P99 step size observed across the
# two real files profiled so far, with margin.
GPU_MAX_STEP_W = 470.0
CPU_MAX_STEP_W = 16.0

# RAPL energy counters (uJ) are fixed-width hardware registers that wrap
# around on overflow. Observed wrap magnitude in real data is ~65.5
# billion uJ. A decrease NOT close to this size is a real monotonicity
# violation (tampering); a decrease close to this size is an expected
# hardware wrap and must not be flagged.
CPU_UJ_WRAP_CEILING = 65_500_000_000
CPU_UJ_WRAP_TOLERANCE = 2_000_000_000

# Score values embedded in VerificationResult
SCORE_TRUSTED      = 0.95
SCORE_FAILED_HARD  = 0.05    # replay, flat/invalid signature, discontinuity
SCORE_FAILED_DRIFT = 0.10    # CUSUM threshold crossed


# ---------------------------------------------------------------------------
# Internal check classes -- ENF side (unchanged from previous version)
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
    Confirms the RAW frequency value is physically plausible, before
    any normalization happens.

    This check exists because _ENFRangeCheck below only inspects the
    normalized signature -- and normalization can absorb an extreme raw
    value as the new local max/min without the resulting shape looking
    abnormal at all. A single spike to 75 Hz, for example, just becomes
    "the highest point in this window" after normalization; nothing in
    the normalized signature reveals that 75 Hz is itself impossible on
    a real power grid. This check catches that gap by looking at the
    actual measurement, the same way _NLRRangeCheck checks raw GPU
    wattage rather than some derived shape of it.
    """

    def check(self, raw_frequency_hz: float) -> tuple[bool, str]:
        deviation = abs(raw_frequency_hz - NOMINAL_HZ)
        if deviation > NOMINAL_TOLERANCE_HZ:
            return False, (
                f"OUT OF NOMINAL RANGE: raw frequency {raw_frequency_hz:.4f} Hz "
                f"deviates {deviation:.4f} Hz from {NOMINAL_HZ} Hz nominal, "
                f"exceeds tolerance {NOMINAL_TOLERANCE_HZ} Hz -- "
                f"not physically plausible on a real power grid"
            )
        return True, "ok"


class _ENFRangeCheck:
    """
    Confirms the anchor signature looks like real ENF.

    Two failure modes:
      Flat signature  -- all values identical (zero variance).
                         Real ENF always has some fluctuation.
                         A fabricated constant signal gets caught here.
      Invalid values  -- any value outside [0, 1].
                         Normalized values must always be in this range.
                         A bug or corruption in the extractor shows here.
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

    Confidence is the Pearson correlation between the current ENF window
    and the previous one. On a real power grid this is always high because
    mechanical turbine inertia prevents abrupt frequency jumps. A sudden
    drop below the hard threshold means the ENF changed discontinuously,
    which is physically impossible and signals injection or fabrication.
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

    Slow drift attacks keep each individual window just within the
    per-window thresholds, so checks 1-3 pass every time. CUSUM catches
    the drift by accumulating the directional bias across many windows.

    When (1 - confidence) consistently exceeds the baseline, the CUSUM
    sum grows until it crosses CUSUM_THRESHOLD and a DRIFT alert fires.
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
        """
        Set baseline from current history.
        Call after a clean warmup period before attack testing begins.
        Resets CUSUM so warmup data does not pollute drift detection.
        """
        if self._history:
            self._baseline = statistics.mean(self._history)
            self._cusum = 0.0

    def reset(self) -> None:
        self._cusum = 0.0


# ---------------------------------------------------------------------------
# Internal check classes -- NLR side (new)
# ---------------------------------------------------------------------------

class _NLRRangeCheck:
    """
    Confirms GPU and CPU power readings are physically plausible.

    A reading above the hardware ceiling cannot occur on real hardware --
    this mirrors the 800W cutoff already applied during ingestion in
    data_loaders.py, applied again here as a verification-time check
    in case Ethan's tampering reintroduces an impossible value after
    ingestion already happened.
    """

    def check(self, record: dict) -> tuple[bool, str]:
        gpu_channels = ["gpu-0[W]", "gpu-1[W]", "gpu-2[W]", "gpu-3[W]"]
        cpu_channels = ["cpu-0[W]", "cpu-1[W]"]

        for ch in gpu_channels:
            val = record.get(ch)
            if val is not None and val > GPU_POWER_CEILING_W:
                return False, (
                    f"OUT OF RANGE: {ch}={val:.2f}W exceeds hardware "
                    f"ceiling {GPU_POWER_CEILING_W}W"
                )
            if val is not None and val < 0:
                return False, f"OUT OF RANGE: {ch}={val:.2f}W is negative"

        for ch in cpu_channels:
            val = record.get(ch)
            if val is not None and val > CPU_POWER_CEILING_W:
                return False, (
                    f"OUT OF RANGE: {ch}={val:.2f}W exceeds hardware "
                    f"ceiling {CPU_POWER_CEILING_W}W"
                )
            if val is not None and val < 0:
                return False, f"OUT OF RANGE: {ch}={val:.2f}W is negative"

        return True, "ok"


class _NLRContinuityCheck:
    """
    Confirms GPU/CPU power does not jump implausibly between consecutive
    aggregated windows (each window = 2 seconds of real time).

    Real GPU power under sustained workload ramps gradually -- a sudden
    spike or drop far beyond the observed P99 step size in clean data
    indicates an injected or fabricated reading rather than genuine
    workload variation.

    Maintains separate previous-value state per channel so it can be
    called once per record with all channels checked together.
    """

    def __init__(self):
        self._prev: dict = {}

    def check(self, record: dict) -> tuple[bool, str]:
        channels = {
            "gpu-0[W]": GPU_MAX_STEP_W, "gpu-1[W]": GPU_MAX_STEP_W,
            "gpu-2[W]": GPU_MAX_STEP_W, "gpu-3[W]": GPU_MAX_STEP_W,
            "cpu-0[W]": CPU_MAX_STEP_W, "cpu-1[W]": CPU_MAX_STEP_W,
        }

        for ch, max_step in channels.items():
            val = record.get(ch)
            if val is None:
                continue

            prev_val = self._prev.get(ch)
            if prev_val is not None:
                step = abs(val - prev_val)
                if step > max_step:
                    self._prev[ch] = val  # still update so we don't cascade-fail
                    return False, (
                        f"DISCONTINUITY: {ch} stepped {step:.2f}W between "
                        f"windows, exceeds max plausible step {max_step}W"
                    )

            self._prev[ch] = val

        return True, "ok"

    def reset(self) -> None:
        self._prev = {}


class _NLRMonotonicityCheck:
    """
    Confirms CPU energy counters (uJ) only increase, except for known
    hardware wraparounds.

    RAPL energy counters are fixed-width hardware registers. They wrap
    to near-zero when they overflow -- observed wrap magnitude in real
    data is consistently ~65.5 billion uJ. A decrease of any OTHER size
    is not explainable by hardware behavior and indicates the value was
    tampered with or fabricated.
    """

    def __init__(self):
        self._prev: dict = {}

    def check(self, record: dict) -> tuple[bool, str]:
        channels = ["cpu-0[uJ]", "cpu-1[uJ]"]

        for ch in channels:
            val = record.get(ch)
            if val is None:
                continue

            prev_val = self._prev.get(ch)
            if prev_val is not None and val < prev_val:
                drop = prev_val - val
                is_expected_wrap = (
                    abs(drop - CPU_UJ_WRAP_CEILING) < CPU_UJ_WRAP_TOLERANCE
                )
                if not is_expected_wrap:
                    self._prev[ch] = val
                    return False, (
                        f"MONOTONICITY VIOLATION: {ch} decreased by "
                        f"{drop:.1f} uJ, not consistent with a known "
                        f"hardware wrap (~{CPU_UJ_WRAP_CEILING:.0f} uJ) -- "
                        f"counter should only increase or wrap"
                    )

            self._prev[ch] = val

        return True, "ok"

    def reset(self) -> None:
        self._prev = {}


# ---------------------------------------------------------------------------
# Public Verifier
# ---------------------------------------------------------------------------

class Verifier:
    """
    Runs all ENF and NLR checks for every (sample, anchor) pair.

    Parameters
    ----------
    component_id : str
        Twin component being verified e.g. "rack_00".
        Must match Hendricks' StateVariable.source_object and
        Marchisano's AttackEvent.target_component so metrics.score()
        can correctly classify true positives.
    warmup_windows : int
        Clean windows before drift baseline is calibrated.
        Keep below the earliest possible attack start.
    strict_ordering : bool
        Enforce strictly increasing timestamps.
        True catches out-of-order replays.
        False catches only exact duplicate timestamps.
    check_nlr : bool
        Whether to run the NLR checks at all. Default True. Set False
        if a given run has no NLR fields available on the sample
        (e.g. ENF-only testing).
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

        # NLR checks
        self._nlr_range_check       = _NLRRangeCheck()
        self._nlr_continuity_check  = _NLRContinuityCheck()
        self._nlr_monotonicity_check = _NLRMonotonicityCheck()

        self._windows_processed: int = 0

    # --- getters ---------------------------------------------------------------

    @property
    def component_id(self) -> str:
        return self._component_id

    @property
    def windows_processed(self) -> int:
        return self._windows_processed

    # --- public API ------------------------------------------------------------

    def verify(self, sample, anchor: AnchorRecord) -> VerificationResult:
        """
        Verify one sample against its anchor.

        Parameters
        ----------
        sample : PowerSample, StateVariable, or dict
            Claimed twin state for this time step.
            Must have a .timestamp attribute (float, elapsed seconds)
            for the ENF checks. For NLR checks, must also expose the
            combined record's GPU/CPU fields -- either as a dict via
            sample (if sample IS the combined dict from
            build_combined_records), or via attribute access matching
            the same field names.
        anchor : AnchorRecord
            ENF anchor from AnchorExtractor.extract() at same timestamp.

        Returns
        -------
        VerificationResult
            All fields match microverse_core.contracts exactly.
            Only FAILED status counts as a detection in metrics.score().
        """
        if hasattr(sample, "timestamp"):
            ts = sample.timestamp
        elif isinstance(sample, dict):
            ts = sample.get("timestamp", sample.get("index", 0))
        else:
            ts = 0

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
        # Check 2: ENF nominal range check -- raw value injection defense
        # Reads the RAW frequency, not the normalized signature, so an
        # extreme single-sample spike can't hide by becoming the new
        # local max/min after normalization. Only runs if the raw value
        # is available on the sample (dict with "FRQ", or an object
        # exposing frequency_hz / FRQ as an attribute).
        # ----------------------------------------------------------------
        raw_freq = None
        if isinstance(sample, dict):
            raw_freq = sample.get("FRQ", sample.get("frequency_hz"))
        else:
            raw_freq = getattr(sample, "FRQ", getattr(sample, "frequency_hz", None))

        if raw_freq is not None:
            passed, reason = self._nominal_range_check.check(raw_freq)
            if not passed:
                return self._make_result(
                    ts, anchor, VerificationStatus.FAILED,
                    SCORE_FAILED_HARD, reason
                )

        # ----------------------------------------------------------------
        # Check 2: ENF range check -- injection defense
        # ----------------------------------------------------------------
        passed, reason = self._range_check.check(anchor.signature)
        if not passed:
            return self._make_result(
                ts, anchor, VerificationStatus.FAILED,
                SCORE_FAILED_HARD, reason
            )

        # ----------------------------------------------------------------
        # Check 3: ENF continuity check -- discontinuity defense
        # ----------------------------------------------------------------
        passed, reason = self._continuity_check.check(anchor.confidence)
        if not passed:
            return self._make_result(
                ts, anchor, VerificationStatus.FAILED,
                anchor.confidence, reason
            )

        # ----------------------------------------------------------------
        # NLR checks 5-7 -- only run if the sample carries NLR fields
        # ----------------------------------------------------------------
        if self._check_nlr:
            record_dict = sample if isinstance(sample, dict) else getattr(sample, "__dict__", {})

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
        # Check 4: Drift monitor -- slow drift defense via CUSUM
        # Record confidence before checking so monitor accumulates
        # even on windows that passed the per-window checks above
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
        # All checks passed -- assign TRUSTED or SUSPECT
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

    # --- private helper --------------------------------------------------------

    def _make_result(
        self,
        timestamp: float,
        anchor: AnchorRecord,
        status: VerificationStatus,
        score: float,
        reason: str,
    ) -> VerificationResult:
        """Instantiates VerificationResult with exact contract field names."""
        return VerificationResult(
            timestamp=timestamp,
            component_id=self._component_id,
            status=status.value,
            score=round(score, 4),
            anchor_ref=anchor.timestamp,
            reason=reason,
        )