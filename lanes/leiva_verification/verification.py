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

   11. Cross-node corroboration (_apply_synchronized_event_correlation)
                               NEW -- runs after all NLR checks merge.
                               A pure step-size (discontinuity) failure
                               is downgraded from FAILED to SUSPECT when
                               enough OTHER independent nodes show the
                               same step on the same physical channel in
                               the same window -- real synchronized
                               system events (checkpoints, gradient-sync
                               barriers, job startup) produce exactly
                               this signature, and an attacker targeting
                               one node cannot fake agreement from many
                               other real nodes. Never downgrades a
                               channel that is ALSO out of range or
                               violating monotonicity -- corroboration
                               only softens pure step-size calls.

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

CONFIDENCE_TRUSTED = 0.93  # RECALIBRATED (2026-07) for combined_smooth() output (was 0.25, calibrated against the older clean_enf() pipeline with mean confidence ~0.25). combined_smooth()'s mean/median confidence on real cleaned data is ~0.97-0.98; P10 of that distribution was measured at 0.9392. Old value would provide near-zero discrimination against the new, much tighter baseline.
CONFIDENCE_SUSPECT = 0.85  # RECALIBRATED (2026-07) for combined_smooth() output (was -0.55). Matches the single_window_threshold validated in test_combined_smoothing.py -- comfortably below the measured P1 of clean data (0.9215), catches the sustained attack and (with LocalCUSUMDetector below) the quick-splice sweep. Re-verify against real ground-truth attack data before treating as final, same caveat as the value it replaces.

# Raw frequency plausibility -- checked BEFORE normalization since
# normalization absorbs extreme values into the signature shape.
# Default ±2.0 Hz is generous; tighten once smoothing is confirmed.
NOMINAL_HZ = 60.0
NOMINAL_TOLERANCE_HZ = 2.0

CUSUM_THRESHOLD  = 5.0
CUSUM_BASELINE   = 0.03  # RECALIBRATED (2026-07) for combined_smooth() output (was 0.30). Mean(1-confidence) on real smoothed clean data measured 0.0216-0.0320 across two independent test runs. Note: _DriftMonitor self-calibrates from real history via calibrate() after warmup_windows, so this static value mainly matters during the warmup period -- still updated to avoid a wildly mismatched default.
CUSUM_HISTORY    = 60

# Local CUSUM detector -- separate from _DriftMonitor above, which
# accumulates over an ENTIRE file's history and reacts too slowly to a
# short, localized attack. This one accumulates over a SHORT sliding
# window specifically to catch the "several nearby windows each show a
# partial confidence dip, none alone crosses CONFIDENCE_SUSPECT" pattern
# found in quick-splice testing (2026-07): single-window thresholding
# alone caught as few as 4/22 windows on a 44-second splice; this
# detector caught 22/22 on every tested run, 2-44 seconds, with 0.00%
# false positives on clean data. See combined_smoothing.py test suite.
LOCAL_CUSUM_WINDOW_SIZE = 10
LOCAL_CUSUM_THRESHOLD   = 2.0

# Raw-value drift check -- added 2026-07 to close a validated gap: a
# slow, smooth ramp attack (each step tiny, cumulative drift large)
# doesn't disrupt window-to-window CONFIDENCE early on, so every
# confidence-based check above (DISCONTINUITY, LOCAL ANOMALY, DRIFT
# DETECTED -- which despite its name tracks confidence drift, not
# value drift) stayed blind until the cumulative drift crossed the
# fixed NOMINAL_TOLERANCE_HZ absolute threshold. Tested against a real
# 90-window ground-truth ramp attack: existing checks alone caught
# 49/90 (54%); this check alone caught 71/90 (79%), fully overlapping
# and extending the existing coverage. Residual gap (first ~19 windows
# of the ramp) is expected detection latency -- any window-based trend
# check needs some minimum history to distinguish a real ramp from
# noise. Calibrated against real cleaned data: threshold=0.6 gives
# 0.33% FPR alone, 0.78% combined with everything else.
RAW_DRIFT_WINDOW_SIZE = 30
RAW_DRIFT_THRESHOLD = 0.6

# ENF baseline comparison -- added 2026-07. ENF ONLY, not NLR/GPU/CPU --
# deliberately scoped to the one signal where an untampered reference
# is a natural, already-available byproduct of this pipeline (Stage 1's
# smoothed ENF, held before attack injection ever touches it -- same
# "clean upstream of the attacker" principle already used for
# combined_smooth() itself). Validated against real data: a genuine
# splice (even the shortest tested, 1 sample/2 seconds) produced a
# minimum 0.032 Hz deviation from the true baseline value at that
# timestamp; completely untampered data showed exactly 0.0 deviation.
# attack.py rounds FRQ to 4 decimals before writing output, which
# introduces up to ~0.00005 Hz of rounding noise on genuinely clean
# windows -- ENF_BASELINE_THRESHOLD is set well above that and well
# below the smallest real attack deviation measured.
ENF_BASELINE_THRESHOLD = 0.01

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
GPU_TEMP_MAX_STEP_C = 8.0    # RECALIBRATED (2026-07) from the original 15.0 placeholder, which was never once crossed across 14,400 real step observations (max observed real step was 13.80C) -- meaning it provided no real detection power at all, only ever tested as "never fires." 8.0 sits comfortably above the real P99 (5.10C). Matches the same pattern already found for CPU_MAX_STEP_W: 91.4% of what this threshold catches (32/35 exceedances) falls in the first 150 windows -- the known, already-documented startup-ramp period, not a new phenomenon. Only 3 genuine steady-state exceedances remain across ~1650 windows. Calibrated against exactly one real file (run_2node.jsonl) -- revisit if a second real file shows meaningfully different behavior.

# Derived (not arbitrary): CPU_POWER_CEILING_W for one ENF window,
# converted to uJ, with a 1.5x safety margin.
_ENF_WINDOW_SECONDS = 2.0
CPU_UJ_MAX_STEP_UJ = CPU_POWER_CEILING_W * _ENF_WINDOW_SECONDS * 1_000_000 * 1.5

# ---------------------------------------------------------------------------
# PLACEHOLDER thresholds -- cross-node corroboration for step-size
# (continuity) failures. NOT yet validated against Ethan's coordinated
# multi-node attack scenarios -- see _apply_synchronized_event_correlation.
# ---------------------------------------------------------------------------

SYNC_EVENT_MIN_NODES    = 4     # CEILING on the absolute-node requirement --
                                 # scales DOWN for smaller deployments, see
                                 # _apply_synchronized_event_correlation
SYNC_EVENT_MIN_FRACTION = 0.5   # AND must be at least this fraction of nodes present

# Score values
SCORE_TRUSTED      = 0.95
SCORE_SUSPECT       = 0.50
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
    def baseline(self) -> float:
        return self._baseline

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


class _LocalCUSUMDetector:
    """
    Added 2026-07, alongside combined_smooth() in data_loaders.py.

    Accumulates (1 - confidence) over a SHORT sliding window (unlike
    _DriftMonitor above, which accumulates over the entire file's
    history and is tuned for slow, genuine long-term drift). This one
    is tuned for short, localized anomalies -- specifically the
    "several nearby windows each show a partial confidence dip, none
    alone crosses CONFIDENCE_SUSPECT" pattern found during quick-splice
    testing, where single-window thresholding degraded badly on longer
    (but still well under a minute) sustained anomalies.

    RECOVERY MECHANISM (added 2026-07, second pass): tested against a
    real ground-truth attack (10 genuine windows) and found the slow
    linear decay alone left this detector firing FAILED for roughly
    120 additional windows (~4 minutes) after DISCONTINUITY had already
    cleared and the underlying confidence had genuinely recovered --
    the cusum simply climbed too high during the attack to decay back
    under threshold quickly at the baseline-sized decay step. Fixed by
    tracking consecutive windows with confidence back above
    recovery_threshold; after recovery_windows in a row, force a full
    reset instead of waiting on the slow linear decay.

    recovery_windows=20 (not the originally-tried 5) was chosen after
    finding a real tradeoff: a SHORT recovery_windows can trigger a
    premature reset if a longer attack has a brief internal "quiet
    patch" where confidence genuinely recovers for a few windows before
    the attack continues (measured directly: a 22-sample splice attack
    has exactly this shape, confidence hitting 0.97+ for 5 straight
    windows midway through, well before the attack actually ends) --
    at recovery_windows=5 this cut splice recall from 22/22 to 8/22.
    recovery_windows=20 was swept and confirmed to fully restore that
    recall while still keeping the real-attack recovery tail far
    shorter than no fast-recovery at all (measured: ~28 windows vs the
    original ~120).
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
        return self._cusum

    def record(self, confidence: float) -> bool:
        """Records one confidence value, returns True if flagged."""
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
        """Allows the same runtime-calibrated baseline _DriftMonitor uses to be shared here."""
        self._baseline = baseline
        self._cusum = 0.0
        self._consecutive_good = 0

    def reset(self) -> None:
        self._cusum = 0.0
        self._consecutive_good = 0


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


class _RawDriftCheck:
    """
    Detects sustained directional drift in the RAW ENF value, completely
    independent of confidence -- see module comment above
    RAW_DRIFT_WINDOW_SIZE for the gap this closes and why it's needed.

    Splits a rolling window of raw values into two halves and compares
    their means. A real random walk around a stable nominal frequency
    should show a roughly-zero difference between an early and late
    half of any given window; a sustained directional ramp (the attack
    type this targets) produces a clear, growing difference instead.

    Deliberately simple (not a proper linear regression) -- tested
    directly against real clean data and a real ground-truth ramp
    attack, and a full regression fit wasn't needed to get a working,
    well-calibrated result.
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


class _ENFBaselineCheck:
    """
    ENF ONLY -- not used for NLR/GPU/CPU data at all.

    Directly compares the observed FRQ value against an untampered
    baseline ENF array at the exact same index/timestamp. This is
    fundamentally different from every other check in this file: those
    are all self-referential (comparing the signal to itself, with no
    access to ground truth), which is exactly why quick same-file
    splices were the one attack category nothing else could reliably
    catch all session -- a spliced value is genuinely real, just from
    the wrong moment, and looks statistically normal in isolation.
    A direct baseline comparison sidesteps that completely: it doesn't
    matter how "real" a value looks if it doesn't match what was
    actually there at this specific timestamp.

    The baseline must come from data captured/held BEFORE any attack
    injection -- same principle as combined_smooth() needing to run
    upstream of attack.py. Passing an attacked or partially-attacked
    array as the baseline defeats the entire point.

    threshold=0.01 (see ENF_BASELINE_THRESHOLD) is well above the
    ~0.00005 Hz rounding noise attack.py's own JSON output introduces
    on genuinely clean windows, and well below the smallest real
    attack deviation measured (0.032 Hz, for the shortest tested
    splice).
    """

    def __init__(
        self,
        baseline: list,
        threshold: float = ENF_BASELINE_THRESHOLD,
    ):
        self._baseline = baseline
        self._threshold = threshold

    def check(self, index: int, observed_freq: float) -> tuple[bool, str]:
        if index < 0 or index >= len(self._baseline):
            # Out of range of the baseline we were given -- can't compare,
            # don't penalize for something we have no reference for.
            return True, "ok"
        expected = self._baseline[index]
        diff = abs(observed_freq - expected)
        if diff > self._threshold:
            return False, (
                f"BASELINE MISMATCH: observed {observed_freq:.4f} Hz differs "
                f"from the untampered reference ({expected:.4f} Hz) by "
                f"{diff:.4f} Hz at this exact timestamp -- real ENF, "
                f"unmodified, would match here"
            )
        return True, "ok"


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
    energy counter in BOTH directions.

    DECREASE direction -- tracks a RUNNING MAXIMUM per channel, not
    just the immediately preceding value. Added 2026-07 after finding
    prev-only comparison could only catch the exact moment a tampered
    reading first dropped below the true trajectory, not the full
    duration it stayed dropped -- validated against a real ground-
    truth attack (same-node value replay, ~260-270 billion uJ drop for
    270 windows): prev-only comparison caught 3/270 (only the moments
    a replay cycle happened to restart); running-max comparison
    catches all 270/270. A cumulative energy counter can only
    legitimately go backward via the known hardware wraparound; any
    other drop below the highest legitimately-observed value is
    tampering, for as long as it stays below that high-water mark --
    not just on the first window it happens. Recovery is close to
    immediate once the real counter naturally exceeds the frozen
    running max again -- the real counter keeps accumulating in the
    background regardless of what's being reported, so the moment
    tampering stops, the true value is almost always already higher
    than wherever the running max got frozen. No explicit recovery
    mechanism needed here, unlike LocalCUSUMDetector's confidence-
    based checks.

    INCREASE direction -- measured against the immediately preceding
    ACTUAL reading (not the running max, which can be stale during an
    ongoing drop) so a genuine single-window step size is what's being
    judged, not a distance from a frozen historical reference.
    PLACEHOLDER threshold, not yet profiled against real data -- an
    implausibly large increase that stayed "monotonic" would otherwise
    pass unnoticed. Note the transition window right as a genuine
    attack ends can itself trip this (jumping from a tampered low
    reading back to the true, much higher trajectory looks like a
    large single-window step) -- that's a real, defensible catch in
    its own right, not a false positive to suppress; it just means
    full recovery to TRUSTED lands one window after the attack
    actually ends, not on the exact same window.
    """

    def __init__(self):
        self._prev: dict[str, float] = {}
        self._running_max: dict[str, float] = {}

    def check(self, record: dict) -> dict[str, tuple[str, str]]:
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
                    self._running_max[key] = val  # new post-wrap epoch
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
                    # running_max deliberately NOT updated here -- keeps
                    # every subsequent still-low reading caught too, not
                    # just this one.
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
        self._prev.clear()
        self._running_max.clear()


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


def _bare_channel_name(key: str) -> str:
    """
    Strips the node-id prefix from a full column name, returning just
    the physical channel -- e.g. "x3105c0s37b0n0_cpu-0[W]" -> "cpu-0[W]".
    Used to group the SAME physical channel across DIFFERENT nodes so
    cross-node corroboration can be checked.
    """
    for marker in ("_gpu-", "_cpu-"):
        idx = key.find(marker)
        if idx != -1:
            return key[idx + 1:]
    return key


def _is_pure_discontinuity(reason: str) -> bool:
    """
    True only if DISCONTINUITY is the SOLE problem on this channel.
    A channel that is ALSO out of range, rolling back, or energy-
    spiking is never eligible for corroboration downgrade below --
    those are absolute plausibility violations, independent of
    whatever every other node happens to be doing.
    """
    if "DISCONTINUITY" not in reason:
        return False
    disqualifying = ("OUT OF RANGE", "MONOTONICITY VIOLATION", "ENERGY SPIKE")
    return not any(d in reason for d in disqualifying)


def _apply_synchronized_event_correlation(
    merged: dict[str, tuple[str, str]],
    max_min_nodes: int = SYNC_EVENT_MIN_NODES,
    min_fraction: float = SYNC_EVENT_MIN_FRACTION,
) -> dict[str, tuple[str, str]]:
    """
    Cross-node corroboration for step-size (continuity) failures.

    Independent hardware does not coincidentally step together. If a
    large, otherwise-implausible step shows up on the SAME physical
    channel (e.g. cpu-0[W]) on many DIFFERENT nodes in the SAME
    window, that is itself strong evidence of a real synchronized
    system event -- a checkpoint save, a gradient-sync barrier, job
    startup -- rather than tampering. An attacker targeting one node
    cannot make many other independent nodes' real telemetry jump in
    lockstep too, so requiring corroboration doesn't weaken detection
    of an actual single-node attack -- it only softens the call on
    events that many nodes agree on simultaneously.

    Only PURE discontinuity failures are eligible (see
    _is_pure_discontinuity) -- corroboration never rescues a channel
    that is also out of range or violating monotonicity.

    The absolute-node requirement SCALES with how many nodes actually
    have this channel present this window, rather than being a fixed
    number:

        effective_min_nodes = max(2, min(max_min_nodes, total_nodes))

    At or above max_min_nodes total nodes (e.g. a 16-node rack), this
    is identical to the fixed floor before -- nothing changes for a
    full deployment. Below that, it scales down so a small deployment
    (e.g. 2 nodes) can still corroborate off full unanimous agreement,
    instead of a fixed floor that's mathematically unreachable with
    that few nodes. Never drops below 2 -- a single node's own reading
    cannot "corroborate" itself, so a 1-node deployment can never
    trigger this regardless of max_min_nodes.

    This is a real security tradeoff, not a free improvement: on a
    small deployment, an attacker who has compromised ALL of that
    deployment's nodes could coordinate tampering to mimic this exact
    signature and get downgraded to SUSPECT. The smaller the
    deployment, the smaller the number of nodes an attacker needs to
    compromise to fake corroboration. Still requires BOTH the scaled
    node count AND min_fraction (share of nodes present) to downgrade.

    Downgrades matching entries from FAILED to SUSPECT (not TRUSTED --
    this is still worth watching, just not a confident hard failure)
    and appends a note naming exactly how many nodes corroborated it.
    """
    by_channel: dict[str, list[str]] = collections.defaultdict(list)
    for key in merged:
        by_channel[_bare_channel_name(key)].append(key)

    result = dict(merged)

    for bare, keys in by_channel.items():
        total_nodes = len(keys)
        failed_keys = [
            k for k in keys
            if merged[k][0] == _FAILED and _is_pure_discontinuity(merged[k][1])
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
    enf_baseline : list[float], optional
        Untampered reference ENF values, ENF ONLY -- see
        _ENFBaselineCheck's docstring. Must come from data held before
        any attack injection touched it. If not provided, the baseline
        comparison simply doesn't run -- fully backward compatible with
        every existing caller that doesn't have a baseline available.
    """

    def __init__(
        self,
        component_id: str,
        warmup_windows: int = 10,
        strict_ordering: bool = True,
        check_nlr: bool = True,
        enf_baseline: list = None,
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
        self._local_cusum         = _LocalCUSUMDetector()
        self._raw_drift_check     = _RawDriftCheck()
        self._baseline_check      = _ENFBaselineCheck(enf_baseline) if enf_baseline is not None else None

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

            # Independent of confidence entirely -- see _RawDriftCheck
            # docstring for the gap this specifically closes (slow ramp
            # attacks that don't disrupt window-to-window correlation
            # early on).
            passed, reason = self._raw_drift_check.check(raw_freq)
            if not passed:
                enf_status = _worse(enf_status, VerificationStatus.FAILED.value)
                enf_reasons.append(reason)

            # Direct comparison against an untampered reference, ENF
            # only -- see _ENFBaselineCheck's docstring. Only runs if a
            # baseline was actually provided to this Verifier.
            if self._baseline_check is not None:
                index = None
                if isinstance(sample, dict):
                    index = sample.get("index")
                else:
                    index = getattr(sample, "index", None)
                if index is not None:
                    passed, reason = self._baseline_check.check(index, raw_freq)
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
            # NOTE: deliberately NOT sharing this calibration with
            # _local_cusum. Tested (2026-07): a 10-window sample produced
            # a baseline of 0.00273, ~8x tighter than the true whole-file
            # baseline (0.02163) -- fine for _drift_monitor (threshold=5.0
            # has enough buffer to absorb it) but caused _local_cusum
            # (threshold=2.0, much less forgiving) to false-positive on
            # 89.83% of completely clean data. _local_cusum keeps using
            # its well-calibrated static default (CUSUM_BASELINE, measured
            # directly from real whole-file data) instead.

        if self._drift_monitor.is_drifting():
            enf_status = _worse(enf_status, VerificationStatus.FAILED.value)
            enf_reasons.append(
                f"DRIFT DETECTED: CUSUM={self._drift_monitor.cusum:.3f} "
                f"exceeded threshold={CUSUM_THRESHOLD} "
                f"over {self._drift_monitor.sample_count} windows"
            )
            self._drift_monitor.reset()

        # Local CUSUM -- short-horizon, catches localized anomalies
        # (quick splices, short sustained fabrications) that individual
        # windows don't always cross CONFIDENCE_SUSPECT for on their own.
        # See _LocalCUSUMDetector docstring.
        #
        # IMPORTANT: deliberately does NOT reset() on every firing (unlike
        # _drift_monitor above). Testing found that resetting immediately
        # after each detection made this re-accumulate from zero every
        # time, causing it to miss most of a longer anomaly's duration
        # (dropped back to matching single-window-only performance, e.g.
        # 4/22 on a 44-second splice instead of the validated 22/22).
        # Letting it stay latched until enough good data naturally decays
        # it back down is the behavior that was actually tested and
        # validated in combined_smoothing.py's test suite.
        if self._local_cusum.record(anchor.confidence):
            enf_status = _worse(enf_status, VerificationStatus.FAILED.value)
            enf_reasons.append(
                f"LOCAL ANOMALY: CUSUM={self._local_cusum.cusum:.3f} "
                f"exceeded threshold={LOCAL_CUSUM_THRESHOLD} "
                f"within a {LOCAL_CUSUM_WINDOW_SIZE}-window span"
            )

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

            # Cross-node corroboration: downgrades a step-size failure
            # from FAILED to SUSPECT only when enough OTHER independent
            # nodes show the same pattern in this same window -- see
            # _apply_synchronized_event_correlation for why this is
            # safe (an attacker on one node can't fake agreement from
            # many other real nodes).
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