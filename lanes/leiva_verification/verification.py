"""
verification.py
----------------
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
    6. LocalCUSUMDetector     short-horizon CUSUM, sticky (doesn't self-reset)
    7. RawDriftCheck          sustained directional trend on the raw value --
                               catches gradual ramp attacks too slow for any
                               single-window check to see
    8. ENFAlternativeCorrelationCheck  windowed Pearson correlation against
                               an independently-noised second ENF stream --
                               simulates the two-sensor architecture the
                               reference papers (ANCHOR-Grid, SAVE) actually
                               use, generated in run_microverse.py's stage_1
                               and held back from attack.py exactly like
                               combined_smooth()'s own output is. Real,
                               validated limitation: mathematically blind to
                               a constant/slowly-varying offset (proved
                               directly -- correlation is invariant to an
                               additive shift), strong against replay-shaped
                               tampering (~99% recall on real data) instead.
                               See its own docstring and ENF_ALT_* constants
                               for full calibration history.
    All eight always run (#8 only if an alternative stream was provided).
    If more than one fires in the same window, the merged ENF result
    reports the worst status and every reason that fired, not just the
    first.

  NLR side (multi-node aware -- one result per physical channel):
    9. NLRRangeCheck            injection defense (impossible GPU/CPU wattage)
   10. NLRContinuityCheck       discontinuity defense (sudden power jumps)
   11. NLRMonotonicityCheck     tamper defense on CPU energy counters --
                                checks BOTH directions: an illegitimate
                                decrease (rollback, hiding real energy use)
                                and an implausibly large increase (energy
                                spike) that a decrease-only check would
                                never catch.
   12. GPUTempRangeCheck        GPU temperature physical plausibility.
   13. GPUTempContinuityCheck   GPU temperature step-size plausibility.
   14. CrossSiblingConsistencyCheck  compares a channel against the median
                                of its siblings on the SAME node -- GPUs
                                (and, more weakly, CPUs) doing the same
                                distributed job track each other tightly.
                                Closes a real gap: a targeted, single-
                                channel replay that leaves siblings
                                untouched is invisible to range/step-size
                                checks but breaks this correlation. No new
                                data used -- only channels already present
                                in the compressed record every consumer
                                (including the digital twin) receives.
   15. NLRSustainedDeviationCheck  GPU wattage/temperature only. Once a
                                step exceeds the continuity threshold, the
                                channel STAYS failed on every subsequent
                                window -- not just the entry transition --
                                until it genuinely returns within a
                                recovery band of its pre-jump value.
                                Closes the "boundary-only" gap continuity
                                checks share: a sustained replay holding a
                                fabricated value steady is otherwise
                                invisible after its first window. Inactive
                                during the documented startup-ramp window
                                (see NLR_STARTUP_RAMP_WINDOWS) -- a
                                legitimate ramp passes through several
                                genuinely different stable levels, so
                                "must return to the original reference" is
                                the wrong test there.
    Every channel present in the record gets its own result, every call --
    a bad gpu-0[W] reading does not stop cpu-1[uJ] from being checked, and
    if multiple checks flag the same channel in the same window, they
    merge into one result naming all of them.

   16. Cross-node corroboration (_apply_synchronized_event_correlation)
                               Runs after all NLR checks merge. A pure
                               step-size (discontinuity) failure is
                               downgraded from FAILED to SUSPECT when
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
                               only softens pure step-size calls. Also
                               requires the step stay within
                               SYNC_EVENT_MAX_STEP_MULTIPLE of the
                               threshold it crossed -- corroboration
                               forgives a borderline step, not an
                               extreme one (added after a real case
                               slipped through: a confirmed 500-600W GPU
                               wattage drop, hitting both nodes in a
                               2-node deployment at once, was getting
                               waved off purely on statistical agreement
                               despite the magnitude alone being
                               implausible as a real synchronized event).

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

Most thresholds in this file have been calibrated against real data --
see each constant's own comment for specifics (what was measured, what
file, what the resulting FPR/recall was). The ones that remain genuine,
undisguised placeholders: GPU_TEMP_FLOOR_C/GPU_TEMP_CEILING_C (physics-
based reasoning, not real-data profiling) and CPU_UJ_MAX_STEP_UJ (a
derived formula, not a profiled value). Everything else -- including
GPU_MAX_STEP_W and GPU_TEMP_MAX_STEP_C, both recalibrated this session
after being found to never fire once against real data -- has real
measurement behind it.
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

# ---------------------------------------------------------------------------
# Independent-sensor simulation via correlation -- added 2026-07, at the
# mentor's direction, to more faithfully match the reference papers'
# actual architecture (independently-measured ENF, compared via
# windowed Pearson correlation) rather than an exact-value diff against
# the same clean copy. The held-back reference itself is legitimate for
# the same reason the removed exact-value baseline check was: both come
# from Stage 1, before any simulated attack touches the data -- the
# difference is HOW the comparison is made, not whether holding a
# reference at all is fair.
#
# The "alternative" ENF stream is generated by adding small,
# independent Gaussian noise to the same clean smoothed signal --
# NOT a time-delayed copy. Distance between two real grid-connected
# sensors does not introduce a time lag the way sound or light would;
# grid frequency is a shared electrical property that updates
# essentially simultaneously across an entire synchronized
# interconnect (the paper's own figure confirms this directly: two
# real sensors 180 miles apart show correlation >0.95 at the SAME
# timestamps, not a lagged match). What actually differs between two
# real sensors is independent local measurement noise, not delay.
#
# RECALIBRATED (2026-07, second pass) from 0.001 to 0.0001. At the
# original 0.001, windowed (40-sample) clean-data correlation had real
# spread (median 0.999, P10 0.993) -- meaningful, but left recall on a
# real replay attack capped around 88.6% even after retuning window
# size and threshold. At 0.0001, clean-data correlation on real data
# tightens dramatically (median 0.99999, min 0.99982 across the whole
# file) -- tight enough to safely raise the threshold far higher (see
# ENF_ALT_CORRELATION_THRESHOLD), which is what actually recovered the
# rest of the gap. A tighter noise floor makes the alternative stream a
# more faithful proxy for the true signal, giving the threshold more
# room to move before natural noise risks a false positive.
ENF_ALT_NOISE_STD = 0.0001

ENF_ALT_WINDOW_SIZE = 40
# See ENF_ALT_CORRELATION_THRESHOLD below for the full recalibration
# history -- window size itself was swept once (5-90) during the first
# pass and 40 held up as the elbow of that curve; unchanged since.
ENF_ALT_CORRELATION_THRESHOLD = 0.999
# RECALIBRATED (2026-07, second pass) from 0.90 to 0.999, alongside
# tightening ENF_ALT_NOISE_STD (0.001 -> 0.0001) -- the two changes
# were tested together, not independently. Real replay attack recall:
# 60.2% (original 10-window/0.8) -> 88.6% (40-window/0.90, first pass)
# -> 99.2% (893/900, this setting). False positives at this setting
# (39/1800 on the attack file) remain FULLY explained, same as before:
# every one falls within the 40-window recovery tail right after the
# attack ends. The only 7 windows still missed sit at the very start of
# the attack (indices 180-186) -- inherent detection latency, since the
# rolling window needs 40 samples of history before correlation means
# anything; not a threshold problem. Confirmed 0 false positives on a
# fully independent clean file (different Dev/Hr, different noise
# seed) at this exact setting.
ENF_ALT_MIN_VARIANCE = 0.000005
# Pearson correlation is numerically unstable when a window is nearly
# flat -- confirmed directly: every real false positive found during
# calibration (6/179 windows on clean data) occurred in a window with
# local variance at or below ~2e-6, where independent noise dominates
# an already-tiny signal and correlation swings wildly (observed as
# low as 0.50) despite the two streams being essentially identical in
# absolute terms. Windows below this variance are skipped entirely,
# same "don't divide by near-zero" principle used elsewhere in this
# file (e.g. the sibling consistency check's sibling_ref <= 1 guard).
#
# HONEST STARTING POINT, not a final answer: threshold and window size
# taken directly from the papers and this initial calibration --
# expected to need adjustment once tested against real attack
# scenarios, per direct instruction not to over-invest before that
# evaluation happens.

# ---------------------------------------------------------------------------
# Tunable thresholds -- NLR side
# TODO: run nlr_profile.py across all five workload modes and update.
# These placeholders come from the P99 step sizes observed in the two
# real files profiled so far.
# ---------------------------------------------------------------------------

GPU_POWER_CEILING_W = 800.0
CPU_POWER_CEILING_W = 800.0
GPU_MAX_STEP_W      = 400.0     # RECALIBRATED (2026-07) from 470.0, which was
                                 # never crossed once across 14,392 real step
                                 # observations (run_2node.jsonl) -- same "never
                                 # fires, zero real detection power" pattern
                                 # already found and fixed for GPU_TEMP_MAX_STEP_C.
                                 # Real max observed: 363.2W (startup-ramp zone,
                                 # idx 141-142, same known volatile period
                                 # documented elsewhere). Steady-state-only
                                 # (idx>150) max: 272.1W, P99.9: 166.2W. 400.0
                                 # sits with real margin (~10%) above the highest
                                 # value ever observed in real data, including
                                 # the volatile ramp period, while being
                                 # meaningfully tighter than the old placeholder.
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

# ---------------------------------------------------------------------------
# Sustained deviation ("sticky" discontinuity) -- added 2026-07. Closes a
# real, confirmed gap: _NLRContinuityCheck only ever compares a value to
# its IMMEDIATELY PRECEDING window, so it correctly catches the single
# window a value jumps, then goes silent the instant the tampered value
# stabilizes -- exactly the "boundary-only, blind to the sustained
# interior" pattern already found and fixed for the uJ energy counter
# (see _NLRMonotonicityCheck), now generalized to a NON-cumulative value
# (wattage, temperature) where a running-max doesn't apply, since these
# are legitimately allowed to go up AND down.
#
# Once a step exceeds the existing GPU_MAX_STEP_W/GPU_TEMP_MAX_STEP_C
# threshold, the channel stays FAILED -- not just for that one window --
# until it genuinely returns within a tolerance band of the value it had
# right before the violation. A channel that never returns (like a
# sustained replay attack holding a fabricated low value for the entire
# attack duration) stays FAILED for the whole time, not just the entry
# transition.
#
# Recovery bands calibrated against real clean data (run_2node.jsonl):
# GPU temp typically wanders ~4.1C (P50) to ~5.2C (P95) around a stable
# point in a rolling 10-window span, max observed 8.0C -- 6.0C sits with
# real margin above P95 while remaining meaningfully tighter than the
# observed max. GPU wattage band (150.0W) taken directly as specified --
# not independently recalibrated against clean data the way temp was;
# worth revisiting if it turns out too tight/loose in practice.
GPU_POWER_RECOVERY_BAND_W = 150.0
GPU_TEMP_RECOVERY_BAND_C  = 6.0

NLR_STARTUP_RAMP_WINDOWS = 150
# _NLRSustainedDeviationCheck does not activate until after this many
# windows. Found necessary, not just cautious: during the documented
# startup ramp, wattage legitimately passes through several genuinely
# different stable levels in quick succession -- "return to the
# original pre-violation reference" is the wrong criterion there, since
# the whole point of a ramp is settling somewhere different. Without
# this exclusion, the check locks onto a reference from early in the
# ramp and never recovers for the rest of the file (confirmed directly:
# a single trigger at idx 133 cascaded into 3000+ false positives
# running all the way to the end of a real file). Matches the same
# ~150-window boundary already established and used throughout this
# project for CPU/GPU startup-ramp behavior. _NLRContinuityCheck still
# catches genuine discontinuities during the ramp exactly as before --
# only this check's sticky/sustained behavior is gated.

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

SYNC_EVENT_MAX_STEP_MULTIPLE = 1.2
# ADDED 2026-07, after a real gap was found: a coordinated attack that
# hits every node in a small (2-node) deployment simultaneously was
# getting corroboration-downgraded from FAILED to SUSPECT even for a
# 500-600W GPU wattage drop -- roughly 1.26x-1.43x the (recalibrated)
# GPU_MAX_STEP_W threshold. A 2-node deployment is the worst case for
# this tradeoff: "2 of 2 nodes agreeing" is trivially satisfied the
# moment both are attacked at once, with no safety margin left the way
# a larger deployment would have.
#
# The fix: corroboration should only be able to explain away a
# BORDERLINE step -- one just past the threshold, plausible as a
# slightly-larger-than-typical legitimate event (checkpoint save,
# sync barrier). A step 1.5-2x past the threshold is a fundamentally
# less plausible claim, and an attacker who controls every node in a
# small deployment can trivially fake "borderline agreement" -- but
# faking a genuinely large, simultaneous jump on real, independent
# hardware is a much bigger ask. 1.2x was chosen specifically to
# exclude the real attack case found (1.26x-1.43x) while still leaving
# room for genuinely borderline legitimate events. HONEST CAVEAT: this
# multiple has not been validated against a real, known legitimate
# large-magnitude synchronized event (checkpoint saves in the data
# profiled so far have all been comfortably under the base threshold,
# not testing this specific boundary) -- a judgment call in the same
# "placeholder, revisit with more data" spirit as the rest of this
# corroboration mechanism's documented tradeoffs.

# ---------------------------------------------------------------------------
# Cross-sibling consistency -- added 2026-07. Fully self-referential and
# fair: uses ONLY channels already present in the compressed data every
# consumer (including the digital twin) already receives -- no new
# columns, no held-back raw data, no hidden reference. GPUs (and, more
# weakly, CPUs) on the SAME node doing the SAME distributed job show
# strong, physically real correlation with each other -- a targeted,
# single-channel replay that leaves siblings untouched breaks that
# correlation, even when the replayed value is itself completely
# plausible in isolation and the attacker has exactly the same access
# to this relationship as the verifier does.
#
# Validated against a real file (run_2node.jsonl) using attack.py's
# actual replay mechanism (not a hand-picked, easier case):
#   GPU wattage:      100% recall (900/900), every GPU, both nodes,
#                      0-1.78% false positives
#   GPU temperature:  100% recall (900/900), every GPU, both nodes,
#                      0% false positives -- tighter than wattage,
#                      makes sense given siblings share the same
#                      physical chassis/cooling environment
#   CPU wattage:       weak (3-33% recall) -- only 2 siblings per node
#                      instead of 4, and CPU workload (data loading,
#                      orchestration) is less tightly synchronized
#                      than GPU compute. Kept anyway: 0% false
#                      positives, genuinely zero-cost when it doesn't
#                      fire, real signal on the cases it does catch.
GPU_POWER_SIBLING_RATIO_MIN = 0.85
GPU_POWER_SIBLING_RATIO_MAX = 1.10
GPU_TEMP_SIBLING_RATIO_MIN  = 0.90
GPU_TEMP_SIBLING_RATIO_MAX  = 1.15
CPU_POWER_SIBLING_RATIO_MIN = 0.75
CPU_POWER_SIBLING_RATIO_MAX = 1.40

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


class _ENFAlternativeCorrelationCheck:
    """
    ENF ONLY. Simulates the reference papers' actual architecture more
    faithfully than a direct value diff: an independently-measured
    second ENF stream, compared via windowed Pearson correlation. See
    ENF_ALT_* constants for full calibration details and reasoning.

    Maintains a rolling window of the last ENF_ALT_WINDOW_SIZE observed
    (possibly tampered) values, and the aligned window from the
    alternative stream, recomputing correlation each time the window is
    full. Skips windows where either signal is nearly flat -- see
    ENF_ALT_MIN_VARIANCE.

    VALIDATED PERFORMANCE (window=40, threshold=0.999, noise_std=0.0001
    -- see the constants' own comments for the two-pass recalibration
    history), against real data:
      Replay attacks:  99.2% (893/900) on a real replay attack -- the
                        7 remaining misses sit at the very start of the
                        attack (inherent detection latency, needs 40
                        windows of history before correlation means
                        anything, not a threshold gap).
      Bias injection:  ~1% (9/900) -- CANNOT meaningfully catch this,
                        regardless of noise/window/threshold tuning.
                        Pearson correlation is mathematically invariant
                        to a constant additive shift (proved directly:
                        corr(x, x+c) = 1.0 for any constant c), so a
                        uniform offset preserves the signal's shape
                        perfectly while moving its level -- exactly
                        what this mechanism structurally cannot see.
      Clean data:      0 false positives on an independent file. The
                        false positives that DO appear (39/1800 on the
                        attack file) all fall within the 40-window
                        recovery tail right after an attack ends, where
                        the rolling window is still partially aging out
                        tampered history -- expected and explained, not
                        a real concern.

    Given the above, this check is now the ONLY ENF cross-verification
    mechanism in this file (the exact-value baseline check it replaced
    was removed 2026-07, once this check's recall was tuned high enough
    to be trusted as the sole ENF cross-check) -- but the bias-injection
    blind spot is real and permanent. Worth keeping in mind for whatever
    this project's writeup says about ENF injection-attack coverage.
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
            return True, "ok"  # too flat for correlation to be meaningful

        mean_o = statistics.mean(observed_window)
        mean_a = statistics.mean(alt_window)
        cov = sum(
            (observed_window[i] - mean_o) * (alt_window[i] - mean_a)
            for i in range(self._window_size)
        )
        denom = (var_observed * var_alt) ** 0.5 * (self._window_size - 1)
        # statistics.variance divides by (n-1), matching cov's own
        # implicit (n-1) scaling above so they cancel consistently
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
    """
    Extracts the node ID prefix from a full column name, e.g.
    'x3102c0s25b0n0_gpu-0[W]' -> 'x3102c0s25b0n0'. Same convention used
    throughout this project (run_microverse.py's node grouping,
    attack.py's own scan_telemetry_schema()).
    """
    if "_gpu-" in key:
        return key.split("_gpu-")[0]
    if "_cpu-" in key:
        return key.split("_cpu-")[0]
    return None


class _CrossSiblingConsistencyCheck:
    """
    Fully self-referential and fair: uses ONLY channels already present
    in the compressed data every consumer (including the digital twin)
    already receives -- no new columns, no held-back raw data, no
    hidden reference the attacker doesn't also have access to.

    GPUs (and, more weakly, CPUs) on the SAME node doing the SAME
    distributed job show strong, physically real correlation with each
    other -- a targeted, single-channel replay that leaves siblings
    untouched breaks that correlation, even when the replayed value is
    itself completely plausible in isolation. Closes the specific gap
    found in a well-chosen, phase-matched single-GPU replay that
    defeated every range/step-size/energy-consistency check already in
    this file.

    Compares each channel against the mean of its siblings on the SAME
    node -- never across nodes, never against history, purely within
    the single record being checked.

    Validated against a real file (run_2node.jsonl) using attack.py's
    actual replay mechanism (not a hand-picked, easier case):
      GPU wattage:     100% recall (900/900), every GPU, both nodes,
                        0-1.78% false positives
      GPU temperature: 100% recall (900/900), every GPU, both nodes,
                        0% false positives -- tighter than wattage,
                        makes sense given siblings share the same
                        physical chassis/cooling environment
      CPU wattage:      weak (3-33% recall) -- only 2 siblings per
                        node instead of 4, and CPU workload is less
                        tightly synchronized than GPU compute. Kept
                        anyway: 0% false positives, genuinely
                        zero-cost when it doesn't fire.

    "-core[W]" channels deliberately excluded -- tested directly and
    found wildly unstable sibling ratios (P99.5 up to ~293x) even on
    genuinely clean data, likely because their values are tiny
    (0.03-0.3W observed) and dominated by measurement noise at that
    scale. Same exclusion already applied to the power/energy check
    for the same reason.
    """

    def check(self, record: dict) -> dict[str, tuple[str, str]]:
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
                    continue  # no siblings on this node to compare against
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
                    # median, not mean -- a single tampered sibling
                    # shouldn't be able to drag the reference point used
                    # to judge its OWN innocent neighbors. With 3+ GPU
                    # siblings this is robust to one bad value; with
                    # only 1 CPU sibling it's unchanged either way.
                    sibling_ref = statistics.median(sibling_vals)
                    if sibling_ref <= 1:
                        continue  # avoid divide-by-near-zero at idle
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
        pass  # fully stateless -- every record checked independently


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


class _NLRSustainedDeviationCheck:
    """
    GPU wattage and temperature only. Closes the "boundary-only" gap
    left by _NLRContinuityCheck: that check compares each value only
    to its immediately preceding window, so it correctly flags the
    single window a value jumps, then goes silent the instant the
    tampered value stabilizes -- a sustained replay that holds a
    fabricated value for hundreds of windows is invisible to it after
    the first one.

    Once a step exceeds GPU_MAX_STEP_W/GPU_TEMP_MAX_STEP_C, the channel
    is marked FAILED and STAYS failed on every subsequent window --
    not just the entry transition -- until it genuinely returns within
    GPU_POWER_RECOVERY_BAND_W/GPU_TEMP_RECOVERY_BAND_C of the value it
    had right before the violation. See the constants above for
    calibration details.

    Does not apply to CPU wattage or CPU energy -- CPU wattage's much
    higher natural volatility (see _CPUPowerEnergyConsistencyCheck's
    calibration notes) hasn't been separately profiled for a sensible
    recovery band, and CPU energy already has its own dedicated,
    better-suited fix (_NLRMonotonicityCheck's running-max, since
    energy is cumulative and wattage/temperature are not).
    """

    def __init__(self):
        self._prev: dict[str, float] = {}
        self._reference: dict[str, float] = {}  # value right before the violation
        self._stuck: dict[str, bool] = {}
        self._window_count = 0

    def check(self, record: dict) -> dict[str, tuple[str, str]]:
        self._window_count += 1
        channels = _find_nlr_keys(record)
        results: dict[str, tuple[str, str]] = {}

        if self._window_count <= NLR_STARTUP_RAMP_WINDOWS:
            # Still track _prev during the ramp so the check has a
            # sensible baseline the instant it activates -- just don't
            # act on anything yet.
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
                    if abs(val - ref) <= recovery_band:
                        self._stuck[key] = False
                        results[key] = (_TRUSTED, "ok")
                    else:
                        results[key] = (_FAILED, (
                            f"SUSTAINED DEVIATION: {key}={val:.2f}{unit} has "
                            f"not recovered -- still {abs(val - ref):.2f}{unit} "
                            f"away from the {ref:.2f}{unit} it held before the "
                            f"original jump (recovery band: "
                            f"+/-{recovery_band}{unit})"
                        ))
                else:
                    step = abs(val - prev)
                    if step > step_threshold:
                        self._stuck[key] = True
                        self._reference[key] = prev
                        results[key] = (_FAILED, (
                            f"SUSTAINED DEVIATION: {key} jumped "
                            f"{step:.2f}{unit} to {val:.2f}{unit} -- will stay "
                            f"FAILED until it returns within "
                            f"+/-{recovery_band}{unit} of {prev:.2f}{unit}"
                        ))

                self._prev[key] = val

        return results

    def reset(self) -> None:
        self._prev.clear()
        self._reference.clear()
        self._stuck.clear()
        self._window_count = 0


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


_DISCONTINUITY_STEP_RE = re.compile(
    r"stepped ([\d.]+)\S* between windows, exceeds max plausible step ([\d.]+)"
)


def _is_within_corroboration_ceiling(
    reason: str, max_multiple: float = SYNC_EVENT_MAX_STEP_MULTIPLE
) -> bool:
    """
    Parses the actual step size and threshold directly out of the
    DISCONTINUITY reason string (see _DISCONTINUITY_STEP_RE) and
    checks the step doesn't exceed max_multiple x the threshold that
    was crossed. See SYNC_EVENT_MAX_STEP_MULTIPLE for the full
    reasoning -- corroboration should only forgive a borderline step,
    not an extreme one, regardless of how many nodes agree on it.

    Returns True (eligible) if the reason can't be parsed at all --
    fails open toward the EXISTING behavior rather than silently
    blocking corroboration on a format this wasn't written to expect.
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
    that is also out of range or violating monotonicity. As of 2026-07,
    also requires the step stay within SYNC_EVENT_MAX_STEP_MULTIPLE x
    the threshold that was crossed (see _is_within_corroboration_ceiling)
    -- corroboration forgives a borderline step, not an extreme one.

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
    enf_alternative : list[float], optional
        Independently-noised second ENF stream, ENF ONLY -- see
        _ENFAlternativeCorrelationCheck's docstring. Must come from data
        held before any attack injection touched it -- same principle
        as combined_smooth() needing to run upstream of attack.py. If
        not provided, the correlation comparison simply doesn't run.
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

        # ENF checks
        self._sequence_guard      = _SequenceGuard(strict_ordering)
        self._nominal_range_check = _ENFNominalRangeCheck()
        self._range_check         = _ENFRangeCheck()
        self._continuity_check    = _ENFContinuityCheck()
        self._drift_monitor       = _DriftMonitor()
        self._local_cusum         = _LocalCUSUMDetector()
        self._raw_drift_check     = _RawDriftCheck()
        self._alt_correlation_check = _ENFAlternativeCorrelationCheck(enf_alternative) if enf_alternative is not None else None

        # NLR checks -- multi-node aware, no configuration needed
        self._nlr_range_check        = _NLRRangeCheck()
        self._nlr_continuity_check   = _NLRContinuityCheck()
        self._nlr_monotonicity_check = _NLRMonotonicityCheck()
        self._cross_sibling_check = _CrossSiblingConsistencyCheck()
        self._sustained_deviation_check = _NLRSustainedDeviationCheck()
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

            # Windowed correlation against an independently-noised
            # second stream -- see _ENFAlternativeCorrelationCheck's
            # docstring. Only runs if an alternative stream was
            # actually provided to this Verifier.
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
                self._cross_sibling_check.check(record_dict),
                self._sustained_deviation_check.check(record_dict),
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