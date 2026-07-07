"""
test_verifier.py
-----------------
Automated regression suite for the rewritten verification.py. Covers
every check class individually (unit-level, contrived inputs) plus
several full end-to-end scenarios through the real AnchorExtractor +
Verifier pipeline (integration-level, synthetic JSONL-shaped data).

This is NOT a replacement for test_attack_detection.py (Ethan's
attack-injection tests) -- it's a regression suite specifically for
confirming the never-stop / full-attribution rewrite works correctly:
  - every original check still catches what it always caught
  - the two brand-new checks (GPU temp, CPU energy upward-spike)
    actually catch what they're supposed to
  - a legitimate RAPL wraparound still passes as TRUSTED (no false
    positive from the new bidirectional monotonicity check)
  - multiple simultaneous problems in the same window/node all get
    reported, merged, with nothing dropped
  - one node's problem never affects another node's result
  - node count (1 vs many) doesn't change any of the above

Run from the repo root:
    python lanes/leiva_verification/test_verifier.py
"""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from anchor import AnchorExtractor
from verification import (
    Verifier,
    _SequenceGuard,
    _ENFNominalRangeCheck,
    _ENFRangeCheck,
    _ENFContinuityCheck,
    _DriftMonitor,
    _NLRRangeCheck,
    _NLRContinuityCheck,
    _NLRMonotonicityCheck,
    _GPUTempRangeCheck,
    _GPUTempContinuityCheck,
    CONFIDENCE_SUSPECT,
    CONFIDENCE_TRUSTED,
    GPU_POWER_CEILING_W,
    CPU_UJ_WRAP_CEILING,
    CUSUM_THRESHOLD,
)

_results = []


def _record(name, passed, detail=""):
    _results.append((name, passed, detail))


def check(name, condition, detail=""):
    try:
        assert condition
        _record(name, True)
    except AssertionError:
        _record(name, False, detail)


# ---------------------------------------------------------------------------
# Helpers to build small synthetic combined records
# ---------------------------------------------------------------------------

def smooth_enf(i, amplitude=0.05, period=20.0):
    """Slow, smooth, deterministic ENF -- keeps confidence high (~1.0)
    for every window not deliberately disturbed, isolating whatever
    this test is actually trying to check."""
    return 60.0 + amplitude * math.sin(i / period)


def base_node_record(node_id="node_A"):
    return {
        f"{node_id}_gpu-0[W]": 70.0, f"{node_id}_gpu-1[W]": 71.0,
        f"{node_id}_gpu-2[W]": 69.0, f"{node_id}_gpu-3[W]": 72.0,
        f"{node_id}_gpu-0[C]": 55.0, f"{node_id}_gpu-1[C]": 56.0,
        f"{node_id}_gpu-2[C]": 54.0, f"{node_id}_gpu-3[C]": 57.0,
        f"{node_id}_cpu-0[W]": 90.0, f"{node_id}_cpu-1[W]": 91.0,
        f"{node_id}_cpu-0[uJ]": 0.0, f"{node_id}_cpu-1[uJ]": 0.0,
    }


def make_records(n, node_ids=("node_A",), enf_fn=smooth_enf, uj_step=1_000_000.0):
    records = []
    for i in range(n):
        r = {"index": i, "FRQ": round(enf_fn(i), 6)}
        for node_id in node_ids:
            base = base_node_record(node_id)
            for k, v in base.items():
                if k.endswith("[uJ]"):
                    v = i * uj_step
                r[k] = v
        records.append(r)
    return records


def run_verifier(records, component_id="rack_00", warmup_windows=10):
    """Runs the full real pipeline (AnchorExtractor + Verifier) over a
    list of records, returns {index: [VerificationResult, ...]}."""
    enf_list = [r["FRQ"] for r in records]
    extractor = AnchorExtractor(enf=enf_list, sample_rate_hz=0.5)
    verifier = Verifier(component_id=component_id, warmup_windows=warmup_windows)

    all_results = {}
    for record in records:
        ts = float(record["index"])
        record = dict(record)
        record["timestamp"] = ts
        anchor = extractor.extract(ts)
        all_results[record["index"]] = verifier.verify(record, anchor)
    return all_results


def find(results_at_index, component_suffix):
    """Finds the result whose component_id ends with the given suffix."""
    for r in results_at_index:
        if r.component_id.endswith(component_suffix):
            return r
    return None


# ---------------------------------------------------------------------------
# 1. Unit-level tests -- individual check classes, contrived inputs
# ---------------------------------------------------------------------------

def test_sequence_guard_replay():
    guard = _SequenceGuard()
    guard.check(1.0)
    guard.check(2.0)
    passed, reason = guard.check(1.0)  # replay
    check("SequenceGuard: replay detected", not passed and "REPLAY" in reason, reason)


def test_sequence_guard_out_of_order():
    guard = _SequenceGuard()
    guard.check(5.0)
    passed, reason = guard.check(3.0)  # went backwards
    check("SequenceGuard: out-of-order detected", not passed and "OUT OF ORDER" in reason, reason)


def test_enf_nominal_range():
    c = _ENFNominalRangeCheck()
    passed, reason = c.check(65.0)  # 5 Hz off nominal, tolerance is 2.0
    check("ENFNominalRangeCheck: catches raw spike", not passed and "NOMINAL" in reason, reason)
    passed, _ = c.check(60.01)
    check("ENFNominalRangeCheck: passes normal reading", passed)


def test_enf_range_flat_signature():
    c = _ENFRangeCheck()
    passed, reason = c.check([0.5] * 11)  # perfectly flat -- fabricated look
    check("ENFRangeCheck: catches flat/fabricated signature", not passed and "FLAT" in reason, reason)
    passed, _ = c.check([0.1, 0.3, 0.5, 0.7, 0.9, 0.6, 0.4, 0.2, 0.5, 0.8, 0.3])
    check("ENFRangeCheck: passes varied signature", passed)


def test_enf_continuity_hard_threshold():
    c = _ENFContinuityCheck()
    passed, reason = c.check(CONFIDENCE_SUSPECT - 0.01)
    check("ENFContinuityCheck: fails below hard threshold", not passed and "DISCONTINUITY" in reason, reason)
    passed, _ = c.check(CONFIDENCE_TRUSTED + 0.01)
    check("ENFContinuityCheck: passes above threshold", passed)


def test_drift_monitor_accumulates_and_resets():
    d = _DriftMonitor()
    # Feed a sustained deviation well above baseline until CUSUM crosses threshold
    for _ in range(200):
        d.record(1.0 - (CUSUM_THRESHOLD))  # deliberately bad confidence every window
        if d.is_drifting():
            break
    check("DriftMonitor: accumulates and eventually flags drift", d.is_drifting(),
          f"cusum={d.cusum:.3f} after {d.sample_count} windows")
    d.reset()
    check("DriftMonitor: reset() clears cusum", d.cusum == 0.0)


def test_nlr_range_check_gpu_overload():
    c = _NLRRangeCheck()
    record = base_node_record()
    record["node_A_gpu-1[W]"] = 950.0  # over ceiling
    results = c.check(record)
    status, reason = results["node_A_gpu-1[W]"]
    check("NLRRangeCheck: catches GPU overload", status == "failed" and "OUT OF RANGE" in reason, reason)
    # every OTHER key in the same call must still be present and trusted
    check("NLRRangeCheck: does not stop after first bad key",
          results["node_A_gpu-0[W]"][0] == "trusted" and results["node_A_cpu-0[W]"][0] == "trusted")


def test_nlr_continuity_check_power_spike():
    c = _NLRContinuityCheck()
    r1 = base_node_record()
    c.check(r1)  # establish baseline
    r2 = dict(r1)
    r2["node_A_gpu-2[W]"] = r1["node_A_gpu-2[W]"] + 900.0  # huge jump, still under ceiling
    results = c.check(r2)
    status, reason = results["node_A_gpu-2[W]"]
    check("NLRContinuityCheck: catches power spike", status == "failed" and "DISCONTINUITY" in reason, reason)


def test_nlr_monotonicity_legitimate_wraparound_passes():
    c = _NLRMonotonicityCheck()
    r1 = {"node_A_cpu-0[uJ]": 65_000_000_000.0}
    c.check(r1)
    r2 = {"node_A_cpu-0[uJ]": 300_000_000.0}  # drop of ~64.7B, within wrap tolerance of 65.5B
    results = c.check(r2)
    status, reason = results["node_A_cpu-0[uJ]"]
    check("NLRMonotonicityCheck: legitimate hardware wraparound passes as trusted",
          status == "trusted", reason)


def test_nlr_monotonicity_illegitimate_rollback_fails():
    c = _NLRMonotonicityCheck()
    r1 = {"node_A_cpu-0[uJ]": 40_000_000_000.0}
    c.check(r1)
    r2 = {"node_A_cpu-0[uJ]": 10_000_000_000.0}  # drop of 30B -- not close to the 65.5B wrap ceiling
    results = c.check(r2)
    status, reason = results["node_A_cpu-0[uJ]"]
    check("NLRMonotonicityCheck: illegitimate rollback fails", status == "failed" and "MONOTONICITY" in reason, reason)


def test_nlr_monotonicity_energy_spike_new_check():
    c = _NLRMonotonicityCheck()
    r1 = {"node_A_cpu-1[uJ]": 1_000_000_000.0}
    c.check(r1)
    r2 = {"node_A_cpu-1[uJ]": 1_000_000_000.0 + 50_000_000_000.0}  # huge but still-increasing jump
    results = c.check(r2)
    status, reason = results["node_A_cpu-1[uJ]"]
    check("NLRMonotonicityCheck: NEW -- catches implausible upward energy spike",
          status == "failed" and "ENERGY SPIKE" in reason, reason)


def test_gpu_temp_range_check_new():
    c = _GPUTempRangeCheck()
    record = base_node_record()
    record["node_A_gpu-3[C]"] = 130.0  # absurd, over ceiling
    results = c.check(record)
    status, reason = results["node_A_gpu-3[C]"]
    check("GPUTempRangeCheck: NEW -- catches out-of-range GPU temperature",
          status == "failed" and "OUT OF RANGE" in reason, reason)


def test_gpu_temp_continuity_check_new():
    c = _GPUTempContinuityCheck()
    r1 = base_node_record()
    c.check(r1)
    r2 = dict(r1)
    r2["node_A_gpu-0[C]"] = r1["node_A_gpu-0[C]"] + 40.0  # instant 40C jump
    results = c.check(r2)
    status, reason = results["node_A_gpu-0[C]"]
    check("GPUTempContinuityCheck: NEW -- catches implausible temperature jump",
          status == "failed" and "DISCONTINUITY" in reason, reason)


# ---------------------------------------------------------------------------
# 2. Integration-level tests -- full AnchorExtractor + Verifier pipeline
# ---------------------------------------------------------------------------

def test_clean_data_everything_trusted():
    records = make_records(30, node_ids=("node_A", "node_B"))
    results = run_verifier(records)
    bad = [r for rs in results.values() for r in rs if r.status != "trusted"]
    check("Integration: fully clean multi-node data is 100% TRUSTED", len(bad) == 0,
          f"{len(bad)} non-trusted results found")


def test_multi_node_isolation_end_to_end():
    records = make_records(20, node_ids=("node_A", "node_B", "node_C"))
    records[10]["node_B_gpu-0[W]"] = 950.0  # attack ONLY on node_B
    results = run_verifier(records)

    r10 = results[10]
    node_b_result = find(r10, "node_B_gpu-0[W]")
    node_a_result = find(r10, "node_A_gpu-0[W]")
    node_c_result = find(r10, "node_C_gpu-0[W]")

    check("Integration: attacked node_B channel is FAILED", node_b_result.status == "failed")
    check("Integration: untouched node_A channel stays TRUSTED", node_a_result.status == "trusted")
    check("Integration: untouched node_C channel stays TRUSTED", node_c_result.status == "trusted")

    # every OTHER node_B channel in the same window must be unaffected too
    other_node_b_ok = all(
        r.status == "trusted" for r in r10
        if r.component_id.startswith("rack_00/node_B_") and not r.component_id.endswith("gpu-0[W]")
    )
    check("Integration: node_B's OTHER channels unaffected by its own gpu-0[W] attack", other_node_b_ok)


def test_simultaneous_multi_component_attack_no_masking():
    records = make_records(20, node_ids=("node_A", "node_B"))
    records[10]["node_A_gpu-1[W]"] = 950.0
    records[10]["node_A_gpu-2[C]"] = 130.0
    records[10]["node_A_cpu-1[uJ]"] = records[9]["node_A_cpu-1[uJ]"] + 50_000_000_000.0
    results = run_verifier(records)

    r10 = results[10]
    gpu_w   = find(r10, "node_A_gpu-1[W]")
    gpu_c   = find(r10, "node_A_gpu-2[C]")
    cpu_uj  = find(r10, "node_A_cpu-1[uJ]")

    check("Integration: simultaneous attack #1 (GPU wattage) caught", gpu_w.status == "failed")
    check("Integration: simultaneous attack #2 (GPU temp) caught", gpu_c.status == "failed")
    check("Integration: simultaneous attack #3 (CPU energy) caught", cpu_uj.status == "failed")

    # node_B must be completely untouched despite 3 simultaneous node_A attacks
    node_b_clean = all(r.status == "trusted" for r in r10 if "node_B" in r.component_id)
    check("Integration: node_B fully clean despite 3 simultaneous node_A attacks", node_b_clean)

    total_results = len(r10)
    check("Integration: result count unaffected by failures (still 1 ENF + 24 NLR channels)",
          total_results == 25, f"got {total_results}")


def test_merge_reports_both_reasons_on_same_channel():
    records = make_records(20, node_ids=("node_A",))
    # gpu-1[W] gets hit by BOTH range check AND continuity check in the same window
    records[10]["node_A_gpu-1[W]"] = 950.0
    results = run_verifier(records)
    r10 = results[10]
    gpu_w = find(r10, "node_A_gpu-1[W]")
    check("Integration: merged result contains BOTH range and continuity reasons",
          gpu_w.status == "failed" and "OUT OF RANGE" in gpu_w.reason and "DISCONTINUITY" in gpu_w.reason,
          gpu_w.reason)


def test_single_node_config():
    records = make_records(15, node_ids=("solo_node",))
    results = run_verifier(records)
    bad = [r for rs in results.values() for r in rs if r.status != "trusted"]
    check("Integration: 1-node config works with zero code changes", len(bad) == 0,
          f"{len(bad)} non-trusted results found")


def test_result_count_scales_with_node_count():
    records_1 = make_records(5, node_ids=("only_node",))
    records_3 = make_records(5, node_ids=("n1", "n2", "n3"))
    results_1 = run_verifier(records_1)
    results_3 = run_verifier(records_3)
    # 1 ENF + 12 NLR channels per node
    check("Integration: 1-node config returns 13 results/window",
          len(results_1[2]) == 13, f"got {len(results_1[2])}")
    check("Integration: 3-node config returns 37 results/window (1 ENF + 3*12 NLR)",
          len(results_3[2]) == 37, f"got {len(results_3[2])}")


def test_enf_nominal_spike_end_to_end():
    def enf_with_spike(i):
        if i == 10:
            return 65.0  # single-sample raw spike, far outside nominal tolerance
        return smooth_enf(i)
    records = make_records(20, node_ids=("node_A",), enf_fn=enf_with_spike)
    results = run_verifier(records)
    enf_result = find(results[10], "/ENF")
    check("Integration: single-sample ENF spike caught by ENFNominalRangeCheck",
          enf_result.status == "failed" and "NOMINAL" in enf_result.reason, enf_result.reason)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    test_fns = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for fn in test_fns:
        try:
            fn()
        except Exception as e:
            _record(fn.__name__, False, f"raised {type(e).__name__}: {e}")

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = [(n, d) for n, ok, d in _results if not ok]

    print(f"\n{'='*70}")
    print("TEST RESULTS")
    print(f"{'='*70}")
    for name, ok, detail in _results:
        mark = "PASS" if ok else "FAIL"
        line = f"  [{mark}] {name}"
        if not ok and detail:
            line += f"\n         -> {detail}"
        print(line)

    print(f"\n{'='*70}")
    print(f"{passed}/{len(_results)} passed")
    if failed:
        print(f"{len(failed)} FAILED:")
        for name, detail in failed:
            print(f"  - {name}: {detail}")
    print(f"{'='*70}\n")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()