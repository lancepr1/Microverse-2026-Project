"""
nlr_profile.py
---------------
Statistical profile of clean (untampered) NLR data to determine
detection thresholds for the new GPU/CPU verification checks.

Mirrors enf_profile.py: run this against your clean recordings before
finalizing any NLR threshold values in the verifier. Accepts multiple
file pairs in a single run so you do not need to invoke this 24 times
by hand.

Usage -- single pair:
    python nlr_profile.py \
        --nvml data/raw/run01_nvml.log \
        --rapl data/raw/run01_rapl.log

Usage -- many pairs, explicit:
    python nlr_profile.py \
        --pairs run01_nvml.log:run01_rapl.log run02_nvml.log:run02_rapl.log

Usage -- many pairs, by folder convention
    (expects files named like *_nvml_*.log and *_rapl_*.log, paired by
    matching the part of the filename after the nvml_/rapl_ prefix):
    python nlr_profile.py --dir data/raw/

Output:
    Per-file breakdown (so you can see if conditions shifted between runs)
    Combined profile across all files (the more statistically robust one)
    Recommended threshold values to paste into verification.py
"""

import argparse
import statistics
from pathlib import Path
from typing import Optional

# reuse the same parsing logic already proven against real files
from microverse_core.data_loaders import (
    _parse_nvml_log,
    _parse_rapl_log,
    _aggregate_gpu,
    _aggregate_cpu,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def percentile(values: list[float], p: float) -> float:
    """p-th percentile (0-100) via linear interpolation."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    idx = (p / 100) * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def step_sizes(values: list[float]) -> list[float]:
    """Absolute difference between each consecutive pair."""
    return [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]


def find_pairs_in_dir(directory: str) -> list[tuple[str, str]]:
    """
    Finds NVML/RAPL pairs in a folder by matching filenames that share
    everything except the nvml_/rapl_ prefix.

    e.g. nvml_wattameter_x3115c0s33b0n0.log
         rapl_wattameter_x3115c0s33b0n0.log
    pair on "wattameter_x3115c0s33b0n0.log"
    """
    d = Path(directory)
    nvml_files = {f.name.replace("nvml_", "", 1): f for f in d.glob("*nvml*.log")}
    rapl_files = {f.name.replace("rapl_", "", 1): f for f in d.glob("*rapl*.log")}

    pairs = []
    for key in nvml_files:
        if key in rapl_files:
            pairs.append((str(nvml_files[key]), str(rapl_files[key])))

    return pairs


# ---------------------------------------------------------------------------
# Per-file stats container
# ---------------------------------------------------------------------------

def collect_channel_stats(
    nvml_path: str,
    rapl_path: str,
    max_windows: Optional[int] = None,
) -> dict:
    """
    Parses one NVML/RAPL pair and returns flat lists of every relevant
    channel, ready to be pooled across files or analyzed individually.
    """
    max_raw = max_windows * 20 if max_windows else None

    raw_gpu = _parse_nvml_log(nvml_path, max_rows=max_raw)
    raw_cpu = _parse_rapl_log(rapl_path, max_rows=max_raw)

    gpu_windows = _aggregate_gpu(raw_gpu)
    cpu_windows = _aggregate_cpu(raw_cpu)

    if max_windows:
        gpu_windows = gpu_windows[:max_windows]
        cpu_windows = cpu_windows[:max_windows]

    gpu_power_all = []
    gpu_temp_all  = []
    for w in gpu_windows:
        gpu_power_all.extend([w.gpu0_w, w.gpu1_w, w.gpu2_w, w.gpu3_w])
        gpu_temp_all.extend([w.gpu0_c, w.gpu1_c, w.gpu2_c, w.gpu3_c])

    cpu_power_all = []
    cpu_uj_all    = []
    cpu_uj_violations = []   # tracks real (non-wraparound) decreases
    cpu_uj_wraps       = []  # tracks expected RAPL counter wraps

    # RAPL energy counters are fixed-width hardware registers that wrap
    # around when they overflow. Observed wrap ceiling in this dataset
    # is ~65.5 billion uJ -- a drop near that magnitude is an expected
    # hardware wraparound, NOT tampering. A drop of any other size is
    # a real monotonicity violation and should be flagged.
    WRAP_CEILING_UJ = 65_500_000_000
    WRAP_TOLERANCE_UJ = 2_000_000_000  # +/- margin around the ceiling

    for i, w in enumerate(cpu_windows):
        cpu_power_all.extend([w.cpu0_w, w.cpu1_w])
        cpu_uj_all.extend([w.cpu0_uj, w.cpu1_uj])
        if i > 0:
            prev = cpu_windows[i - 1]
            for channel_name, cur_val, prev_val in [
                ("cpu0", w.cpu0_uj, prev.cpu0_uj),
                ("cpu1", w.cpu1_uj, prev.cpu1_uj),
            ]:
                if cur_val < prev_val:
                    drop_size = prev_val - cur_val
                    is_expected_wrap = abs(drop_size - WRAP_CEILING_UJ) < WRAP_TOLERANCE_UJ
                    if is_expected_wrap:
                        cpu_uj_wraps.append((i, channel_name, drop_size))
                    else:
                        cpu_uj_violations.append((i, channel_name, drop_size))

    # per-channel step sizes (within each window list, channel by channel)
    gpu0_steps = step_sizes([w.gpu0_w for w in gpu_windows])
    gpu1_steps = step_sizes([w.gpu1_w for w in gpu_windows])
    gpu2_steps = step_sizes([w.gpu2_w for w in gpu_windows])
    gpu3_steps = step_sizes([w.gpu3_w for w in gpu_windows])
    gpu_step_pool = gpu0_steps + gpu1_steps + gpu2_steps + gpu3_steps

    cpu0_steps = step_sizes([w.cpu0_w for w in cpu_windows])
    cpu1_steps = step_sizes([w.cpu1_w for w in cpu_windows])
    cpu_step_pool = cpu0_steps + cpu1_steps

    return {
        "n_windows": len(gpu_windows),
        "gpu_power": gpu_power_all,
        "gpu_temp": gpu_temp_all,
        "gpu_step": gpu_step_pool,
        "cpu_power": cpu_power_all,
        "cpu_uj": cpu_uj_all,
        "cpu_step": cpu_step_pool,
        "cpu_uj_violations": cpu_uj_violations,
        "cpu_uj_wraps": cpu_uj_wraps,
    }


def print_summary(label: str, stats: dict) -> None:
    print(f"\n{'─'*60}")
    print(f"{label}")
    print(f"{'─'*60}")
    print(f"  windows                : {stats['n_windows']}")

    gp = stats["gpu_power"]
    print(f"\n  GPU power (W)")
    print(f"    mean / std           : {statistics.mean(gp):.2f} / {statistics.stdev(gp):.2f}")
    print(f"    min / max            : {min(gp):.2f} / {max(gp):.2f}")
    print(f"    P99                  : {percentile(gp, 99):.2f}")

    gt = stats["gpu_temp"]
    print(f"\n  GPU temperature (C)")
    print(f"    mean / std           : {statistics.mean(gt):.2f} / {statistics.stdev(gt):.2f}")
    print(f"    min / max            : {min(gt):.2f} / {max(gt):.2f}")

    gs = stats["gpu_step"]
    if gs:
        print(f"\n  GPU power step between windows (W)")
        print(f"    mean / std           : {statistics.mean(gs):.3f} / {statistics.stdev(gs):.3f}")
        print(f"    max                  : {max(gs):.3f}")
        print(f"    P99                  : {percentile(gs, 99):.3f}")

    cp = stats["cpu_power"]
    print(f"\n  CPU power (W)")
    print(f"    mean / std           : {statistics.mean(cp):.2f} / {statistics.stdev(cp):.2f}")
    print(f"    min / max            : {min(cp):.2f} / {max(cp):.2f}")
    print(f"    P99                  : {percentile(cp, 99):.2f}")

    cs = stats["cpu_step"]
    if cs:
        print(f"\n  CPU power step between windows (W)")
        print(f"    mean / std           : {statistics.mean(cs):.3f} / {statistics.stdev(cs):.3f}")
        print(f"    max                  : {max(cs):.3f}")
        print(f"    P99                  : {percentile(cs, 99):.3f}")

    print(f"\n  CPU uJ monotonicity     : "
          f"{'OK -- always increasing' if stats['cpu_uj_monotonic'] else 'VIOLATED -- check parsing'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Statistical profile of clean NLR data")
    parser.add_argument("--nvml", help="Single NVML .log path")
    parser.add_argument("--rapl", help="Single RAPL .log path")
    parser.add_argument(
        "--pairs", nargs="+",
        help="Multiple pairs as nvml_path:rapl_path nvml_path:rapl_path ..."
    )
    parser.add_argument("--dir", help="Folder to auto-discover nvml/rapl pairs in")
    parser.add_argument(
        "--max-windows", type=int, default=None,
        help="Cap each file to this many aggregated windows (default: no cap)"
    )
    args = parser.parse_args()

    # build the list of (nvml, rapl) pairs to process
    pairs: list[tuple[str, str]] = []

    if args.nvml and args.rapl:
        pairs.append((args.nvml, args.rapl))
    if args.pairs:
        for p in args.pairs:
            nvml, rapl = p.split(":")
            pairs.append((nvml, rapl))
    if args.dir:
        found = find_pairs_in_dir(args.dir)
        print(f"Found {len(found)} nvml/rapl pairs in {args.dir}")
        pairs.extend(found)

    if not pairs:
        print("ERROR: no file pairs given. Use --nvml/--rapl, --pairs, or --dir")
        return

    print(f"\nProcessing {len(pairs)} file pair(s)...")

    # ----------------------------------------------------------------
    # Per-file breakdown
    # ----------------------------------------------------------------
    all_stats = []
    for i, (nvml_path, rapl_path) in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] {Path(nvml_path).name}")
        stats = collect_channel_stats(nvml_path, rapl_path, args.max_windows)
        all_stats.append(stats)
        print_summary(f"FILE {i+1}: {Path(nvml_path).name}", stats)

    # ----------------------------------------------------------------
    # Combined pool across all files
    # ----------------------------------------------------------------
    combined = {
        "n_windows": sum(s["n_windows"] for s in all_stats),
        "gpu_power": sum((s["gpu_power"] for s in all_stats), []),
        "gpu_temp":  sum((s["gpu_temp"]  for s in all_stats), []),
        "gpu_step":  sum((s["gpu_step"]  for s in all_stats), []),
        "cpu_power": sum((s["cpu_power"] for s in all_stats), []),
        "cpu_uj":    sum((s["cpu_uj"]    for s in all_stats), []),
        "cpu_step":  sum((s["cpu_step"]  for s in all_stats), []),
        "cpu_uj_monotonic": all(s["cpu_uj_monotonic"] for s in all_stats),
    }
    print_summary(f"COMBINED ACROSS ALL {len(pairs)} FILES", combined)

    # ----------------------------------------------------------------
    # Recommended thresholds
    # ----------------------------------------------------------------
    gpu_step_max = percentile(combined["gpu_step"], 99) * 1.2 if combined["gpu_step"] else 0
    cpu_step_max = percentile(combined["cpu_step"], 99) * 1.2 if combined["cpu_step"] else 0
    gpu_power_ceiling = max(800.0, percentile(combined["gpu_power"], 99.9) * 1.1)
    cpu_power_ceiling = max(800.0, percentile(combined["cpu_power"], 99.9) * 1.1)

    print(f"\n{'='*60}")
    print("RECOMMENDED NLR THRESHOLDS")
    print(f"{'='*60}")
    print(f"  GPU_POWER_CEILING_W = {gpu_power_ceiling:.1f}   "
          f"# hard ceiling, vendor TDP + margin")
    print(f"  CPU_POWER_CEILING_W = {cpu_power_ceiling:.1f}")
    print(f"  GPU_MAX_STEP_W      = {gpu_step_max:.2f}   "
          f"# P99 step x 1.2 margin")
    print(f"  CPU_MAX_STEP_W      = {cpu_step_max:.2f}")
    print(f"  CPU_UJ_MUST_INCREASE = True  "
          f"# {'confirmed in clean data' if combined['cpu_uj_monotonic'] else 'WARNING: violated in clean data, check parsing'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()