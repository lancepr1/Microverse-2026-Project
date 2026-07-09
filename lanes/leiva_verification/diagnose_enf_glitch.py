"""
diagnose_enf_glitch.py
-----------------------
Characterizes the known ENF false-positive cluster near the end of the
1-hour AFRL recording (confirmed by calibrate_thresholds.py and
verify_file.py to be the dominant source of ENF-side false positives --
741 of 820 total FAILED results in the last full run, concentrated
around indices ~1682-1798).

Pulls the RAW Hz values in that range and checks for specific glitch
signatures, since which one it turns out to be determines which
remediation actually fits:

  (a) STUCK/REPEATED VALUE  -- sensor or capture pipeline dropped out
      and is repeating a stale reading. Fix: smooth/interpolate at
      ingestion (data_loaders.py), since the raw data itself is bad.

  (b) SUDDEN LEVEL SHIFT    -- one or two large single-sample jumps,
      then stable (possibly at a wrong level) afterward. Could be a
      real grid event OR a capture artifact (e.g. clock desync,
      Raspberry Pi buffer glitch). Fix depends on which -- worth
      cross-checking against a public grid-frequency event log for
      the exact UTC timestamp if it looks real.

  (c) SUSTAINED HIGH NOISE  -- values genuinely wander more than
      elsewhere, no obvious single bad sample or stuck value. Could
      be a real noisy stretch (the ENF window-correlation confidence
      metric is known from earlier profiling to be sensitive to even
      normal noise). Fix: this may not need touching at ingestion at
      all -- could be a verifier-side calibration question instead
      (does CONFIDENCE_SUSPECT need to account for genuinely noisier
      real stretches, not just glitches).

Usage:
    python lanes/leiva_verification/diagnose_enf_glitch.py \
        --path "/path/to/ENF_data/Dev1_ENF_Hr01.csv" \
        --start 1650 --end 1800
"""

import argparse
import csv
import statistics
from collections import Counter
from pathlib import Path


def load_enf_with_raw_rows(path: str) -> list[tuple[int, list, float]]:
    """
    Same parsing as load_enf_raw(), but also keeps the raw CSV row
    alongside each parsed value -- so a parsing bug (wrong column read,
    malformed row, stray extra field) is directly visible rather than
    hidden behind an already-parsed float. If the "glitch" turns out
    to be a parsing artifact rather than a real sensor/capture problem,
    this is the cheapest possible fix -- a parsing correction, not a
    smoothing/interpolation decision.
    """
    out: list[tuple[int, list, float]] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # skip metadata row
        for row in reader:
            if not row:
                continue
            try:
                out.append((len(out), row, float(row[1])))
            except (ValueError, IndexError):
                out.append((len(out), row, float("nan")))
    return out


def load_enf_raw(path: str) -> list[float]:
    """Mirrors data_loaders.load_enf() exactly -- same file format,
    same skip-metadata-row logic -- so what this script sees is
    exactly what the real pipeline sees, nothing different."""
    values: list[float] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # skip metadata row
        for row in reader:
            if not row:
                continue
            try:
                values.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    return values


def main():
    parser = argparse.ArgumentParser(
        description="Characterize the known ENF glitch signature"
    )
    parser.add_argument("--path", required=True, help="Path to the ENF CSV file")
    parser.add_argument("--start", type=int, default=1650, help="First suspect index")
    parser.add_argument("--end", type=int, default=1800, help="Last suspect index (exclusive)")
    parser.add_argument(
        "--show-raw-rows", action="store_true",
        help="Print the exact CSV row (not just the parsed float) for every "
             "index in range -- catches parsing bugs (wrong column, malformed "
             "row, extra fields) that would look identical to a real glitch "
             "once parsed into a plain float"
    )
    args = parser.parse_args()

    if not Path(args.path).exists():
        print(f"ERROR: file not found: {args.path}")
        return

    values = load_enf_raw(args.path)
    print(f"Loaded {len(values)} ENF samples\n")

    if args.end > len(values):
        print(f"WARNING: --end {args.end} exceeds file length {len(values)}, clamping")
        args.end = len(values)

    baseline = values[:args.start] + values[args.end:]
    window = values[args.start:args.end]

    if not baseline or not window:
        print("ERROR: empty baseline or window -- check --start/--end")
        return

    # ------------------------------------------------------------
    # Baseline vs. suspect-window comparison
    # ------------------------------------------------------------
    print("=" * 70)
    print(f"BASELINE  (every index OUTSIDE [{args.start}, {args.end}))")
    print("=" * 70)
    print(f"  mean = {statistics.mean(baseline):.4f} Hz   std = {statistics.stdev(baseline):.4f} Hz")
    print(f"  min  = {min(baseline):.4f} Hz   max = {max(baseline):.4f} Hz")

    print(f"\n{'=' * 70}")
    print(f"SUSPECT WINDOW  (indices [{args.start}, {args.end}))")
    print("=" * 70)
    print(f"  mean = {statistics.mean(window):.4f} Hz   std = {statistics.stdev(window):.4f} Hz")
    print(f"  min  = {min(window):.4f} Hz   max = {max(window):.4f} Hz")

    steps = [abs(window[i + 1] - window[i]) for i in range(len(window) - 1)]
    print(f"\n  step sizes within window: mean={statistics.mean(steps):.4f} Hz  "
          f"max={max(steps):.4f} Hz")

    # ------------------------------------------------------------
    # Signature check (a): stuck / repeated value
    # ------------------------------------------------------------
    counts = Counter(round(v, 4) for v in window)
    most_common_val, most_common_count = counts.most_common(1)[0]
    repeat_fraction = most_common_count / len(window)
    print(f"\n{'-' * 70}")
    print("SIGNATURE CHECK (a): stuck/repeated value")
    print(f"{'-' * 70}")
    print(f"  most repeated value: {most_common_val} Hz, appears "
          f"{most_common_count}/{len(window)} times ({repeat_fraction:.1%})")
    if repeat_fraction > 0.10:
        print("  --> SUSPICIOUS: looks like a stuck/stale sensor reading")
    else:
        print("  --> no dominant repeated value -- doesn't look stuck")

    # ------------------------------------------------------------
    # Signature check (b): sudden level shift (one or two big jumps,
    # then relatively stable afterward, vs. constant churn)
    # ------------------------------------------------------------
    big_steps = [(i, s) for i, s in enumerate(steps) if s > 1.0]
    print(f"\n{'-' * 70}")
    print("SIGNATURE CHECK (b): sudden level shift")
    print(f"{'-' * 70}")
    print(f"  steps > 1.0 Hz: {len(big_steps)} occurrence(s) out of {len(steps)}")
    if 1 <= len(big_steps) <= 3:
        print("  --> SUSPICIOUS: a small number of large jumps -- consistent with "
              "a discrete glitch/dropout event rather than continuous noise")
    elif len(big_steps) > 3:
        print("  --> multiple large jumps scattered through the window -- "
              "looks more like sustained instability than one discrete event")
    else:
        print("  --> no single large jump found")

    # ------------------------------------------------------------
    # Signature check (c): sustained high noise (compare window std
    # to baseline std directly)
    # ------------------------------------------------------------
    baseline_std = statistics.stdev(baseline)
    window_std = statistics.stdev(window)
    ratio = window_std / baseline_std if baseline_std else float("inf")
    print(f"\n{'-' * 70}")
    print("SIGNATURE CHECK (c): sustained high noise vs. baseline")
    print(f"{'-' * 70}")
    print(f"  window std / baseline std = {ratio:.2f}x")
    if ratio > 3:
        print("  --> SUSPICIOUS: window is dramatically noisier than the rest "
              "of the file, not just a couple of bad samples")
    else:
        print("  --> window noise level is roughly comparable to baseline")

    # ------------------------------------------------------------
    # Raw values for manual inspection
    # ------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("RAW VALUES  (index: Hz, step from previous)")
    print("=" * 70)
    prev = None
    for i, v in zip(range(args.start, args.end), window):
        step_str = f"  step={abs(v - prev):.4f}" if prev is not None else ""
        marker = "  <-- large step" if prev is not None and abs(v - prev) > 1.0 else ""
        print(f"  [{i:5d}] {v:.4f} Hz{step_str}{marker}")
        prev = v

    if args.show_raw_rows:
        print(f"\n{'=' * 70}")
        print("RAW CSV ROWS  (exact fields as read from the file, before parsing)")
        print("=" * 70)
        with_rows = load_enf_with_raw_rows(args.path)
        for idx, row, parsed in with_rows[args.start:args.end]:
            flag = ""
            if len(row) != 2:
                flag = f"  <-- {len(row)} FIELDS (expected 2)"
            print(f"  [{idx:5d}] raw_row={row}  parsed={parsed:.4f}{flag}")

    print(f"\n{'=' * 70}")
    print("Bring this output to the remediation decision -- see the module")
    print("docstring for how each signature maps to a fix (smooth at")
    print("ingestion / verifier-side handling / leave as real data).")
    print("=" * 70)


if __name__ == "__main__":
    main()