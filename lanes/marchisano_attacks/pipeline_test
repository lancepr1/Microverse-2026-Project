"""
pipeline_test.py
-----------------
One script that does the whole thing:
  1. Takes the three raw file paths (ENF, NVML, RAPL)
  2. Calls data_loaders.py to ingest and merge them
  3. Writes the combined JSONL
  4. Reads it back and prints every record to the console

Run from the repo root:
    python scripts/pipeline_test.py \
        --enf-path "path/to/your/enf.csv" \
        --nvml-path "path/to/nvml_wattameter_xxxxx.log" \
        --rapl-path "path/to/rapl_wattameter_xxxxx.log" \
        --out-path "data/combined/run01.jsonl"

--out-path is optional, defaults to data/combined/run01.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microverse_core.data_loaders import (
    load_enf,
    load_nlr,
    build_combined_records,
    write_combined_jsonl,
    read_combined_jsonl,
)


def main():
    parser = argparse.ArgumentParser(description="Ingest ENF + NVML + RAPL and print the result")
    parser.add_argument("--enf-path", required=True, help="Path to the ENF CSV file")
    parser.add_argument("--nvml-path", required=True, help="Path to the NVML .log file")
    parser.add_argument("--rapl-path", required=True, help="Path to the RAPL .log file")
    parser.add_argument(
        "--out-path", default="data/combined/run01.jsonl",
        help="Where to write the combined JSONL file"
    )
    args = parser.parse_args()

    # ---- ingest ----
    enf = load_enf(args.enf_path)
    gpu_windows, cpu_windows = load_nlr(
        nvml_path=args.nvml_path,
        rapl_path=args.rapl_path,
    )
    records = build_combined_records(enf, gpu_windows, cpu_windows)
    write_combined_jsonl(records, args.out_path)

    # ---- print ----
    for record in read_combined_jsonl(args.out_path):
        print(record)


if __name__ == "__main__":
    main()