"""
pipeline_test.py
-----------------
Ingests ENF + NLR data for a 16-node configuration, merges them into
a combined JSONL file, then reads it back and prints every record to
the console.

Works for any node count (1 to 16+) -- discovers pairs automatically
from the folder by SLURM job ID so only one hour's files are picked up
even if the folder contains multiple hours.

Run from the repo root:
    python lanes/marchisano_attacks/pipeline_test.py

Edit the CONFIG block below to point at your actual file paths.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from microverse_core.data_loaders import (
    load_enf,
    discover_nlr_pairs,
    load_nlr_multi,
    build_combined_records,
    write_combined_jsonl,
    read_combined_jsonl,
)


# ---------------------------------------------------------------------------
# CONFIG -- edit these before running
# ---------------------------------------------------------------------------

ENF_PATH = (
    "/home/brandon/Desktop/ENF-ML (CNN+MAMBA)/Data/Dev1_ENF_Hr01.csv"
)

NLR_FOLDER = (
    "/home/brandon/Desktop/00_raw_datasets/training_llama2_70b_lora/16node/"
)

# SLURM job ID shared by all 16 nodes for this run.
# All 160 files in the folder share slurmid_10742842 -- change this
# if you want a different hour/run from the same folder.
SLURM_ID = "10742842"

# Where to write the combined JSONL output
OUT_PATH = "data/combined/run_16node.jsonl"

# ---------------------------------------------------------------------------


def main():
    # ---- Step 1: load ENF ----
    print(f"Loading ENF from {ENF_PATH} ...")
    enf = load_enf(ENF_PATH)
    print(f"  -> {len(enf)} ENF readings")

    # ---- Step 2: discover and load all 16 node pairs ----
    print(f"\nDiscovering NLR pairs in {NLR_FOLDER}")
    print(f"  filtering by slurm_id={SLURM_ID} ...")
    pairs = discover_nlr_pairs(NLR_FOLDER, slurm_id=SLURM_ID)
    print(f"  -> {len(pairs)} node pair(s) found")

    print("\nLoading and aggregating NLR data ...")
    node_windows = load_nlr_multi(pairs)

    # ---- Step 3: merge into combined records ----
    print("\nMerging ENF + NLR into combined records ...")
    records = build_combined_records(enf, node_windows)
    print(f"  -> {len(records)} records")
    print(f"  -> {len(records[0])} columns per record")
    print(f"  -> columns: {list(records[0].keys())[:4]} ... "
          f"{list(records[0].keys())[-2:]}")

    # ---- Step 4: write to JSONL ----
    print(f"\nWriting to {OUT_PATH} ...")
    write_combined_jsonl(records, OUT_PATH)
    print(f"  -> done")

    # ---- Step 5: read back and print ----
    print(f"\nReading back from {OUT_PATH} and printing to console ...\n")
    for record in read_combined_jsonl(OUT_PATH):
        print(record)


if __name__ == "__main__":
    main()