"""
pipeline_test.py
-----------------
Ingests ENF + NLR data for a 16-node configuration, merges them into
a combined JSONL file, then reads it back and prints every record to
the console.

Works for any node count (1 to 16+) -- discovers pairs automatically
from the folder by SLURM job ID so only one hour's files are picked up
even if the folder contains multiple hours.

============================================================
HOW TO RUN THIS ON YOUR OWN MACHINE
============================================================

Raw data is NOT distributed with this repo -- it stays wherever it
already lives on your machine. There is no shared data/ folder to
populate; you just point this script at your own local paths.

1. Clone the repo:
     git clone <repo-url>
     cd Microverse-2026-Project

2. Python 3.10+ is all you need. No third-party packages required --
   this script and everything it imports is standard library only.

3. Edit the CONFIG block below to point ENF_PATH and NLR_FOLDER at
   wherever YOUR raw ENF/NLR files actually live on your machine.
   Everyone running this will have these files in a different place --
   that's expected, just edit the paths to match your own setup.

4. Pick the right NLR_FOLDER/SLURM_ID pair for your data:
     - Training data (SLURM-tagged filenames, e.g. multiple hours in
       one folder): use the training lines as-is, set SLURM_ID to the
       specific hour you want.
     - Inference data (no SLURM ID in filenames, e.g.
       "nvml_wattameter_x3115c0s33b0n0.log"): use the inference lines,
       set SLURM_ID = None. Each inference run must be in its OWN
       folder -- there's no filename token to disambiguate multiple
       runs sharing one folder.

5. Run it. OUT_PATH is anchored to this file's own location on disk
   (not your current directory or where your raw data lives), so the
   combined output always lands in this repo's data/combined/
   regardless of where you run the command from:
     python lanes/marchisano_attacks/pipeline_test.py

6. What you should see:
     Loading ENF from ...          -> N ENF readings
     Discovering NLR pairs in ...  -> N node pair(s) found
     Merging ENF + NLR ...         -> N records, N columns each
     Writing to data/combined/run_16node.jsonl -> done
     Every record printed to console as a final sanity check

TROUBLESHOOTING
  FileNotFoundError on ENF_PATH or NLR_FOLDER
    -> Your CONFIG paths don't point at real files on this machine.
       Edit ENF_PATH/NLR_FOLDER below to match where your data
       actually lives.

  ModuleNotFoundError: No module named 'microverse_core'
    -> Confirm the microverse_core/ folder sits directly at the repo
       root and wasn't moved or renamed.

  ValueError: Duplicate NVML/RAPL file for node ...
    -> Two different runs are mixed in the same folder. For training
       data, set SLURM_ID to the specific hour you want. For inference
       data there's no SLURM ID to filter by at all -- move each run
       into its own separate folder instead.
============================================================
"""


import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from microverse_core.data_loaders import (
    load_enf,
    clean_enf,
    discover_nlr_pairs,
    load_nlr_multi,
    build_combined_records,
    write_combined_jsonl,
    read_combined_jsonl,
)


# ---------------------------------------------------------------------------
# CONFIG -- EDIT THESE to point at wherever your raw data actually lives
# on YOUR machine. These paths are personal to your setup -- everyone
# running this script will have different values here.
# ---------------------------------------------------------------------------

ENF_PATH = "/home/brandon/Desktop/ENF-ML (CNN+MAMBA)/Data/Dev1_ENF_Hr01.csv"

# --- Training run (SLURM-tagged filenames, e.g. multiple hours co-located
#     in one folder) ---
NLR_FOLDER = "/home/brandon/Desktop/00_raw_datasets/training_llama2_70b_lora/2node/"
SLURM_ID = "10742795"

# --- Inference run (old-style filenames, e.g. "nvml_wattameter_x3115c0s33b0n0.log")
#     no SLURM ID in the filename at all. Node ID extraction already works
#     for this naming convention with zero code changes -- just don't pass
#     a SLURM filter. Since there's no run-identifying token in these
#     filenames, EACH INFERENCE RUN MUST GET ITS OWN FOLDER -- there's no
#     way to disambiguate multiple runs sharing one folder.
#     Uncomment to use instead of the training config above:
#
# NLR_FOLDER = "/path/to/your/nlr_data/inference_llama2_70b_lora/16node_run1/"
# SLURM_ID = None

# NODE_COUNT / NODE_IDS -- optional. Both default to None, meaning
# "ingest every node discover_nlr_pairs finds" (today's behavior,
# unchanged). Set ONE of these to ingest only a subset -- useful for
# comparing verifier behavior at different real node counts without
# touching raw data or needing a separate script:
#   NODE_COUNT = 2        -> first 2 nodes found, sorted alphabetically
#   NODE_IDS = ["x3105c0s37b0n0", "x3105c0s41b0n0"]  -> these exact nodes
NODE_COUNT = None
NODE_IDS = None

# Output filename reflects how many nodes were actually ingested, so
# a subset run never silently overwrites your full-rack baseline file.
OUT_PATH = REPO_ROOT / "data" / "combined" / "run_2node.jsonl"

# ---------------------------------------------------------------------------


def main():
    # ---- Step 1: load ENF, then clean it -- BEFORE this file goes ----
    # ---- anywhere near attack injection. Must stay in this exact  ----
    # ---- position: clean_enf() should never run on a file that's  ----
    # ---- already been through attack injection (verified 2026-07 -- ----
    # ---- doing so silently erases spike-type attacks with zero    ----
    # ---- detection). verify_file.py never imports clean_enf at all,----
    # ---- so this is the only place it should ever be called.      ----
    print(f"Loading ENF from {ENF_PATH} ...")
    enf = load_enf(ENF_PATH)
    print(f"  -> {len(enf)} ENF readings")

    print("Cleaning ENF (physical-range check + Hampel filter) ...")
    enf = clean_enf(enf)
    print(f"  -> done")

    # ---- Step 2: discover node pairs, optionally narrow to a subset ----
    print(f"\nDiscovering NLR pairs in {NLR_FOLDER}")
    print(f"  filtering by slurm_id={SLURM_ID} ...")
    pairs = discover_nlr_pairs(NLR_FOLDER, slurm_id=SLURM_ID)
    print(f"  -> {len(pairs)} node pair(s) found")

    out_path = OUT_PATH
    if NODE_IDS:
        pairs = [p for p in pairs if p[0] in NODE_IDS]
        missing = set(NODE_IDS) - {p[0] for p in pairs}
        if missing:
            raise ValueError(f"Requested node(s) not found: {sorted(missing)}")
        print(f"  -> narrowed to {len(pairs)} requested node(s): {[p[0] for p in pairs]}")
        out_path = REPO_ROOT / "data" / "combined" / f"run_{len(pairs)}node.jsonl"
    elif NODE_COUNT:
        pairs = sorted(pairs, key=lambda p: p[0])[:NODE_COUNT]
        print(f"  -> narrowed to first {len(pairs)} node(s): {[p[0] for p in pairs]}")
        out_path = REPO_ROOT / "data" / "combined" / f"run_{len(pairs)}node.jsonl"

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
    print(f"\nWriting to {out_path} ...")
    write_combined_jsonl(records, out_path)
    print(f"  -> done")

    # ---- Step 5: read back and print ----
    print(f"\nReading back from {out_path} and printing to console ...\n")
    for record in read_combined_jsonl(out_path):
        print(record)


if __name__ == "__main__":
    main()