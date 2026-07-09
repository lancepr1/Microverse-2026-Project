"""
run_microverse.py
-------------------
WORKING NAME -- this is meant to become the renamed, CLI-driven
successor to pipeline_test.py, per the requested architecture:

    ingest -> smooth -> attack injection -> verify + annotate -> fork
    to (scoreboard, dashboard, digital twin)

Lives at scripts/run_microverse.py -- one level below the repo root.
All internal paths (data/combined, lanes/marchisano_attacks/,
lanes/leiva_verification/outputs/) are anchored explicitly to the
repo root regardless of where you invoke this script from, so it
works whether you run it from the repo root or from inside scripts/.
The one thing that does still matter: attack.py itself is invoked
with cwd set to the repo root, so ITS OWN relative paths resolve
correctly too.

STATUS OF EACH STAGE (2026-07):
    Stage 1 (ingest + smooth):    REAL, tested, working.
    Stage 2 (attack injection):   Wired up against attack.py's current
                                   (still-being-developed) behavior --
                                   confirm it still works if that
                                   changes on Ethan's end.
    Stage 3 (verify + annotate):  REAL, tested, working -- this is the
                                   same logic as verify_file.py,
                                   refactored into an importable
                                   function so this script can call it
                                   in-process instead of shelling out.
    Stage 4 (fork to 3 outputs):  PLACEHOLDER destinations, REAL file
                                   writes -- writes to
                                   lanes/leiva_verification/outputs/.
                                   Replace with the real dashboard/
                                   digital-twin integration once their
                                   expected interface is confirmed.

CLI (run from anywhere, though the repo root is the convention used
everywhere else in this project):
    python scripts/run_microverse.py \\
        --workload-type {inference,training} \\
        --enf-path /path/to/ENF.csv \\
        --nlr-folder /path/to/nlr/data/ \\
        --slurm-id ID          # required for training, ignored for inference
        [--node-count N]       # first N nodes found, sorted alphabetically
        [--node-ids ID1 ID2 ...]   # exact nodes instead of --node-count
        [--component-id rack_00]
        [--output-dir data/combined]

Workload type matters beyond just labeling the run -- it changes HOW
NLR data is discovered. Training runs use a SLURM job ID to find the
right log files; inference runs use old-style logs with no SLURM ID
at all (discover_nlr_pairs(folder, slurm_id=None)). Get this wrong
and node discovery silently returns nothing or the wrong files.
"""

import argparse
import json
import sys
from pathlib import Path

# Lives in scripts/run_microverse.py -- one level below repo root, so
# .parent.parent (not .parent) is needed to reach the repo root where
# microverse_core/ and lanes/ actually are.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "lanes" / "leiva_verification"))

from microverse_core.data_loaders import (
    load_enf,
    combined_smooth,
    discover_nlr_pairs,
    load_nlr_multi,
    build_combined_records,
    write_combined_jsonl,
    read_combined_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Microverse-2026-Project: full ingestion-to-verification pipeline"
    )
    parser.add_argument(
        "--workload-type", choices=["inference", "training"], required=True,
        help="Determines how NLR data is discovered -- training runs use a "
             "SLURM job ID, inference runs don't."
    )
    parser.add_argument(
        "--node-count", type=int, default=None,
        help="Use the first N discovered nodes (sorted alphabetically). "
             "Omit to use every node found."
    )
    parser.add_argument(
        "--node-ids", nargs="+", default=None,
        help="Use these exact node IDs instead of --node-count. Overrides "
             "--node-count if both are given."
    )
    parser.add_argument("--enf-path", required=True, help="Path to the raw ENF CSV file")
    parser.add_argument("--nlr-folder", required=True, help="Path to the folder containing NLR (GPU/CPU) log files")
    parser.add_argument(
        "--slurm-id", default=None,
        help="Required when --workload-type training. Ignored for inference."
    )
    parser.add_argument("--component-id", default="rack_00")
    parser.add_argument(
        "--output-dir", default="data/combined",
        help="Where to write intermediate and final pipeline files"
    )
    args = parser.parse_args()

    if args.workload_type == "training" and not args.slurm_id:
        parser.error("--slurm-id is required when --workload-type training")

    # Anchor to repo root explicitly if given as a relative path -- makes
    # this robust to being invoked from anywhere, not just the repo root
    # (matters more now that this script itself lives one level deeper,
    # in scripts/, than the rest of the project's convention assumes).
    output_dir = Path(args.output_dir)
    args.output_dir = str(output_dir if output_dir.is_absolute() else _REPO_ROOT / output_dir)

    return args


def stage_1_ingest_and_smooth(args) -> Path:
    """
    Load raw ENF, smooth it (Hampel outlier correction + Butterworth
    lowpass -- see combined_smooth() in data_loaders.py), load NLR data
    for the requested nodes, combine into one JSONL.

    combined_smooth() MUST run here, before attack injection ever
    touches the data -- validated (2026-07) that smoothing downstream
    of an attack silently erases it with zero detection. This is the
    one ordering rule in the whole pipeline that must never move.
    """
    print(f"[1/4] Ingesting ENF from {args.enf_path} ...")
    enf = load_enf(args.enf_path)
    enf = combined_smooth(enf)

    print(f"[1/4] Discovering NLR pairs in {args.nlr_folder} "
          f"(workload_type={args.workload_type}) ...")
    slurm_id = args.slurm_id if args.workload_type == "training" else None
    pairs = discover_nlr_pairs(args.nlr_folder, slurm_id=slurm_id)

    if args.node_ids:
        pairs = [p for p in pairs if p[0] in args.node_ids]
        missing = set(args.node_ids) - {p[0] for p in pairs}
        if missing:
            print(f"[1/4] WARNING: requested node IDs not found: {missing}")
    elif args.node_count:
        pairs = sorted(pairs, key=lambda p: p[0])[:args.node_count]

    if not pairs:
        raise RuntimeError(
            "No NLR node pairs found -- check --nlr-folder, --workload-type, "
            "and --slurm-id are all correct for this run."
        )

    print(f"[1/4] Using {len(pairs)} node(s): {sorted(p[0] for p in pairs)}")
    node_windows = load_nlr_multi(pairs)
    records = build_combined_records(enf, node_windows)

    # NOTE: attack.py currently expects a file literally named
    # "2node.jsonl" in this folder (confirmed 2026-07, not a dynamic
    # convention as far as we know) -- meaning as of right now this
    # pipeline only actually works end-to-end for 2-node runs,
    # regardless of what --node-count is set to. If you run with a
    # different node count, stage 1 will still succeed, but attack.py
    # will silently read the WRONG (or a stale/missing) file unless
    # this naming assumption is confirmed to scale, or attack.py is
    # updated to accept a real --input argument like everything else
    # in this project.
    out_path = Path(args.output_dir) / "2node.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_combined_jsonl(records, str(out_path))
    print(f"[1/4] Wrote {len(records)} records -> {out_path}")
    if len(pairs) != 2:
        print(f"[1/4] WARNING: ingested {len(pairs)} nodes, but attack.py "
              f"currently expects a fixed '2node.jsonl' filename regardless "
              f"of node count -- confirm this actually works for non-2-node runs.")
    return out_path


def stage_2_inject_attacks(clean_path: Path, args) -> Path:
    """
    Confirmed 2026-07: attack.py is invoked with no CLI arguments at
    all -- `python lanes/marchisano_attacks/attack.py`.

    INPUT: currently reads a hardcoded data/combined/2node.jsonl.
    STILL BEING ACTIVELY DEVELOPED on the attack.py side -- eventually
    it will take whatever file is present in data/combined/ instead of
    a fixed name. stage_1 writing to "2node.jsonl" specifically is a
    provisional assumption tied to attack.py's CURRENT behavior, not
    a stable contract -- revisit this once that change lands.

    OUTPUT: written to lanes/marchisano_attacks/outputs/, filename
    varies by which attack type got selected (e.g.
    attack_203_check.jsonl) -- NOT predictable in advance, so this
    can't just be a hardcoded path the way input is. Found instead by
    snapshotting the output directory before running attack.py, then
    diffing after, and picking whatever new file appeared.

    NOTE the "_check" suffix pattern matches the ground-truth-labeled
    files used throughout this project's testing (an "attack" column
    with 0/1 labels). If that column is present in what attack.py
    outputs, it will pass through untouched into anchor_verified.jsonl
    at stage 3 -- Verifier only reads known field names (FRQ, node-
    prefixed channels, index) and ignores anything else, so this is
    harmless functionally, and arguably useful for the scoreboard
    (ground truth sitting right next to our verdict in the same file).
    BUT confirm with whoever owns the scoreboard whether they want
    that ground-truth column present in what they receive, or want it
    stripped first -- not something to decide silently either way.
    """
    import subprocess

    output_dir = _REPO_ROOT / "lanes" / "marchisano_attacks" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    before = set(output_dir.glob("attack_*.jsonl"))

    print(f"[2/4] Running attack.py (reads {clean_path} implicitly, no args passed) ...")
    subprocess.run(
        ["python", "lanes/marchisano_attacks/attack.py"],
        check=True,
        cwd=str(_REPO_ROOT),  # ensures attack.py's own relative paths resolve
                               # correctly regardless of where run_microverse.py
                               # itself was invoked from
    )

    after = set(output_dir.glob("attack_*.jsonl"))
    new_files = after - before

    if not new_files:
        raise FileNotFoundError(
            f"attack.py ran, but no new attack_*.jsonl file appeared in "
            f"{output_dir}. Its output filename varies by attack type and "
            f"couldn't be found automatically this way -- check attack.py's "
            f"actual current behavior."
        )
    if len(new_files) > 1:
        print(f"[2/4] WARNING: multiple new files appeared at once: "
              f"{sorted(new_files)} -- using the most recently modified one, "
              f"but this ambiguity is worth understanding, not just working around.")

    attacked_path = max(new_files, key=lambda p: p.stat().st_mtime)
    print(f"[2/4] attack.py finished -> {attacked_path}")
    return attacked_path


def stage_3_verify_and_annotate(attacked_path: Path, args) -> Path:
    """
    Runs verification, produces the status-annotated JSONL -- same
    logic as verify_file.py (0.0=trusted, 0.5=suspect, 1.0=failed,
    worst-of across every component checked each window).
    """
    from anchor import AnchorExtractor
    from verification import Verifier

    print(f"[3/4] Verifying {attacked_path} ...")
    records = list(read_combined_jsonl(str(attacked_path)))
    enf_list = [r["FRQ"] for r in records]
    extractor = AnchorExtractor(enf=enf_list, sample_rate_hz=0.5)
    verifier = Verifier(component_id=args.component_id, warmup_windows=10, check_nlr=True)

    STATUS_RANK = {"trusted": 0, "suspect": 1, "failed": 2}
    STATUS_SCORE = {"trusted": 0.0, "suspect": 0.5, "failed": 1.0}

    out_path = Path(args.output_dir) / "anchor_verified.jsonl"
    with open(out_path, "w") as out_fh:
        for record in records:
            scoreboard_record = dict(record)
            record = dict(record)
            # See verify_file.py for why this exact conversion matters --
            # AnchorExtractor needs real elapsed seconds, not raw index.
            record["timestamp"] = float(record["index"]) / 0.5
            anchor = extractor.extract(record["timestamp"])
            results = verifier.verify(record, anchor)

            worst = "trusted"
            for result in results:
                if STATUS_RANK[result.status] > STATUS_RANK[worst]:
                    worst = result.status
            scoreboard_record["status"] = STATUS_SCORE[worst]

            out_fh.write(json.dumps(scoreboard_record) + "\n")

    print(f"[3/4] Wrote {len(records)} verified records -> {out_path}")
    return out_path


def stage_4_fork_outputs(verified_path: Path, args) -> None:
    """
    PLACEHOLDER. Sends the verified, annotated JSONL to three
    destinations: scoreboard, dashboard, digital twin.

    Writes to lanes/leiva_verification/outputs/ -- currently three
    file copies to conventionally-named paths. Replace with the real
    dashboard/digital-twin integration once their expected interface
    (file path? socket? HTTP endpoint?) is confirmed with McCray/Baron.
    """
    import shutil
    out_dir = _REPO_ROOT / "lanes" / "leiva_verification" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        "scoreboard": out_dir / "for_scoreboard.jsonl",
        "dashboard": out_dir / "for_dashboard.jsonl",
        "digital_twin": out_dir / "for_digital_twin.jsonl",
    }
    for name, dest in destinations.items():
        shutil.copy(str(verified_path), str(dest))

    print(f"[4/4] Forked output to:")
    for name, dest in destinations.items():
        print(f"       {name:14s} -> {dest}")


def main():
    args = parse_args()
    clean_path = stage_1_ingest_and_smooth(args)
    attacked_path = stage_2_inject_attacks(clean_path, args)
    verified_path = stage_3_verify_and_annotate(attacked_path, args)
    stage_4_fork_outputs(verified_path, args)


if __name__ == "__main__":
    main()