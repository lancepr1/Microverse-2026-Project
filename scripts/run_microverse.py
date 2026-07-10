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

USAGE (2026-07 redesign -- fully interactive, no CLI flags to remember):
    python scripts/run_microverse.py

You'll be walked through, in order:
    1. Path to your raw datasets root folder (e.g. .../00_raw_datasets/)
    2. Which dataset to ingest -- listed from whatever folders are
       actually present there (training_*, inference_*)
    3. How many nodes -- listed from whatever node-count folders exist
       under the chosen dataset. This is NOT a fixed list -- different
       datasets genuinely have different options (confirmed:
       training_llama2_70b_lora only has 2/4/8/16node;
       training_stable_diffusion has 1/2/4/8/16node too). Always
       discovered from the real folders present, never hardcoded.
    4. If (and only if) a training dataset was chosen: which SLURM job
       ID -- discovered by scanning the selected node folder's actual
       filenames for the "..._slurmid_XXXXX_..." pattern, since one
       node folder can contain multiple separate training runs.
       Inference datasets skip this entirely -- they don't use one.
    5. Path to the ENF CSV file
    6. Component ID (defaults to rack_00)

Then runs the full pipeline. When it reaches attack.py, that script's
OWN interactive prompts take over the terminal directly -- this
script doesn't try to pass it arguments, just launches it and lets
you answer its prompts as normal.

Workload type matters beyond just labeling the run -- it changes HOW
NLR data is discovered. Training runs use a SLURM job ID to find the
right log files; inference runs use old-style logs with no SLURM ID
at all (discover_nlr_pairs(folder, slurm_id=None)).
"""

import argparse
import json
import re
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


def prompt_choice(prompt_text: str, options: list) -> str:
    """Prints a numbered list, prompts until a valid choice is made."""
    print(f"\n{prompt_text}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        raw = input(f"Enter a number [1-{len(options)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("  Invalid choice, try again.")


def prompt_path(prompt_text: str, default: str = None) -> Path:
    """Prompts for a filesystem path, re-asking until it actually exists."""
    suffix = f" [{default}]" if default else ""
    while True:
        raw = input(f"{prompt_text}{suffix}: ").strip()
        if not raw and default:
            raw = default
        path = Path(raw).expanduser()
        if path.exists():
            return path
        print(f"  Path not found: {path} -- try again.")


def _node_sort_key(folder_name: str) -> int:
    """Sorts '1node'/'2node'/'4node'/... numerically, not alphabetically
    (alphabetical would put '16node' before '2node')."""
    digits = re.sub(r"\D", "", folder_name)
    return int(digits) if digits else 0


def discover_slurm_ids(node_folder: Path) -> list:
    """
    Scans filenames in node_folder for the 'slurmid_XXXXX_' pattern
    (matches the naming convention used throughout this project, e.g.
    nvml_wattameter_emissions_parsed_slurmid_10742842_node_...) and
    returns the distinct IDs actually present, sorted. A training
    dataset's node folder can contain multiple separate runs (multiple
    SLURM jobs), which is why this is a selection, not a single fixed
    value.
    """
    pattern = re.compile(r"slurmid_(\d+)_")
    ids = set()
    for f in node_folder.iterdir():
        m = pattern.search(f.name)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def discover_enf_devices(enf_folder: Path) -> dict:
    """
    Scans enf_folder for files matching 'DevN_ENF_HrNN.csv' and returns
    {device_label: {hour_label: filepath}}, e.g.
        {"Dev1": {"Hr01": Path(...), "Hr02": Path(...), ...}, "Dev2": {...}}
    Devices and hours are whatever's actually present -- not assumed
    to be Dev1-3 or any fixed hour range.
    """
    pattern = re.compile(r"^(Dev\d+)_ENF_(Hr\d+)\.csv$", re.IGNORECASE)
    devices: dict = {}
    for f in enf_folder.iterdir():
        m = pattern.match(f.name)
        if m:
            dev, hr = m.group(1), m.group(2)
            devices.setdefault(dev, {})[hr] = f
    return devices


def gather_inputs():
    """
    Interactive replacement for CLI flags -- walks the user through:
    which dataset, how many nodes (options differ per dataset -- e.g.
    training_llama2_70b_lora only has 2/4/8/16node, while
    training_stable_diffusion has 1/2/4/8/16node -- discovered from
    the real folders present, never hardcoded), and for training
    datasets specifically, which SLURM job ID (inference datasets
    don't use one at all).
    """
    print("=" * 70)
    print("Microverse-2026-Project -- Data Ingestion")
    print("=" * 70)

    datasets_root = prompt_path("Path to your raw datasets folder")
    dataset_options = sorted(p.name for p in datasets_root.iterdir() if p.is_dir())
    if not dataset_options:
        raise RuntimeError(f"No dataset folders found in {datasets_root}")

    dataset_name = prompt_choice("Which dataset do you want to ingest?", dataset_options)
    dataset_path = datasets_root / dataset_name
    workload_type = "training" if "training" in dataset_name.lower() else "inference"
    print(f"  -> workload_type = {workload_type}")

    node_options = sorted(
        (p.name for p in dataset_path.iterdir() if p.is_dir()),
        key=_node_sort_key,
    )
    if not node_options:
        raise RuntimeError(f"No node-count folders found in {dataset_path}")

    node_folder_name = prompt_choice(
        f"How many nodes? (options found under {dataset_name})", node_options
    )
    node_folder = dataset_path / node_folder_name

    slurm_id = None
    if workload_type == "training":
        slurm_ids = discover_slurm_ids(node_folder)
        if not slurm_ids:
            raise RuntimeError(
                f"No SLURM job IDs found in {node_folder} -- expected filenames "
                f"matching '..._slurmid_XXXXX_...'"
            )
        slurm_id = prompt_choice(
            f"Which SLURM job ID? (found in {node_folder_name})", slurm_ids
        )
    else:
        print("  (no SLURM ID needed -- inference datasets don't use one)")

    enf_folder = prompt_path("\nPath to your ENF data folder")
    devices = discover_enf_devices(enf_folder)
    if not devices:
        raise RuntimeError(
            f"No files matching 'DevN_ENF_HrNN.csv' found in {enf_folder}"
        )

    device_options = sorted(devices.keys(), key=_node_sort_key)
    device = prompt_choice("Which recording device?", device_options)

    hour_options = sorted(devices[device].keys(), key=_node_sort_key)
    hour = prompt_choice(f"Which hour? (found for {device})", hour_options)
    enf_path = devices[device][hour]
    print(f"  -> using {enf_path.name}")

    # component_id is just a LABEL for which simulated rack this run
    # represents -- it's not read from the ENF file or the NLR data,
    # and doesn't need to relate to which Dev/hour was picked above.
    # Every verification result gets prefixed with it (e.g.
    # "rack_00/ENF", "rack_00/x3102c0s25b0n0_cpu-0[W]") so results from
    # different racks can be told apart if this project ever verifies
    # more than one at once. Almost always just the default.
    print(
        "\nComponent ID -- a label for which simulated rack this run "
        "represents (not related to the ENF device/hour you just picked). "
        "Every result gets prefixed with it, e.g. 'rack_00/ENF'. "
        "Leave blank for the default unless you're specifically "
        "simulating more than one rack."
    )
    component_id = input("Component ID [rack_00]: ").strip() or "rack_00"

    args = argparse.Namespace(
        workload_type=workload_type,
        nlr_folder=str(node_folder),
        node_folder_name=node_folder_name,
        slurm_id=slurm_id,
        enf_path=str(enf_path),
        component_id=component_id,
        node_count=None,  # already resolved by folder selection -- discover_nlr_pairs
        node_ids=None,    # will naturally find exactly what's in node_folder
        output_dir=str(_REPO_ROOT / "data" / "combined"),
    )
    print()
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

    # Named after the actual node folder the user selected (e.g.
    # "run_4node.jsonl", "run_1node.jsonl") -- traceable to the real
    # choice made in gather_inputs(), rather than a hardcoded guess.
    #
    # CONFIRMED 2026-07 from reading attack.py's actual source: it
    # constructs exactly f"run_{args.nodes}node.jsonl", where
    # args.nodes comes from its own --nodes CLI argument (default 2).
    # stage_2 passes --nodes matching this exact filename, so any
    # node count works correctly now -- no longer limited to 2-node
    # runs the way earlier versions of this pipeline assumed.
    out_filename = f"run_{getattr(args, 'node_folder_name', None) or '2node'}"
    out_path = Path(args.output_dir) / f"{out_filename}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_combined_jsonl(records, str(out_path))
    print(f"[1/4] Wrote {len(records)} records -> {out_path}")
    return out_path


def stage_2_inject_attacks(clean_path: Path, args) -> Path:
    """
    Confirmed 2026-07 from reading attack.py's actual source:

    INPUT: takes a real CLI argument, --nodes N (default 2), used to
    construct exactly f"run_{N}node.jsonl" in data/combined/. This
    generalizes cleanly -- no more "only works for 2-node runs"
    limitation. Passed here as whatever node count was actually
    selected in gather_inputs(), kept in sync with stage_1's output
    filename since both derive from the same node_folder_name.

    OUTPUT: writes TWO files per run to lanes/marchisano_attacks/outputs/:
    attack_{id}.jsonl (no ground truth) AND attack_{id}_check.jsonl
    (same data, plus a 0/1 "attack" ground-truth column). {id} varies
    by which scenario got selected (interactively, or randomly for
    medium/hard preset scenarios) -- genuinely unpredictable in
    advance, so still found via before/after directory diffing.

    IMPORTANT FIX: must specifically glob "attack_*_check.jsonl", not
    the broader "attack_*.jsonl" -- the broad pattern matches BOTH
    files written each run, which would incorrectly trigger the
    "multiple new files" ambiguity path on every single run. The
    _check version is deliberately the one used here (not the plain
    version) -- it carries ground truth alongside our verdict, which
    is genuinely useful for the scoreboard, though worth confirming
    with whoever owns it whether they want that ground-truth column
    visible or stripped before it reaches them.
    """
    import subprocess

    output_dir = _REPO_ROOT / "lanes" / "marchisano_attacks" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    before = set(output_dir.glob("attack_*_check.jsonl"))

    node_count = _node_sort_key(getattr(args, "node_folder_name", None) or "2node")

    print(f"[2/4] Launching attack.py --nodes {node_count} "
          f"(reads {clean_path}) ...")
    print(f"[2/4] attack.py has its own interactive prompts -- answer those directly below.\n")
    subprocess.run(
        ["python", "lanes/marchisano_attacks/attack.py", "--nodes", str(node_count)],
        check=True,
        cwd=str(_REPO_ROOT),  # ensures attack.py's own relative paths resolve
                               # correctly regardless of where run_microverse.py
                               # itself was invoked from
    )

    after = set(output_dir.glob("attack_*_check.jsonl"))
    new_files = after - before

    if not new_files:
        raise FileNotFoundError(
            f"attack.py ran, but no new attack_*_check.jsonl file appeared in "
            f"{output_dir}. Its output filename varies by scenario and "
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
    from microverse_core.io_records import write_records

    print(f"[3/4] Verifying {attacked_path} ...")
    records = list(read_combined_jsonl(str(attacked_path)))
    enf_list = [r["FRQ"] for r in records]
    extractor = AnchorExtractor(enf=enf_list, sample_rate_hz=0.5)
    verifier = Verifier(component_id=args.component_id, warmup_windows=10, check_nlr=True)

    STATUS_RANK = {"trusted": 0, "suspect": 1, "failed": 2}
    STATUS_SCORE = {"trusted": 0.0, "suspect": 0.5, "failed": 1.0}

    # Every VerificationResult verifier.verify() returns (per-node,
    # per-metric -- see verification.py's merged NLR/GPU checks), not just
    # the collapsed worst-of status below. Collected here and written to
    # runs/<component_id>/verification.jsonl so the dashboard's
    # verification_feed.py has real per-node detail to read instead of
    # falling back to "--" for every sample -- see PIPELINE_SETUP.txt /
    # McCray's dashboard lane for the reader side.
    all_results = []

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
            all_results.extend(results)

            worst = "trusted"
            for result in results:
                if STATUS_RANK[result.status] > STATUS_RANK[worst]:
                    worst = result.status
            scoreboard_record["status"] = STATUS_SCORE[worst]

            out_fh.write(json.dumps(scoreboard_record) + "\n")

    verification_path = write_records(args.component_id, "verification", all_results)

    print(f"[3/4] Wrote {len(records)} verified records -> {out_path}")
    print(f"[3/4] Wrote {len(all_results)} per-component verification results -> {verification_path}")
    return out_path


def stage_4_fork_outputs(verified_path: Path, args) -> None:
    """
    PLACEHOLDER. Sends the verified, annotated JSONL to three
    destinations: scoreboard, dashboard, digital twin.

    Writes to lanes/leiva_verification/outputs/ -- currently three
    file copies to conventionally-named paths. Replace with the real
    dashboard/digital-twin integration once their expected interface
    (file path? socket? HTTP endpoint?) is confirmed with McCray/Baron.

    The dashboard copy is additionally normalized in place: the pipeline's
    node columns are named after whatever raw identifier the source data
    used (e.g. SLURM hostnames like "x3105c0s37b0n0"), but the dashboard's
    node-detection only recognizes "node00".."nodeNN" naming (see
    lanes/mccray_dashboard/dashboard/data_feed.py's _NODE_PREFIX_RE). Without
    this step every dashboard run after a fresh pipeline run would show 0
    nodes and the Operator tab's per-node callbacks would error with
    "Callback error with no output from input" (no node-card components to
    route the output to).

    runs/<component_id>/verification.jsonl (written by
    stage_3_verify_and_annotate) is normalized the same way, using this
    exact same rename mapping -- its VerificationResult.component_id values
    embed the same raw node identifiers for_dashboard.jsonl just had
    renamed, since both come from the same source recording. Without this,
    the dashboard's per-node status lookup (verification_feed.py) would
    never match anything, because it looks for the "node00"-style ids, not
    the raw ones.
    """
    import shutil
    import sys
    sys.path.insert(0, str(_REPO_ROOT / "lanes" / "mccray_dashboard" / "tools"))
    from normalize_node_ids import normalize as normalize_node_ids
    from normalize_node_ids import normalize_verification_component_ids

    out_dir = _REPO_ROOT / "lanes" / "leiva_verification" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        "scoreboard": out_dir / "for_scoreboard.jsonl",
        "dashboard": out_dir / "for_dashboard.jsonl",
        "digital_twin": out_dir / "for_digital_twin.jsonl",
    }
    for name, dest in destinations.items():
        shutil.copy(str(verified_path), str(dest))

    rename = normalize_node_ids(destinations["dashboard"], destinations["dashboard"])
    verification_path = _REPO_ROOT / "runs" / args.component_id / "verification.jsonl"
    normalize_verification_component_ids(rename, verification_path)

    print(f"[4/4] Forked output to:")
    for name, dest in destinations.items():
        print(f"       {name:14s} -> {dest}")
    if rename:
        print(f"[4/4] Normalized {len(rename)} node id(s) in for_dashboard.jsonl "
              f"for the dashboard (e.g. {next(iter(rename.items()))})")


def main():
    args = gather_inputs()
    clean_path = stage_1_ingest_and_smooth(args)
    attacked_path = stage_2_inject_attacks(clean_path, args)
    verified_path = stage_3_verify_and_annotate(attacked_path, args)
    stage_4_fork_outputs(verified_path, args)


if __name__ == "__main__":
    main()