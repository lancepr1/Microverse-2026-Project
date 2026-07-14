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
    Stage 2 (attack injection):   Wired up against attack.py's real
                                   confirmed behavior (--nodes CLI arg,
                                   reads run_{N}node.jsonl, writes both
                                   a plain and a _check file). Only the
                                   plain file (no ground truth) is ever
                                   fed to verification -- see
                                   stage_2_inject_attacks()'s docstring
                                   for why that boundary matters.
    Stage 3 (verify + fork):      REAL, tested, working. Verifies and
                                   writes directly to all three
                                   destinations in one pass -- no
                                   intermediate "anchor_verified.jsonl"
                                   file, since all three destinations
                                   were always identical copies of it
                                   anyway. for_digital_twin.jsonl is now
                                   genuinely consumed by Baron's
                                   main_run.py (stage 4 below). Also
                                   writes runs/<component_id>/verification.jsonl
                                   with real per-node/per-metric detail
                                   for the dashboard's verification_feed.py,
                                   and normalizes node ids in for_dashboard.jsonl
                                   and verification.jsonl to the "node00"
                                   style the dashboard expects -- see this
                                   function's docstring below.
    Stage 4 (launch digital twin): REAL. Launches Blender with
                                   main_run.py, which reads
                                   for_digital_twin.jsonl (just written
                                   by stage 3) and plays it back live,
                                   coloring nodes/ENF green/yellow/red
                                   by verification status. Not yet
                                   tested inside actual Blender from
                                   this end -- main_run.py's own logic
                                   was validated separately, but the
                                   full launch-from-here path hasn't
                                   been run for real yet.

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
import os
import json
import random
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

    # Fixed, conventional locations -- not prompted for anymore. See
    # README.md's "Where to put your data" section for the exact
    # expected layout. Home directory resolved via os.path.expanduser,
    # portable across usernames/machines rather than hardcoded to one
    # person's account.
    home = os.path.expanduser("~")
    datasets_root = Path(home) / "Projects" / "00_raw_datasets"
    if not datasets_root.exists():
        raise RuntimeError(
            f"Expected your raw NLR datasets at {datasets_root}, but that "
            f"folder doesn't exist. See README.md's \"Where to put your "
            f"data\" section -- create that folder and put your dataset "
            f"folders (training_*, inference_*) inside it."
        )
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
    if node_options:
        node_folder_name = prompt_choice(
            f"How many nodes? (options found under {dataset_name})", node_options
        )
        node_folder = dataset_path / node_folder_name
    else:
        # No Nnode subfolders under this dataset (confirmed 2026-07:
        # not every dataset has them the way the two training datasets
        # tested earlier did -- inference_offline_llama3_70b has files
        # directly inside it instead). Use the dataset folder itself.
        # The ACTUAL node count gets determined later from what
        # discover_nlr_pairs() really finds there, in stage_1 -- not
        # guessed from a folder name that may not exist.
        print(f"  (no node-count subfolders under {dataset_name} -- using its files directly)")
        node_folder = dataset_path
        node_folder_name = None

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

    enf_folder = Path(home) / "Projects" / "ENF-ML (CNN+MAMBA)" / "Data"
    if not enf_folder.exists():
        raise RuntimeError(
            f"Expected your ENF data at {enf_folder}, but that folder "
            f"doesn't exist. See README.md's \"Where to put your data\" "
            f"section -- create that folder and put your DevN_ENF_HrNN.csv "
            f"files inside it."
        )
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

    # Generates the independently-noised second ENF stream used by
    # _ENFAlternativeCorrelationCheck in stage_3 -- simulates a
    # genuinely independent sensor reading, NOT a delayed copy (grid
    # frequency is a shared electrical property that updates
    # essentially simultaneously across a synchronized interconnect;
    # what actually differs between two real sensors is independent
    # local measurement noise, not distance-based delay). Generated
    # here, before attack injection ever runs, from this exact clean
    # array -- same "clean upstream of the attacker" principle as
    # combined_smooth() itself, and the same requirement
    # _ENFAlternativeCorrelationCheck's docstring documents. This
    # array is held by the pipeline and passed directly to the
    # verifier in stage_3 -- attack.py never sees it.
    from verification import ENF_ALT_NOISE_STD
    _enf_noise_rng = random.Random(getattr(args, "enf_alt_seed", None))
    args.enf_alternative = [
        v + _enf_noise_rng.gauss(0, ENF_ALT_NOISE_STD) for v in enf
    ]

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

    # Node count comes from what discover_nlr_pairs() ACTUALLY found
    # (len(pairs)) -- not from a folder name, which may not exist at
    # all (confirmed 2026-07: not every dataset has Nnode subfolders,
    # e.g. inference_offline_llama3_70b has files directly inside it
    # instead). This is robust to either folder structure. Stored on
    # args so stage_2 can pass the same real count to attack.py's
    # --nodes without needing to re-derive or guess it.
    #
    # CONFIRMED 2026-07 from reading attack.py's actual source: it
    # constructs exactly f"run_{args.nodes}node.jsonl", where
    # args.nodes comes from its own --nodes CLI argument (default 2).
    args.actual_node_count = len(pairs)
    out_filename = f"run_{args.actual_node_count}node"
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
    limitation. Passed here as args.actual_node_count, which stage_1
    sets from len(pairs) -- the REAL count of nodes discover_nlr_pairs()
    actually found, not a count parsed from a folder name (which may
    not exist at all -- confirmed 2026-07 that not every dataset has
    Nnode subfolders the way the two training datasets tested earlier
    did). Kept in sync with stage_1's output filename since both
    derive from this same real count.

    OUTPUT: writes TWO files per run to lanes/marchisano_attacks/outputs/:
    attack_{id}.jsonl (no ground truth) AND attack_{id}_check.jsonl
    (same data, plus a 0/1 "attack" ground-truth column). {id} varies
    by which scenario got selected (interactively, or randomly for
    medium/hard preset scenarios) -- not knowable in advance, so
    found via before/after directory diffing (globbing
    specifically on "attack_*_check.jsonl" -- the broader
    "attack_*.jsonl" would match both files written each run and
    incorrectly trigger the "multiple new files" ambiguity path every
    single time).

    CRITICAL DESIGN RULE (2026-07, corrected after review): the
    verifier must NEVER be given the ground-truth "attack" column --
    that defeats the entire point of independent verification, even
    though our Verifier class happens to ignore unknown fields today
    (fragile to rely on that staying true). So this function returns
    BOTH paths as a tuple: (plain_path, check_path). Only plain_path
    (found by stripping "_check" off the discovered check_path's name)
    goes anywhere near stage_3/verification. check_path is carried
    forward untouched, used only at the very end for scoring -- see
    stage_4, which sends it to the scoreboard specifically and nowhere
    else.
    """
    import subprocess
    import time

    output_dir = _REPO_ROOT / "lanes" / "marchisano_attacks" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # NOT a before/after set-diff -- FIXED (2026-07) after a real run
    # revealed the bug: Easy Mode produces DETERMINISTIC filenames
    # (e.g. attack_easy_2.jsonl, always the same for the same choices),
    # unlike Medium/Hard's randomized scenario IDs (attack_201.jsonl
    # etc). If a file with that exact name already existed from ANY
    # earlier run, attack.py just silently overwrites it -- it's not
    # "new" by set membership, so a before/after diff finds nothing
    # even though attack.py succeeded and genuinely wrote fresh data.
    # Modification time is robust to this regardless of whether the
    # filename is reused or genuinely new. Small negative buffer on
    # the start time avoids any sub-second clock-precision edge case.
    start_time = time.time() - 1.0

    node_count = args.actual_node_count

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

    new_files = [
        f for f in output_dir.glob("attack_*_check.jsonl")
        if f.stat().st_mtime >= start_time
    ]

    if not new_files:
        raise FileNotFoundError(
            f"attack.py ran, but no attack_*_check.jsonl file in "
            f"{output_dir} was modified during this run. Its output "
            f"filename varies by scenario and couldn't be found "
            f"automatically this way -- check attack.py's actual "
            f"current behavior."
        )
    if len(new_files) > 1:
        print(f"[2/4] WARNING: multiple files modified at once: "
              f"{sorted(new_files)} -- using the most recently modified one, "
              f"but this ambiguity is worth understanding, not just working around.")

    check_path = max(new_files, key=lambda p: p.stat().st_mtime)

    # CRITICAL: the verifier must NEVER see the ground-truth "attack"
    # column -- that would undermine the entire point of independent
    # verification. attack.py writes a PLAIN file alongside the _check
    # one (same scenario ID, no ground truth added) -- that's what
    # gets fed into stage 3, not this _check file. check_path is only
    # used here, internally, to reliably locate plain_path -- Ethan
    # handles ground truth/scoring separately with his own copy of the
    # _check file, so nothing downstream of this function needs it.
    plain_path = check_path.parent / check_path.name.replace("_check.jsonl", ".jsonl")
    if not plain_path.exists():
        raise FileNotFoundError(
            f"Found {check_path} but its plain (no-ground-truth) counterpart "
            f"{plain_path} doesn't exist -- attack.py is expected to write "
            f"both every run. Verification cannot proceed safely without "
            f"the plain file, since the _check file must never be what the "
            f"verifier actually processes."
        )

    print(f"[2/4] attack.py finished -> {plain_path} (fed to verifier)")
    return plain_path


def stage_3_verify_and_fork(attacked_path: Path, args) -> None:
    """
    Runs verification and writes the result DIRECTLY to all three
    destinations in one pass -- no "anchor_verified.jsonl" intermediate
    file, since all three destinations were always going to be
    identical copies of it anyway.

    OUTPUT COLUMNS (2026-07 redesign, replacing the old single
    catch-all "status" field): no overall summary field anymore.
    Instead:
        ENF_status              -- worst-of every ENF-related check
                                    (confidence, drift, raw-value
                                    trend, and the new baseline
                                    comparison)
        {node_id}_status         -- one per node ACTUALLY PRESENT in
                                    this run's data, worst-of just that
                                    node's own metrics (GPU watts/temps,
                                    CPU watts/energy). Variable count --
                                    a 1-node run gets 1 column, a
                                    16-node run gets 16, driven by the
                                    real data, never hardcoded.
    All values use the same 0.0/0.5/1.0 (trusted/suspect/failed)
    encoding as before. Node ID is parsed from each result's
    component_id using the same convention attack.py's own
    scan_telemetry_schema() uses (split on "_gpu"/"_cpu") -- kept
    consistent with the rest of the project rather than inventing a
    new parsing rule.

    Also collects every VerificationResult verifier.verify() returns
    (per-node, per-metric -- see verification.py's merged NLR/GPU
    checks), not just the collapsed worst-of status above, and writes
    them to runs/<component_id>/verification.jsonl so the dashboard's
    verification_feed.py has real per-node detail to read instead of
    falling back to "--" for every sample -- see PIPELINE_SETUP.txt /
    McCray's dashboard lane for the reader side.

    Destinations (lanes/leiva_verification/outputs/):
        for_scoreboard.jsonl
        for_dashboard.jsonl
        for_digital_twin.jsonl
    Replace with the real dashboard/digital-twin integration once
    their expected interface (file path? socket? HTTP endpoint?) is
    confirmed with McCray/Baron.

    NORMALIZATION: the pipeline's node columns are named after whatever
    raw identifier the source data used (e.g. SLURM hostnames like
    "x3105c0s37b0n0"), but the dashboard's node-detection only
    recognizes "node00".."nodeNN" naming (see
    lanes/mccray_dashboard/dashboard/data_feed.py's _NODE_PREFIX_RE).
    Without this step, every dashboard run after a fresh pipeline run
    would show 0 nodes and the Operator tab's per-node callbacks would
    error with "Callback error with no output from input" (no node-card
    components to route the output to). runs/<component_id>/verification.jsonl
    is normalized the same way, using this exact same rename mapping --
    its VerificationResult.component_id values embed the same raw node
    identifiers for_dashboard.jsonl just had renamed, since both come
    from the same source recording. Without this, the dashboard's
    per-node status lookup (verification_feed.py) would never match
    anything, because it looks for the "node00"-style ids, not the raw
    ones.

    Ground truth is never included here -- verifies attacked_path,
    which is the PLAIN file with no "attack" column (see stage_2).
    Ethan maintains his own ground-truth copy for scoring/metrics
    separately -- not this pipeline's concern.
    """
    from anchor import AnchorExtractor
    from verification import Verifier
    from microverse_core.io_records import write_records
    sys.path.insert(0, str(_REPO_ROOT / "lanes" / "mccray_dashboard" / "tools"))
    from normalize_node_ids import normalize as normalize_node_ids
    from normalize_node_ids import normalize_verification_component_ids

    print(f"[3/4] Verifying {attacked_path} ...")
    records = list(read_combined_jsonl(str(attacked_path)))
    enf_list = [r["FRQ"] for r in records]
    extractor = AnchorExtractor(enf=enf_list, sample_rate_hz=0.5)
    verifier = Verifier(
        component_id=args.component_id,
        warmup_windows=10,
        check_nlr=True,
        enf_alternative=getattr(args, "enf_alternative", None),
    )

    STATUS_RANK = {"trusted": 0, "suspect": 1, "failed": 2}
    STATUS_SCORE = {"trusted": 0.0, "suspect": 0.5, "failed": 1.0}

    def node_id_of(component_id: str) -> str:
        """Strips the leading 'rack_00/' prefix, then extracts the node
        ID the same way attack.py's scan_telemetry_schema() does."""
        name = component_id.split("/", 1)[-1]
        if "_gpu" in name:
            return name.split("_gpu")[0]
        if "_cpu" in name:
            return name.split("_cpu")[0]
        return None  # ENF, or anything else not tied to a specific node

    # Determine which nodes are actually present ONCE, from the first
    # record's own field names -- fixed for the whole run, so every
    # output record gets exactly the same set of columns.
    sample_keys = [k for k in records[0].keys() if k not in ("index", "FRQ", "timestamp")]
    node_ids = sorted({
        (k.split("_gpu")[0] if "_gpu" in k else k.split("_cpu")[0])
        for k in sample_keys if "_gpu" in k or "_cpu" in k
    })
    print(f"[3/4] {len(node_ids)} node column(s): {node_ids}")

    out_dir = _REPO_ROOT / "lanes" / "leiva_verification" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        "scoreboard": out_dir / "for_scoreboard.jsonl",
        "dashboard": out_dir / "for_dashboard.jsonl",
        "digital_twin": out_dir / "for_digital_twin.jsonl",
    }
    handles = {name: open(path, "w") for name, path in destinations.items()}

    # Every VerificationResult verifier.verify() returns (per-node,
    # per-metric), not just the collapsed worst-of status written to
    # the destinations above -- collected here and written to
    # runs/<component_id>/verification.jsonl once the loop below finishes.
    all_results = []

    try:
        for record in records:
            scoreboard_record = dict(record)
            record = dict(record)
            # See verify_file.py for why this exact conversion matters --
            # AnchorExtractor needs real elapsed seconds, not raw index.
            record["timestamp"] = float(record["index"]) / 0.5
            anchor = extractor.extract(record["timestamp"])
            results = verifier.verify(record, anchor)
            all_results.extend(results)

            enf_status = "trusted"
            node_status = {nid: "trusted" for nid in node_ids}

            for result in results:
                nid = node_id_of(result.component_id)
                if nid is None:
                    if STATUS_RANK[result.status] > STATUS_RANK[enf_status]:
                        enf_status = result.status
                elif nid in node_status:
                    if STATUS_RANK[result.status] > STATUS_RANK[node_status[nid]]:
                        node_status[nid] = result.status

            scoreboard_record["ENF_status"] = STATUS_SCORE[enf_status]
            for nid in node_ids:
                scoreboard_record[f"{nid}_status"] = STATUS_SCORE[node_status[nid]]

            line = json.dumps(scoreboard_record) + "\n"
            for fh in handles.values():
                fh.write(line)
    finally:
        for fh in handles.values():
            fh.close()

    verification_path = write_records(args.component_id, "verification", all_results)

    rename = normalize_node_ids(destinations["dashboard"], destinations["dashboard"])
    normalize_verification_component_ids(rename, Path(verification_path))

    print(f"[3/4] Wrote {len(records)} verified records to 3 destinations:")
    for name, dest in destinations.items():
        print(f"       {name:14s} -> {dest}")
    print(f"[3/4] Wrote {len(all_results)} per-component verification results -> {verification_path}")
    if rename:
        print(f"[3/4] Normalized {len(rename)} node id(s) in for_dashboard.jsonl "
              f"and verification.jsonl for the dashboard (e.g. {next(iter(rename.items()))})")


def stage_4_launch_digital_twin() -> None:
    """
    Launches Blender with main_run.py (repo root -- Baron's
    lane), which reads for_digital_twin.jsonl (just written by stage 3)
    and plays it back live, one record every 2 seconds, coloring each
    node and the grid anchor green/yellow/red by verification status.

    Deliberately NOT run in --background mode -- that's Blender's
    headless mode with no visible viewport at all, which would defeat
    the entire point of watching the twin update live. Runs with a
    normal, visible Blender window instead.

    main_run.py has its own interactive prompt for the .blend file path
    (remembers the last-used path across runs) -- inherits this
    process's stdin/stdout so that prompt works correctly, same pattern
    already used for attack.py's own interactive prompts in stage_2.
    """
    import subprocess

    print(f"[4/4] Launching Blender with the digital twin ...")
    print(f"[4/4] main_run.py has its own prompt for the .blend file path -- answer that directly below.\n")
    subprocess.run(
        ["blender", "--python", "main_run.py"],
        check=True,
        cwd=str(_REPO_ROOT),
    )


def main():
    args = gather_inputs()
    clean_path = stage_1_ingest_and_smooth(args)
    attacked_path = stage_2_inject_attacks(clean_path, args)
    stage_3_verify_and_fork(attacked_path, args)
    stage_4_launch_digital_twin()


if __name__ == "__main__":
    main()
