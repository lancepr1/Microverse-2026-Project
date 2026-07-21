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
                                   for why that boundary matters. Now
                                   also returns check_path (previously
                                   discarded after internal use), since
                                   stage 4 below needs it.
    Stage 3 (verify + fork):      REAL, tested, working. Verifies and
                                   writes directly to all three
                                   destinations in one pass -- no
                                   intermediate "anchor_verified.jsonl"
                                   file, since all three destinations
                                   were always identical copies of it
                                   anyway. for_digital_twin.jsonl is
                                   genuinely consumed by Baron's
                                   main_run.py (stage 6 below).
                                   for_dashboard.jsonl is consumed
                                   DIRECTLY by McCray's dashboard --
                                   data_feed.py reads its raw hostname
                                   node-id columns as-is now, no
                                   normalization step exists anywhere in
                                   this pipeline.
    Stage 4 (evaluate detection): NEW (2026-07) -- a different thing
                                   from the old, now-removed Stage 4 that
                                   used to live at this number
                                   (node-id normalization; see prior
                                   git history, not repeated here). This
                                   one runs scripts/metrics.py against
                                   this run's own ground truth
                                   (check_path, from stage 2) and
                                   for_scoreboard.jsonl (from stage 3),
                                   printing a precision/recall/F1/FPR/
                                   time-to-detection report straight to
                                   the console. This is now deliberately
                                   the ONLY console output this pipeline
                                   produces by default -- see VERBOSE
                                   below, which silences every other
                                   stage's routine narration so a run's
                                   console output IS this evaluation
                                   report, not buried under step-by-step
                                   status. metrics.py's assumed location
                                   (scripts/metrics.py) has not been
                                   confirmed against the real repo layout
                                   yet -- see stage_4_evaluate_detection()'s
                                   own docstring.
    Stage 5 (launch dashboard):   Launches McCray's Dash app
                                   (lanes/mccray_dashboard/dashboard/
                                   main.py) as a background process --
                                   non-blocking, since stage 6 (Blender)
                                   blocks until the viewport window is
                                   closed and both need to run at once.
                                   Terminated in main()'s finally block
                                   when Blender exits, so it never
                                   orphans.
    Stage 6 (launch digital twin): REAL. Launches Blender with
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

CHANGED (2026-07): all data now lives in a fixed, repo-relative location
-- data/rawdata/ at the repo root, the same path on every teammate's
machine regardless of OS or username, not committed to git (space
constraints). No path typing needed for any of it; you're walked
through, in order:
    1. Which dataset to ingest -- listed from whatever folders are
       actually present under data/rawdata/00_raw_datasets/
       (training_*, inference_*)
    2. How many nodes -- listed from whatever node-count folders exist
       under the chosen dataset. This is NOT a fixed list -- different
       datasets genuinely have different options (confirmed:
       training_llama2_70b_lora only has 2/4/8/16node;
       training_stable_diffusion has 1/2/4/8/16node too). Always
       discovered from the real folders present, never hardcoded.
    3. If (and only if) a training dataset was chosen: which SLURM job
       ID -- discovered by scanning the selected node folder's actual
       filenames for the "..._slurmid_XXXXX_..." pattern, since one
       node folder can contain multiple separate training runs.
       Inference datasets skip this entirely -- they don't use one.
    4. Which ENF recording device and hour -- listed from whatever
       DevN_ENF_HrNN.csv files are actually present under
       data/rawdata/ENF-ML (CNN+MAMBA)/Data/
    5. Component ID (defaults to rack_00)

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
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

# Lives in scripts/run_microverse.py -- one level below repo root, so
# .parent.parent (not .parent) is needed to reach the repo root where
# microverse_core/ and lanes/ actually are.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "lanes" / "leiva_verification"))

# McCray's dashboard lane -- _DASHBOARD_DIR is where main.py/data_feed.py
# live (confirmed by triangulating data_feed.py's own "../../leiva_verification"
# path math against the repo-root math the (now-deleted)
# tools/generate_verification.py used, not guessed). Never imported here --
# McCray's dashboard package must stay importable standalone with no
# sibling-lane code present (see that package's own coupling-guard test),
# so this script only ever launches it as a subprocess.
#
# REMOVED (2026-07, cleanup pass): _DASHBOARD_TOOLS_DIR and its sys.path
# insert. This script used to import tools/normalize_node_ids.py for a
# now-deleted node-id normalization stage -- data_feed.py discovers real
# node ids directly from for_dashboard.jsonl now, so there's nothing left
# in lanes/mccray_dashboard/tools/ for this script to call at all (that
# whole directory has been deleted from the dashboard lane).
_MCCRAY_DASHBOARD_ROOT = _REPO_ROOT / "lanes" / "mccray_dashboard"
_DASHBOARD_DIR = _MCCRAY_DASHBOARD_ROOT / "dashboard"

from microverse_core.data_loaders import (
    load_enf,
    combined_smooth,
    discover_nlr_pairs,
    load_nlr_multi,
    build_combined_records,
    write_combined_jsonl,
    read_combined_jsonl,
)

# CHANGED (2026-07): the "[N/5] doing X ..." progress narration scattered
# through every stage below was genuinely useful while the dashboard
# wasn't wired up yet -- it was the only visibility into what the
# pipeline was doing. Now that the dashboard shows live results as soon
# as it launches, that narration is redundant console noise. Gated behind
# VERBOSE (default off) instead of deleted outright, so a future bug can
# get that visibility back with a one-line flip instead of re-adding
# print statements throughout the file. WARNING lines, the interactive
# prompts in gather_inputs(), and the dashboard-shutdown notice at the
# end of main() are deliberately NOT gated -- those indicate a real
# problem, are part of the actual interactive UI, or explain an
# otherwise-silent side effect, not routine status.
VERBOSE = False


def vprint(*args, **kwargs) -> None:
    if VERBOSE:
        print(*args, **kwargs)


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


# REMOVED (2026-07): prompt_path() used to live here -- a free-text path
# prompt for the raw datasets/ENF folders. Confirmed unused anywhere in
# this file once those became fixed, auto-discovered locations under
# data/rawdata/ (see gather_inputs() below) -- dead code, removed rather
# than left behind to avoid implying paths are still user-typed anywhere.


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

    # CHANGED (2026-07): was Path(home) / "Projects" / "00_raw_datasets" --
    # home-directory-relative, meaning every teammate had to create a
    # "Projects" folder in their own home directory and put data there
    # by hand. Now repo-relative (data/rawdata/), matching the actual
    # folder structure everyone's local clone has -- works identically
    # on Windows/Mac/Linux with zero per-person setup, since _REPO_ROOT
    # is self-locating (see its own comment above) and pathlib handles
    # path separators correctly on every OS. Data lives in this folder
    # locally on every machine but is NOT committed to git (space
    # constraints) -- see .gitignore.
    datasets_root = _REPO_ROOT / "data" / "rawdata" / "00_raw_datasets"
    if not datasets_root.exists():
        raise RuntimeError(
            f"Expected your raw NLR datasets at {datasets_root}, but that "
            f"folder doesn't exist. See README.md's \"Where to put your "
            f"data\" section -- create data/rawdata/00_raw_datasets/ at "
            f"the repo root and put your dataset folders (training_*, "
            f"inference_*) inside it."
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

    # CHANGED (2026-07): same repo-relative move as datasets_root above.
    enf_folder = _REPO_ROOT / "data" / "rawdata" / "ENF-ML (CNN+MAMBA)" / "Data"
    if not enf_folder.exists():
        raise RuntimeError(
            f"Expected your ENF data at {enf_folder}, but that folder "
            f"doesn't exist. See README.md's \"Where to put your data\" "
            f"section -- create data/rawdata/ENF-ML (CNN+MAMBA)/Data/ at "
            f"the repo root and put your DevN_ENF_HrNN.csv files inside it."
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
    vprint(f"[1/5] Ingesting ENF from {args.enf_path} ...")
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

    vprint(f"[1/5] Discovering NLR pairs in {args.nlr_folder} "
          f"(workload_type={args.workload_type}) ...")
    slurm_id = args.slurm_id if args.workload_type == "training" else None
    pairs = discover_nlr_pairs(args.nlr_folder, slurm_id=slurm_id)

    if args.node_ids:
        pairs = [p for p in pairs if p[0] in args.node_ids]
        missing = set(args.node_ids) - {p[0] for p in pairs}
        if missing:
            vprint(f"[1/5] WARNING: requested node IDs not found: {missing}")
    elif args.node_count:
        pairs = sorted(pairs, key=lambda p: p[0])[:args.node_count]

    if not pairs:
        raise RuntimeError(
            "No NLR node pairs found -- check --nlr-folder, --workload-type, "
            "and --slurm-id are all correct for this run."
        )

    vprint(f"[1/5] Using {len(pairs)} node(s): {sorted(p[0] for p in pairs)}")
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
    vprint(f"[1/5] Wrote {len(records)} records -> {out_path}")
    return out_path


def stage_2_inject_attacks(clean_path: Path, args) -> tuple[Path, Path]:
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
    medium/hard preset scenarios) -- genuinely unpredictable in
    advance, so found via before/after directory diffing (globbing
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
    goes anywhere near stage_3/verification.

    CHANGED (2026-07): check_path used to be discarded after being used
    internally to locate plain_path -- now actually returned, since
    stage_4 (metrics evaluation, new this week) needs it to find the
    matching ground-truth file for scoring. Still never touches
    verification itself -- stage_3 only ever receives plain_path.
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

    vprint(f"[2/6] Launching attack.py --nodes {node_count} "
          f"(reads {clean_path}) ...")
    vprint(f"[2/6] attack.py has its own interactive prompts -- answer those directly below.\n")
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
        vprint(f"[2/6] WARNING: multiple files modified at once: "
              f"{sorted(new_files)} -- using the most recently modified one, "
              f"but this ambiguity is worth understanding, not just working around.")

    check_path = max(new_files, key=lambda p: p.stat().st_mtime)

    # CRITICAL: the verifier must NEVER see the ground-truth "attack"
    # column -- that would undermine the entire point of independent
    # verification. attack.py writes a PLAIN file alongside the _check
    # one (same scenario ID, no ground truth added) -- that's what
    # gets fed into stage 3, not this _check file. check_path is
    # carried forward (see CHANGED note above) purely for stage 4's
    # scoring use -- nothing else downstream of stage 2 touches it.
    plain_path = check_path.parent / check_path.name.replace("_check.jsonl", ".jsonl")
    if not plain_path.exists():
        raise FileNotFoundError(
            f"Found {check_path} but its plain (no-ground-truth) counterpart "
            f"{plain_path} doesn't exist -- attack.py is expected to write "
            f"both every run. Verification cannot proceed safely without "
            f"the plain file, since the _check file must never be what the "
            f"verifier actually processes."
        )

    vprint(f"[2/6] attack.py finished -> {plain_path} (fed to verifier)")
    return plain_path, check_path


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

    PLACEHOLDER destinations -- currently just three real files with
    identical content:
        for_scoreboard.jsonl
        for_dashboard.jsonl
        for_digital_twin.jsonl
    Replace with the real dashboard/digital-twin integration once
    their expected interface (file path? socket? HTTP endpoint?) is
    confirmed with McCray/Baron.

    Ground truth is never included here -- verifies attacked_path,
    which is the PLAIN file with no "attack" column (see stage_2).
    Ethan maintains his own ground-truth copy for scoring/metrics
    separately -- not this pipeline's concern.
    """
    from anchor import AnchorExtractor
    from verification import Verifier

    vprint(f"[3/6] Verifying {attacked_path} ...")
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
    vprint(f"[3/6] {len(node_ids)} node column(s): {node_ids}")

    out_dir = _REPO_ROOT / "lanes" / "leiva_verification" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        "scoreboard": out_dir / "for_scoreboard.jsonl",
        "dashboard": out_dir / "for_dashboard.jsonl",
        "digital_twin": out_dir / "for_digital_twin.jsonl",
    }
    handles = {name: open(path, "w") for name, path in destinations.items()}

    try:
        for record in records:
            scoreboard_record = dict(record)
            record = dict(record)
            # See verify_file.py for why this exact conversion matters --
            # AnchorExtractor needs real elapsed seconds, not raw index.
            record["timestamp"] = float(record["index"]) / 0.5
            anchor = extractor.extract(record["timestamp"])
            results = verifier.verify(record, anchor)

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

    vprint(f"[3/6] Wrote {len(records)} verified records to 3 destinations:")
    for name, dest in destinations.items():
        vprint(f"       {name:14s} -> {dest}")


def stage_4_evaluate_detection(check_path: Path, args) -> None:
    """
    NEW (2026-07). Runs the scoring/evaluation tool against this run's
    own ground-truth file (check_path, from stage 2) and
    for_scoreboard.jsonl (just written by stage 3), letting its
    multi-layer precision/recall/F1/FPR/time-to-detection report print
    straight to the console. This is now deliberately the ONLY thing
    this pipeline prints to the console by default -- see VERBOSE at
    the top of this file, which now gates every other stage's routine
    narration. The intent: a run's console output should BE this
    evaluation report, not routine step-by-step status buried around
    it.

    CONFIRMED LOCATION (2026-07): microverse_core/metrics.py -- fixes an
    earlier wrong guess (scripts/metrics.py, which doesn't exist; the
    stage silently found nothing and skipped, and its own "not found"
    warning was mistakenly gated behind vprint too, so a real run could
    finish with zero indication the report never ran at all). Lines up
    with microverse_core/__init__.py's own docstring, which already
    listed "the scoring metrics" among what that package owns --
    should have been the first place checked. The not-found warning
    below is now a plain, always-visible print(), not vprint() -- a
    missing evaluation report is exactly the kind of thing that must
    never fail silently, unlike the routine step narration VERBOSE
    gates everywhere else in this file.

    Scenario ID is derived directly from check_path's own filename
    (e.g. "attack_304_check.jsonl" -> "304",
    "attack_easy_2_check.jsonl" -> "easy_2") -- exactly the format
    metrics.py's own resolve_project_paths() already expects (its own
    docstring gives "304" or "easy_1" as examples), so this reuses
    metadata attack.py already encoded in the filename rather than
    re-deriving or guessing scenario identity a second time.

    Deliberately does NOT use check=True / propagate a failure here --
    a missing or malformed scoreboard file failing evaluation shouldn't
    block the dashboard or digital twin from still launching after it;
    those are independent concerns from scoring. metrics.py's own
    main() already prints a clear error and exits non-zero on a real
    problem (missing files, row-count mismatch, index misalignment) --
    that output is not suppressed, so a failure here is still visible,
    just not fatal to the rest of the run.
    """
    _METRICS_SCRIPT = _REPO_ROOT / "microverse_core" / "metrics.py"

    scenario_id = check_path.name.removeprefix("attack_").removesuffix("_check.jsonl")

    if not _METRICS_SCRIPT.exists():
        print(f"[4/6] WARNING: metrics.py not found at {_METRICS_SCRIPT} -- "
              f"skipping the evaluation report. If it's been moved, update "
              f"_METRICS_SCRIPT's location in stage_4_evaluate_detection().")
        return

    subprocess.run(
        [sys.executable, str(_METRICS_SCRIPT), "--id", scenario_id],
        cwd=str(_REPO_ROOT),
    )


def stage_5_launch_dashboard():
    """
    Launches McCray's Dash app (lanes/mccray_dashboard/dashboard/main.py)
    as a background process -- non-blocking, via Popen rather than run(),
    since stage 5 (Blender) blocks the main thread until its viewport
    window is closed and both processes need to be alive at the same
    time. Returns the Popen handle so main() can terminate it cleanly
    once Blender exits, rather than leaving an orphaned dashboard server
    running after the pipeline itself has finished.

    Launched with cwd=_DASHBOARD_DIR (not _REPO_ROOT) -- main.py's own
    imports (ui.layout, data_feed, etc.) are relative to that directory,
    the same way attack.py's and main_run.py's relative paths require
    cwd=_REPO_ROOT for THEM specifically in stages 2 and 5.

    REMOVED (2026-07, cleanup pass): the normalization stage that used to
    run before this one is gone entirely -- data_feed.py now discovers
    real node ids directly from for_dashboard.jsonl's own raw hostname
    column prefixes (e.g. "x3105c0s37b0n0_gpu-0[W]"), so there's nothing
    left for this pipeline to prepare before launching the dashboard.
    tools/normalize_node_ids.py, tools/generate_verification.py, and
    verification_feed.py have all been deleted from the dashboard lane to
    match (not just marked obsolete anymore).

    FIXED (2026-07, after a real run): a Popen() call returning
    successfully only means the process SPAWNED, not that it stayed
    alive -- a real run hit this exact gap: main.py crashed immediately
    on `ModuleNotFoundError: No module named 'dash'` (dependencies not
    installed in this environment), but the pipeline printed "Dashboard
    launched" anyway and carried on into Blender with a dead dashboard
    process in the background. DASHBOARD_STARTUP_GRACE_S gives the
    process a moment to either crash (common: missing deps, port already
    held by a stale process from a previous run -- see McCray's own
    dashboard README step 11) or survive; if it's already dead by then,
    this raises instead of returning a stale, misleading "success".
    """
    DASHBOARD_STARTUP_GRACE_S = 1.5

    vprint(f"[5/6] Launching dashboard (lanes/mccray_dashboard/dashboard/main.py) ...")
    process = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=str(_DASHBOARD_DIR),
    )

    time.sleep(DASHBOARD_STARTUP_GRACE_S)
    if process.poll() is not None:
        raise RuntimeError(
            f"[5/6] Dashboard process exited immediately (code "
            f"{process.returncode}) -- see the traceback printed above "
            f"for the real cause. Two common ones: (1) dashboard "
            f"dependencies aren't installed in this environment -- "
            f"`source .venv/bin/activate && pip install -r requirements.txt` "
            f"from the repo root, then re-run; (2) a stale dashboard "
            f"process from a previous run is still holding the port -- "
            f"`pkill -9 -f main.py` first, then re-run."
        )

    vprint(f"[5/6] Dashboard launched (pid={process.pid}) -- see main.py's own "
          f"app.run() call for the port (Dash defaults to 127.0.0.1:8050 "
          f"unless that's been changed).")
    return process


def _resolve_blend_file() -> Path:
    """
    Scans data/rawdata/ for a .blend file, prompting if more than one
    exists. MOVED here (2026-07) from main_run.py's own
    _find_blend_file() -- necessary once stage_6 below started piping
    and filtering Blender's stdout in real time to strip out its
    per-tick render engine noise (see that function's own comment for
    why). input()'s prompt text has no trailing newline, so a
    line-buffered read loop over a piped stdout would never flush it
    to the terminal -- Blender would sit there waiting for a response
    to a question you can't see, a silent deadlock, not a hypothetical
    risk. Resolving the choice HERE, in this process's own normal
    terminal (never piped), and handing the result to Blender via
    MICROVERSE_BLEND_FILE (main_run.py already checks this env var
    first, before its own scan) means main_run.py never needs
    interactive input through that pipe at all. main_run.py's own
    _find_blend_file() is left in place as a fallback for anyone
    running `blender --python main_run.py` directly, standalone,
    outside this pipeline -- just never reached when launched from
    here.
    """
    blend_dir = _REPO_ROOT / "data" / "rawdata"
    if not blend_dir.is_dir():
        raise RuntimeError(
            f"Expected {blend_dir} to exist -- see README.md's "
            f"\"Where to put your data\" section. Put your .blend file "
            f"directly inside data/rawdata/ at the repo root."
        )

    blend_files = sorted(f for f in blend_dir.iterdir() if f.suffix.lower() == ".blend")

    if not blend_files:
        raise RuntimeError(
            f"No .blend file found in {blend_dir}. See README.md's "
            f"\"Where to put your data\" section -- put your .blend "
            f"file directly inside data/rawdata/ at the repo root."
        )

    if len(blend_files) == 1:
        return blend_files[0]

    print(f"\nMultiple .blend files found in {blend_dir}:")
    for i, f in enumerate(blend_files, 1):
        print(f"  {i}. {f.name}")
    while True:
        raw = input(f"Which one? [1-{len(blend_files)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(blend_files):
            return blend_files[int(raw) - 1]
        print("  Invalid choice, try again.")


def stage_6_launch_digital_twin() -> None:
    """
    Launches Blender with main_run.py (repo root -- Baron's
    lane), which reads for_digital_twin.jsonl (just written by stage 3)
    and plays it back live, one record every 2 seconds, coloring each
    node and the grid anchor green/yellow/red by verification status.

    Deliberately NOT run in --background mode -- that's Blender's
    headless mode with no visible viewport at all, which would defeat
    the entire point of watching the twin update live. Runs with a
    normal, visible Blender window instead.

    CHANGED (2026-07): .blend file selection now happens HERE, before
    Blender is even launched, via _resolve_blend_file() above -- see
    its own docstring for why. The result is passed through
    MICROVERSE_BLEND_FILE so main_run.py's own scan is skipped
    entirely.

    CHANGED AGAIN (2026-07): switched from subprocess.run (inherits
    stdout directly) to Popen with a real-time line filter. Blender's
    own internal render engine logs two lines on EVERY viewport capture
    tick ("Saved: '...'" / "OpenGL Render written to '...'") --
    completely independent of anything in main_run.py's own print
    statements (already gated behind that file's own VERBOSE flag --
    this is a SEPARATE noise source, Blender's engine itself, not our
    code, and could never have been silenced by fixing our own print()
    calls). Filtered here line-by-line rather than redirecting stdout
    to DEVNULL entirely, since that blunt approach would also swallow
    genuine Blender errors/tracebacks along with the noise -- everything
    else still passes through live, unfiltered, the moment each line
    arrives. Bypassed (shows everything, unfiltered) when VERBOSE=True,
    same override every other stage in this file already respects.

    Still fundamentally a BLOCKING step -- process.wait() below still
    holds this function open until Blender's window is closed, exactly
    like the old subprocess.run(check=True) did, so the pipeline still
    "finishes" exactly when the person closes the Blender window.
    """
    vprint(f"[6/6] Launching Blender with the digital twin ...")

    blend_path = os.environ.get("MICROVERSE_BLEND_FILE")
    if not blend_path:
        blend_path = str(_resolve_blend_file())

    env = os.environ.copy()
    env["MICROVERSE_BLEND_FILE"] = blend_path

    _BLENDER_NOISE_PATTERNS = ("Saved: '", "OpenGL Render written to")
    command = ["blender", "--python", "main_run.py"]

    process = subprocess.Popen(
        command,
        cwd=str(_REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        stripped = line.rstrip("\n")
        if not VERBOSE and any(p in stripped for p in _BLENDER_NOISE_PATTERNS):
            continue
        print(stripped, flush=True)

    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def main():
    args = gather_inputs()
    clean_path = stage_1_ingest_and_smooth(args)
    attacked_path, check_path = stage_2_inject_attacks(clean_path, args)
    stage_3_verify_and_fork(attacked_path, args)
    stage_4_evaluate_detection(check_path, args)

    dashboard_process = stage_5_launch_dashboard()
    try:
        stage_6_launch_digital_twin()
    finally:
        # Blender closed (or crashed/raised) -- don't leave the dashboard
        # server running as an orphaned background process either way.
        if dashboard_process.poll() is None:
            vprint("[6/6] Blender exited -- stopping the dashboard process ...")
            dashboard_process.terminate()
            try:
                dashboard_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dashboard_process.kill()


if __name__ == "__main__":
    main()