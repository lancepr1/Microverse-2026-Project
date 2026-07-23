"""Runs the full Microverse pipeline end to end: ingest, attack, verify, evaluate, display.

    ingest -> smooth -> attack injection -> verify + annotate -> fork
    to (scoreboard, dashboard, digital twin)

Lives at scripts/run_microverse.py. All internal paths are anchored
explicitly to the repo root regardless of where this script is
invoked from. See .readme/run_microverse.md for per-stage status
history and design rationale.

Usage:
    python scripts/run_microverse.py

Fully interactive -- no CLI flags. Walks through: which dataset,
how many nodes, which SLURM job (training datasets only), which ENF
recording device/hour, and a component ID label. When the pipeline
reaches the attack-injection stage, that script's own interactive
prompts take over the terminal directly.
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "lanes" / "leiva_verification"))

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

VERBOSE = False


def vprint(*args, **kwargs) -> None:
    """Prints only when VERBOSE is True.

    Args:
        *args: Passed through to print().
        **kwargs: Passed through to print().
    """
    if VERBOSE:
        print(*args, **kwargs)


def prompt_choice(prompt_text: str, options: list) -> str:
    """Prints a numbered list and prompts until a valid choice is made.

    Args:
        prompt_text: Prompt shown above the list of options.
        options: Choices to present.

    Returns:
        str: The chosen option.
    """
    print(f"\n{prompt_text}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        raw = input(f"Enter a number [1-{len(options)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("  Invalid choice, try again.")


def _node_sort_key(folder_name: str) -> int:
    """Extracts the numeric prefix from a folder name for correct numeric sorting.

    Args:
        folder_name: A folder name such as "16node".

    Returns:
        int: The numeric prefix, so "16node" sorts after "2node"
        rather than before it.
    """
    digits = re.sub(r"\D", "", folder_name)
    return int(digits) if digits else 0


def discover_slurm_ids(node_folder: Path) -> list:
    """Scans a node folder's filenames for distinct SLURM job IDs present.

    Args:
        node_folder: Folder to scan.

    Returns:
        list[str]: Distinct SLURM job IDs found, sorted. A training
        dataset's node folder can contain multiple separate runs,
        which is why this returns a list rather than a single value.
    """
    pattern = re.compile(r"slurmid_(\d+)_")
    ids = set()
    for f in node_folder.iterdir():
        m = pattern.search(f.name)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def discover_enf_devices(enf_folder: Path) -> dict:
    """Scans a folder for ENF recording files and groups them by device and hour.

    Args:
        enf_folder: Folder to scan for "DevN_ENF_HrNN.csv" files.

    Returns:
        dict: {device_label: {hour_label: filepath}}, built entirely
        from whatever files are actually present.
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
    """Interactively walks the user through every choice needed to start a run.

    Discovers available datasets, node counts, SLURM job IDs (training
    datasets only), and ENF recording device/hour entirely from what's
    actually present on disk -- nothing is hardcoded.

    Returns:
        argparse.Namespace: Populated with everything the pipeline
        stages need: workload_type, nlr_folder, node_folder_name,
        slurm_id, enf_path, component_id, node_count, node_ids, and
        output_dir.

    Raises:
        RuntimeError: If an expected data folder doesn't exist, or
        contains no usable files.
    """
    print("=" * 70)
    print("Microverse-2026-Project -- Data Ingestion")
    print("=" * 70)

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
        node_count=None,
        node_ids=None,
        output_dir=str(_REPO_ROOT / "data" / "combined"),
    )
    print()
    return args
def stage_1_ingest_and_smooth(args) -> Path:
    """Loads raw ENF, smooths it, loads NLR data, and writes one combined JSONL.

    combined_smooth() must run here, before attack injection ever
    touches the data -- smoothing downstream of an attack silently
    erases it with zero detection. This is the one ordering rule in
    the whole pipeline that must never move.

    Also generates the independently-noised reference ENF stream used
    by the correlation check in stage 3, from this same clean array,
    before attack injection ever runs -- see .readme/run_microverse.md
    for why this must happen here specifically.

    Args:
        args: Namespace from gather_inputs().

    Returns:
        Path: Path to the written combined JSONL file.

    Raises:
        RuntimeError: If no NLR node pairs are found for the given
            folder/workload type/SLURM ID combination.
    """
    vprint(f"[1/5] Ingesting ENF from {args.enf_path} ...")
    enf = load_enf(args.enf_path)
    enf = combined_smooth(enf)

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

    args.actual_node_count = len(pairs)
    out_filename = f"run_{args.actual_node_count}node"
    out_path = Path(args.output_dir) / f"{out_filename}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_combined_jsonl(records, str(out_path))
    vprint(f"[1/5] Wrote {len(records)} records -> {out_path}")
    return out_path


def stage_2_inject_attacks(clean_path: Path, args) -> tuple[Path, Path]:
    """Runs the attack-injection script and locates both files it writes.

    The attack script writes two files per run: a plain file (no
    ground truth) and a "_check" file (same data, plus a 0/1 "attack"
    ground-truth column). Only the plain file is ever fed to
    verification -- the verifier must never see the ground-truth
    column, or that would defeat the entire point of independent
    verification. See .readme/run_microverse.md for how the output
    file is located and why modification time, not a before/after
    diff, is used.

    Args:
        clean_path: Path to the combined JSONL written by stage 1.
            Read directly by the attack script, not passed as an
            argument.
        args: Namespace from gather_inputs(); must have
            actual_node_count set by stage 1.

    Returns:
        tuple[Path, Path]: (plain_path, check_path). Only plain_path
        is used by stage 3; check_path is carried forward for stage 4's
        scoring use only.

    Raises:
        FileNotFoundError: If no matching output file can be found
            after the attack script runs, or if its plain-file
            counterpart is missing.
    """
    import subprocess
    import time

    output_dir = _REPO_ROOT / "lanes" / "marchisano_attacks" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time() - 1.0

    node_count = args.actual_node_count

    vprint(f"[2/6] Launching attack.py --nodes {node_count} "
          f"(reads {clean_path}) ...")
    vprint(f"[2/6] attack.py has its own interactive prompts -- answer those directly below.\n")
    subprocess.run(
        ["python", "lanes/marchisano_attacks/attack.py", "--nodes", str(node_count)],
        check=True,
        cwd=str(_REPO_ROOT),
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
    """Verifies every record and writes the annotated result to three destinations.

    Writes directly to all three destinations in one pass -- no
    intermediate file, since all three were always identical copies of
    it anyway. Never includes ground truth: verifies `attacked_path`,
    the plain file with no "attack" column (see stage_2_inject_attacks()).

    Output columns: `ENF_status` (worst-of every ENF-related check)
    and one `{node_id}_status` per node actually present in this run's
    data (worst-of just that node's own GPU/CPU metrics). All values
    use the 0.0/0.5/1.0 (trusted/suspect/failed) encoding.

    Args:
        attacked_path: Path to the plain (no ground truth) JSONL
            written by stage 2.
        args: Namespace from gather_inputs(); must have component_id
            and, optionally, enf_alternative set.
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
        """Extracts the node ID from a component_id, or None if it's not node-specific.

        Args:
            component_id: e.g. "rack_00/x3102c0s25b0n0_gpu-0[W]".

        Returns:
            Optional[str]: The node ID, or None for ENF or anything
            else not tied to a specific node.
        """
        name = component_id.split("/", 1)[-1]
        if "_gpu" in name:
            return name.split("_gpu")[0]
        if "_cpu" in name:
            return name.split("_cpu")[0]
        return None

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
    """Runs the scoring tool and prints its evaluation report to the console.

    Deliberately the only thing this pipeline prints to the console by
    default -- see VERBOSE at the top of this file, which gates every
    other stage's routine narration. The scenario ID is derived
    directly from check_path's filename (e.g.
    "attack_304_check.jsonl" -> "304").

    Does not propagate a failure here (no check=True) -- a missing or
    malformed scoreboard file failing evaluation shouldn't block the
    dashboard or digital twin from still launching; those are
    independent concerns from scoring. The scoring tool's own main()
    already prints a clear error and exits non-zero on a real problem,
    so a failure here is still visible, just not fatal to the rest of
    the run.

    Args:
        check_path: Path to the ground-truth "_check" file from stage 2.
        args: Namespace from gather_inputs(). Unused directly, kept
            for signature consistency with the other stages.
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
    """Launches the dashboard app as a non-blocking background process.

    Uses Popen rather than run(), since stage 6 (Blender) blocks the
    main thread until its viewport window is closed, and both
    processes need to be alive at the same time. Waits
    DASHBOARD_STARTUP_GRACE_S after launch to confirm the process
    actually stayed alive (a successful Popen() call only means the
    process spawned, not that it didn't immediately crash) -- raises
    if it's already dead by then, rather than returning a stale,
    misleading "success".

    Returns:
        subprocess.Popen: Handle to the running dashboard process, so
        main() can terminate it cleanly once Blender exits.

    Raises:
        RuntimeError: If the dashboard process exits within the
            startup grace period.
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
    """Scans data/rawdata/ for a .blend file, prompting if more than one exists.

    Resolves the choice in this process's own normal (never piped)
    terminal, rather than leaving it to Blender's own file-scanning
    logic -- see .readme/run_microverse.md for why that distinction
    matters once Blender's stdout is being piped and filtered.

    Returns:
        Path: The chosen .blend file.

    Raises:
        RuntimeError: If data/rawdata/ doesn't exist, or contains no
            .blend files.
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
    """Launches Blender with the digital twin scene, blocking until it's closed.

    Reads for_digital_twin.jsonl (written by stage 3) and plays it
    back live, coloring each node and the grid anchor by verification
    status. Runs with a normal, visible Blender window (not
    --background), since the entire point is watching the twin update
    live.

    Filters Blender's own internal render-engine log lines (which
    print on every viewport capture tick, independent of anything in
    the digital twin script's own output) out of the piped stdout in
    real time, rather than redirecting stdout to DEVNULL entirely --
    that blunter approach would also swallow genuine Blender errors
    and tracebacks. Bypassed (shows everything unfiltered) when
    VERBOSE=True.

    Raises:
        subprocess.CalledProcessError: If the Blender process exits
            with a non-zero return code.
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


def main() -> None:
    """Runs the full pipeline: gather inputs, then every stage in order."""
    args = gather_inputs()
    clean_path = stage_1_ingest_and_smooth(args)
    attacked_path, check_path = stage_2_inject_attacks(clean_path, args)
    stage_3_verify_and_fork(attacked_path, args)
    stage_4_evaluate_detection(check_path, args)

    dashboard_process = stage_5_launch_dashboard()
    try:
        stage_6_launch_digital_twin()
    finally:
        if dashboard_process.poll() is None:
            vprint("[6/6] Blender exited -- stopping the dashboard process ...")
            dashboard_process.terminate()
            try:
                dashboard_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dashboard_process.kill()


if __name__ == "__main__":
    main()