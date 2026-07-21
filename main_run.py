import sys
import os
import json
import importlib
import tempfile
import bpy

# CHANGED (2026-07): matches the same VERBOSE/vprint convention added to
# scripts/run_microverse.py -- the per-frame "Grid: X Hz | ..." line
# below printed on every single simulation tick (every 2 seconds),
# which is exactly the kind of routine narration that's redundant now
# that the dashboard and digital twin show the same information live.
# Default off. Real warnings (missing scene objects, per-frame write
# failures) and the interactive .blend-file disambiguation prompt are
# deliberately NOT gated -- those indicate a real problem or are the
# actual interactive UI, not routine status.
VERBOSE = False


def vprint(*args, **kwargs) -> None:
    if VERBOSE:
        print(*args, **kwargs)


# ==========================================================================
# 📂 REPOSITORY PATH ALIGNMENT
# ==========================================================================
# Was hardcoded to Baron's own home directory -- broke the moment this
# ran on anyone else's machine (ModuleNotFoundError on blender_bridge,
# since /home/Baron/... simply doesn't exist elsewhere). Self-locating
# now: this file lives at the repo root, so its own real location on
# disk IS the repo root, on any machine, no editing needed per person.
#
# MOVED (2026-07): this used to be defined AFTER _find_blend_file() ran
# (see below) -- but that function now needs REPO_ROOT itself, to scan
# data/rawdata/ instead of ~/Projects/, so this has to come first.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _find_blend_file():
    """
    Scans data/rawdata/ (repo-relative -- same convention
    scripts/run_microverse.py's gather_inputs() already uses for the raw
    NLR datasets and ENF data, not prompted for anymore -- see
    README.md's "Where to put your data" section) for a .blend file.
    Auto-selects if exactly one is found, the common case; if multiple
    exist, asks which one (a real choice, not location bookkeeping); if
    none exist, fails with a clear, actionable message rather than a
    confusing downstream crash.

    CHANGED (2026-07): was ~/Projects/, home-directory-relative --
    meant every teammate had to create a "Projects" folder in their own
    home directory and put the .blend file there by hand, a different
    real path per person/OS. Now repo-relative (data/rawdata/), the
    exact same folder scripts/run_microverse.py already reads the raw
    datasets and ENF data from -- one shared location, works identically
    on Windows/Mac/Linux with zero per-person setup. Data lives in this
    folder locally on every machine but is NOT committed to git (space
    constraints) -- see .gitignore.
    """
    projects_dir = os.path.join(REPO_ROOT, "data", "rawdata")
    if not os.path.isdir(projects_dir):
        raise FileNotFoundError(
            f"Expected {projects_dir} to exist -- see README.md's "
            f"\"Where to put your data\" section. Put your .blend file "
            f"directly inside data/rawdata/ at the repo root."
        )

    blend_files = sorted(f for f in os.listdir(projects_dir) if f.lower().endswith(".blend"))

    if not blend_files:
        raise FileNotFoundError(
            f"No .blend file found in {projects_dir}. See README.md's "
            f"\"Where to put your data\" section -- put your .blend file "
            f"directly inside data/rawdata/ at the repo root."
        )

    if len(blend_files) == 1:
        chosen = blend_files[0]
        vprint(f"🔎 Found .blend file: {chosen}")
    else:
        print(f"\nMultiple .blend files found in {projects_dir}:")
        for i, f in enumerate(blend_files, 1):
            print(f"  {i}. {f}")
        while True:
            raw = input(f"Which one? [1-{len(blend_files)}]: ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(blend_files):
                chosen = blend_files[int(raw) - 1]
                break
            print("  Invalid choice, try again.")

    return os.path.join(projects_dir, chosen)


# ==========================================================================
# 🏠 TARGET BLENDER FILE AUTO-LOADER (STEP 1 ADDITION)
# ==========================================================================
# MICROVERSE_BLEND_FILE, if set, skips the scan entirely -- useful for
# non-interactive/automated runs. Otherwise auto-discovered from
# data/rawdata/ via _find_blend_file above.
TARGET_BLEND_FILE = os.environ.get("MICROVERSE_BLEND_FILE") or _find_blend_file()

# If the runner launches a default blank project, dynamically open your file
if bpy.data.filepath != TARGET_BLEND_FILE:
    if os.path.exists(TARGET_BLEND_FILE):
        vprint(f"🔄 Hot-loading target twin project: {TARGET_BLEND_FILE}")
        bpy.ops.wm.open_mainfile(filepath=TARGET_BLEND_FILE)
    else:
        raise FileNotFoundError(
            f"❌ Could not find the target .blend file at: {TARGET_BLEND_FILE}\n"
            f"This script only writes data onto objects that already exist in "
            f"a real scene -- it cannot create the rack/component models "
            f"itself. Get a copy of the real .blend file onto this machine, "
            f"then run again and provide its correct path when prompted."
        )


if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# FIXED (2026-07): was adding microverse_core itself to sys.path and
# importing blender_bridge as a bare top-level module -- but
# blender_bridge.py (and io_records.py) use RELATIVE imports internally
# (from .contracts import ...), which only resolve correctly when
# Python treats microverse_core as a real package. Only REPO_ROOT goes
# on sys.path now; microverse_core is imported AS a package below, so
# its own internal relative imports work the way they're written.
from microverse_core import blender_bridge, io_records
from microverse_core.contracts import StateVariable

# Force fresh reloading so updates in VS Code are pushed to Blender instantly
importlib.reload(blender_bridge)
importlib.reload(io_records)


# ==========================================================================
# 📦 COMPATIBILITY WRAPPER FOR TELEMETRY LOGGING
# ==========================================================================
class Record:
    def __init__(self, obj: str, prop: str, val: float, ts: float):
        self.obj = obj
        self.prop = prop
        self.val = val
        self.ts = ts

    def to_dict(self):
        """Satisfies the strict interface expected by io_records.py."""
        return {
            "object": self.obj,
            "property": self.prop,
            "value": self.val,
            "timestamp": self.ts
        }


# ==========================================================================
# ⏱️ PRE-EXISTING SCENE LINKED SIMULATION ENGINE
# ==========================================================================
class RealTimeSimulationRunner:
    def __init__(self, run_id: str):
        self.run_id = run_id
        # Fixed path -- this is the verifier's actual output now, not a raw
        # ingestion file. Every record here already carries ENF_status and
        # one {node_id}_status field per node present in the run.
        self.file_path = os.path.join(
            REPO_ROOT, "lanes", "leiva_verification", "outputs", "for_digital_twin.jsonl"
        )
        self.step = 0
        self.file_handle = None

        vprint(f"\n🚀 Connecting Ingestion Engine to Pre-Existing Scene for Run: {run_id}")
        vprint(f"📂 Source Data Path: {self.file_path}")

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Critical Error: Missing data file at {self.file_path}")
        self.file_handle = open(self.file_path, 'r')

        # --- DYNAMIC NODE DISCOVERY -------------------------------------
        # Was a hardcoded 2-node list -- now discovered from the file's
        # own first record, same "_gpu"/"_cpu" split convention used
        # throughout the rest of this project (attack.py's own
        # scan_telemetry_schema(), run_microverse.py's column grouping).
        # Works unchanged whether this run has 1 node or 16 -- nothing
        # here needs to know the count in advance.
        first_line = self.file_handle.readline()
        if not first_line.strip():
            raise ValueError(f"{self.file_path} is empty -- nothing to discover nodes from.")
        first_record = json.loads(first_line)
        self.nodes = sorted({
            (k.split("_gpu")[0] if "_gpu" in k else k.split("_cpu")[0])
            for k in first_record.keys()
            if "_gpu" in k or "_cpu" in k
        })
        vprint(f"🔎 Discovered {len(self.nodes)} node(s) from data: {self.nodes}")
        # Rewind so process_next_line() sees this same first record again --
        # discovery shouldn't consume a real simulation step.
        self.file_handle.seek(0)

        # Match your workspace hierarchy explicitly
        self.grid_anchor_name = "grid_anchor.001"

        # --- VERIFY MULTI-NODE HIERARCHY MATCHING ---
        # GPU/CPU counts per node are fixed (4 GPUs, 2 CPUs) -- confirmed
        # this doesn't vary, so only the NODE LIST itself needs to be
        # dynamic, not the per-node topology.
        missing_components = []

        for node in self.nodes:
            for c in range(2):
                cpu_id = f"{node}_cpu-{c}"
                if cpu_id not in bpy.data.objects:
                    missing_components.append(cpu_id)

            for g in range(4):
                gpu_id = f"{node}_gpu-{g}"
                if gpu_id not in bpy.data.objects:
                    missing_components.append(gpu_id)

        if self.grid_anchor_name not in bpy.data.objects:
            missing_components.append(self.grid_anchor_name)

        if missing_components:
            print(f"⚠️ WARNING: The following objects were not found in your scene tree: {missing_components}")
            print(f"⚠️ This usually means the scene doesn't yet have objects for every node this run's data actually contains -- add them, or this run's node count/IDs won't fully display.")
        else:
            vprint("✨ Success! All hardware components matched your workspace tree layout perfectly.")


    def process_next_line(self):
        """Timer callback loop passing raw primitives directly to workers."""
        if not self.file_handle:
            return None
            
        line = self.file_handle.readline()
        if not line or not line.strip():
            vprint(f"✅ Real-time asset stream complete. Reached end of file.")
            self.file_handle.close()
            return None
            
        data = json.loads(line)
        ts = float(data.get("index", self.step)) * 2.0
        frame_records = []

        # --------------------------------------------------------------
        # 🔴 DEBUGGER BREAKPOINT DIRECTION
        # PLACE YOUR BREAKPOINT ON THE LINE BELOW!
        # --------------------------------------------------------------
        current_snapshot_index = data.get("index", self.step)

        # Tracks exactly what happened this frame, for the console
        # summary at the end -- lets you confirm correctness directly
        # from terminal output, without needing to describe what the
        # viewport looks like.
        status_summary = []
        write_failures = []

        def _safe_write(obj_name, fn, *fn_args):
            """
            A single missing or misnamed object used to crash the ENTIRE
            real-time loop on its very next write -- one bad object name
            killed the whole simulation, not just that object. Now
            catches and records the failure, keeps going with everything
            else. Failures print in this frame's summary so a real
            naming mismatch is still loud and visible, just not fatal.
            """
            try:
                fn(obj_name, *fn_args)
                return True
            except KeyError:
                write_failures.append(obj_name)
                return False

        # Direct layout iteration across server keys
        for node in self.nodes:
            # Verification status for this whole node -- worst-of every
            # metric belonging to it, computed upstream by the verifier.
            # Applied to EVERY sub-component object this node has (all
            # CPUs and GPUs), since there's no single node-level parent
            # object in the current scene to target instead -- the whole
            # node visibly lights up together.
            node_status = data.get(f"{node}_status")
            if node_status is not None:
                label = {0.0: "TRUSTED/green", 0.5: "SUSPECT/yellow", 1.0: "FAILED/red"}.get(node_status, f"unexpected({node_status})")
                status_summary.append(f"{node}={label}")

            # --- 1. DIRECT CPU TELEMETRY DATA ENTRY MATCHING ---
            for cpu_idx in range(2):
                p_val = data.get(f"{node}_cpu-{cpu_idx}[W]", 0.0)
                c_val = data.get(f"{node}_cpu-{cpu_idx}-core[W]", 0.0)
                
                target_cpu_name = f"{node}_cpu-{cpu_idx}"
                
                _safe_write(target_cpu_name, blender_bridge.set_state, StateVariable(name="power_draw_w", value=p_val, unit="W", timestamp=ts, source_object=target_cpu_name))
                _safe_write(target_cpu_name, blender_bridge.set_state, StateVariable(name="core_power_w", value=c_val, unit="W", timestamp=ts, source_object=target_cpu_name))
                if node_status is not None:
                    _safe_write(target_cpu_name, blender_bridge.set_status_color, node_status)
                
                frame_records.append(Record(obj=target_cpu_name, prop="power_draw_w", val=p_val, ts=ts))
                frame_records.append(Record(obj=target_cpu_name, prop="core_power_w", val=c_val, ts=ts))

            # --- 2. DIRECT GPU TELEMETRY DATA ENTRY MATCHING ---
            for g_idx in range(4):
                p_val = data.get(f"{node}_gpu-{g_idx}[W]", 0.0)
                t_val = data.get(f"{node}_gpu-{g_idx}[C]", 0.0)
                
                target_gpu_name = f"{node}_gpu-{g_idx}"
                
                _safe_write(target_gpu_name, blender_bridge.set_state, StateVariable(name="power_draw_w", value=p_val, unit="W", timestamp=ts, source_object=target_gpu_name))
                _safe_write(target_gpu_name, blender_bridge.set_state, StateVariable(name="temp_c", value=t_val, unit="C", timestamp=ts, source_object=target_gpu_name))
                if node_status is not None:
                    _safe_write(target_gpu_name, blender_bridge.set_status_color, node_status)
                
                frame_records.append(Record(obj=target_gpu_name, prop="power_draw_w", val=p_val, ts=ts))
                frame_records.append(Record(obj=target_gpu_name, prop="temp_c", val=t_val, ts=ts))

        # --- 3. UPDATE REGIONAL GRID ANCHOR ---
        freq_val = data.get("FRQ", 60.0)
        enf_status = data.get("ENF_status")
        _safe_write(self.grid_anchor_name, blender_bridge.set_state, StateVariable(name="grid_frequency_hz", value=freq_val, unit="Hz", timestamp=ts, source_object=self.grid_anchor_name))
        if enf_status is not None:
            _safe_write(self.grid_anchor_name, blender_bridge.set_status_color, enf_status)
            label = {0.0: "TRUSTED/green", 0.5: "SUSPECT/yellow", 1.0: "FAILED/red"}.get(enf_status, f"unexpected({enf_status})")
            status_summary.append(f"ENF={label}")
        frame_records.append(Record(obj=self.grid_anchor_name, prop="grid_frequency_hz", val=freq_val, ts=ts))
        
        # --- 4. EXPORT SNAPSHOTS TO RUNS DIRECTORY ---
        io_records.write_records(self.run_id, "power", frame_records)

        vprint(f"⏰ [Line Entry {data.get('index')}] Grid: {freq_val:.4f} Hz | {' | '.join(status_summary)}")
        if write_failures:
            print(f"   ⚠️ {len(write_failures)} object write(s) failed (not in scene): {sorted(set(write_failures))}")
        
        # Redraw viewport layout live
        for area in bpy.context.screen.areas:
            if area.type == 'VIEWPORT_3D':
                area.tag_redraw()

        # Feeds the dashboard's "Live Digital Twin" panel -- see
        # _capture_viewport()'s docstring for why this exact path.
        _capture_viewport()

        self.step += 1
        return 2.0


def _capture_viewport(filepath=None):
    """
    Grabs a fast OpenGL capture of the 3D viewport and writes it to disk
    as a PNG -- this exact path is what the dashboard's
    lanes/mccray_dashboard/dashboard/ui/blender_feed.py panel polls
    (mtime-based) to show a "Live Digital Twin" preview. Uses
    render.opengl (a viewport snapshot) rather than a full render --
    cheap enough to call every simulation tick.

    CHANGED (2026-07): default filepath was a hardcoded "/tmp/..." --
    /tmp doesn't exist on Windows at all, which would have silently
    broken the whole Live Digital Twin panel for the new Windows
    teammate (this function would raise, or Blender would error trying
    to write to a nonexistent root path). tempfile.gettempdir() resolves
    to the correct OS temp directory on Windows/Mac/Linux alike. Both
    this file and ui/blender_feed.py's IMG_PATH must resolve to the
    SAME actual path for the hand-off to work -- they're two separate
    processes (Blender and the dashboard) on the same machine, so this
    only works because tempfile.gettempdir() is deterministic per-OS,
    not because they coordinate directly.
    """
    if filepath is None:
        filepath = os.path.join(tempfile.gettempdir(), "blender_viewport.png")

    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        scene = bpy.context.scene
                        scene.render.image_settings.file_format = 'PNG'
                        scene.render.filepath = filepath
                        with bpy.context.temp_override(window=window, area=area, region=region):
                            # view_context=True captures what's actually
                            # visible in the viewport (current camera
                            # angle/zoom), not the scene's render camera.
                            bpy.ops.render.opengl(write_still=True, view_context=True)
                        return
    print("⚠️ No VIEW_3D area found -- skipping viewport capture for this tick.")


# Initialize layout link using your pre-existing environment file structure
sim_runner = RealTimeSimulationRunner(run_id="frl_lean_direct_run")
bpy.app.timers.register(sim_runner.process_next_line)