import sys
import os
import json
import importlib
import bpy

# ==========================================================================
# 🏠 TARGET BLENDER FILE AUTO-LOADER (STEP 1 ADDITION)
# ==========================================================================
TARGET_BLEND_FILE = "/home/Baron/Documents/Blender Files/Data_Center_Twin3.blend"

# If the runner launches a default blank project, dynamically open your file
if bpy.data.filepath != TARGET_BLEND_FILE:
    if os.path.exists(TARGET_BLEND_FILE):
        print(f"🔄 Hot-loading target twin project: {TARGET_BLEND_FILE}")
        bpy.ops.wm.open_mainfile(filepath=TARGET_BLEND_FILE)
    else:
        print(f"❌ Error: Could not locate your file path at: {TARGET_BLEND_FILE}")


# ==========================================================================
# 📂 REPOSITORY PATH ALIGNMENT
# ==========================================================================
REPO_ROOT = "/home/Baron/Projects/Microverse-2026-Project"
CORE_DIR = f"{REPO_ROOT}/microverse_core"

for path in [REPO_ROOT, CORE_DIR]:
    if path not in sys.path:
        sys.path.append(path)

# Import workers directly from your local repository filesystem
import blender_bridge
import io_records
from contracts import StateVariable

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
    def __init__(self, run_id: str, source_filename: str):
        self.run_id = run_id
        self.file_path = os.path.join(REPO_ROOT, "data", "combined", source_filename)
        self.step = 0
        self.file_handle = None
        
        print(f"\n🚀 Connecting Ingestion Engine to Pre-Existing Scene for Run: {run_id}")
        print(f"📂 Source Data Path: {self.file_path}")
        
        # Exact nodes identified inside your cluster logging network
        self.nodes = ["x3102c0s25b0n0", "x3102c0s5b0n0"]
        
        # Match your workspace hierarchy explicitly
        self.grid_anchor_name = "grid_anchor.001"
        
        # --- VERIFY MULTI-NODE HIERARCHY MATCHING ---
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
        else:
            print("✨ Success! All hardware components matched your workspace tree layout perfectly.")

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Critical Error: Missing data file at {self.file_path}")
        self.file_handle = open(self.file_path, 'r')

    def process_next_line(self):
        """Timer callback loop passing raw primitives directly to workers."""
        if not self.file_handle:
            return None
            
        line = self.file_handle.readline()
        if not line or not line.strip():
            print(f"✅ Real-time asset stream complete. Reached end of file.")
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

        # Direct layout iteration across server keys
        for node in self.nodes:
            
            # --- 1. DIRECT CPU TELEMETRY DATA ENTRY MATCHING ---
            for cpu_idx in range(2):
                p_val = data.get(f"{node}_cpu-{cpu_idx}[W]", 0.0)
                c_val = data.get(f"{node}_cpu-{cpu_idx}-core[W]", 0.0)
                
                target_cpu_name = f"{node}_cpu-{cpu_idx}"
                
                blender_bridge.set_state(target_cpu_name, StateVariable(name="power_draw_w", value=p_val, unit="W", timestamp=ts, source_object=target_cpu_name))
                blender_bridge.set_state(target_cpu_name, StateVariable(name="core_power_w", value=c_val, unit="W", timestamp=ts, source_object=target_cpu_name))
                
                frame_records.append(Record(obj=target_cpu_name, prop="power_draw_w", val=p_val, ts=ts))
                frame_records.append(Record(obj=target_cpu_name, prop="core_power_w", val=c_val, ts=ts))

            # --- 2. DIRECT GPU TELEMETRY DATA ENTRY MATCHING ---
            for g_idx in range(4):
                p_val = data.get(f"{node}_gpu-{g_idx}[W]", 0.0)
                t_val = data.get(f"{node}_gpu-{g_idx}[C]", 0.0)
                
                target_gpu_name = f"{node}_gpu-{g_idx}"
                
                blender_bridge.set_state(target_gpu_name, StateVariable(name="power_draw_w", value=p_val, unit="W", timestamp=ts, source_object=target_gpu_name))
                blender_bridge.set_state(target_gpu_name, StateVariable(name="temp_c", value=t_val, unit="C", timestamp=ts, source_object=target_gpu_name))
                
                frame_records.append(Record(obj=target_gpu_name, prop="power_draw_w", val=p_val, ts=ts))
                frame_records.append(Record(obj=target_gpu_name, prop="temp_c", val=t_val, ts=ts))

        # --- 3. UPDATE REGIONAL GRID ANCHOR ---
        freq_val = data.get("FRQ", 60.0)
        blender_bridge.set_state(self.grid_anchor_name, StateVariable(name="grid_frequency_hz", value=freq_val, unit="Hz", timestamp=ts, source_object=self.grid_anchor_name)) 
        frame_records.append(Record(obj=self.grid_anchor_name, prop="grid_frequency_hz", val=freq_val, ts=ts))
        
        # --- 4. EXPORT SNAPSHOTS TO RUNS DIRECTORY ---
        io_records.write_records(self.run_id, "power", frame_records)

        print(f"⏰ [Line Entry {data.get('index')}] Grid: {freq_val:.4f} Hz | Data synchronized smoothly to twin components.")
        
        # Redraw viewport layout live
        for area in bpy.context.screen.areas:
            if area.type == 'VIEWPORT_3D':
                area.tag_redraw()

        self.step += 1
        return 2.0


# Initialize layout link using your pre-existing environment file structure
sim_runner = RealTimeSimulationRunner(run_id="frl_lean_direct_run", source_filename="run_2node.jsonl")
bpy.app.timers.register(sim_runner.process_next_line)