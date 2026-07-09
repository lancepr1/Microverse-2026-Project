import sys
import os
import json
import importlib
import bpy

# ==========================================================================
# 📂 REPOSITORY PATH ALIGNMENT
# ==========================================================================
REPO_ROOT = "/home/Baron/Projects/Microverse-2026-Project"
CORE_DIR = f"{REPO_ROOT}/microverse_core"

import sys
for path in [REPO_ROOT, CORE_DIR]:
    if path not in sys.path:
        sys.path.append(path)

# Import workers directly from your local repository filesystem
import blender_bridge
import io_records

# Force fresh reloading so updates in VS Code are pushed to Blender instantly
importlib.reload(blender_bridge)
importlib.reload(io_records)

# ==========================================================================
# 🏗️ ASSET VALIDATION LAYER
# ==========================================================================
def verify_and_grab_asset(obj_name: str, fallback_type: str = "CUBE", location: tuple = (0, 0, 0)) -> bpy.types.Object:
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    if obj_name in bpy.data.objects:
        return bpy.data.objects[obj_name]
    if fallback_type == "ICOSPHERE":
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1.0, location=location)
    else:
        bpy.ops.mesh.primitive_cube_add(size=1.2, location=location)
    new_obj = bpy.context.active_object
    new_obj.name = obj_name
    return new_obj


# ==========================================================================
# ⏱️ DIRECT-DATA SIMULATION ENGINE (1 line every 2 seconds)
# ==========================================================================
class RealTimeSimulationRunner:
    def __init__(self, run_id: str, source_filename: str):
        self.run_id = run_id
        # Points directly to the data folder verified in your VS Code workspace tree
        self.file_path = os.path.join(REPO_ROOT, "data", source_filename)
        self.step = 0
        self.file_handle = None
        
        print(f"\n🚀 Initializing Git-Linked Telemetry Loop for Run: {run_id}")
        print(f"📂 Source Data Path: {self.file_path}")
        
        # 1. Spawn the 2 CPUs (Left side)
        verify_and_grab_asset("cpu_00", "CUBE", location=(-4.0, 0.0, 1.0))
        verify_and_grab_asset("cpu_01", "CUBE", location=(-2.0, 0.0, 1.0))
        
        # 2. Spawn the 4 GPUs (Middle)
        for i in range(4):
            verify_and_grab_asset(f"gpu_{i:02d}", "CUBE", location=(i * 2.0, 0.0, 1.0))
            
        # 3. Spawn the Grid Anchor (Right side)
        verify_and_grab_asset("grid_anchor", "ICOSPHERE", location=(8.0, 0.0, 1.0))
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Critical Error: Missing data file at {self.file_path}")
        self.file_handle = open(self.file_path, 'r')

    def process_next_line(self):
        """Timer callback loop passing raw primitives directly to workers."""
        if not self.file_handle:
            return None
            
        line = self.file_handle.readline()
        if not line or not line.strip():
            print(f"✅ Real-time run execution complete. Stream reached end of file.")
            self.file_handle.close()
            return None
            
        data = json.loads(line)
        ts = float(data.get("index", self.step)) * 2.0
        frame_records = []

        # --- 1. UPDATE CPU SYSTEM CUSTOM PROPERTIES ---
        for i in range(2):
            cpu_name = f"cpu_{i:02d}"
            p_val = data.get(f"cpu-{i}[W]", 0.0)
            c_val = data.get(f"cpu-{i}-core[W]", 0.0)
            
            blender_bridge.set_state(cpu_name, "power_draw_w", p_val, timestamp=ts)
            blender_bridge.set_state(cpu_name, "core_power_w", c_val, timestamp=ts)
            
            frame_records.append({"object": cpu_name, "property": "power_draw_w", "value": p_val, "timestamp": ts})
            frame_records.append({"object": cpu_name, "property": "core_power_w", "value": c_val, "timestamp": ts})

        # --- 2. UPDATE GPU SYSTEM CUSTOM PROPERTIES ---
        for i in range(4):
            gpu_name = f"gpu_{i:02d}"
            p_val = data.get(f"gpu-{i}[W]", 0.0)
            t_val = data.get(f"gpu-{i}[C]", 0.0)
            
            blender_bridge.set_state(gpu_name, "power_draw_w", p_val, timestamp=ts)
            blender_bridge.set_state(gpu_name, "temp_c", t_val, timestamp=ts)
            
            frame_records.append({"object": gpu_name, "property": "power_draw_w", "value": p_val, "timestamp": ts})
            frame_records.append({"object": gpu_name, "property": "temp_c", "value": t_val, "timestamp": ts})

        # --- 3. UPDATE REGIONAL GRID ANCHOR ---
        freq_val = data.get("FRQ", 60.0)
        blender_bridge.set_state("grid_anchor", "grid_frequency_hz", freq_val, timestamp=ts)
        frame_records.append({"object": "grid_anchor", "property": "grid_frequency_hz", "value": freq_val, "timestamp": ts})

        # --- 4. EXPORT SNAPSHOTS TO RUNS DIRECTORY ---
        io_records.write_records(self.run_id, "power", frame_records)

        print(f"⏰ [Line Entry {data.get('index')}] Grid: {freq_val:.4f} Hz | Active tracking processing from disk updates.")
        
        # Redraw viewport layout live
        for area in bpy.context.screen.areas:
            if area.type == 'VIEWPORT_3D':
                area.tag_redraw()

        self.step += 1
        return 2.0


# Initialize runner using the direct repository file mapping tracking architecture
sim_runner = RealTimeSimulationRunner(run_id="frl_lean_direct_run", source_filename="run01.jsonl")
bpy.app.timers.register(sim_runner.process_next_line)