#!/usr/bin/env python3
import os
import json
import sys
import random
import argparse

# =====================================================================
# PATH CONFIGURATION
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../data/combined"))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

# =====================================================================
# SCENARIO PRESET DICTIONARIES (For Medium & Hard Blind Testing)
# =====================================================================
PRESET_SCENARIOS = {
    2: [  # Medium Scenarios
        {
            "id": "201",
            "name": "Scenario 201", # GPU bias injection
            "type": "injection",
            "target_group": "gpus",
            "start_pct": 0.30,
            "end_pct": 0.70,
            "params": {"type": "bias", "value": 20.0}
        },
        {
            "id": "202",
            "name": "Scenario 202", # All value replay attack
            "type": "replay",
            "target_group": "all",
            "start_pct": 0.60,
            "end_pct": 0.75,
            "params": {"source_start_pct": 0.10}
        },        
        {
            "id": "203",
            "name": "Scenario 203", # Scaled injection to look like active GPU's are idle
            "type": "injection",
            "target_group": "gpu_watts", # Optimized to target strictly wattage metrics
            "start_pct": 0.10,
            "end_pct": 0.9,
            "params": {
                "type": "scale",
                "value": 0.1216,
                "condition_type": "over",
                "conditional_val": 550
            }
        },
        {
            "id": "204",
            "name": "Scenario 204", # Conditional fake activity 
            "type": "injection",  
            "target_group": "gpus",
            "start_pct": 0.20,
            "end_pct": 0.80,
            "params": {
                "type": "absolute",  
                "value": 600.0, 
                "condition_type": "under", 
                "condition_val": 200.0
            }
        },
    ],
    3: [  # Hard Scenarios
        {
            "id": "301",
            "name": "Scenario 301", # Slow small drift in cpu usage
            "type": "drift",
            "target_group": "cpus",
            "start_pct": 0.15,
            "end_pct": 0.85,
            "params": {"type": "additive", "max_value": 0.04}
        },
        {
            "id": "302",
            "name": "Scenario 302", # Instant 3% CPU increase
            "type": "injection",
            "target_group": "cpus",
            "start_pct": 0.40,
            "end_pct": 0.55,
            "params": {"type": "scale", "value": 1.03}
        },
        {
            "id": "303",
            "name": "Scenario 303", # Short period masking workload with idle replay
            "type": "replay",
            "target_group": "gpu_watts", # Optimized to target strictly wattage metrics
            "start_pct": 0.40,
            "end_pct": 0.43,
            "params": {
                "source_start_pct": 0.01,   
                "condition_type": "above",  
                "condition_val": 500.0      
            }
        },
        {
            "id": "304",
            "name": "Scenario 304", 
            "type": "replay",  
            "target_group": "gpu_watts",
            "start_pct": 0.10,
            "end_pct": 0.90,
            "params": {
                "source_start_pct": 0.01,   
                "source_end_pct": 0.03,     
                "condition_type": "above",  
                "condition_val": 100.0      
            }
        },
        {
            "id": "305",
            "name": "Scenario 305", 
            "type": "injection",  
            "target_group": "gpu_watts",
            "start_pct": 0.10,
            "end_pct": 0.90,
            "params": {
                "type": "absolute_jitter",  
                "value": 73.5,              
                "jitter": 0.6,              
                "condition_type": "above",  
                "condition_val": 100.0      
            }
        }
    ]
}

# =====================================================================
# DYNAMIC SCHEMA PARSER (Upgraded for Wattage and Thermal Separation)
# =====================================================================
def scan_telemetry_schema(first_row):
    gpu_all = []
    gpu_watts = []
    gpu_temps = []
    cpu_cols = []
    all_cols = []
    node_ids = set()

    for key in first_row.keys():
        if key == "index":
            continue
        all_cols.append(key)
        
        if "_gpu" in key:
            gpu_all.append(key)
            node_ids.add(key.split("_gpu")[0])
            
            # Granular suffix parsing matching your telemetry tags
            if "[W]" in key:
                gpu_watts.append(key)
            elif "[C]" in key:
                gpu_temps.append(key)
                
        elif "_cpu" in key:
            cpu_cols.append(key)
            node_ids.add(key.split("_cpu")[0])

    return {
        "frq": ["FRQ"] if "FRQ" in first_row else [],
        "gpus": gpu_all,          # Combined (Wattage + Temp)
        "gpu_watts": gpu_watts,    # Strictly Wattage [W]
        "gpu_temps": gpu_temps,    # Strictly Celsius Temperature [C]
        "cpus": cpu_cols,
        "all": all_cols,
        "discovered_nodes": sorted(list(node_ids))
    }

# =====================================================================
# CONDITIONAL HELPER UTILITY
# =====================================================================
def should_apply_modification(current_val, condition_type, condition_val):
    if condition_type is None or condition_val is None:
        return True
    
    c_type = str(condition_type).strip().lower()
    
    if c_type in ["under", "below"] and current_val < condition_val:
        return True
    if c_type in ["above", "over"] and current_val > condition_val:
        return True
        
    return False

# =====================================================================
# PARAMETERIZED ATTACK ENGINES
# =====================================================================
def apply_no_change(data):
    return [row.copy() for row in data], [0] * len(data)

def apply_replay_attack(data, start, end, total_rows, params, condition_type=None, condition_val=None, targets=None):
    modified_data, attack_labels = [], []
    
    source_start_pct = params.get("source_start_pct", 0.0)
    source_end_pct = params.get("source_end_pct", source_start_pct + 0.05)
    if source_end_pct > 1.0:
        source_end_pct = 1.0
        
    source_start_idx = int(total_rows * source_start_pct)
    source_end_idx = int(total_rows * source_end_pct)
    source_window_len = source_end_idx - source_start_idx
    
    if source_window_len <= 0:
        source_window_len = 1  

    for i, row in enumerate(data):
        new_row = row.copy()
        is_attack = 0
        if start <= i < end:
            source_idx = source_start_idx + ((i - start) % source_window_len)
            if 0 <= source_idx < len(data):
                source_row = data[source_idx]
                for key in row.keys():
                    if key != "index" and key in source_row and isinstance(row[key], (int, float)):
                        # If targeting a sub-group, bypass columns that aren't inside our targets list
                        if targets is not None and key not in targets:
                            continue
                        if should_apply_modification(row[key], condition_type, condition_val):
                            new_row[key] = source_row[key]
                            is_attack = 1
        modified_data.append(new_row)
        attack_labels.append(is_attack)
    return modified_data, attack_labels

def apply_injection_attack(data, start, end, targets, params, condition_type=None, condition_val=None):
    modified_data, attack_labels = [], []
    injection_type = params.get("type", "bias")
    value = params.get("value", 0.0)
    jitter_amt = params.get("jitter", 0.5)

    for i, row in enumerate(data):
        new_row = row.copy()
        is_attack = 0
        if start <= i < end:
            for col in targets:
                if col in new_row and isinstance(new_row[col], (int, float)):
                    if should_apply_modification(new_row[col], condition_type, condition_val):
                        is_attack = 1
                        if injection_type == "bias":
                            new_row[col] += value
                        elif injection_type == "scale":
                            new_row[col] *= value
                        elif injection_type == "absolute":
                            new_row[col] = value  
                        elif injection_type == "absolute_jitter":
                            new_row[col] = value + random.uniform(-jitter_amt, jitter_amt)
        modified_data.append(new_row)
        attack_labels.append(is_attack)
    return modified_data, attack_labels

def apply_drift_attack(data, start, end, targets, params, condition_type=None, condition_val=None):
    modified_data, attack_labels = [], []
    drift_type = params.get("type", "additive")
    max_value = params.get("max_value", 0.0)
    duration = end - start if (end - start) > 0 else 1
    
    for i, row in enumerate(data):
        new_row = row.copy()
        is_attack = 0
        if start <= i < end:
            factor = (i - start) / duration
            for col in targets:
                if col in new_row and isinstance(new_row[col], (int, float)):
                    if should_apply_modification(new_row[col], condition_type, condition_val):
                        is_attack = 1
                        if drift_type == "additive":
                            new_row[col] += max_value * factor
                        elif drift_type == "multiplicative":
                            new_row[col] *= (1.0 + (max_value * factor))
        modified_data.append(new_row)
        attack_labels.append(is_attack)
    return modified_data, attack_labels

# =====================================================================
# MAIN RUNNER EXECUTION
# =====================================================================
def main():
    # Setup command line argument parser
    parser = argparse.ArgumentParser(description="Inject anomaly scenarios into dynamic telemetry datasets.")
    parser.add_argument(
        "--nodes", 
        type=int, 
        default=2, 
        help="Specify target topology cluster size to parse from data/combined directory (e.g., 2 or 16)"
    )
    args = parser.parse_args()

    # Construct file path using dynamically provided nodes count
    input_file = os.path.join(DATA_DIR, f"run_{args.nodes}node.jsonl")

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    print(f"Reading input data from '{input_file}'...")
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]

    total_rows = len(data)
    if total_rows == 0:
        print("Error: Input data file is empty.")
        sys.exit(1)

    schema = scan_telemetry_schema(data[0])
    print(f"-> Scan Complete. Discovered {len(schema['discovered_nodes'])} distinct servers/nodes.")
    print(f"   - Total GPU Metric Columns Found: {len(schema['gpus'])}")
    print(f"   - Isolated Wattage Columns [W]: {len(schema['gpu_watts'])}")
    print(f"   - Isolated Thermal Columns [C]: {len(schema['gpu_temps'])}")
    print(f"-> Total timeline entries: {total_rows} frames.\n")

    print("Select Detection Evaluation Difficulty:")
    print(" [1] Easy (Manual Proof of Concept Configuration)")
    print(" [2] Medium")
    print(" [3] Hard")
    
    try:
        difficulty = int(input("Enter choice (1-3): "))
    except ValueError:
        difficulty = 1

    attack_type = "none"
    target_group = "all"
    start_pct, end_pct = 0.0, 0.0
    run_params = {}
    output_id = "0"

    # -----------------------------------------------------------------
    # PATHWAY A: EASY MODE (Manual Configuration)
    # -----------------------------------------------------------------
    if difficulty == 1:
        print("\n--- Easy Mode Configuration Options ---")
        print("Select Attack Engine:")
        print(" [0] Normal\n [1] Extreme Injection (POC)\n [2] Replay\n [3] Injection\n [4] Drift")
        try:
            choice = int(input("Select Option (0-4): "))
        except ValueError:
            choice = 0

        type_map = {0: "none", 1: "extreme_shortcut", 2: "replay", 3: "injection", 4: "drift"}
        selected_mode = type_map.get(choice, "none")
        output_id = f"easy_{choice}"

        if selected_mode != "none":
            print("\nSelect Target Metric Category:")
            print(" [1] Frequency (FRQ)")
            print(" [2] All GPU Metrics (Watts + Temp)")
            print(" [3] GPU Wattage Only ([W])")
            print(" [4] GPU Temperature Only ([C])")
            print(" [5] All Discovered CPUs")
            print(" [6] Everything")
            try:
                tgt_choice = int(input("Select Option (1-6): "))
            except ValueError:
                tgt_choice = 6

            tgt_map = {1: "frq", 2: "gpus", 3: "gpu_watts", 4: "gpu_temps", 5: "cpus", 6: "all"}
            target_group = tgt_map.get(tgt_choice, "all")

            start_pct, end_pct = 0.10, 0.60
            
            if selected_mode == "extreme_shortcut":
                attack_type = "injection"
                run_params = {"type": "absolute", "value": 0.0} 
                
            elif selected_mode == "replay":
                attack_type = "replay"
                run_params = {"source_start_pct": 0.01}
                
            elif selected_mode == "injection":
                attack_type = "injection"
                print("\nSelect Injection Sub-Type:")
                print(" [1] Bias (Additive Shift)")
                print(" [2] Scale (Multiplicative Shift)")
                print(" [3] Absolute (Custom Fixed Override)")
                try:
                    inj_choice = int(input("Select Option (1-3): "))
                except ValueError:
                    inj_choice = 1
                
                inj_map = {1: "bias", 2: "scale", 3: "absolute"}
                inj_sub_type = inj_map.get(inj_choice, "bias")
                
                default_val = 10.0 if inj_sub_type == "bias" else (1.05 if inj_sub_type == "scale" else 0.0)
                try:
                    val_input = input(f"Enter injection value (default {default_val}): ")
                    inj_val = float(val_input) if val_input.strip() else default_val
                except ValueError:
                    inj_val = default_val
                
                run_params = {"type": inj_sub_type, "value": inj_val}
                
            elif selected_mode == "drift":
                attack_type = "drift"
                run_params = {"type": "additive", "max_value": 50.0}

    # -----------------------------------------------------------------
    # PATHWAY B: MEDIUM / HARD MODE (Automated Scenarios)
    # -----------------------------------------------------------------
    else:
        scenarios_pool = PRESET_SCENARIOS.get(difficulty, PRESET_SCENARIOS[2])
        chosen_scenario = random.choice(scenarios_pool)
        
        output_id = chosen_scenario["id"]
        print(f"\n>>> Launching Blind Evaluation Run: {chosen_scenario['name']} <<<")
        
        attack_type = chosen_scenario["type"]
        target_group = chosen_scenario["target_group"]
        start_pct = chosen_scenario["start_pct"]
        end_pct = chosen_scenario["end_pct"]
        run_params = chosen_scenario["params"]

    # -----------------------------------------------------------------
    # EXECUTION OF MATH ENGINE PIPELINE
    # -----------------------------------------------------------------
    start_idx = int(total_rows * start_pct)
    end_idx = int(total_rows * end_pct)
    targets = schema.get(target_group, schema["all"])

    c_type = run_params.get("condition_type", None)
    c_val = run_params.get("condition_val", run_params.get("conditional_val", None))

    if attack_type == "none":
        modified_data, attack_labels = apply_no_change(data)
    elif attack_type == "replay":
        modified_data, attack_labels = apply_replay_attack(data, start_idx, end_idx, total_rows, run_params, c_type, c_val, targets=targets)
    elif attack_type == "injection":
        modified_data, attack_labels = apply_injection_attack(data, start_idx, end_idx, targets, run_params, c_type, c_val)
    elif attack_type == "drift":
        modified_data, attack_labels = apply_drift_attack(data, start_idx, end_idx, targets, run_params, c_type, c_val)

    # -----------------------------------------------------------------
    # OUTPUT GENERATION (With Dynamic Precision Guard)
    # -----------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_main = os.path.join(OUTPUT_DIR, f"attack_{output_id}.jsonl")
    output_check = os.path.join(OUTPUT_DIR, f"attack_{output_id}_check.jsonl")

    print(f"\nWriting files to: '{OUTPUT_DIR}'...")
    with open(output_main, "w") as f_main, open(output_check, "w") as f_check:
        for row, is_attack in zip(modified_data, attack_labels):
            # Formatted inline guard to preserve raw FRQ float data
            formatted_row = {
                k: (v if k == "FRQ" else (round(v, 1) if "uJ" in k else round(v, 4))) if isinstance(v, float) else v 
                for k, v in row.items()
            }
            
            f_main.write(json.dumps(formatted_row) + "\n")
            
            check_row = formatted_row.copy()
            check_row["attack"] = is_attack
            f_check.write(json.dumps(check_row) + "\n")

    print(f"Successfully generated 'attack_{output_id}.jsonl' and validation blueprint.")

if __name__ == "__main__":
    main()