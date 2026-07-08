#!/usr/bin/env python3
import json
import os
import sys

# =====================================================================
# CONFIGURATION BLOCK
# Change these parameters to alter attack behaviors, ranges, and types.
# =====================================================================

INPUT_FILE = "../../data/combined/run_2node.jsonl"  # Path to your input data file
SELECTED_OPTION = int(input("Enter which scenario you'd like to test (0-4): "))            # Choose option: 0, 1, 2, 3, or 4
if SELECTED_OPTION < 0 or SELECTED_OPTION > 4:
	print("Please select a number 0-4")
	SELECTED_OPTION = int(input("Enter which scenario you'd like to test (0-4): "))

# Target column(s) for math-based manipulations (e.g., 'FRQ' for ENF data)
TARGET_COLUMNS = ["FRQ"]

# --- Option 1: Extreme Change Parameters ---
EXTREME_START = 20
EXTREME_END = 40
EXTREME_VALUE = 0.0            # Extreme floor/ceiling value to test proof of concept

# --- Option 2: Replay Attack Parameters ---
REPLAY_START = 30
REPLAY_END = 40
REPLAY_SOURCE_START = 10      # Historical window index to clone and replay later

# --- Option 3: Injection Attack Parameters ---
INJECTION_START = 20
INJECTION_END = 40
INJECTION_TYPE = "bias"        # Options: "bias" (constant addition) or "scale" (multiplication)
INJECTION_BIAS = 10           # Value added to target column if type is "bias"
INJECTION_SCALE = 1.08         # Value multiplied by target column if type is "scale"

# --- Option 4: Drift Attack Parameters ---
DRIFT_START = 10
DRIFT_END = 100
DRIFT_MAX_VALUE = 5         # The peak error value reached at DRIFT_END
DRIFT_TYPE = "additive"        # Options: "additive" or "multiplicative"

# =====================================================================
# MODULAR ATTACK IMPLEMENTATIONS
# =====================================================================

def apply_no_change(data):
    """Option 0: Returns clean copy of data and zeroed attack flags."""
    modified_data = [row.copy() for row in data]
    attack_labels = [0] * len(data)
    return modified_data, attack_labels

def apply_extreme_change(data):
    """Option 1: Overwrites target metrics with a harsh constant value."""
    modified_data = []
    attack_labels = []
    for i, row in enumerate(data):
        new_row = row.copy()
        is_attack = 0
        if EXTREME_START <= i < EXTREME_END:
            is_attack = 1
            for col in TARGET_COLUMNS:
                if col in new_row:
                    new_row[col] = EXTREME_VALUE
        modified_data.append(new_row)
        attack_labels.append(is_attack)
    return modified_data, attack_labels

def apply_replay_attack(data):
    """
    Option 2: Clones a past time window and overwrites a future window.
    Preserves original index telemetry sequences to avoid timeline breakages.
    """
    modified_data = []
    attack_labels = []
    for i, row in enumerate(data):
        new_row = row.copy()
        is_attack = 0
        if REPLAY_START <= i < REPLAY_END:
            is_attack = 1
            source_idx = REPLAY_SOURCE_START + (i - REPLAY_START)
            if 0 <= source_idx < len(data):
                source_row = data[source_idx]
                # Replace all environment/sensor readings, but keep sequence index unchanged
                for key in row.keys():
                    if key != "index":
                        new_row[key] = source_row.get(key, row[key])
        modified_data.append(new_row)
        attack_labels.append(is_attack)
    return modified_data, attack_labels

def apply_injection_attack(data):
    """Option 3: Injects a static bias or a scaling factor factor into data."""
    modified_data = []
    attack_labels = []
    for i, row in enumerate(data):
        new_row = row.copy()
        is_attack = 0
        if INJECTION_START <= i < INJECTION_END:
            is_attack = 1
            for col in TARGET_COLUMNS:
                if col in new_row and isinstance(new_row[col], (int, float)):
                    if INJECTION_TYPE == "bias":
                        new_row[col] += INJECTION_BIAS
                    elif INJECTION_TYPE == "scale":
                        new_row[col] *= INJECTION_SCALE
        modified_data.append(new_row)
        attack_labels.append(is_attack)
    return modified_data, attack_labels

def apply_drift_attack(data):
    """Option 4: Implements a slow linear accumulation of data measurement drift."""
    modified_data = []
    attack_labels = []
    duration = DRIFT_END - DRIFT_START
    for i, row in enumerate(data):
        new_row = row.copy()
        is_attack = 0
        if DRIFT_START <= i < DRIFT_END:
            is_attack = 1
            # Calculate slope fraction from 0.0 to 1.0
            factor = (i - DRIFT_START) / duration
            for col in TARGET_COLUMNS:
                if col in new_row and isinstance(new_row[col], (int, float)):
                    if DRIFT_TYPE == "additive":
                        new_row[col] += DRIFT_MAX_VALUE * factor
                    elif DRIFT_TYPE == "multiplicative":
                        new_row[col] *= (1.0 + (DRIFT_MAX_VALUE * factor))
        modified_data.append(new_row)
        attack_labels.append(is_attack)
    return modified_data, attack_labels

# =====================================================================
# MAIN RUNNER EXECUTION
# =====================================================================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        sys.exit(1)

    print(f"Reading input data from '{INPUT_FILE}'...")
    with open(INPUT_FILE, "r") as f:
        data = [json.loads(line) for line in f]

    print(f"Processing Option {SELECTED_OPTION}...")
    if SELECTED_OPTION == 0:
        modified_data, attack_labels = apply_no_change(data)
    elif SELECTED_OPTION == 1:
        modified_data, attack_labels = apply_extreme_change(data)
    elif SELECTED_OPTION == 2:
        modified_data, attack_labels = apply_replay_attack(data)
    elif SELECTED_OPTION == 3:
        modified_data, attack_labels = apply_injection_attack(data)
    elif SELECTED_OPTION == 4:
        modified_data, attack_labels = apply_drift_attack(data)
    else:
        print("Invalid choice! Choose an option between 0 and 4.")
        sys.exit(1)

    # Establish generic obfuscated output filenames
    output_main = f"attack_{SELECTED_OPTION}.jsonl"
    output_check = f"attack_{SELECTED_OPTION}_check.jsonl"

    print(f"Writing outputs: '{output_main}' and '{output_check}'...")
    with open(output_main, "w") as f_main, open(output_check, "w") as f_check:
        for row, is_attack in zip(modified_data, attack_labels):
            # 1. Output altered stream file (Identical format to original)
            f_main.write(json.dumps(row) + "\n")
            
            # 2. Output verification file with a flat, obscure 'attack' column added
            check_row = row.copy()
            check_row["attack"] = is_attack
            f_check.write(json.dumps(check_row) + "\n")

    print("Success! Data simulation pipeline execution complete.")

if __name__ == "__main__":
    main()
