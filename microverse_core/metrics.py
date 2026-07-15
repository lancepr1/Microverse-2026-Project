#!/usr/bin/env python3
import os
import json
import sys
import argparse

# =====================================================================
# STATIC PATH RESOLUTION MECHANICS
# =====================================================================
def resolve_project_paths(scenario_id):
    """
    Traverses upwards from the script's physical location to find the 
    enclosing project root containing the 'lanes' tree topology.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current = script_dir
    root_dir = None

    # Walk up parent folders to anchor the universal project root
    while current:
        if os.path.isdir(os.path.join(current, "lanes")):
            root_dir = current
            break
        parent = os.path.dirname(current)
        if parent == current:  # Hit system root boundary
            break
        current = parent

    # Fallback to standard 2-level-up offset if layout markers are missing
    if not root_dir:
        root_dir = os.path.normpath(os.path.join(script_dir, "../.."))

    attacks_dir = os.path.join(root_dir, "lanes", "marchisano_attacks", "outputs")
    verification_dir = os.path.join(root_dir, "lanes", "leiva_verification", "outputs")

    # Handle flexible id inputs (e.g., handles '1' gracefully if file is 'attack_easy_1_check.jsonl')
    truth_file = os.path.join(attacks_dir, f"attack_{scenario_id}_check.jsonl")
    easy_truth_file = os.path.join(attacks_dir, f"attack_easy_{scenario_id}_check.jsonl")

    if not os.path.exists(truth_file) and os.path.exists(easy_truth_file):
        truth_file = easy_truth_file

    detector_file = os.path.join(verification_dir, "for_scoreboard.jsonl")

    return truth_file, detector_file

# =====================================================================
# MATHEMATICAL EVALUATION ENGINE
# =====================================================================
def calculate_confusion_matrix(truth_list, pred_list):
    """
    Computes standard binary classification metrics from compiled time series.
    """
    tp, fp, tn, fn = 0, 0, 0, 0

    for t, p in zip(truth_list, pred_list):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 1 and p == 0:
            fn += 1

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, 
        "f1": f1_score, "fpr": fpr
    }

def calculate_time_to_detection(truth_list, pred_list):
    """
    Extracts explicit contiguous windows of active ground-truth attacks and
    measures the telemetry frame offset latency until the first detector flag.
    """
    windows = []
    start_idx = None

    # Step 1: Isolate active attack frames into discrete bounded intervals
    for i, t in enumerate(truth_list):
        if t == 1 and start_idx is None:
            start_idx = i
        elif t == 0 and start_idx is not None:
            windows.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None:
        windows.append((start_idx, len(truth_list) - 1))

    if not windows:
        return []

    window_results = []
    
    # Step 2: Measure tracking delay inside each specific attack window
    for w_idx, (w_start, w_end) in enumerate(windows):
        detected_frame = None
        for i in range(w_start, w_end + 1):
            if pred_list[i] == 1:
                detected_frame = i
                break
        
        if detected_frame is not None:
            latency = detected_frame - w_start
            window_results.append({
                "window": w_idx + 1,
                "start": w_start,
                "end": w_end,
                "ttd": latency
            })
        else:
            window_results.append({
                "window": w_idx + 1,
                "start": w_start,
                "end": w_end,
                "ttd": "Infinite / Undetected"
            })

    return window_results

# =====================================================================
# DASHBOARD RENDERING LAYOUT
# =====================================================================
def print_evaluation_dashboard(scenario_id, truth_file, detector_file, results_by_layer):
    """
    Generates a structured, unified diagnostics panel comparing framework performance.
    """
    print("=" * 80)
    print(f"       MULTI-LAYER DETECTOR EVALUATION REPORT: SCENARIO {scenario_id}")
    print("=" * 80)
    print(f"Ground Truth Reference : {os.path.basename(truth_file)}")
    print(f"Detector Output Source : {os.path.basename(detector_file)}")
    print("=" * 80)
    
    for layer_name, data in results_by_layer.items():
        strict_m = data["strict_m"]
        lenient_m = data["lenient_m"]
        strict_ttd = data["strict_ttd"]
        lenient_ttd = data["lenient_ttd"]
        
        print(f"\n >>> LAYER: {layer_name} <<<")
        print("-" * 80)
        row_fmt = " {:<28} | {:<22} | {:<22}"
        print(row_fmt.format("Metric Parameter", "Strict Mode (1.0)", "Lenient Mode (>=0.5)"))
        print("-" * 80)
        print(row_fmt.format("True Positives (TP)", strict_m["tp"], lenient_m["tp"]))
        print(row_fmt.format("False Positives (FP)", strict_m["fp"], lenient_m["fp"]))
        print(row_fmt.format("True Negatives (TN)", strict_m["tn"], lenient_m["tn"]))
        print(row_fmt.format("False Negatives (FN)", strict_m["fn"], lenient_m["fn"]))
        print("-" * 80)
        print(row_fmt.format("Precision", f"{strict_m['precision']:.4%}", f"{lenient_m['precision']:.4%}"))
        print(row_fmt.format("Recall (Sensitivity)", f"{strict_m['recall']:.4%}", f"{lenient_m['recall']:.4%}"))
        print(row_fmt.format("F1-Score", f"{strict_m['f1']:.4%}", f"{lenient_m['f1']:.4%}"))
        print(row_fmt.format("False Positive Rate (FPR)", f"{strict_m['fpr']:.4%}", f"{lenient_m['fpr']:.4%}"))
        print("-" * 80)
        
        # Print latency profiles if attack windows were found
        if strict_ttd:
            print(" Latency Profiles: Time-To-Detection (TTD)")
            ttd_fmt = "   Window #{:<2} [Frames {:>4}-{:<4}] | Strict TTD: {:<12} | Lenient TTD: {:<12}"
            for s_t, l_t in zip(strict_ttd, lenient_ttd):
                s_val = f"{s_t['ttd']} frames" if isinstance(s_t['ttd'], int) else s_t['ttd']
                l_val = f"{l_t['ttd']} frames" if isinstance(l_t['ttd'], int) else l_t['ttd']
                print(ttd_fmt.format(s_t['window'], s_t['start'], s_t['end'], s_val, l_val))
            
            s_valid = [x['ttd'] for x in strict_ttd if isinstance(x['ttd'], int)]
            l_valid = [x['ttd'] for x in lenient_ttd if isinstance(x['ttd'], int)]
            s_avg = f"{sum(s_valid)/len(s_valid):.2f} frames" if s_valid else "N/A"
            l_avg = f"{sum(l_valid)/len(l_valid):.2f} frames" if l_valid else "N/A"
            print(f"   Average Response Delay   | Strict Avg: {s_avg:<10} | Lenient Avg: {l_avg:<10}")
        else:
            print(" Latency Profiles: No active attack windows registered on this layer.")
        print("-" * 80)
    
    print("=" * 80)
    print(" METRIC GLOSSARY & OPERATIONAL INTERPRETATION")
    print("-" * 80)
    print(" • Precision                : Out of all frames flagged as anomalous by the")
    print("                              detector, what % were actually true attacks?")
    print("                              (Higher value = high alert trustworthiness/reliability)")
    print(" • Recall (Sensitivity)     : Out of all total attack frames that actually happened,")
    print("                              what % did the detector catch?")
    print("                              (Higher value = complete coverage, low attack leakages)")
    print(" • F1-Score                 : The harmonic mean of Precision and Recall, providing")
    print("                              a unified balance rating for imbalanced datasets.")
    print(" • False Positive Rate (FPR): What % of clean, completely normal baseline frames were")
    print("                              wrongly flagged as anomalies? (Lower is better to avoid")
    print("                              operational noise and user alert fatigue).")
    print(" • Time-to-Detection (TTD) : The frame-offset delta between the exact start of an")
    print("                              attack phase and the detector's first flagged alert.")
    print("                              (0 frames = instant, immediate edge-trigger response)")
    print("=" * 80)

# =====================================================================
# MAIN RUNNER EXECUTION
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection telemetry profiles against system ground truth.")
    parser.add_argument(
        "--id", 
        required=True, 
        help="Specify the target Scenario Evaluation ID (e.g., '304' or 'easy_1')"
    )
    args = parser.parse_args()

    truth_path, detector_path = resolve_project_paths(args.id)

    # File Existence Verification Gates
    if not os.path.exists(truth_path):
        print(f"Error: Missing ground truth file at path:\n '{truth_path}'")
        sys.exit(1)
    if not os.path.exists(detector_path):
        print(f"Error: Missing detector scoreboard output file at path:\n '{detector_path}'")
        sys.exit(1)

    # Ingest Stream Data Arrays
    truth_rows = []
    with open(truth_path, "r") as f:
        for line in f:
            if line.strip():
                truth_rows.append(json.loads(line))

    detector_rows = []
    with open(detector_path, "r") as f:
        for line in f:
            if line.strip():
                detector_rows.append(json.loads(line))

    # Synchronous Integrity Inspection Gate
    if len(truth_rows) != len(detector_rows):
        print(f"Error Frame Alignment Mismatch: Ground Truth contains {len(truth_rows)} rows, "
              f"but Detector Output contains {len(detector_rows)} rows.")
        sys.exit(1)

    # Identify all base subsystems present across the files
    truth_keys = list(truth_rows[0].keys())
    detector_keys = list(detector_rows[0].keys())

    # Build unique sets of base layers (e.g., "ENF", "x3115c0s33b0n0")
    bases = set()
    for k in truth_keys:
        if k.endswith("_attack") and k != "attack":
            bases.add(k[:-7])
    for k in detector_keys:
        if k.endswith("_status") and k != "status":
            bases.add(k[:-7])

    results_by_layer = {}

    # Setup execution layers: Always evaluate Global/System, then individual sub-layers
    layers_to_eval = [("Global / System", None)]
    for base in sorted(list(bases)):
        layers_to_eval.append((base, base))

    for layer_name, base in layers_to_eval:
        ground_truth_attack = []
        strict_predictions = []
        lenient_predictions = []

        for idx, (t_row, d_row) in enumerate(zip(truth_rows, detector_rows)):
            if t_row.get("index") != d_row.get("index"):
                print(f"Error Chronological Mismatch at file offset row {idx}: "
                      f"Ground Truth index is {t_row.get('index')}, but Detector index is {d_row.get('index')}.")
                sys.exit(1)

            # --- RESOLVE TRUTH VALUE ---
            if base is None:
                # Global / System: Use "attack", fallback to max of any "_attack" column
                specific_attacks = [t_row[k] for k in t_row if k.endswith("_attack") and k != "attack"]
                if "attack" in t_row:
                    truth_val = int(t_row["attack"])
                elif specific_attacks:
                    truth_val = int(max(specific_attacks))
                else:
                    truth_val = 0
            else:
                # Sub-layer Specific: Use "{base}_attack", fallback to global "attack"
                truth_val = int(t_row.get(f"{base}_attack", t_row.get("attack", 0)))

            # --- RESOLVE DETECTOR VALUE ---
            if base is None:
                # Global / System: Use "status", fallback to max of any "_status" column
                status_keys = [k for k in d_row if k.endswith("_status") and k != "status"]
                if "status" in d_row:
                    status_val = float(d_row["status"])
                elif status_keys:
                    status_val = float(max(d_row[k] for k in status_keys))
                else:
                    status_val = 0.0
            else:
                # Sub-layer Specific: Use "{base}_status", fallback to generic "status"
                status_val = float(d_row.get(f"{base}_status", d_row.get("status", 0.0)))

            ground_truth_attack.append(truth_val)
            
            # Mapping Binary Strict Target Layer (status == 1.0)
            strict_predictions.append(1 if status_val == 1.0 else 0)
            
            # Mapping Binary Lenient Target Layer (status >= 0.5)
            lenient_predictions.append(1 if status_val >= 0.5 else 0)

        # Compute Statistical Metrics Profiles
        strict_metrics = calculate_confusion_matrix(ground_truth_attack, strict_predictions)
        lenient_metrics = calculate_confusion_matrix(ground_truth_attack, lenient_predictions)

        # Compute Latency Metrics Profiles
        strict_ttd_profile = calculate_time_to_detection(ground_truth_attack, strict_predictions)
        lenient_ttd_profile = calculate_time_to_detection(ground_truth_attack, lenient_predictions)

        results_by_layer[layer_name] = {
            "strict_m": strict_metrics,
            "lenient_m": lenient_metrics,
            "strict_ttd": strict_ttd_profile,
            "lenient_ttd": lenient_ttd_profile
        }

    # Render Results Dashboard Panel
    print_evaluation_dashboard(
        args.id, truth_path, detector_path, results_by_layer
    )

if __name__ == "__main__":
    main()