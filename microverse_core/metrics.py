"""Evaluates detection performance against ground truth and prints a report.

Compares a detector's scoreboard output against a ground-truth attack
file, computing precision/recall/F1/FPR and time-to-detection per
layer (global, plus per node/channel), and appends the results to a
persistent CSV history. See .readme/metrics.md for the CSV schema and
known limitations.

Example:
    python microverse_core/metrics.py --id 304 --difficulty hard
"""

import os
import csv
import json
import sys
import argparse
import datetime


def resolve_project_paths(scenario_id):
    """Resolves the ground-truth and detector output paths for one scenario.

    Args:
        scenario_id: Scenario identifier, e.g. "304" or "easy_1".

    Returns:
        tuple[str, str]: (truth_file, detector_file) paths.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current = script_dir
    root_dir = None

    while current:
        if os.path.isdir(os.path.join(current, "lanes")):
            root_dir = current
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    if not root_dir:
        root_dir = os.path.normpath(os.path.join(script_dir, "../.."))

    attacks_dir = os.path.join(root_dir, "lanes", "marchisano_attacks", "outputs")
    verification_dir = os.path.join(root_dir, "lanes", "leiva_verification", "outputs")

    truth_file = os.path.join(attacks_dir, f"attack_{scenario_id}_check.jsonl")
    easy_truth_file = os.path.join(attacks_dir, f"attack_easy_{scenario_id}_check.jsonl")

    if not os.path.exists(truth_file) and os.path.exists(easy_truth_file):
        truth_file = easy_truth_file

    detector_file = os.path.join(verification_dir, "for_scoreboard.jsonl")

    return truth_file, detector_file


def _resolve_root_dir():
    """Finds the repo root by walking up from this file until a 'lanes' folder is found.

    Returns:
        str: Path to the repo root.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current = script_dir
    root_dir = None
    while current:
        if os.path.isdir(os.path.join(current, "lanes")):
            root_dir = current
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    if not root_dir:
        root_dir = os.path.normpath(os.path.join(script_dir, "../.."))
    return root_dir


def calculate_confusion_matrix(truth_list, pred_list):
    """Computes standard binary classification metrics from two label sequences.

    Args:
        truth_list: Ground-truth binary labels (0/1).
        pred_list: Predicted binary labels (0/1), same length.

    Returns:
        dict: tp, fp, tn, fn, precision, recall, f1, and fpr.
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
    """Measures detection latency for every contiguous ground-truth attack window.

    Args:
        truth_list: Ground-truth binary labels (0/1).
        pred_list: Predicted binary labels (0/1), same length.

    Returns:
        list[dict]: One entry per attack window, each with `window`,
        `start`, `end`, and `ttd` (frame offset to first detection, or
        the string "Infinite / Undetected" if never detected).
    """
    windows = []
    start_idx = None

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


def append_to_history_csv(scenario_id, difficulty, truth_file, detector_file, results_by_layer):
    """Appends one row per evaluated layer to runs/detection_history.csv.

    Purely additive -- does not affect the console report produced by
    print_evaluation_dashboard(). See .readme/metrics.md for the
    difficulty-tier auto-detection caveat.

    Args:
        scenario_id: Scenario identifier for this run.
        difficulty: "easy", "medium", "hard", or "unknown".
        truth_file: Path to the ground-truth file used.
        detector_file: Path to the detector output file used.
        results_by_layer: Per-layer results as built by main().
    """
    root_dir = _resolve_root_dir()
    history_dir = os.path.join(root_dir, "runs")
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, "detection_history.csv")

    file_exists = os.path.exists(history_path)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    fieldnames = [
        "timestamp", "scenario_id", "difficulty", "layer",
        "strict_tp", "strict_fp", "strict_tn", "strict_fn",
        "strict_precision", "strict_recall", "strict_f1", "strict_fpr",
        "lenient_tp", "lenient_fp", "lenient_tn", "lenient_fn",
        "lenient_precision", "lenient_recall", "lenient_f1", "lenient_fpr",
        "strict_avg_ttd", "lenient_avg_ttd",
        "truth_file", "detector_file",
    ]

    with open(history_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for layer_name, data in results_by_layer.items():
            strict_m = data["strict_m"]
            lenient_m = data["lenient_m"]
            strict_ttd = data["strict_ttd"]
            lenient_ttd = data["lenient_ttd"]

            s_valid = [x["ttd"] for x in strict_ttd if isinstance(x["ttd"], int)]
            l_valid = [x["ttd"] for x in lenient_ttd if isinstance(x["ttd"], int)]
            strict_avg_ttd = round(sum(s_valid) / len(s_valid), 3) if s_valid else ""
            lenient_avg_ttd = round(sum(l_valid) / len(l_valid), 3) if l_valid else ""

            writer.writerow({
                "timestamp": timestamp,
                "scenario_id": scenario_id,
                "difficulty": difficulty,
                "layer": layer_name,
                "strict_tp": strict_m["tp"], "strict_fp": strict_m["fp"],
                "strict_tn": strict_m["tn"], "strict_fn": strict_m["fn"],
                "strict_precision": round(strict_m["precision"], 6),
                "strict_recall": round(strict_m["recall"], 6),
                "strict_f1": round(strict_m["f1"], 6),
                "strict_fpr": round(strict_m["fpr"], 6),
                "lenient_tp": lenient_m["tp"], "lenient_fp": lenient_m["fp"],
                "lenient_tn": lenient_m["tn"], "lenient_fn": lenient_m["fn"],
                "lenient_precision": round(lenient_m["precision"], 6),
                "lenient_recall": round(lenient_m["recall"], 6),
                "lenient_f1": round(lenient_m["f1"], 6),
                "lenient_fpr": round(lenient_m["fpr"], 6),
                "strict_avg_ttd": strict_avg_ttd,
                "lenient_avg_ttd": lenient_avg_ttd,
                "truth_file": os.path.basename(truth_file),
                "detector_file": os.path.basename(detector_file),
            })

    print(f"\n📊 Logged {len(results_by_layer)} layer(s) to {history_path} "
          f"(difficulty recorded as: {difficulty})")


def print_evaluation_dashboard(scenario_id, truth_file, detector_file, results_by_layer):
    """Prints a formatted, per-layer detection evaluation report to the console.

    Args:
        scenario_id: Scenario identifier for this run.
        truth_file: Path to the ground-truth file used.
        detector_file: Path to the detector output file used.
        results_by_layer: Per-layer results as built by main().
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


def main() -> None:
    """Parses CLI arguments, evaluates the scenario, prints the report, and logs history."""
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection telemetry profiles against system ground truth.")
    parser.add_argument(
        "--id",
        required=True,
        help="Specify the target Scenario Evaluation ID (e.g., '304' or 'easy_1')"
    )
    parser.add_argument(
        "--difficulty",
        default=None,
        choices=["easy", "medium", "hard"],
        help="Optional. Recorded in runs/detection_history.csv for this run. "
             "Auto-detected as 'easy' when the scenario id itself indicates "
             "Easy mode -- Medium and Hard cannot be told apart automatically, "
             "so pass this explicitly for those if you want the history log "
             "to reflect it correctly."
    )
    args = parser.parse_args()

    truth_path, detector_path = resolve_project_paths(args.id)

    if not os.path.exists(truth_path):
        print(f"Error: Missing ground truth file at path:\n '{truth_path}'")
        sys.exit(1)
    if not os.path.exists(detector_path):
        print(f"Error: Missing detector scoreboard output file at path:\n '{detector_path}'")
        sys.exit(1)

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

    if len(truth_rows) != len(detector_rows):
        print(f"Error Frame Alignment Mismatch: Ground Truth contains {len(truth_rows)} rows, "
              f"but Detector Output contains {len(detector_rows)} rows.")
        sys.exit(1)

    truth_keys = list(truth_rows[0].keys())
    detector_keys = list(detector_rows[0].keys())

    bases = set()
    for k in truth_keys:
        if k.endswith("_attack") and k != "attack":
            bases.add(k[:-7])
    for k in detector_keys:
        if k.endswith("_status") and k != "status":
            bases.add(k[:-7])

    results_by_layer = {}

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

            if base is None:
                specific_attacks = [t_row[k] for k in t_row if k.endswith("_attack") and k != "attack"]
                if "attack" in t_row:
                    truth_val = int(t_row["attack"])
                elif specific_attacks:
                    truth_val = int(max(specific_attacks))
                else:
                    truth_val = 0
            else:
                truth_val = int(t_row.get(f"{base}_attack", t_row.get("attack", 0)))

            if base is None:
                status_keys = [k for k in d_row if k.endswith("_status") and k != "status"]
                if "status" in d_row:
                    status_val = float(d_row["status"])
                elif status_keys:
                    status_val = float(max(d_row[k] for k in status_keys))
                else:
                    status_val = 0.0
            else:
                status_val = float(d_row.get(f"{base}_status", d_row.get("status", 0.0)))

            ground_truth_attack.append(truth_val)
            strict_predictions.append(1 if status_val == 1.0 else 0)
            lenient_predictions.append(1 if status_val >= 0.5 else 0)

        strict_metrics = calculate_confusion_matrix(ground_truth_attack, strict_predictions)
        lenient_metrics = calculate_confusion_matrix(ground_truth_attack, lenient_predictions)

        strict_ttd_profile = calculate_time_to_detection(ground_truth_attack, strict_predictions)
        lenient_ttd_profile = calculate_time_to_detection(ground_truth_attack, lenient_predictions)

        results_by_layer[layer_name] = {
            "strict_m": strict_metrics,
            "lenient_m": lenient_metrics,
            "strict_ttd": strict_ttd_profile,
            "lenient_ttd": lenient_ttd_profile
        }

    print_evaluation_dashboard(
        args.id, truth_path, detector_path, results_by_layer
    )

    difficulty = args.difficulty
    if difficulty is None:
        difficulty = "easy" if "easy" in args.id.lower() else "unknown"
    append_to_history_csv(args.id, difficulty, truth_path, detector_path, results_by_layer)


if __name__ == "__main__":
    main()