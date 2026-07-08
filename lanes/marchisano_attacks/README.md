# Data Center Telemetry Attack Simulator

This lane contains a pipeline tool to simulate attacks in a transmission scenario.

## What It Does
The script reads an authentic data center telemetry log (`.jsonl` format) and introduces controlled anomalies to simulate various sensor and transmission tampering vectors. It currently offers 5 different scenarios.

To ensure blind testing conditions for downstream digital twin detection mechanisms, the script generates two outputs (`attack_#.jsonl` and `attack_#_check.jsonl`) that hide the specific type of attack vector used. The frst file is to be used for verification and display, while the check is used for metrics.

## How to Run It
1. Ensure your source file (e.g., `run_2node.jsonl`) is in the data/combined/ directory.
2. Run python3 attack.py and input your scenario (0-4).
