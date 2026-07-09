# Data Center Telemetry Attack Simulator

This lane contains a pipeline tool to simulate attacks in a transmission scenario.

## What It Does
The script takes in a combined data file from `data/combined/` in the format `run_#node.jsonl`. It provides 3 options of difficulty for generating attacks: Easy, Medium, Hard. Easy allows for manual configuration where the user knows the attacks, this is meant for Proof of Concept tests. Medium and Hard difficulties choose from preset attacks that range from smaller changes in values to following model signatures. It then generates outputs in the form of `attack_ID#.jsonl` and `attack_ID#_check.jsonl`. The first is used for the verification, blender, and dashboard. The second is used for scoring metrics.

## How to Run It
1. Ensure your source file (e.g., `run_2node.jsonl`) is in the data/combined/ directory
2. Run `python3 attack.py
3. Select from the three difficulties
4. If easy is picked, go through configuration options to pick the attack and values you want to test
5. Otherwise, wait for output to be generated.

## Who to ask
Refer to Ethan archisano for any questions on the script.

## Legend

* ID #
    * 1-100 - Easy
    * 201-300 - Medium
    * 301-400 - Hard
* Input File
    * data/combined/run_#node.jsonl
* Output Files
    * lanes/marchisano_attacks/outputs/attack_ID#.jsonl
    * lanes/marchisano_attacks/outputs/attack_ID#_check.jsonl
