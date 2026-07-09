"""
tools/normalize_node_ids.py -- rewrites a combined-format JSONL file
(lanes/leiva_verification/outputs/for_dashboard.jsonl by default) so every
node-prefixed column uses the dashboard's "node00".."nodeNN" naming instead
of whatever raw node identifier the source data used (e.g. SLURM hostnames
like "x3105c0s37b0n0").

Each row is one polling interval for the whole rack: shared columns
("index", "FRQ", and whatever else isn't node-prefixed, e.g. "attack",
"status") plus one block of "<node_id>_gpu-N[...]" / "<node_id>_cpu-N[...]"
columns per node. This script never assumes what <node_id> looks like --
it finds every column matching "<anything>_gpu-N[...]" or
"<anything>_cpu-N[...]", collects the distinct prefixes, sorts them, and
renumbers them node00, node01, ... in that sorted order (the same
alphabetical-by-original-id scheme tools/split_16node.py already uses, so
numbering stays stable across regeneration). Every other column, and the
row order, is left untouched.

This keeps the dashboard's node-detection regex in data_feed.py
(_NODE_PREFIX_RE = r"^(node\\d+)_") simple and lets ui/*.py's "Node 00".."Node
15" display labels keep working, regardless of what naming scheme upstream
data uses.

    python lanes/mccray_dashboard/tools/normalize_node_ids.py
    python lanes/mccray_dashboard/tools/normalize_node_ids.py \\
        --input data/combined/run_16node.jsonl --output /tmp/normalized.jsonl
"""
import argparse
import json
import re
from pathlib import Path

_NODE_COLUMN_RE = re.compile(r"^(.+?)_(?:gpu|cpu)-\d+(?:-core)?\[")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PATH = _REPO_ROOT / "lanes" / "leiva_verification" / "outputs" / "for_dashboard.jsonl"


def _discover_node_ids(rows: list[dict]) -> list[str]:
    ids = set()
    for row in rows:
        for key in row:
            match = _NODE_COLUMN_RE.match(key)
            if match:
                ids.add(match.group(1))
    return sorted(ids)


def normalize(input_path: Path, output_path: Path) -> dict[str, str]:
    rows = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    raw_ids = _discover_node_ids(rows)
    rename = {raw_id: f"node{i:02d}" for i, raw_id in enumerate(raw_ids)}

    with open(output_path, "w") as out_fh:
        for row in rows:
            new_row = {}
            for key, value in row.items():
                match = _NODE_COLUMN_RE.match(key)
                if match and match.group(1) in rename:
                    raw_id = match.group(1)
                    key = rename[raw_id] + key[len(raw_id):]
                new_row[key] = value
            out_fh.write(json.dumps(new_row) + "\n")

    return rename


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite a combined-format JSONL file so node-prefixed "
                    "columns use node00..nodeNN naming instead of raw "
                    "hostnames/ids."
    )
    parser.add_argument("--input", type=Path, default=_DEFAULT_PATH,
                         help="Path to the JSONL file to normalize")
    parser.add_argument("--output", type=Path, default=None,
                         help="Path to write the normalized JSONL to "
                              "(defaults to overwriting --input in place, "
                              "since these are regenerated pipeline outputs)")
    args = parser.parse_args()
    output_path = args.output or args.input

    rename = normalize(args.input, output_path)
    if not rename:
        print(f"No node-prefixed columns found in {args.input} -- nothing to rename.")
        return

    for raw_id, node_id in rename.items():
        print(f"{raw_id} -> {node_id}")
    print(f"Wrote {len(rename)} node(s) -> {output_path}")


if __name__ == "__main__":
    main()
