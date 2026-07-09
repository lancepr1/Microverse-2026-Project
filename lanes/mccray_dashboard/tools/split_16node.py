"""
tools/split_16node.py -- splits the combined multi-node recording
(data/combined/run_16node.jsonl) into one single-node JSONL file per node,
in the same schema as data/run01.jsonl, so the dashboard's existing
single-node TelemetrySample parser (see models.py) can load any of them
unchanged.

Each line of run_16node.jsonl is one polling interval for the whole rack:
"index" and "FRQ" (shared across all 16 nodes -- one rack-level ENF/PDU
reading per interval, not per node), plus 16 blocks of
"<hostname>_gpu-N[...]" / "<hostname>_cpu-N[...]" columns, one block per
node, e.g. "x3105c0s37b0n0_gpu-0[W]". This script regroups those columns by
hostname, strips the "<hostname>_" prefix, and re-attaches the shared
index/FRQ, so each output row has the same "index"/"FRQ"/"gpu-N[...]"/
"cpu-N[...]" shape as run01.jsonl. One output file is written per node,
named "node00.jsonl".."node15.jsonl" (numbered by sorting the source
hostnames alphabetically, the same order data_feed.list_node_ids() already
sorts in, so the numbering stays stable across regeneration), to data/
(see data/README.md for the full writeup). The original hostname is kept
in the printed summary for traceability back to the source recording.

    python lanes/mccray_dashboard/tools/split_16node.py
"""
import argparse
import json
import re
from pathlib import Path

_NODE_KEY_RE = re.compile(r"^(x\d+c\d+s\d+b\d+n\d+)_(.+)$")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_INPUT = _REPO_ROOT / "data" / "combined" / "run_16node.jsonl"
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"


def split(input_path: Path, output_dir: Path) -> dict[str, tuple[str, int]]:
    per_node_records: dict[str, list[dict]] = {}

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            shared = {"index": record["index"], "FRQ": record["FRQ"]}

            nodes_in_record: dict[str, dict] = {}
            for key, value in record.items():
                match = _NODE_KEY_RE.match(key)
                if not match:
                    continue
                hostname, field = match.groups()
                nodes_in_record.setdefault(hostname, dict(shared))[field] = value

            for hostname, node_record in nodes_in_record.items():
                per_node_records.setdefault(hostname, []).append(node_record)

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for i, (hostname, records) in enumerate(sorted(per_node_records.items())):
        node_id = f"node{i:02d}"
        out_path = output_dir / f"{node_id}.jsonl"
        with open(out_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        counts[node_id] = (hostname, len(records))

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Split data/combined/run_16node.jsonl into one "
                    "single-node JSONL file per node under data/."
    )
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT,
                         help="Path to the combined 16-node JSONL recording")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR,
                         help="Directory to write node00.jsonl..node15.jsonl files into")
    args = parser.parse_args()

    counts = split(args.input, args.output_dir)
    for node_id, (hostname, count) in counts.items():
        print(f"{node_id} ({hostname}): {count} rows -> {args.output_dir / (node_id + '.jsonl')}")


if __name__ == "__main__":
    main()
