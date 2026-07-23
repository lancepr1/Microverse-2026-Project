"""A simple file-based message bus for the contract records.

Each lane appends its records to a JSONL file under `runs/<run_id>/`
and reads the files it depends on. Every run produces a durable,
replayable artifact rather than requiring a live service. See
.readme/io_records.md for the file layout and design rationale.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Type, TypeVar

from .contracts import RECORD_TYPES, _Record

T = TypeVar("T", bound=_Record)

RUNS_DIR = os.environ.get("MICROVERSE_RUNS", "runs")


def _path(run_id: str, name: str) -> str:
    """Builds (and ensures the existence of) the path for one record file.

    Args:
        run_id: Identifier for this run.
        name: Record stream name, e.g. "anchors".

    Returns:
        str: Path to runs/<run_id>/<name>.jsonl.
    """
    d = os.path.join(RUNS_DIR, run_id)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{name}.jsonl")


def write_records(run_id: str, name: str, records: Iterable[_Record]) -> str:
    """Appends records to runs/<run_id>/<name>.jsonl.

    Args:
        run_id: Identifier for this run.
        name: Record stream name, e.g. "anchors".
        records: Records to append. Each is written as one line,
            tagged with its type so the reader can reconstruct it.

    Returns:
        str: Path to the file written.
    """
    path = _path(run_id, name)
    with open(path, "a") as fh:
        for rec in records:
            line = {"_type": type(rec).__name__, "data": rec.to_dict()}
            fh.write(json.dumps(line) + "\n")
    return path


def read_records(run_id: str, name: str, expect: Type[T] = None) -> list:
    """Reads records back from runs/<run_id>/<name>.jsonl into their dataclass types.

    Args:
        run_id: Identifier for this run.
        name: Record stream name, e.g. "anchors".
        expect: If given, only records of this exact type are
            returned; others are skipped.

    Returns:
        list: Reconstructed record instances, or an empty list if the
        file doesn't exist.
    """
    path = _path(run_id, name)
    out: list = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            line = json.loads(raw)
            cls = RECORD_TYPES[line["_type"]]
            if expect is not None and cls is not expect:
                continue
            out.append(cls.from_dict(line["data"]))
    return out