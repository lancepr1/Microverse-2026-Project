"""
io_records.py: a dead-simple file bus for the contract records.

Week three is when components start talking. Rather than stand up a live
service on day one, each lane appends its records to a JSONL file under runs/
and reads the files it depends on. It is boring, debuggable, and gives every
run a permanent artifact you can replay. Move to something fancier only if a
real need shows up.

    runs/<run_id>/power.jsonl          PowerSample      (loaders)
    runs/<run_id>/anchors.jsonl        AnchorRecord     (Leiva)
    runs/<run_id>/verification.jsonl   VerificationResult (Leiva)
    runs/<run_id>/attacks.jsonl        AttackEvent      (Marchisano)
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Type, TypeVar

from .contracts import RECORD_TYPES, _Record

T = TypeVar("T", bound=_Record)

RUNS_DIR = os.environ.get("MICROVERSE_RUNS", "runs")


def _path(run_id: str, name: str) -> str:
    d = os.path.join(RUNS_DIR, run_id)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{name}.jsonl")


def write_records(run_id: str, name: str, records: Iterable[_Record]) -> str:
    """Append records to runs/<run_id>/<name>.jsonl. Each line is one record,
    tagged with its type so the reader can reconstruct it."""
    path = _path(run_id, name)
    with open(path, "a") as fh:
        for rec in records:
            line = {"_type": type(rec).__name__, "data": rec.to_dict()}
            fh.write(json.dumps(line) + "\n")
    return path


def read_records(run_id: str, name: str, expect: Type[T] = None) -> list:
    """Read records back into their dataclass types."""
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
