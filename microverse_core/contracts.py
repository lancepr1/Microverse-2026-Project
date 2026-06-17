"""
contracts.py: the interface agreements between every lane.

This is the most important file in the integration layer. If the five lanes
agree on these record shapes in week one, they can build in parallel and the
week-3 integration does not turn into a rewrite.

Rules of the road:
  1. Change these only by pull request, with a heads-up at the all-hands.
     Every lane imports this module, so a silent rename here breaks four
     people downstream.
  2. Keep this file pure standard library. No bpy, no numpy. It must import
     cleanly in CI and inside Blender's bundled Python.

Who owns what (producer -> consumer):
  StateVariable      Hendricks produces, McCray and Leiva consume.
  PowerSample        data_loaders produces, Hendricks consumes.
  AnchorRecord       Leiva produces, McCray displays.
  VerificationResult Leiva produces, McCray displays, Marchisano scores against.
  AttackEvent        Marchisano produces, used as ground truth for metrics.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Optional


# --------------------------------------------------------------------------
# Enumerations. Use these instead of bare strings so a typo fails loudly.
# --------------------------------------------------------------------------

class WorkloadClass(str, Enum):
    LLM_INFERENCE = "llm_inference"
    LLM_TRAINING = "llm_training"
    IMAGE_GENERATION = "image_generation"
    IDLE = "idle"


class AnchorType(str, Enum):
    ENF = "enf"                # electric network frequency fingerprint
    POWER_SIGNATURE = "power"  # per-component power-draw signature
    CLOCK = "clock"            # trusted timestamp


class VerificationStatus(str, Enum):
    TRUSTED = "trusted"   # reading is consistent with its anchor
    SUSPECT = "suspect"   # borderline, score below threshold
    FAILED = "failed"     # reading contradicts its anchor -> alert


class AttackClass(str, Enum):
    REPLAY = "replay"
    INJECTION = "injection"
    DRIFT = "drift"


# --------------------------------------------------------------------------
# Serialization mixin. Gives every record to_dict / to_json / from_dict so the
# file-based exchange in io_records.py and the dashboard speak the same format.
# --------------------------------------------------------------------------

def _coerce_enum(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


class _Record:
    def to_dict(self) -> dict:
        return {k: _coerce_enum(v) for k, v in asdict(self).items()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)

    @classmethod
    def from_json(cls, s: str):
        return cls.from_dict(json.loads(s))


# --------------------------------------------------------------------------
# The records.
# --------------------------------------------------------------------------

@dataclass
class StateVariable(_Record):
    """A single named property exposed by an object in the digital twin.

    Hendricks writes these onto rack / PDU / cooling objects in Blender.
    McCray reads them for the dashboard. Leiva reads them as the 'claimed
    state' to check against an anchor.
    """
    name: str                      # e.g. "power_draw_w", "temp_c", "load_pct"
    value: float
    unit: str                      # e.g. "W", "C", "%"
    source_object: str             # Blender object name, e.g. "rack_03"
    timestamp: float = field(default_factory=time.time)
    workload_class: Optional[str] = None  # WorkloadClass value, if applicable


@dataclass
class PowerSample(_Record):
    """One time-step of measured power, as produced by the NLR loader."""
    timestamp: float
    node_id: str
    power_w: float
    workload_class: str            # WorkloadClass value


@dataclass
class AnchorRecord(_Record):
    """Leiva's output: the authenticated fingerprint extracted from a stream.

    The doc specifies the format as (timestamp, signature, confidence).
    `signature` is a list of floats (e.g. an ENF feature vector) so it stays
    JSON-serializable across the file bus.
    """
    timestamp: float
    anchor_type: str               # AnchorType value
    signature: list               # feature vector
    confidence: float              # 0.0 to 1.0
    source: str = "enf_dataset"


@dataclass
class VerificationResult(_Record):
    """Leiva's output: did the claimed twin state match its anchor?

    Marchisano needs this shape to know what counts as a 'detection' before
    he can attack it. A FAILED status during an active AttackEvent window is a
    true positive; a FAILED status outside any window is a false positive.
    """
    timestamp: float
    component_id: str              # e.g. "rack_03"
    status: str                    # VerificationStatus value
    score: float                   # similarity to anchor, higher = more trusted
    anchor_ref: Optional[float] = None  # timestamp of the anchor used
    reason: str = ""


@dataclass
class AttackEvent(_Record):
    """Marchisano's output and the ground truth for the metrics.

    Until week 6 the specific params stay between Marchisano and Dr. Xiang
    (see the 'adversarial within the team' norm). The *shape* is shared now so
    the test harness can score detections without knowing attack internals.
    """
    attack_id: str
    attack_class: str              # AttackClass value
    target_component: str
    start_ts: float
    end_ts: float
    params: dict = field(default_factory=dict)  # opaque to everyone but Marchisano


# Records that travel over the file bus, registered by name for io_records.
RECORD_TYPES = {
    "StateVariable": StateVariable,
    "PowerSample": PowerSample,
    "AnchorRecord": AnchorRecord,
    "VerificationResult": VerificationResult,
    "AttackEvent": AttackEvent,
}
