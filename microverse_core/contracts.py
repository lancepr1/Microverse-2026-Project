"""Interface contracts shared by every lane in the project.

Defines the record shapes (dataclasses) that flow between lanes over
the file bus (see io_records.py), plus the enumerations used
throughout to avoid bare-string typos. Pure standard library only --
must import cleanly in plain Python, CI, and inside Blender's bundled
Python.

See .readme/contracts.md for ownership (which lane produces/consumes
each record type) and the change-management convention for this file.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Optional


class WorkloadClass(str, Enum):
    """Enumerates the workload types a component can be running."""
    LLM_INFERENCE = "llm_inference"
    LLM_TRAINING = "llm_training"
    IMAGE_GENERATION = "image_generation"
    IDLE = "idle"


class AnchorType(str, Enum):
    """Enumerates the physical signal types an anchor can be derived from."""
    ENF = "enf"
    POWER_SIGNATURE = "power"
    CLOCK = "clock"


class VerificationStatus(str, Enum):
    """Enumerates the trust levels a VerificationResult can carry."""
    TRUSTED = "trusted"
    SUSPECT = "suspect"
    FAILED = "failed"


class AttackClass(str, Enum):
    """Enumerates the attack categories an AttackEvent can represent."""
    REPLAY = "replay"
    INJECTION = "injection"
    DRIFT = "drift"


def _coerce_enum(value: Any) -> Any:
    """Returns an Enum member's value, or passes any other value through unchanged.

    Args:
        value: Any value, possibly an Enum member.

    Returns:
        Any: `value.value` if `value` is an Enum member, else `value`.
    """
    return value.value if isinstance(value, Enum) else value


class _Record:
    """Mixin giving every dataclass record to_dict/to_json/from_dict/from_json."""

    def to_dict(self) -> dict:
        """Returns:
            dict: This record's fields, with any Enum values coerced to
            their plain string value.
        """
        return {k: _coerce_enum(v) for k, v in asdict(self).items()}

    def to_json(self) -> str:
        """Returns:
            str: This record serialized as a JSON string.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict):
        """Builds an instance from a plain dict.

        Args:
            d: Field values, as produced by to_dict().

        Returns:
            An instance of the calling class.
        """
        return cls(**d)

    @classmethod
    def from_json(cls, s: str):
        """Builds an instance from a JSON string.

        Args:
            s: JSON string, as produced by to_json().

        Returns:
            An instance of the calling class.
        """
        return cls.from_dict(json.loads(s))


@dataclass
class StateVariable(_Record):
    """A single named property exposed by an object in the digital twin.

    Attributes:
        name: Property name, e.g. "power_draw_w", "temp_c", "load_pct".
        value: Numeric value.
        unit: Unit of `value`, e.g. "W", "C", "%".
        source_object: Blender object name, e.g. "rack_03".
        timestamp: Unix timestamp. Defaults to the current time.
        workload_class: WorkloadClass value, if applicable.
    """
    name: str
    value: float
    unit: str
    source_object: str
    timestamp: float = field(default_factory=time.time)
    workload_class: Optional[str] = None


@dataclass
class PowerSample(_Record):
    """One time-step of measured power, as produced by the NLR loader.

    Attributes:
        timestamp: Unix timestamp.
        node_id: Identifier of the node this sample came from.
        power_w: Measured power in watts.
        workload_class: WorkloadClass value.
    """
    timestamp: float
    node_id: str
    power_w: float
    workload_class: str


@dataclass
class AnchorRecord(_Record):
    """An authenticated fingerprint extracted from a physical signal stream.

    Attributes:
        timestamp: Elapsed seconds this anchor corresponds to.
        anchor_type: AnchorType value.
        signature: Feature vector (e.g. a normalized ENF window).
        confidence: Value in [0.0, 1.0].
        source: Identifies where this anchor was derived from.
    """
    timestamp: float
    anchor_type: str
    signature: list
    confidence: float
    source: str = "enf_dataset"


@dataclass
class VerificationResult(_Record):
    """The result of checking one component's claimed state against its anchor.

    Attributes:
        timestamp: Unix timestamp.
        component_id: Identifies what was checked, e.g. "rack_03".
        status: VerificationStatus value.
        score: Similarity to the anchor; higher means more trusted.
        anchor_ref: Timestamp of the anchor used, if applicable.
        reason: Human-readable explanation of the result.
    """
    timestamp: float
    component_id: str
    status: str
    score: float
    anchor_ref: Optional[float] = None
    reason: str = ""


@dataclass
class AttackEvent(_Record):
    """A simulated attack window, used as ground truth for scoring.

    Attributes:
        attack_id: Unique identifier for this attack.
        attack_class: AttackClass value.
        target_component: Identifier of the component targeted.
        start_ts: Unix timestamp the attack began.
        end_ts: Unix timestamp the attack ended.
        params: Attack-specific parameters.
    """
    attack_id: str
    attack_class: str
    target_component: str
    start_ts: float
    end_ts: float
    params: dict = field(default_factory=dict)


RECORD_TYPES = {
    "StateVariable": StateVariable,
    "PowerSample": PowerSample,
    "AnchorRecord": AnchorRecord,
    "VerificationResult": VerificationResult,
    "AttackEvent": AttackEvent,
}