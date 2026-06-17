"""Tests that lock the contracts. If these break, a lane interface changed."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from microverse_core.contracts import (  # noqa: E402
    AnchorRecord, AnchorType, AttackClass, AttackEvent,
    StateVariable, VerificationResult, VerificationStatus,
)
from microverse_core import metrics  # noqa: E402


def test_state_variable_roundtrip():
    sv = StateVariable(name="power_draw_w", value=512.0, unit="W",
                       source_object="rack_00", timestamp=1.0)
    assert StateVariable.from_json(sv.to_json()) == sv


def test_anchor_roundtrip():
    a = AnchorRecord(timestamp=1.0, anchor_type=AnchorType.ENF.value,
                     signature=[60.0, 60.01], confidence=0.8)
    assert AnchorRecord.from_json(a.to_json()) == a


def test_metrics_perfect_detection():
    atk = AttackEvent("a1", AttackClass.REPLAY.value, "rack_00", 10.0, 20.0)
    results = [
        VerificationResult(15.0, "rack_00", VerificationStatus.FAILED.value, 0.1),
        VerificationResult(5.0, "rack_00", VerificationStatus.TRUSTED.value, 0.9),
    ]
    m = metrics.score(results, [atk])
    assert m.true_positives == 1
    assert m.false_positives == 0
    assert m.false_negatives == 0
    assert m.recall == 1.0


def test_metrics_false_positive_and_missed():
    atk = AttackEvent("a1", AttackClass.DRIFT.value, "rack_00", 10.0, 20.0)
    results = [
        # a failed detection outside any attack window -> false positive
        VerificationResult(50.0, "rack_00", VerificationStatus.FAILED.value, 0.1),
    ]
    m = metrics.score(results, [atk])
    assert m.false_positives == 1
    assert m.false_negatives == 1  # the attack was never caught


def test_time_to_detection():
    atk = AttackEvent("a1", AttackClass.INJECTION.value, "rack_00", 10.0, 20.0)
    results = [
        VerificationResult(13.0, "rack_00", VerificationStatus.FAILED.value, 0.1),
        VerificationResult(17.0, "rack_00", VerificationStatus.FAILED.value, 0.1),
    ]
    assert metrics.time_to_detection(results, atk) == 3.0
