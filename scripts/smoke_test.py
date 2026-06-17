"""
smoke_test.py: the whole pipeline in one runnable file, no Blender required.

This is the onboarding artifact. A new intern runs:

    python scripts/smoke_test.py

and sees the shape of the entire system: a workload power trace, a fake anchor,
fake verification output, a planted attack, and the resulting detection
metrics. None of it is real verification or real attacks. It exists so that on
day one everyone can see how the contracts connect, and so CI has something to
fail on if a contract changes underneath the team.

Each lane replaces its fake stage with the real one:
  Hendricks  -> real NLR-driven twin state instead of synthetic_power_profile
  Leiva      -> real ENF anchoring + verification instead of fake_verify
  Marchisano -> real attacks instead of plant_attack
  McCray     -> real dashboard instead of the print at the end
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from microverse_core.contracts import (  # noqa: E402
    AnchorRecord, AnchorType, AttackClass, AttackEvent,
    VerificationResult, VerificationStatus, WorkloadClass,
)
from microverse_core.data_loaders import (  # noqa: E402
    synthetic_power_profile, synthetic_enf,
)
from microverse_core import io_records, metrics  # noqa: E402

RUN_ID = "smoke"
COMPONENT = "rack_00"


def fake_anchor(enf, t):
    """Stand-in for Leiva: turn an ENF window into an anchor record."""
    window = enf[max(0, int(t) - 5):int(t) + 5]
    sig = window or [60.0]
    return AnchorRecord(
        timestamp=t,
        anchor_type=AnchorType.ENF.value,
        signature=sig,
        confidence=0.9,
        source="synthetic_enf",
    )


def fake_verify(sample, anchor, under_attack):
    """Stand-in for Leiva's verification. Here it 'detects' simply by being
    told the ground truth; the real one compares state to the anchor."""
    status = VerificationStatus.FAILED if under_attack else VerificationStatus.TRUSTED
    return VerificationResult(
        timestamp=sample.timestamp,
        component_id=COMPONENT,
        status=status.value,
        score=0.2 if under_attack else 0.95,
        anchor_ref=anchor.timestamp,
        reason="state inconsistent with anchor" if under_attack else "ok",
    )


def plant_attack(start, end):
    """Stand-in for Marchisano: declare a replay attack window as ground truth."""
    return AttackEvent(
        attack_id="atk_demo_01",
        attack_class=AttackClass.REPLAY.value,
        target_component=COMPONENT,
        start_ts=start,
        end_ts=end,
        params={"note": "synthetic demo attack"},
    )


def main():
    # 1. workload power trace (Hendricks' input) + ENF (Leiva's input)
    power = synthetic_power_profile(WorkloadClass.LLM_INFERENCE, COMPONENT,
                                    seconds=60, hz=2, seed=1)
    enf = synthetic_enf(seconds=60, hz=2, seed=1)
    io_records.write_records(RUN_ID, "power", power)

    # 2. plant one attack from t=20s to t=30s (Marchisano)
    attack = plant_attack(20.0, 30.0)
    io_records.write_records(RUN_ID, "attacks", [attack])

    # 3. anchor + verify each sample (Leiva)
    anchors, results = [], []
    for s in power:
        a = fake_anchor(enf, s.timestamp)
        under = attack.start_ts <= s.timestamp <= attack.end_ts
        anchors.append(a)
        results.append(fake_verify(s, a, under))
    io_records.write_records(RUN_ID, "anchors", anchors)
    io_records.write_records(RUN_ID, "verification", results)

    # 4. score (shared metrics) -- McCray would render this
    m = metrics.score(results, [attack])
    ttd = metrics.time_to_detection(results, attack)

    print("=== Microverse smoke test ===")
    print(f"samples: {len(power)}   attack window: "
          f"{attack.start_ts:.0f}-{attack.end_ts:.0f}s")
    print(f"metrics: {m.as_dict()}")
    print(f"time-to-detection: {ttd:.2f}s" if ttd is not None else "missed")
    print(f"records written under runs/{RUN_ID}/")

    # CI assertion: the planted attack must be caught with no false alarms
    assert m.recall == 1.0 and m.false_positives == 0, "pipeline wiring broken"
    print("OK")


if __name__ == "__main__":
    main()
