"""
metrics.py: score verification output against attack ground truth.

The paper's quantitative claim is precision / recall / F1 / time-to-detection
on the three attack classes. Putting the scoring here, in the integration
layer, means Marchisano, Leiva and the paper all use the exact same definition
of a true positive. Marchisano owns the evaluation design and is free to
extend this; it lives here so there is one source of truth for the numbers.

Definitions:
  A VerificationResult with status FAILED is a positive detection.
  It is a TRUE positive if its timestamp falls inside any AttackEvent window
    that targets the same component.
  It is a FALSE positive otherwise.
  An AttackEvent with no FAILED detection inside its window is a FALSE negative
    (a missed attack).
  time-to-detection: seconds from an attack's start_ts to the first FAILED
    detection inside its window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .contracts import AttackEvent, VerificationResult, VerificationStatus


@dataclass
class Metrics:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def _in_window(result: VerificationResult, attack: AttackEvent) -> bool:
    return (
        result.component_id == attack.target_component
        and attack.start_ts <= result.timestamp <= attack.end_ts
    )


def score(
    results: list[VerificationResult],
    attacks: list[AttackEvent],
) -> Metrics:
    failed = [r for r in results if r.status == VerificationStatus.FAILED.value]

    tp = 0
    fp = 0
    matched_attacks = set()
    for r in failed:
        hit = next((a for a in attacks if _in_window(r, a)), None)
        if hit is not None:
            tp += 1
            matched_attacks.add(hit.attack_id)
        else:
            fp += 1

    fn = sum(1 for a in attacks if a.attack_id not in matched_attacks)
    return Metrics(true_positives=tp, false_positives=fp, false_negatives=fn)


def time_to_detection(
    results: list[VerificationResult],
    attack: AttackEvent,
) -> Optional[float]:
    """Seconds from attack start to first failed detection, or None if missed."""
    hits = sorted(
        (r.timestamp for r in results
         if r.status == VerificationStatus.FAILED.value and _in_window(r, attack))
    )
    return (hits[0] - attack.start_ts) if hits else None
