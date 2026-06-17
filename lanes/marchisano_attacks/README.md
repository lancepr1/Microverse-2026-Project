# Marchisano: adversarial scenarios and attack implementation

## What it does
Designs and runs replay, injection, and drift attacks against Leiva's
verification, and measures detection. Produces `AttackEvent` records that serve
as ground truth for `microverse_core.metrics`.

## How to run
Emit an `AttackEvent` per attack (class, target component, start/end window,
opaque params). Score with `metrics.score` and `metrics.time_to_detection`. Your
attacks must be plausible against real NLR workload baselines, not synthetic
data, once the dataset is available.

## Norm (important)
You and Leiva are adversarial by design. Do not share specific attack signatures
with Leiva until week 6. Discuss attack internals with Dr. Xiang in 1:1s, not in
team meetings. The shared contract is only the `AttackEvent` shape, not its
params.

## Who to ask
Dr. Xiang (methodology, evaluation), Lance (test harness integration).

## Week-1 deliverable
One-page attack-scenario sketch: what each of the three attacks does at the data
level, attacker-success vs defender-success, and which NLR workload category
each targets.
