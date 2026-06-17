# Leiva: sensor anchors and verification logic

## What it does
Extracts an authenticated anchor from each sensor stream (ENF first) and checks
whether the twin's claimed state is consistent with that anchor. Produces
`AnchorRecord` and `VerificationResult` records.

## How to run
Consume ENF via `microverse_core.data_loaders` (synthetic until the real set
lands). Read claimed twin state through `blender_bridge.get_state`. Emit
`VerificationResult` with status TRUSTED / SUSPECT / FAILED so Marchisano can
score against it. See the fake stages in `scripts/smoke_test.py` for the exact
shapes; replace them with the real anchoring and comparison.

## Who to ask
Dr. Qu (methodology, ANCHOR-Grid), Lance (repo and integration).

## Week-1 deliverable
Two-paragraph design memo: how the anchor extractor reads ENF, what it outputs
(timestamp, signature, confidence), how verification compares an anchor to a
claimed twin state.
