# Hendricks: digital-twin physical model

## What it does
Builds the Blender DT of the data center (racks, PDUs, cooling) and drives each
object's state variables from real NLR power profiles. This is the canvas every
other lane runs on. Replaces `microverse_core/scene_builder.py` (placeholder).

## How to run
Develop against the placeholder first:
`blender --background --python scripts/build_starter_scene.py`. Write twin state
with `microverse_core.blender_bridge.set_state`, using the NLR loader for real
values. Keep object naming consistent (`rack_NN`, `pdu_NN`, `cooling_NN`) so the
dashboard and verification can find your objects.

## Who to ask
Dr. Xiang (dynamics, NLR mapping), Dr. Qu (architecture fit), Lance (Blender
mechanics), Sahadat (what a real rack looks like).

## Week-1 deliverable
(a) one-page sketch of components mapped to NLR workload categories;
(b) Blender scene with room geometry blocked in plus a script that loads one
NLR profile and exposes it as a state variable on a placeholder rack.
