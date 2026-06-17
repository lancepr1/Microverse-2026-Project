"""
scene_builder.py: a PLACEHOLDER digital twin so nobody is blocked in week one.

The doc makes Hendricks's DT skeleton the critical-path artifact of week two,
and says Leiva, McCray and Marchisano all depend on it existing. This module
buys those three a target to develop against before the real skeleton lands.

It builds a grid of labelled boxes (racks, PDUs, a cooling unit), each carrying
twin state via blender_bridge.set_state. It is intentionally crude. Hendricks
replaces it. Do not let it drift into being the real model: its only job is to
expose the state-variable interface so the contracts can be exercised end to
end on day one.

Run it with:
    blender --background --python scripts/build_starter_scene.py
"""
from __future__ import annotations

import time

from .blender_bridge import set_state
from .contracts import StateVariable, WorkloadClass


def _new_box(bpy, name: str, location, size=1.0):
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def build_placeholder_twin(n_racks: int = 6) -> list[str]:
    """Block in a small room of racks plus PDUs and a cooling unit.

    Returns the list of object names that carry twin state.
    """
    import bpy

    # clear the default scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    created: list[str] = []
    now = time.time()

    # a row of racks
    for i in range(n_racks):
        name = f"rack_{i:02d}"
        _new_box(bpy, name, location=(i * 2.0, 0.0, 1.0), size=1.0)
        # seed placeholder state. real values come from NLR traces later.
        set_state(name, StateVariable(
            name="power_draw_w", value=120.0, unit="W",
            source_object=name, timestamp=now,
            workload_class=WorkloadClass.IDLE.value,
        ))
        set_state(name, StateVariable(
            name="temp_c", value=22.0, unit="C",
            source_object=name, timestamp=now,
        ))
        set_state(name, StateVariable(
            name="load_pct", value=0.0, unit="%",
            source_object=name, timestamp=now,
        ))
        created.append(name)

    # two power distribution units
    for j in range(2):
        name = f"pdu_{j:02d}"
        _new_box(bpy, name, location=(j * 6.0, -2.5, 0.5), size=0.8)
        set_state(name, StateVariable(
            name="power_draw_w", value=400.0, unit="W",
            source_object=name, timestamp=now,
        ))
        created.append(name)

    # one cooling unit
    _new_box(bpy, "cooling_00", location=(n_racks, 2.5, 1.0), size=1.4)
    set_state("cooling_00", StateVariable(
        name="temp_c", value=18.0, unit="C",
        source_object="cooling_00", timestamp=now,
    ))
    created.append("cooling_00")

    print(f"[scene_builder] placeholder twin built: {len(created)} objects")
    return created
