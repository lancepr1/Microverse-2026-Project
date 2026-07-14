"""
blender_bridge.py: the one place that touches bpy.

State variables live on Blender objects as custom properties. Every lane that
needs to read or write twin state goes through this module instead of poking
obj["..."] directly, so if the storage convention changes we change it once.

bpy is imported lazily inside each function. That means contracts.py,
data_loaders.py, metrics.py and the test suite all import fine in plain Python
or CI; only the functions that actually drive Blender require Blender.
"""
from __future__ import annotations

from typing import Optional

from .contracts import StateVariable


def _bpy():
    try:
        import bpy  # noqa: WPS433  (intentional lazy import)
        return bpy
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This function must run inside Blender "
            "(blender --background --python your_script.py)."
        ) from exc


def get_object(name: str):
    bpy = _bpy()
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise KeyError(f"No object named {name!r} in the scene.")
    return obj


def set_state(obj_name: str, var: StateVariable) -> None:
    """Write a StateVariable onto an object as a namespaced custom property.

    We prefix with 'mv_' so twin state never collides with Blender's own
    properties or a modeler's ad-hoc fields.
    """
    obj = get_object(obj_name)
    obj[f"mv_{var.name}"] = var.value
    obj[f"mv_{var.name}__unit"] = var.unit
    obj[f"mv_{var.name}__ts"] = var.timestamp
    if var.workload_class is not None:
        obj["mv_workload_class"] = var.workload_class


# Verification status color mapping. 0.0/0.5/1.0 = trusted/suspect/failed,
# matching the numeric encoding used throughout the verification pipeline.
STATUS_COLORS = {
    0.0: (0.0, 1.0, 0.0, 1.0),   # trusted -> green
    0.5: (1.0, 1.0, 0.0, 1.0),   # suspect -> yellow
    1.0: (1.0, 0.0, 0.0, 1.0),   # failed  -> red
}
_STATUS_FALLBACK_COLOR = (0.5, 0.5, 0.5, 1.0)  # unexpected value -> gray, not silently wrong


def set_status_color(obj_name: str, status: float) -> None:
    """
    Sets an object's built-in Viewport Display Color (obj.color) based on a
    verification status value. Uses Blender's native per-object color
    property rather than touching materials directly -- visible immediately
    under "Object Color" viewport shading with no extra setup required, and
    any material can optionally read it later via an Object Info node if
    finer control is wanted.

    Also writes the raw numeric status as a normal state variable (via
    set_state, same as every other twin property) so it stays queryable/
    loggable, not just visually represented.
    """
    obj = get_object(obj_name)
    obj.color = STATUS_COLORS.get(status, _STATUS_FALLBACK_COLOR)
    set_state(obj_name, StateVariable(
        name="verification_status",
        value=status,
        unit="status",
        source_object=obj_name,
    ))


def get_state(obj_name: str, var_name: str) -> StateVariable:
    obj = get_object(obj_name)
    key = f"mv_{var_name}"
    if key not in obj.keys():
        raise KeyError(f"{obj_name!r} has no state variable {var_name!r}.")
    return StateVariable(
        name=var_name,
        value=float(obj[key]),
        unit=str(obj.get(f"{key}__unit", "")),
        source_object=obj_name,
        timestamp=float(obj.get(f"{key}__ts", 0.0)),
        workload_class=obj.get("mv_workload_class"),
    )


def list_state(obj_name: str) -> list[str]:
    """Names of every state variable currently exposed on an object."""
    obj = get_object(obj_name)
    names = []
    for key in obj.keys():
        if key.startswith("mv_") and "__" not in key and key != "mv_workload_class":
            names.append(key[len("mv_"):])
    return names


def all_twin_objects(prefix: Optional[str] = None) -> list[str]:
    """Object names that carry twin state. Optionally filter by name prefix
    (e.g. 'rack', 'pdu', 'cooling')."""
    bpy = _bpy()
    out = []
    for obj in bpy.data.objects:
        has_state = any(k.startswith("mv_") for k in obj.keys())
        if has_state and (prefix is None or obj.name.startswith(prefix)):
            out.append(obj.name)
    return out