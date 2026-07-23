"""The one module that touches bpy.

State variables live on Blender objects as custom properties. Every
lane that needs to read or write twin state goes through this module
instead of touching `obj[...]` directly, so the storage convention
only needs to change in one place if it ever needs to change at all.

bpy is imported lazily inside each function, so this module (and
anything that imports it, like contracts.py's callers) imports
cleanly in plain Python or CI; only the functions that actually drive
Blender require Blender itself.
"""

from __future__ import annotations

from typing import Optional

from .contracts import StateVariable


def _bpy():
    """Imports and returns the bpy module, or raises a clear error if unavailable.

    Returns:
        module: The bpy module.

    Raises:
        RuntimeError: If bpy is not importable (not running inside Blender).
    """
    try:
        import bpy  # noqa: WPS433
        return bpy
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This function must run inside Blender "
            "(blender --background --python your_script.py)."
        ) from exc


def get_object(name: str):
    """Looks up a Blender object by name.

    Args:
        name: Object name.

    Returns:
        The Blender object.

    Raises:
        KeyError: If no object with that name exists in the scene.
    """
    bpy = _bpy()
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise KeyError(f"No object named {name!r} in the scene.")
    return obj


def set_state(obj_name: str, var: StateVariable) -> None:
    """Writes a StateVariable onto an object as namespaced custom properties.

    Args:
        obj_name: Name of the Blender object to write onto.
        var: The state variable to write. Stored under an "mv_" prefix
            so twin state never collides with Blender's own properties
            or a modeler's ad-hoc fields.
    """
    obj = get_object(obj_name)
    obj[f"mv_{var.name}"] = var.value
    obj[f"mv_{var.name}__unit"] = var.unit
    obj[f"mv_{var.name}__ts"] = var.timestamp
    if var.workload_class is not None:
        obj["mv_workload_class"] = var.workload_class


STATUS_COLORS = {
    0.0: (0.0, 1.0, 0.0, 1.0),
    0.5: (1.0, 1.0, 0.0, 1.0),
    1.0: (1.0, 0.0, 0.0, 1.0),
}
_STATUS_FALLBACK_COLOR = (0.5, 0.5, 0.5, 1.0)


def set_status_color(obj_name: str, status: float) -> None:
    """Sets an object's viewport display color based on a verification status.

    Uses Blender's native per-object color property rather than
    touching materials directly, so the color is visible immediately
    under "Object Color" viewport shading with no extra setup. Also
    writes the raw numeric status as a normal state variable via
    set_state(), so it stays queryable/loggable, not just visually
    represented.

    Args:
        obj_name: Name of the Blender object to color.
        status: 0.0 (trusted), 0.5 (suspect), or 1.0 (failed). Any
            other value falls back to gray.
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
    """Reads one state variable back from an object.

    Args:
        obj_name: Name of the Blender object.
        var_name: Name of the state variable to read.

    Returns:
        StateVariable: The requested variable.

    Raises:
        KeyError: If the object has no such state variable.
    """
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
    """Lists the names of every state variable currently exposed on an object.

    Args:
        obj_name: Name of the Blender object.

    Returns:
        list[str]: State variable names, without the "mv_" prefix.
    """
    obj = get_object(obj_name)
    names = []
    for key in obj.keys():
        if key.startswith("mv_") and "__" not in key and key != "mv_workload_class":
            names.append(key[len("mv_"):])
    return names


def all_twin_objects(prefix: Optional[str] = None) -> list[str]:
    """Lists every object in the scene that carries twin state.

    Args:
        prefix: If given, only object names starting with this prefix
            are returned (e.g. "rack", "pdu", "cooling").

    Returns:
        list[str]: Matching object names.
    """
    bpy = _bpy()
    out = []
    for obj in bpy.data.objects:
        has_state = any(k.startswith("mv_") for k in obj.keys())
        if has_state and (prefix is None or obj.name.startswith(prefix)):
            out.append(obj.name)
    return out