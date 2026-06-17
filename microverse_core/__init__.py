"""
microverse_core: the integration layer for the Secure AI Data Center Microverse.

Lance owns this package. It holds the shared contracts every lane builds
against, the Blender bridge, a placeholder twin, the data loaders, the file
bus, and the scoring metrics. The goal is that the five lanes plug into stable
interfaces instead of into each other's half-finished code.
"""
from . import contracts, data_loaders, io_records, metrics

# blender_bridge and scene_builder import bpy lazily, so importing them here is
# safe outside Blender; their functions just raise if bpy is missing.
from . import blender_bridge, scene_builder  # noqa: F401

__all__ = [
    "contracts",
    "data_loaders",
    "io_records",
    "metrics",
    "blender_bridge",
    "scene_builder",
]

__version__ = "0.1.0"
