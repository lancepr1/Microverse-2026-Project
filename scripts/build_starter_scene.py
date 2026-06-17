"""
build_starter_scene.py: run inside Blender to generate the placeholder twin.

    blender --background --python scripts/build_starter_scene.py -- --save twin.blend

The '--' separates Blender's args from ours. With --save it writes a .blend so
McCray and Leiva have a file to load. Without it, just builds in memory (useful
for a quick check that the bridge works).
"""
import sys
import os

# make the repo importable when Blender runs this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from microverse_core.scene_builder import build_placeholder_twin  # noqa: E402
from microverse_core.blender_bridge import list_state             # noqa: E402


def _arg(flag, default=None):
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    return argv[argv.index(flag) + 1] if flag in argv else default


def main():
    import bpy
    names = build_placeholder_twin(n_racks=6)
    print(f"[build_starter_scene] objects with state: {names}")
    print(f"[build_starter_scene] rack_00 exposes: {list_state('rack_00')}")

    save_path = _arg("--save")
    if save_path:
        save_path = os.path.abspath(save_path)
        bpy.ops.wm.save_as_mainfile(filepath=save_path)
        print(f"[build_starter_scene] saved to {save_path}")


if __name__ == "__main__":
    main()
