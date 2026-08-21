"""
Kimodo Blender Bridge
=====================
Runs NVIDIA Kimodo or NVIDIA ARDY in a managed subprocess for AI-powered
human(oid) motion generation directly inside Blender.

Features:
  • Two interchangeable backends, selected in the Connection panel:
      Kimodo — SOMA skeleton at 30 FPS, one-shot diffusion
      ARDY   — Core skeleton at 20 FPS, autoregressive and long-sequence
  • Generate motion from text prompts via a persistent bridge process
    (bridge_server.py / ardy_bridge.py, JSON over stdin/stdout —
    model loads once)
  • Automatic BVH import into Blender armature
  • Custom bone-mapping retargeting to any existing rig
  • Constraint-based retargeting with one-click bake
  • Save / load bone mapping presets

Requirements:
  • Blender 4.2+ (tested on 4.x and 5.x)
  • NVIDIA GPU (CUDA); each backend is installed automatically into its own
    venv (~/.kimodo-venv, ~/.ardy-venv)
    See: https://github.com/nv-tlabs/kimodo and
         https://github.com/nv-tlabs/ardy

"""

bl_info = {
    "name":        "Kimodo Blender Bridge",
    "author":      "Lewdineer",
    "version":     (1, 7, 1),
    "blender":     (4, 2, 0),
    "location":    "View3D › Sidebar (N-Panel) › Kimodo",
    "description": "Generate human motion with NVIDIA Kimodo or ARDY. "
                   "Runs the model in a managed background process.",
    "doc_url":     "https://github.com/nv-tlabs/kimodo",
    "tracker_url": "https://github.com/nv-tlabs/kimodo/issues",
    "category":    "Animation",
    "support":     "COMMUNITY",
}

import bpy

# Sub-modules (imported after bl_info for Blender's enable/disable system)
from . import properties, operators, ui_list, panels, constraints, timeline
from . import setup_operator
from . import ardy_setup
from . import ardy_stream
from . import subprocess_client as sc


def register():
    properties.register()
    operators.register()
    setup_operator.register()
    ardy_setup.register()
    ardy_stream.register()
    ui_list.register()
    panels.register()
    timeline.register()


def unregister():
    # Kill the bridge process so we don't leave orphaned GPU processes
    sc.stop()
    timeline.unregister()
    panels.unregister()
    ui_list.unregister()
    ardy_stream.unregister()
    ardy_setup.unregister()
    setup_operator.unregister()
    operators.unregister()
    properties.unregister()
