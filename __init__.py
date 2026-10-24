"""
Kimodo Blender Bridge
=====================
Runs NVIDIA Kimodo in a managed subprocess for AI-powered human(oid)
motion generation directly inside Blender.

Features:
  • Generate motion from text prompts, either in a local bridge process
    (bridge_server.py) or on a remote motionmcp-kimodo server (HTTP URL)
  • Automatic BVH import into Blender armature
  • Custom bone-mapping retargeting to any existing rig
  • Constraint-based retargeting with one-click bake
  • Save / load bone mapping presets

Requirements:
  • Blender 4.2+ (tested on 4.x and 5.x)
  • NVIDIA GPU (CUDA); Kimodo is installed automatically into ~/.kimodo-venv
    See: https://github.com/nv-tlabs/kimodo

"""

bl_info = {
    "name":        "Kimodo Blender Bridge",
    "author":      "Lewdineer",
    "version":     (1, 6, 0),
    "blender":     (4, 2, 0),
    "location":    "View3D › Sidebar (N-Panel) › Kimodo",
    "description": "Generate human motion with NVIDIA Kimodo AI, "
                   "locally or on a remote server.",
    "doc_url":     "https://github.com/nv-tlabs/kimodo",
    "tracker_url": "https://github.com/nv-tlabs/kimodo/issues",
    "category":    "Animation",
    "support":     "COMMUNITY",
}

import bpy

# Sub-modules (imported after bl_info for Blender's enable/disable system)
from . import properties, operators, ui_list, panels, constraints, timeline
from . import setup_operator
from . import subprocess_client as sc
from . import remote_client as rc


def register():
    properties.register()
    operators.register()
    setup_operator.register()
    ui_list.register()
    panels.register()
    timeline.register()


def unregister():
    # Kill the local bridge process so we don't leave orphaned GPU processes.
    rc.disconnect()
    sc.stop()
    timeline.unregister()
    panels.unregister()
    ui_list.unregister()
    setup_operator.unregister()
    operators.unregister()
    properties.unregister()
