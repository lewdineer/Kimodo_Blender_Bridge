"""
Kimodo Blender Bridge
=====================
Connects Blender to a running NVIDIA Kimodo Gradio demo for AI-powered
human(oid) motion generation directly inside Blender.

Features:
  • Generate motion from text prompts via the Kimodo Gradio REST API
  • Automatic BVH import into Blender armature
  • Custom bone-mapping retargeting to any existing rig
  • Constraint-based retargeting with one-click bake
  • Save / load bone mapping presets

Requirements:
  • Blender 4.0+ (tested on 4.x and 5.x)
  • Kimodo demo running locally (or remotely with port forwarding)
    See: https://github.com/nv-tlabs/kimodo

"""

bl_info = {
    "name":        "Kimodo Blender Bridge",
    "author":      "Lewdineer",
    "version":     (1, 4, 0),
    "blender":     (4, 2, 0),
    "location":    "View3D › Sidebar (N-Panel) › Kimodo",
    "description": "Generate human motion with NVIDIA Kimodo AI. "
                   "Connects to a running Kimodo Gradio instance.",
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


def register():
    properties.register()
    operators.register()
    setup_operator.register()
    ui_list.register()
    panels.register()
    timeline.register()


def unregister():
    # Kill the bridge process so we don't leave orphaned GPU processes
    sc.stop()
    timeline.unregister()
    panels.unregister()
    ui_list.unregister()
    setup_operator.unregister()
    operators.unregister()
    properties.unregister()
