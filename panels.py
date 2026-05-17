"""
Kimodo Blender Bridge — Panels
All N-panel UI panels in the 3D Viewport → Kimodo tab.
"""

import os

import bpy
import json
from bpy.types import Panel


# ---------------------------------------------------------------------------
# Base class — common settings
# ---------------------------------------------------------------------------

class KIMODO_PanelBase:
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = 'Kimodo'


# ---------------------------------------------------------------------------
# Panel 1: Connection
# ---------------------------------------------------------------------------

class KIMODO_PT_Connection(KIMODO_PanelBase, Panel):
    bl_label    = "⚙  Connection"
    bl_idname   = "KIMODO_PT_Connection"
    bl_order    = 10

    def draw(self, context):
        layout = self.layout
        s = context.scene.kimodo

        running = s.is_connected

        # --- Auto-install section ---
        from . import setup_operator as so

        if so.is_installing():
            box = layout.box()
            box.label(text="Installing Kimodo…", icon='TIME')
            box.label(text=so.install_status())
            layout.separator(factor=0.5)

        elif so.install_failed() or (so.venv_exists() and not so.is_installed()):
            # install_failed()  → failed this session
            # venv_exists() but not is_installed() → partial venv from a
            # previous session (no sentinel file); treat it the same way.
            box = layout.box()
            if so.needs_python():
                box.label(text="Python 3.10–3.12 required!", icon='ERROR')
                box.separator(factor=0.3)
                box.label(text="No compatible Python was found on your system.")
                box.label(text="Click below to download the Python 3.12 installer (Windows 64-bit).")
                box.label(text='Run it, tick "Add Python to PATH".')
                box.label(text="On Windows: run the installer as Administrator.")
                box.label(text="Then restart Blender before clicking Retry Install.", icon='ERROR')
                box.separator(factor=0.3)
                box.operator("kimodo.open_python_download",
                             text="Download Python 3.12 Installer", icon='URL')
                box.separator(factor=0.3)
            else:
                box.label(text="Installation incomplete", icon='ERROR')
                if so.install_status():
                    box.label(text=so.install_status(), icon='BLANK1')
            box.operator("kimodo.install_kimodo",
                         text="Retry Install", icon='FILE_REFRESH')
            box.operator("kimodo.reset_venv",
                         text="Reset Venv", icon='TRASH')
            layout.separator(factor=0.5)

        elif not so.is_installed():
            box = layout.box()
            has_gpu = so.has_nvidia_gpu()
            if not has_gpu:
                box.label(text="NVIDIA GPU required!", icon='ERROR')
                box.label(text="Kimodo only works with NVIDIA GPUs (CUDA).")
                box.label(text="AMD and Intel GPUs are not supported.")
                box.separator(factor=0.3)
            else:
                box.label(text="Kimodo not installed", icon='INFO')
            box.label(text="Installs to:  ~/.kimodo-venv/")
            box.label(text="Requires:  Python 3.10+, ~8 GB disk, internet")
            row = box.row()
            row.enabled = has_gpu
            row.operator("kimodo.install_kimodo", icon='IMPORT')
            layout.separator(factor=0.5)

        elif not s.python_executable or not os.path.isfile(s.python_executable):
            box = layout.box()
            box.label(text="Kimodo venv ready", icon='CHECKMARK')
            box.operator("kimodo.use_installed_kimodo", icon='CONSOLE')
            layout.separator(factor=0.5)

        # --- Python executable ---
        col = layout.column(align=True)
        col.label(text="Kimodo Python:", icon='CONSOLE')
        row = col.row(align=True)
        row.prop(s, "python_executable", text="")
        row.enabled = not running

        col.label(text="  Leave blank to auto-detect from PATH / sibling venv",
                  icon='INFO')

        layout.separator(factor=0.5)

        # --- Model selector ---
        row = layout.row(align=True)
        row.label(text="Model:", icon='ARMATURE_DATA')
        row.prop(s, "kimodo_model", text="")
        row.enabled = not running

        layout.separator(factor=0.5)

        # --- Start / Stop buttons ---
        if running:
            layout.operator("kimodo.stop_kimodo",
                            text="Stop Kimodo", icon='CANCEL')
        else:
            layout.operator("kimodo.start_kimodo",
                            text="Start Kimodo", icon='PLAY')

        # --- Status ---
        status_row = layout.row()
        if running:
            status_row.label(text=s.connection_status, icon='CHECKMARK')
        elif s.connection_status in ("Not started", "Stopped"):
            status_row.label(text=s.connection_status, icon='RADIOBUT_OFF')
        else:
            # Loading or error
            is_err = s.connection_status.startswith("Failed") or \
                     s.connection_status.startswith("Error")
            status_row.label(
                text=s.connection_status,
                icon='ERROR' if is_err else 'TIME',
            )

        # --- Hint: text encoder ---
        if not running:
            layout.separator(factor=0.5)
            box = layout.box()
            box.label(text="Tip: GPU < 16 GB VRAM?", icon='INFO')
            box.label(text="Run first:  kimodo_textencoder --device cpu")
            box.label(text="This offloads the text encoder to CPU.")


# ---------------------------------------------------------------------------
# Panel 2: Motion Segments (timeline bars)
# ---------------------------------------------------------------------------

class KIMODO_PT_Segments(KIMODO_PanelBase, Panel):
    bl_label  = "🎞  Motion Segments"
    bl_idname = "KIMODO_PT_Segments"
    bl_order  = 15

    def draw(self, context):
        layout = self.layout
        s      = context.scene.kimodo

        # --- Toolbar ---
        row = layout.row(align=True)
        row.operator("kimodo.add_segment",    text="Add",       icon='ADD')
        row.operator("kimodo.remove_segment", text="Remove",    icon='REMOVE')
        row.separator()
        row.operator("kimodo.duplicate_segment", text="", icon='DUPLICATE')
        row.operator("kimodo.move_segment_up",   text="", icon='TRIA_UP')
        row.operator("kimodo.move_segment_down", text="", icon='TRIA_DOWN')

        layout.separator(factor=0.5)


        row = layout.row(align=True)
        row.label(text="Model:")
        row.prop(s, "model_type", expand=True)
        # Show seed for the selected segment if available
        if s.motion_segments and 0 <= s.segment_index < len(s.motion_segments):
            selected_seg = s.motion_segments[s.segment_index]
            row.prop(selected_seg, "seed", text="Seed")
        else:
            row.prop(s, "seed", text="Global Seed")

        # --- FPS warning ---
        scene_fps = context.scene.render.fps / context.scene.render.fps_base
        if abs(scene_fps - 30.0) > 0.01:
            fps_box = layout.box()
            fps_box.alert = True
            fps_box.label(text=f"Scene is {scene_fps:.4g} FPS — Kimodo needs 30 FPS", icon='ERROR')
            fps_box.operator("kimodo.set_to_30fps", text="Set to 30 FPS", icon='RECOVER_LAST')

        # --- Segment list ---
        if not s.motion_segments:
            col = layout.column()
            col.label(text="No segments yet.", icon='INFO')
            col.label(text="Click Add to create one.")
            return

        for i, seg in enumerate(s.motion_segments):
            is_active     = (i == s.segment_index)
            is_generating = (i == s.generating_segment_index)

            box = layout.box()
            #box.alert = is_active   # blue highlight on active segment

            # --- Header row: [enabled] [color] [label] [select] [generate] ---
            header = box.row(align=True)

            header.prop(seg, "enabled", text="", emboss=False,
                        icon='CHECKBOX_HLT' if seg.enabled else 'CHECKBOX_DEHLT')

            # Segment title: prompt preview + frame range
            title = f"  {seg.prompt[:28]}{'…' if len(seg.prompt) > 28 else ''}"
            header.label(text=title)

            frames_label = f"{seg.start_frame}–{seg.end_frame}"
            header.label(text=frames_label)

            if is_generating:
                header.label(text="", icon='TIME')
            elif seg.generated:
                header.label(text="", icon='CHECKMARK')

            # Select button — sets this as the active segment
            #select_icon = 'RADIOBUT_ON' if is_active else 'RADIOBUT_OFF'
            #op_sel = header.operator("kimodo.select_segment", text="", icon=select_icon, emboss=False)
            #op_sel.index = i

            # Remove button
            op_rem = header.operator("kimodo.remove_segment_by_index", text="", icon='X', emboss=False)
            op_rem.index = i

            # --- Body: always visible ---
            col = box.column(align=True)
            col.prop(seg, "prompt", text="")

            row2 = col.row(align=True)
            start_sub = row2.row(align=True)
            start_sub.enabled = (i == 0)
            start_sub.prop(seg, "start_frame", text="Start")
            row2.prop(seg, "end_frame",   text="End")

            fps = context.scene.render.fps / context.scene.render.fps_base
            dur = (seg.end_frame - seg.start_frame + 1) / fps
            col.label(text=f"  {dur:.1f}s · {seg.end_frame - seg.start_frame + 1} frames",
                      icon='TIME')



        layout.separator()

        # --- Generate buttons ---
        layout.prop(s, "bvh_standard_tpose", icon='ARMATURE_DATA')

        # Reuse armature eyedropper + auto-pick button
        reuse_row = layout.row(align=True)
        reuse_row.prop(s, "reuse_armature", text="Reuse", icon='ARMATURE_DATA')
        reuse_row.operator("kimodo.pick_latest_armature", text="", icon='SORTTIME')

        gen_row = layout.row(align=True)
        gen_row.enabled = s.is_connected and not s.is_generating
        #gen_row.operator("kimodo.generate_segment",      text="Generate Selected", icon='PLAY')
        #use tpose button below
        
        gen_row.scale_y = 2
        gen_row.operator("kimodo.generate_all_segments", text="Generate Motion",icon='PLAY')

        if s.is_generating:
            box2 = layout.box()
            box2.label(text=s.generation_progress or "Working…", icon='TIME')
            box2.operator("kimodo.cancel_generation", text="Cancel", icon='X')



# ---------------------------------------------------------------------------
# Panel 3: Generate (single prompt, kept for quick use)
# ---------------------------------------------------------------------------

class KIMODO_PT_Generate(KIMODO_PanelBase, Panel):
    bl_label   = "🎬  Quick Generate"
    bl_idname  = "KIMODO_PT_Generate"
    bl_order   = 18
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        s = context.scene.kimodo

        # Disable panel while generating
        layout.enabled = not s.is_generating

        # Model selector
        row = layout.row(align=True)
        row.label(text="Model:")
        row.prop(s, "model_type", expand=True)

        # Prompt
        layout.label(text="Prompt:")
        layout.prop(s, "prompt", text="")

        # Duration + Seed
        split = layout.split(factor=0.6)
        split.prop(s, "duration", slider=True)
        split.prop(s, "seed")

        # Output format
        row = layout.row(align=True)
        row.label(text="Output:")
        row.prop(s, "output_format", expand=True)

        # BVH T-pose option (only show for BVH format)
        if s.output_format == "bvh":
            layout.prop(s, "bvh_standard_tpose", icon='ARMATURE_DATA')

        layout.separator()

        # Reuse armature eyedropper + auto-pick button
        reuse_row = layout.row(align=True)
        reuse_row.prop(s, "reuse_armature", text="Reuse", icon='ARMATURE_DATA')
        reuse_row.operator("kimodo.pick_latest_armature", text="", icon='SORTTIME')

        layout.separator()

        # Generate / Cancel button
        if s.is_generating:
            col = layout.column()
            col.enabled = True   # re-enable for cancel
            col.operator("kimodo.cancel_generation", text="⏹  Cancel", icon='X')
            # Progress
            box = layout.box()
            box.label(text=s.generation_progress or "Working…", icon='TIME')
        else:
            connected_icon = 'PLAY' if s.is_connected else 'UNLINKED'
            row = layout.column()
            
            row.enabled = s.is_connected
            
            scene_fps = context.scene.render.fps / context.scene.render.fps_base
            
            if abs(scene_fps - 30.0) > 0.01:
                fps_box = row.box()
                fps_box.alert = True
                fps_box.label(text=f"Scene is {scene_fps:.4g} FPS — Kimodo needs 30 FPS", icon='ERROR')
                fps_box.operator("kimodo.set_to_30fps", text="Set to 30 FPS", icon='RECOVER_LAST')

            row.scale_y = 2
            row.operator("kimodo.generate", text="Generate Motion", icon=connected_icon)
            if s.generation_progress:
                layout.label(text=s.generation_progress,
                             icon='CHECKMARK' if "Done" in s.generation_progress else 'ERROR')

        # Manual import fallback
        layout.separator()
        box = layout.box()
        box.label(text="Manual Import", icon='IMPORT')
        row = box.row()
        row.prop(s, "last_bvh_path", text="BVH Path")
        row.operator("kimodo.import_bvh", text="", icon='FILE_FOLDER').filepath = ""


# ---------------------------------------------------------------------------
# Panel 3: Motion Constraints
# ---------------------------------------------------------------------------

class KIMODO_PT_Constraints(KIMODO_PanelBase, Panel):
    bl_label   = "🎯  Motion Constraints"
    bl_idname  = "KIMODO_PT_Constraints"
    bl_order   = 25

    def draw(self, context):
        layout = self.layout
        s = context.scene.kimodo

        # --- Quick-add buttons ---
        layout.label(text="Add Constraint at Current Frame:", icon='ADD')
        grid = layout.grid_flow(row_major=True, columns=3, even_columns=True, align=True)

        add_types = [
            ('root2d',      "Root XZ",    'EMPTY_ARROWS'),
            ('fullbody',    "Full-Body",  'ARMATURE_DATA'),
            ('left_hand',   "L.Hand",     'VIEW_PAN'),
            ('right_hand',  "R.Hand",     'VIEW_PAN'),
            ('left_foot',   "L.Foot",     'SNAP_FACE'),
            ('right_foot',  "R.Foot",     'SNAP_FACE'),
        ]
        for ctype, label, icon in add_types:
            op = grid.operator("kimodo.add_constraint", text=label, icon=icon)
            op.constraint_type = ctype

        # Fullbody tip — shown when Full-Body button is in the grid
        has_fullbody = any(ci.constraint_type == 'fullbody' for ci in s.motion_constraints)
        if not has_fullbody:
            tip = layout.box()
            tip.label(text="Full-Body tip:", icon='INFO')
            tip.label(text="Select an armature first, then click Full-Body.")
            tip.label(text="Or generate once — source arm will be duplicated.")

        layout.separator()
        n = len(s.motion_constraints)
        if n:
            layout.label(text=f"{n} constraint{'s' if n != 1 else ''}:", icon='SEQUENCE')
        else:
            layout.label(text="No constraints — motion is unconstrained", icon='INFO')

        for i, ci in enumerate(s.motion_constraints):
            box = layout.box()
            row = box.row(align=True)

            # Enabled toggle
            row.prop(ci, "enabled", text="", emboss=False,
                     icon='CHECKBOX_HLT' if ci.enabled else 'CHECKBOX_DEHLT')

            # Type icon
            type_icons = {
                'root2d': 'EMPTY_ARROWS', 'fullbody': 'ARMATURE_DATA',
                'left_hand': 'VIEW_PAN',  'right_hand': 'VIEW_PAN',
                'left_foot': 'SNAP_FACE', 'right_foot': 'SNAP_FACE',
            }
            row.label(text="", icon=type_icons.get(ci.constraint_type, 'DOT'))

            # Type selector
            row.prop(ci, "constraint_type", text="")

            # Frame
            row.label(text="F:")
            row.prop(ci, "frame", text="")

            # Go to frame
            op_goto = row.operator("kimodo.goto_constraint_frame", text="", icon='TIME')
            op_goto.frame = ci.frame

            # Select object
            op_sel = row.operator("kimodo.select_constraint_object", text="", icon='RESTRICT_SELECT_OFF')
            op_sel.index = i

            # Remove
            op_rem = row.operator("kimodo.remove_constraint", text="", icon='X')
            op_rem.delete_object = False

            # --- Object picker row — type-aware ---
            sub = box.row(align=True)

            if ci.constraint_type == 'fullbody':
                # Armature picker with validation warning
                obj = ci.marker_object
                if obj and obj.type != 'ARMATURE':
                    # Wrong type — show error
                    sub.alert = True
                    sub.label(text="⚠ Not an armature!", icon='ERROR')
                    sub.prop(ci, "marker_object", text="Fix →")
                elif not obj:
                    # Nothing set — show hint
                    sub.alert = True
                    sub.label(text="Select an armature above then click Full-Body again", icon='INFO')
                    sub.prop(ci, "marker_object", text="Or set →")
                else:
                    # Valid armature — show normally with bone count hint
                    bone_n = len(obj.data.bones)
                    sub.prop(ci, "marker_object", text=f"Pose Ref  ({bone_n} bones)")
                    sub.operator(
                        "kimodo.select_constraint_object",
                        text="", icon='EDITMODE_HLT',
                    ).index = i
            else:
                # Regular Empty picker for all other types
                sub.prop(ci, "marker_object", text="Marker")

            # root2d heading extras
            if ci.constraint_type == 'root2d':
                sub2 = box.row(align=True)
                sub2.prop(ci, "include_heading", text="Heading")
                if ci.include_heading:
                    sub2.prop(ci, "heading_angle", text="")

        layout.separator()

        # --- Settings ---
        box = layout.box()
        box.label(text="Settings", icon='PREFERENCES')
        row = box.row(align=True)
        row.prop(s, "kimodo_fps")
        row.prop(s, "auto_canonicalize", toggle=True, text="Auto-Origin")

        layout.separator()

        # --- Actions ---
        row = layout.row(align=True)
        row.operator("kimodo.preview_constraints_json", icon='TEXT', text="Preview JSON")
        row.operator("kimodo.clear_constraints",        icon='TRASH', text="Clear All")


# ---------------------------------------------------------------------------
# Panel 4: Retarget
# ---------------------------------------------------------------------------

class KIMODO_PT_Retarget(KIMODO_PanelBase, Panel):
    bl_label   = "🦴  Retarget"
    bl_idname  = "KIMODO_PT_Retarget"
    bl_order   = 30

    def draw(self, context):
        layout = self.layout
        s = context.scene.kimodo

        # Armature pickers
        box = layout.box()
        box.label(text="Armatures", icon='ARMATURE_DATA')
        box.prop(s, "source_armature", text="Source (Kimodo)")
        box.prop(s, "target_armature", text="Target (Your Rig)")
        box.prop(s, "retarget_root_bone", text="Root Bone")

        layout.separator()

        # Bone mapping list
        layout.label(text="Bone Mapping:", icon='BONE_DATA')

        if s.source_armature and s.target_armature:
            # Auto-match button
            layout.operator("kimodo.auto_map_bones",
                            text="Auto-Match Bones", icon='SHADERFX')

        row = layout.row()
        row.template_list(
            "KIMODO_UL_BoneMappings", "",
            s, "bone_mappings",
            s, "bone_mapping_index",
            rows=6,
        )

        col = row.column(align=True)
        col.operator("kimodo.add_bone_mapping",    text="", icon='ADD')
        col.operator("kimodo.remove_bone_mapping", text="", icon='REMOVE')

        layout.separator()

        # Apply / Remove constraints
        row = layout.row(align=True)
        row.operator("kimodo.apply_retargeting",  text="Apply Constraints", icon='CONSTRAINT_BONE')
        row.operator("kimodo.remove_retargeting", text="",                  icon='X')

        layout.separator()

        # Bake section
        box = layout.box()
        box.label(text="Bake Animation", icon='RENDER_ANIMATION')
        row = box.row(align=True)
        row.prop(s, "bake_start_frame", text="Start")
        row.prop(s, "bake_end_frame",   text="End")
        box.operator("kimodo.bake_retargeting",
                     text="Bake & Remove Constraints", icon='NLA_PUSHDOWN')

        layout.separator()

        # Presets
        box = layout.box()
        box.label(text="Bone Map Presets", icon='PRESET')
        row = box.row(align=True)
        row.prop(s, "preset_name", text="")
        row.operator("kimodo.save_preset", text="", icon='FILE_TICK')
        row.operator("kimodo.load_preset", text="", icon='IMPORT').preset_name = s.preset_name

        # List saved presets
        try:
            prefs = context.preferences.addons[__package__].preferences
            from . import retarget as rt
            preset_names = rt.list_presets(prefs)
        except Exception:
            preset_names = []

        if preset_names:
            col = box.column(align=True)
            for name in preset_names:
                row2 = col.row(align=True)
                op_load = row2.operator("kimodo.load_preset",   text=name, icon='IMPORT')
                op_load.preset_name = name
                op_del  = row2.operator("kimodo.delete_preset", text="",   icon='TRASH')
                op_del.preset_name = name

        # File export / import
        row = box.row(align=True)
        row.operator("kimodo.export_preset_file", text="Export to File", icon='EXPORT')
        row.operator("kimodo.import_preset_file", text="Import from File", icon='IMPORT')


# ---------------------------------------------------------------------------
# Panel 4: Help / About
# ---------------------------------------------------------------------------

class KIMODO_PT_Help(KIMODO_PanelBase, Panel):
    bl_label   = "ℹ  Help"
    bl_idname  = "KIMODO_PT_Help"
    bl_order   = 90
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Quick Start:", icon='QUESTION')
        col.separator()
        col.label(text="1. Install Kimodo via python virtual environment(https://github.com/nv-tlabs/kimodo)")
        col.label(text="2. Set Python path → Start Kimodo")
        col.label(text="   (model loads once, stays loaded)")
        col.separator()
        col.label(text="3. Enter prompt → Generate Motion")
        col.label(text="4. In Retarget tab, Pick source + target rigs")
        col.label(text="5. Auto-Match → Apply Constraints")
        col.label(text="6. If it does not match, manually add your control bones")
        col.label(text="7. Click Apply Constraints")
        col.label(text="8. Bake when satisfied")
        col.separator()
        col.label(text="Low VRAM? Run separately while having the Kimodo virtual environment activated:", icon='INFO')
        col.label(text="  while in kimodo folder: \"source venv/bin/activate\"")
        col.label(text="  kimodo_textencoder --device cpu")
        col.separator()
        col.label(text="Docs & Source:", icon='URL')
        col.label(text="github.com/nv-tlabs/kimodo")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = [
    KIMODO_PT_Connection,
    KIMODO_PT_Segments,
    KIMODO_PT_Generate,
    KIMODO_PT_Constraints,
    KIMODO_PT_Retarget,
    KIMODO_PT_Help,
]


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
