"""
Kimodo Blender Bridge — Operators
All bpy.ops.kimodo.* operators.
"""

import bpy
import os
import json
import threading
import random
import time
from bpy.types import Operator
from bpy.props import StringProperty, BoolProperty, IntProperty

from . import subprocess_client as sc
from . import retarget as rt
from . import constraints as cmod


# ---------------------------------------------------------------------------
# Shared async state (thread → modal operator communication)
# ---------------------------------------------------------------------------

_generation_state = {
    "running": False,
    "done": False,
    "success": False,
    "result": "",       # file path on success, error message on failure
    "progress": "",
}


def _reset_state():
    _generation_state.update(running=False, done=False,
                              success=False, result="", progress="")


_HISTORY_MAX = 20


def _push_history(s, prompt: str, seed: int, duration: float, bvh_path: str) -> None:
    """Prepend a new entry to generation_history, keeping newest-first, capped at _HISTORY_MAX."""
    import datetime
    entry = s.generation_history.add()
    entry.prompt    = prompt
    entry.seed      = seed
    entry.duration  = duration
    entry.bvh_path  = bvh_path
    entry.timestamp = datetime.datetime.now().isoformat(timespec='seconds')
    # Move the new item (appended at the end) to index 0
    last = len(s.generation_history) - 1
    for i in range(last, 0, -1):
        s.generation_history.move(i, i - 1)
    # Trim to max size
    while len(s.generation_history) > _HISTORY_MAX:
        s.generation_history.remove(len(s.generation_history) - 1)


# ---------------------------------------------------------------------------
# Bridge start/stop state  (thread → modal operator communication)
# ---------------------------------------------------------------------------

_start_state = {
    "running": False,
    "done":    False,
    "success": False,
    "message": "",
}


def _reset_start_state():
    _start_state.update(running=False, done=False, success=False, message="")


# ---------------------------------------------------------------------------
# Connection operators
# ---------------------------------------------------------------------------

class KIMODO_OT_StartKimodo(Operator):
    """Load the Kimodo model in the background and keep it ready for generation"""
    bl_idname = "kimodo.start_kimodo"
    bl_label  = "Start Kimodo"

    _timer  = None
    _thread = None

    def _run_start(self, python_exe: str, model_name: str):
        def progress(msg):
            _start_state["message"] = msg

        success, msg = sc.start(python_exe, model_name, progress_callback=progress)
        _start_state["success"] = success
        _start_state["message"] = msg
        _start_state["done"]    = True
        _start_state["running"] = False

    def invoke(self, context, event):
        s = context.scene.kimodo

        if sc.is_running():
            self.report({'INFO'}, "Kimodo is already running.")
            return {'CANCELLED'}

        _reset_start_state()
        _start_state["running"] = True
        s.is_connected      = False
        s.connection_status = "Starting…"

        self._thread = threading.Thread(
            target=self._run_start,
            args=(s.python_executable, s.kimodo_model),
            daemon=True,
        )
        self._thread.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    # Allow calling without a click event (e.g. from scripts)
    def execute(self, context):
        return self.invoke(context, None)

    def modal(self, context, event):
        s = context.scene.kimodo

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        s.connection_status = _start_state["message"]
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        if not _start_state["done"]:
            return {'RUNNING_MODAL'}

        context.window_manager.event_timer_remove(self._timer)

        if _start_state["success"]:
            s.is_connected      = True
            s.connection_status = _start_state["message"]
            self.report({'INFO'}, f"Kimodo ready: {_start_state['message']}")
        else:
            s.is_connected      = False
            s.connection_status = _start_state["message"]
            self.report({'ERROR'}, f"Kimodo failed to start: {_start_state['message']}")

        return {'FINISHED'}

    def cancel(self, context):
        context.window_manager.event_timer_remove(self._timer)


class KIMODO_OT_StopKimodo(Operator):
    """Shut down the Kimodo bridge process and free GPU memory"""
    bl_idname = "kimodo.stop_kimodo"
    bl_label  = "Stop Kimodo"

    def execute(self, context):
        sc.stop()
        s = context.scene.kimodo
        s.is_connected      = False
        s.connection_status = "Stopped"
        self.report({'INFO'}, "Kimodo bridge stopped.")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Generation operators
# ---------------------------------------------------------------------------

class KIMODO_OT_Generate(Operator):
    """Generate motion with Kimodo. Runs in background thread."""
    bl_idname = "kimodo.generate"
    bl_label = "Generate Motion"

    _timer = None
    _thread = None

    def _run_generation(self, prompt, duration, seed, fmt, constraints_json=None, bvh_standard_tpose=False):
        """Runs in background thread."""
        def progress_cb(msg):
            _generation_state["progress"] = msg

        success, result = sc.generate_motion(
            prompt=prompt,
            duration=duration,
            seed=seed,
            output_format=fmt,
            constraints_json=constraints_json,
            bvh_standard_tpose=bvh_standard_tpose,
            progress_callback=progress_cb,
        )
        _generation_state["success"] = success
        _generation_state["result"] = result
        _generation_state["done"] = True
        _generation_state["running"] = False

    def invoke(self, context, event):
        s = context.scene.kimodo

        if not sc.is_running():
            self.report({'WARNING'}, "Kimodo is not running — click 'Start Kimodo' first.")
            return {'CANCELLED'}
        if s.is_generating:
            self.report({'WARNING'}, "Already generating — please wait.")
            return {'CANCELLED'}

        # Resolve seed
        seed = s.seed if s.seed >= 0 else random.randint(0, 2**31)
        self._resolved_seed = seed

        # Launch background thread
        _reset_state()
        _generation_state["running"] = True
        s.is_generating = True
        s.generation_progress = "Starting…"

        # Build constraints JSON if any are defined
        constraints_json = None
        enabled_constraints = [c for c in s.motion_constraints if c.enabled and c.marker_object]
        if enabled_constraints:
            try:
                constraints_data = cmod.build_constraints_json(
                    s.motion_constraints,
                    context.scene,
                    kimodo_fps=s.kimodo_fps,
                    auto_canonicalize=s.auto_canonicalize,
                )
                constraints_json = json.dumps(constraints_data)
                s.constraint_json_preview = constraints_json
            except Exception as e:
                self.report({'WARNING'}, f"Constraint build failed (generating without): {e}")

        self._thread = threading.Thread(
            target=self._run_generation,
            args=(
                s.prompt,
                s.duration,
                seed,
                s.output_format,
                constraints_json,
                s.bvh_standard_tpose,
            ),
            daemon=True,
        )
        self._thread.start()

        # Add modal timer to poll thread
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        s = context.scene.kimodo

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        # Update progress display
        s.generation_progress = _generation_state.get("progress", "")
        # Force N-panel redraw
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        if not _generation_state["done"]:
            return {'RUNNING_MODAL'}

        # Generation finished
        context.window_manager.event_timer_remove(self._timer)
        s.is_generating = False

        if _generation_state["success"]:
            file_path = _generation_state["result"]
            s.last_bvh_path = file_path
            s.generation_progress = "Done ✓"
            _push_history(s, s.prompt, self._resolved_seed, s.duration, file_path)
            # Auto-import if BVH
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".bvh":
                bpy.ops.kimodo.import_bvh('EXEC_DEFAULT', filepath=file_path)
                self.report({'INFO'}, f"Motion generated and imported from {os.path.basename(file_path)}")
            else:
                self.report({'INFO'}, f"Motion saved to {file_path} (NPZ — import manually)")
        else:
            s.generation_progress = "Failed ✗"
            self.report({'ERROR'}, f"Generation failed: {_generation_state['result']}")

        return {'FINISHED'}

    def cancel(self, context):
        context.window_manager.event_timer_remove(self._timer)
        context.scene.kimodo.is_generating = False
        context.scene.kimodo.generation_progress = "Cancelled"
        _generation_state["running"] = False


class KIMODO_OT_CancelGeneration(Operator):
    """Cancel ongoing generation (marks as cancelled; thread will finish naturally)"""
    bl_idname = "kimodo.cancel_generation"
    bl_label = "Cancel"

    def execute(self, context):
        _generation_state["running"] = False
        context.scene.kimodo.is_generating = False
        context.scene.kimodo.generation_progress = "Cancelled"
        self.report({'INFO'}, "Cancellation requested.")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

def _apply_to_existing_source(s, new_arm: bpy.types.Object) -> bpy.types.Object:
    """
    If reuse_armature is set, transfer the action from new_arm to it and
    delete new_arm.  Returns the armature that should be used going forward.
    """
    existing = s.reuse_armature
    if not existing or existing.type != 'ARMATURE' or existing == new_arm:
        return new_arm  # nothing to reuse — keep the freshly imported one

    # Transfer the action (the data-block survives object deletion).
    if new_arm.animation_data and new_arm.animation_data.action:
        if not existing.animation_data:
            existing.animation_data_create()
        existing.animation_data.action = new_arm.animation_data.action

    # Remove the temporary stand-in armature.
    bpy.data.objects.remove(new_arm, do_unlink=True)
    return existing


def _iter_action_fcurves(action):
    """Yield all FCurves from an action.

    Handles both the legacy API (action.fcurves, Blender < 4.4) and the
    layered action API (action.layers → strips → channelbag, Blender 4.4+).
    """
    if hasattr(action, 'fcurves'):
        # Legacy system — still present on older Blender builds.
        yield from action.fcurves
        return
    # Layered system (Blender 4.4+): layers → strips → channelbag(slot).
    for layer in getattr(action, 'layers', []):
        for strip in layer.strips:
            for slot in getattr(action, 'slots', []):
                try:
                    cb = strip.channelbag(slot)
                    if cb is not None:
                        yield from cb.fcurves
                except Exception:
                    pass


def _scale_root_location_keyframes(arm: bpy.types.Object, scale_factor: float) -> None:
    """Scale root-bone location keyframe values to compensate for an applied armature scale.

    When a SOMA armature has been scaled and the scale applied, the bone
    coordinate space changes.  Multiplying the root/hip location fcurve
    values by scale_factor restores the expected world-space displacement.
    Only runs when scale_factor differs meaningfully from 1.0.
    """
    if abs(scale_factor - 1.0) < 1e-5:
        return

    action = arm.animation_data and arm.animation_data.action
    if not action:
        return

    root_name = None
    for name in ("Hips", "hips", "Hip", "pelvis", "Pelvis"):
        if arm.pose.bones.get(name):
            root_name = name
            break
    if root_name is None:
        return

    dp = f'pose.bones["{root_name}"].location'
    for fc in _iter_action_fcurves(action):
        if fc.data_path == dp:
            for kp in fc.keyframe_points:
                kp.co[1]           *= scale_factor
                kp.handle_left[1]  *= scale_factor
                kp.handle_right[1] *= scale_factor
            fc.update()


# ---------------------------------------------------------------------------
# Import operators
# ---------------------------------------------------------------------------

class KIMODO_OT_ImportBVH(Operator):
    """Import a BVH file and register it as the Kimodo source armature"""
    bl_idname = "kimodo.import_bvh"
    bl_label = "Import BVH"

    filepath: StringProperty(subtype='FILE_PATH')

    def execute(self, context):
        s = context.scene.kimodo
        path = self.filepath or s.last_bvh_path

        if not path or not os.path.exists(path):
            self.report({'ERROR'}, f"File not found: {path}")
            return {'CANCELLED'}

        # Remember which objects exist before import
        before = set(bpy.context.scene.objects)

        bpy.ops.import_anim.bvh(
            filepath=path,
            axis_forward='-Z',
            axis_up='Y',
            target='ARMATURE',
            global_scale=0.01,   # BVH is usually in cm; Blender expects meters
            frame_start=1,
            use_fps_scale=False,
            update_scene_fps=False,
            update_scene_duration=True,
            use_cyclic=False,
            rotate_mode='NATIVE',
        )

        # Find newly added armature
        after = set(bpy.context.scene.objects)
        new_objs = after - before
        new_arm = next((o for o in new_objs if o.type == 'ARMATURE'), None)

        if new_arm:
            new_arm.name = "Kimodo_Source"
            new_arm["kimodo_source"] = True
            new_arm["kimodo_creation_time"] = time.time()
            new_arm = _apply_to_existing_source(s, new_arm)
            _scale_root_location_keyframes(new_arm, s.soma_scale_factor)
            s.source_armature = new_arm
            s.reuse_armature = new_arm
            self.report({'INFO'}, f"Imported '{new_arm.name}' with {len(new_arm.data.bones)} bones")
        else:
            self.report({'WARNING'}, "BVH imported but no armature found in scene.")

        return {'FINISHED'}

    def invoke(self, context, event):
        if self.filepath:
            return self.execute(context)
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}




# ---------------------------------------------------------------------------
# Retargeting operators
# ---------------------------------------------------------------------------

class KIMODO_OT_AutoMapBones(Operator):
    """Auto-match bone names between Kimodo source and target armature"""
    bl_idname = "kimodo.auto_map_bones"
    bl_label = "Auto-Match Bones"

    def execute(self, context):
        s = context.scene.kimodo
        if not s.source_armature:
            self.report({'ERROR'}, "Set the Source Armature first.")
            return {'CANCELLED'}
        if not s.target_armature:
            self.report({'ERROR'}, "Set the Target Armature first.")
            return {'CANCELLED'}

        pairs = rt.auto_build_mapping(s.source_armature, s.target_armature, s.model_type)
        s.bone_mappings.clear()

        for src, tgt in pairs:
            item = s.bone_mappings.add()
            item.source_bone = src
            item.target_bone = tgt
            item.enabled = True

        self.report({'INFO'}, f"Auto-matched {len(pairs)} bone pairs")
        return {'FINISHED'}


class KIMODO_OT_AddBoneMapping(Operator):
    """Add a new empty bone mapping row"""
    bl_idname = "kimodo.add_bone_mapping"
    bl_label = "Add Bone Pair"

    def execute(self, context):
        s = context.scene.kimodo
        item = s.bone_mappings.add()
        item.source_bone = ""
        item.target_bone = ""
        item.enabled = True
        s.bone_mapping_index = len(s.bone_mappings) - 1
        return {'FINISHED'}


class KIMODO_OT_RemoveBoneMapping(Operator):
    """Remove the selected bone mapping row"""
    bl_idname = "kimodo.remove_bone_mapping"
    bl_label = "Remove Bone Pair"

    def execute(self, context):
        s = context.scene.kimodo
        idx = s.bone_mapping_index
        if 0 <= idx < len(s.bone_mappings):
            s.bone_mappings.remove(idx)
            s.bone_mapping_index = max(0, idx - 1)
        return {'FINISHED'}


class KIMODO_OT_ApplyRetargeting(Operator):
    """Apply Copy Rotation/Location constraints to drive target rig from Kimodo motion"""
    bl_idname = "kimodo.apply_retargeting"
    bl_label = "Apply Retargeting"

    def execute(self, context):
        s = context.scene.kimodo
        if not s.source_armature:
            self.report({'ERROR'}, "Set Source Armature.")
            return {'CANCELLED'}
        if not s.target_armature:
            self.report({'ERROR'}, "Set Target Armature.")
            return {'CANCELLED'}
        if not s.bone_mappings:
            self.report({'ERROR'}, "No bone mappings defined. Use Auto-Match or add manually.")
            return {'CANCELLED'}

        pairs = [(item.source_bone, item.target_bone, item.enabled, item.retarget_mode)
                 for item in s.bone_mappings]

        n, warnings = rt.apply_retargeting_constraints(
            s.source_armature, s.target_armature, pairs, s.retarget_root_bone
        )

        for w in warnings:
            self.report({'WARNING'}, w)

        self.report({'INFO'}, f"Applied retargeting constraints to {n} bones")
        return {'FINISHED'}


class KIMODO_OT_RemoveRetargeting(Operator):
    """Remove all Kimodo retargeting constraints from target armature"""
    bl_idname = "kimodo.remove_retargeting"
    bl_label = "Remove Constraints"

    def execute(self, context):
        s = context.scene.kimodo
        if not s.target_armature:
            self.report({'ERROR'}, "Set Target Armature.")
            return {'CANCELLED'}
        n = rt.remove_retargeting_constraints(s.target_armature)
        self.report({'INFO'}, f"Removed {n} Kimodo constraints")
        return {'FINISHED'}


class KIMODO_OT_BakeRetargeting(Operator):
    """Bake the retargeted animation into keyframes and remove constraints"""
    bl_idname = "kimodo.bake_retargeting"
    bl_label = "Bake Animation"

    def execute(self, context):
        s = context.scene.kimodo
        if not s.target_armature:
            self.report({'ERROR'}, "Set Target Armature.")
            return {'CANCELLED'}

        success = rt.bake_retargeted_animation(
            s.target_armature,
            s.bake_start_frame,
            s.bake_end_frame,
        )
        if success:
            self.report({'INFO'}, "Animation baked successfully ✓")
        else:
            self.report({'ERROR'}, "Bake failed — check console for details")
        return {'FINISHED'} if success else {'CANCELLED'}


# ---------------------------------------------------------------------------
# Preset operators
# ---------------------------------------------------------------------------

class KIMODO_OT_SavePreset(Operator):
    """Save current bone mapping as a named preset"""
    bl_idname = "kimodo.save_preset"
    bl_label = "Save Preset"

    def execute(self, context):
        s = context.scene.kimodo
        prefs = context.preferences.addons[__package__].preferences
        name = s.preset_name.strip()
        if not name:
            self.report({'ERROR'}, "Enter a preset name first.")
            return {'CANCELLED'}

        pairs = [{"src": item.source_bone, "tgt": item.target_bone,
                  "en": item.enabled, "mode": item.retarget_mode}
                 for item in s.bone_mappings]
        rt.save_preset(prefs, name, pairs)
        self.report({'INFO'}, f"Preset '{name}' saved ({len(pairs)} bone pairs)")
        return {'FINISHED'}


class KIMODO_OT_LoadPreset(Operator):
    """Load a saved bone mapping preset"""
    bl_idname = "kimodo.load_preset"
    bl_label = "Load Preset"

    preset_name: StringProperty()

    def execute(self, context):
        s = context.scene.kimodo
        prefs = context.preferences.addons[__package__].preferences
        name = self.preset_name or s.preset_name.strip()

        pairs = rt.load_preset(prefs, name)
        if pairs is None:
            self.report({'ERROR'}, f"Preset '{name}' not found.")
            return {'CANCELLED'}

        s.bone_mappings.clear()
        for p in pairs:
            item = s.bone_mappings.add()
            item.source_bone   = p.get("src", "")
            item.target_bone   = p.get("tgt", "")
            item.enabled       = p.get("en", True)
            item.retarget_mode = p.get("mode", "COPY_ROTATION")

        self.report({'INFO'}, f"Loaded preset '{name}' ({len(pairs)} bone pairs)")
        return {'FINISHED'}


class KIMODO_OT_DeletePreset(Operator):
    """Delete a saved bone mapping preset"""
    bl_idname = "kimodo.delete_preset"
    bl_label = "Delete Preset"

    preset_name: StringProperty()

    def execute(self, context):
        prefs = context.preferences.addons[__package__].preferences
        name = self.preset_name
        try:
            presets = json.loads(prefs.saved_presets)
            if name in presets:
                del presets[name]
                prefs.saved_presets = json.dumps(presets)
                self.report({'INFO'}, f"Deleted preset '{name}'")
            else:
                self.report({'WARNING'}, f"Preset '{name}' not found")
        except Exception as e:
            self.report({'ERROR'}, str(e))
        return {'FINISHED'}


class KIMODO_OT_ExportPresetFile(Operator):
    """Export the current bone mapping to a JSON file"""
    bl_idname  = "kimodo.export_preset_file"
    bl_label   = "Export Bone Map"

    filepath:   StringProperty(subtype='FILE_PATH')
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    def invoke(self, context, event):
        s = context.scene.kimodo
        self.filepath = (s.preset_name.strip() or "bone_map") + ".json"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        s = context.scene.kimodo
        pairs = [{"src": item.source_bone, "tgt": item.target_bone,
                  "en": item.enabled, "mode": item.retarget_mode}
                 for item in s.bone_mappings]
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(pairs, f, indent=2)
            self.report({'INFO'}, f"Exported {len(pairs)} bone pairs to {self.filepath}")
        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}


class KIMODO_OT_ImportPresetFile(Operator):
    """Import a bone mapping from a JSON file"""
    bl_idname  = "kimodo.import_preset_file"
    bl_label   = "Import Bone Map"

    filepath:   StringProperty(subtype='FILE_PATH')
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        s = context.scene.kimodo
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                pairs = json.load(f)
            if not isinstance(pairs, list):
                self.report({'ERROR'}, "File does not contain a JSON array.")
                return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Import failed: {e}")
            return {'CANCELLED'}

        s.bone_mappings.clear()
        for p in pairs:
            item = s.bone_mappings.add()
            item.source_bone   = p.get("src", "")
            item.target_bone   = p.get("tgt", "")
            item.enabled       = p.get("en", True)
            item.retarget_mode = p.get("mode", "COPY_ROTATION")

        self.report({'INFO'}, f"Imported {len(pairs)} bone pairs from {self.filepath}")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Motion segment operators
# ---------------------------------------------------------------------------

from .properties import _SEGMENT_COLORS


class KIMODO_OT_SelectSegment(Operator):
    """Set a segment as the active one"""
    bl_idname = "kimodo.select_segment"
    bl_label  = "Select Segment"
    index: IntProperty()

    def execute(self, context):
        s = context.scene.kimodo
        if 0 <= self.index < len(s.motion_segments):
            s.segment_index = self.index
        return {'FINISHED'}


class KIMODO_OT_RemoveSegmentByIndex(Operator):
    """Remove a specific segment by index"""
    bl_idname = "kimodo.remove_segment_by_index"
    bl_label  = "Remove Segment"
    index: IntProperty()

    def execute(self, context):
        s = context.scene.kimodo
        if 0 <= self.index < len(s.motion_segments):
            s.motion_segments.remove(self.index)
            s.segment_index = max(0, min(s.segment_index, len(s.motion_segments) - 1))
            _tag_timeline_redraw(context)
        return {'FINISHED'}


class KIMODO_OT_AddSegment(Operator):
    """Add a new motion segment at the current frame"""
    bl_idname = "kimodo.add_segment"
    bl_label = "Add Segment"

    def execute(self, context):
        s = context.scene.kimodo
        scene = context.scene

        # Auto-assign a cycling colour
        color = _SEGMENT_COLORS[len(s.motion_segments) % len(_SEGMENT_COLORS)]

        # Place segment after the last existing one, or at current frame
        start = scene.frame_current
        if s.motion_segments:
            last = max(s.motion_segments, key=lambda seg: seg.end_frame)
            start = last.end_frame + 1
            

        fps = scene.render.fps / scene.render.fps_base
        end = start + int(fps * 5) - 1  # default 5 seconds

        seg = s.motion_segments.add()
        seg.prompt      = "a person walks forward"
        seg.start_frame = start
        seg.end_frame   = end
        seg.model_type  = s.model_type
        seg.seed        = s.seed
        seg.color       = color
        seg.enabled     = True

        s.segment_index = len(s.motion_segments) - 1

        # Force timeline redraw
        _tag_timeline_redraw(context)
        return {'FINISHED'}


class KIMODO_OT_RemoveSegment(Operator):
    """Remove the selected motion segment"""
    bl_idname = "kimodo.remove_segment"
    bl_label = "Remove Segment"

    def execute(self, context):
        s = context.scene.kimodo
        idx = s.segment_index
        if 0 <= idx < len(s.motion_segments):
            s.motion_segments.remove(idx)
            s.segment_index = max(0, idx - 1)
            _tag_timeline_redraw(context)
        return {'FINISHED'}


class KIMODO_OT_DuplicateSegment(Operator):
    """Duplicate the selected segment and place it immediately after"""
    bl_idname = "kimodo.duplicate_segment"
    bl_label = "Duplicate Segment"

    def execute(self, context):
        s = context.scene.kimodo
        idx = s.segment_index
        if not (0 <= idx < len(s.motion_segments)):
            return {'CANCELLED'}

        src = s.motion_segments[idx]
        duration = src.end_frame - src.start_frame

        new_seg = s.motion_segments.add()
        new_seg.prompt      = src.prompt
        new_seg.start_frame = src.end_frame + 1
        new_seg.end_frame   = src.end_frame + 1 + duration
        new_seg.model_type  = src.model_type
        new_seg.seed        = src.seed
        new_seg.color       = src.color
        new_seg.enabled     = src.enabled

        s.segment_index = len(s.motion_segments) - 1
        _tag_timeline_redraw(context)
        return {'FINISHED'}


class KIMODO_OT_MoveSegmentUp(Operator):
    """Move segment up in list (earlier in sequence)"""
    bl_idname = "kimodo.move_segment_up"
    bl_label = "Move Up"

    def execute(self, context):
        s = context.scene.kimodo
        idx = s.segment_index
        if idx > 0:
            s.motion_segments.move(idx, idx - 1)
            s.segment_index -= 1
        return {'FINISHED'}


class KIMODO_OT_MoveSegmentDown(Operator):
    """Move segment down in list (later in sequence)"""
    bl_idname = "kimodo.move_segment_down"
    bl_label = "Move Down"

    def execute(self, context):
        s = context.scene.kimodo
        idx = s.segment_index
        if idx < len(s.motion_segments) - 1:
            s.motion_segments.move(idx, idx + 1)
            s.segment_index += 1
        return {'FINISHED'}


class KIMODO_OT_SyncSeeds(Operator):
    """Sync all segment seeds with global seed setting"""
    bl_idname = "kimodo.sync_seeds"
    bl_label = "Sync Seeds"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.kimodo
        global_seed = s.seed
        
        for seg in s.motion_segments:
            seg.seed = global_seed
            
        self.report({'INFO'}, f"Updated {len(s.motion_segments)} segments with seed {global_seed}")
        return {'FINISHED'}


class KIMODO_OT_GenerateSegment(Operator):
    """Generate motion for the selected segment"""
    bl_idname = "kimodo.generate_segment"
    bl_label = "Generate Selected"

    _timer  = None
    _thread = None
    _target_segment_idx: int = -1

    def invoke(self, context, event):
        s = context.scene.kimodo
        idx = s.segment_index

        if not (0 <= idx < len(s.motion_segments)):
            self.report({'ERROR'}, "No segment selected.")
            return {'CANCELLED'}
        if not sc.is_running():
            self.report({'WARNING'}, "Kimodo is not running — click 'Start Kimodo' first.")
            return {'CANCELLED'}
        if s.is_generating:
            self.report({'WARNING'}, "Already generating — please wait.")
            return {'CANCELLED'}

        seg = s.motion_segments[idx]
        self._target_segment_idx = idx
        return self._start_generation(context, s, seg)

    def _start_generation(self, context, s, seg):
        import random as _random

        seed = seg.seed if seg.seed >= 0 else _random.randint(0, 2**31)
        self._resolved_seed = seed
        fps  = context.scene.render.fps / context.scene.render.fps_base
        duration = (seg.end_frame - seg.start_frame + 1) / fps
        self._segment_duration = duration

        # Build constraints for this segment if any exist
        constraints_json = _build_segment_constraints(context, seg)

        _reset_state()
        _generation_state["running"] = True
        s.is_generating = True
        s.generation_progress = f"Generating: {seg.prompt[:40]}…"
        s.generating_segment_index = self._target_segment_idx

        self._thread = threading.Thread(
            target=self._run_generation,
            args=(seg.prompt, duration, seed, s.output_format, constraints_json, s.bvh_standard_tpose),
            daemon=True,
        )
        self._thread.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _run_generation(self, prompt, duration, seed, fmt, constraints_json, bvh_standard_tpose):
        def progress_cb(msg):
            _generation_state["progress"] = msg

        success, result = sc.generate_motion(
            prompt=prompt,
            duration=duration,
            seed=seed,
            output_format=fmt,
            constraints_json=constraints_json,
            bvh_standard_tpose=bvh_standard_tpose,
            progress_callback=progress_cb,
        )
        _generation_state["success"] = success
        _generation_state["result"]  = result
        _generation_state["done"]    = True
        _generation_state["running"] = False

    def modal(self, context, event):
        s = context.scene.kimodo
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        s.generation_progress = _generation_state.get("progress", "")
        _tag_timeline_redraw(context)
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        if not _generation_state["done"]:
            return {'RUNNING_MODAL'}

        context.window_manager.event_timer_remove(self._timer)
        s.is_generating = False
        s.generating_segment_index = -1

        if _generation_state["success"]:
            file_path = _generation_state["result"]
            seg = s.motion_segments[self._target_segment_idx]
            seg.last_bvh_path = file_path
            seg.generated = True
            s.generation_progress = "Done ✓"
            _push_history(s, seg.prompt, self._resolved_seed, self._segment_duration, file_path)

            if os.path.splitext(file_path)[1].lower() == ".bvh":
                bpy.ops.kimodo.import_bvh_at_frame(
                    'EXEC_DEFAULT',
                    filepath=file_path,
                    start_frame=seg.start_frame,
                    label=seg.prompt[:30],
                )
                self.report({'INFO'}, f"Segment '{seg.prompt[:40]}' generated ✓")
            else:
                self.report({'INFO'}, f"NPZ saved to {file_path}")
        else:
            s.generation_progress = "Failed ✗"
            self.report({'ERROR'}, f"Generation failed: {_generation_state['result']}")

        _tag_timeline_redraw(context)
        return {'FINISHED'}

    def cancel(self, context):
        context.window_manager.event_timer_remove(self._timer)
        context.scene.kimodo.is_generating = False
        context.scene.kimodo.generating_segment_index = -1
        _generation_state["running"] = False


def _enforce_segment_continuity(ordered_segs):
    """
    Make segments contiguous: each segment starts the frame after the previous ends.
    Preserves each segment's duration (frame count). Returns True if any were changed.
    """
    changed = False
    prev_end = None
    for seg in ordered_segs:
        if prev_end is not None:
            expected_start = prev_end + 1
            if seg.start_frame != expected_start:
                duration_frames = seg.end_frame - seg.start_frame
                seg.start_frame = expected_start
                seg.end_frame   = expected_start + duration_frames
                changed = True
        prev_end = seg.end_frame
    return changed


def _build_multi_prompt_constraints(context, first_start_frame: int) -> "str | None":
    """Build scene constraints JSON for multi-prompt generation.

    Uses first_start_frame as the Kimodo sequence origin so constraint frame
    indices align with the combined BVH that starts at that Blender frame.
    """
    s = context.scene.kimodo
    enabled = [c for c in s.motion_constraints if c.enabled and c.marker_object]
    if not enabled:
        return None
    try:
        data = cmod.build_constraints_json(
            s.motion_constraints, context.scene,
            kimodo_fps=s.kimodo_fps,
            auto_canonicalize=s.auto_canonicalize,
            scene_start_override=first_start_frame,
        )
        return json.dumps(data) if data else None
    except Exception:
        return None


class KIMODO_OT_GenerateAllSegments(Operator):
    """Generate all enabled segments as a single multi-prompt sequence with smooth transitions"""
    bl_idname = "kimodo.generate_all_segments"
    bl_label  = "Generate All Segments"

    _timer           = None
    _thread          = None
    _segment_indices: list = []
    _start_frame: int = 1

    def invoke(self, context, event):
        s = context.scene.kimodo
        if not sc.is_running():
            self.report({'WARNING'}, "Kimodo is not running — click 'Start Kimodo' first.")
            return {'CANCELLED'}
        if s.is_generating:
            self.report({'WARNING'}, "Already generating.")
            return {'CANCELLED'}

        # Collect enabled segments sorted chronologically
        ordered = sorted(
            [(i, seg) for i, seg in enumerate(s.motion_segments) if seg.enabled],
            key=lambda x: x[1].start_frame,
        )
        if not ordered:
            self.report({'WARNING'}, "No enabled segments.")
            return {'CANCELLED'}

        # Auto-fix any gaps or overlaps between segments
        ordered_segs = [seg for _, seg in ordered]
        if _enforce_segment_continuity(ordered_segs):
            self.report({'INFO'},
                "Segment frame ranges adjusted to be contiguous (no gaps/overlaps).")

        self._segment_indices = [i for i, _ in ordered]
        self._start_frame = ordered[0][1].start_frame

        fps = context.scene.render.fps / context.scene.render.fps_base
        prompts   = [seg.prompt for _, seg in ordered]
        durations = [(seg.end_frame - seg.start_frame + 1) / fps for _, seg in ordered]

        # Use the first enabled segment's seed (or random if unset)
        first_seg = ordered[0][1]
        seed = first_seg.seed if first_seg.seed >= 0 else random.randint(0, 2**31)

        # Build constraints relative to the start of the combined sequence
        constraints_json = _build_multi_prompt_constraints(context, self._start_frame)

        s.is_generating = True
        s.generating_segment_index = self._segment_indices[0]
        _reset_state()
        _generation_state["running"] = True
        s.generation_progress = (
            f"Generating {len(ordered)} segments as multi-prompt sequence…"
        )

        num_transition_frames = s.num_transition_frames

        self._thread = threading.Thread(
            target=self._run_all,
            args=(prompts, durations, seed, s.output_format,
                  constraints_json, s.bvh_standard_tpose, num_transition_frames),
            daemon=True,
        )
        self._thread.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _run_all(self, prompts, durations, seed, fmt, constraints_json, bvh_standard_tpose,
                 num_transition_frames=5):
        def progress_cb(msg):
            _generation_state["progress"] = msg

        success, result = sc.generate_motion_multi(
            prompts=prompts,
            durations=durations,
            seed=seed,
            output_format=fmt,
            constraints_json=constraints_json,
            bvh_standard_tpose=bvh_standard_tpose,
            num_transition_frames=num_transition_frames,
            progress_callback=progress_cb,
        )
        _generation_state["success"] = success
        _generation_state["result"]  = result
        _generation_state["done"]    = True
        _generation_state["running"] = False

    def modal(self, context, event):
        s = context.scene.kimodo
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        s.generation_progress = _generation_state.get("progress", "")
        _tag_timeline_redraw(context)
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        if not _generation_state["done"]:
            return {'RUNNING_MODAL'}

        context.window_manager.event_timer_remove(self._timer)
        s.is_generating = False
        s.generating_segment_index = -1

        if _generation_state["success"]:
            file_path = _generation_state["result"]

            # Mark all segments as generated and store the shared path
            for idx in self._segment_indices:
                seg = s.motion_segments[idx]
                seg.last_bvh_path = file_path
                seg.generated = True

            if os.path.splitext(file_path)[1].lower() == ".bvh":
                label = f"{len(self._segment_indices)}-prompt"
                bpy.ops.kimodo.import_bvh_at_frame(
                    'EXEC_DEFAULT',
                    filepath=file_path,
                    start_frame=self._start_frame,
                    label=label,
                )

            n = len(self._segment_indices)
            s.generation_progress = f"All {n} segments generated ✓"
            self.report({'INFO'}, f"Multi-prompt generation complete ({n} segments) ✓")
        else:
            s.generation_progress = "Failed ✗"
            self.report({'ERROR'}, f"Generation failed: {_generation_state['result']}")

        _tag_timeline_redraw(context)
        return {'FINISHED'}

    def cancel(self, context):
        context.window_manager.event_timer_remove(self._timer)
        context.scene.kimodo.is_generating = False
        context.scene.kimodo.generating_segment_index = -1
        _generation_state["running"] = False


# ---------------------------------------------------------------------------
# Frame-aware BVH import (offsets keyframes to start_frame)
# ---------------------------------------------------------------------------

class KIMODO_OT_ImportBVHAtFrame(Operator):
    """Import BVH and shift its keyframes to start at start_frame"""
    bl_idname = "kimodo.import_bvh_at_frame"
    bl_label  = "Import BVH at Frame"

    filepath:    StringProperty(subtype='FILE_PATH')
    start_frame: IntProperty(default=1)
    label:       StringProperty(default="")

    def execute(self, context):
        if not os.path.exists(self.filepath):
            self.report({'ERROR'}, f"File not found: {self.filepath}")
            return {'CANCELLED'}

        before = set(bpy.data.objects)

        bpy.ops.import_anim.bvh(
            filepath=self.filepath,
            axis_forward='-Z', axis_up='Y',
            target='ARMATURE',
            global_scale=0.01,
            frame_start=self.start_frame,
            use_fps_scale=False,
            update_scene_fps=False,
            update_scene_duration=False,
            use_cyclic=False,
            rotate_mode='NATIVE',
        )

        after   = set(bpy.data.objects)
        new_arm = next((o for o in after - before if o.type == 'ARMATURE'), None)

        if new_arm:
            name = f"Kimodo_{self.label}" if self.label else "Kimodo_Source"
            new_arm.name = name
            new_arm["kimodo_source"] = True
            new_arm["kimodo_creation_time"] = time.time()
            s = context.scene.kimodo
            new_arm = _apply_to_existing_source(s, new_arm)
            _scale_root_location_keyframes(new_arm, s.soma_scale_factor)
            s.source_armature = new_arm
            s.reuse_armature = new_arm

        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Helpers shared by segment operators
# ---------------------------------------------------------------------------

def _tag_timeline_redraw(context):
    """Force Timeline and DopeSheet areas to redraw."""
    for area in context.screen.areas:
        if area.type == 'DOPESHEET_EDITOR':
            area.tag_redraw()


def _build_segment_constraints(context, seg) -> str | None:
    """Build constraints JSON for a segment (reuses scene-level constraints for now)."""
    s = context.scene.kimodo
    enabled = [c for c in s.motion_constraints if c.enabled and c.marker_object]
    if not enabled:
        return None
    try:
        from . import constraints as cmod
        data = cmod.build_constraints_json(
            s.motion_constraints, context.scene,
            kimodo_fps=s.kimodo_fps,
            auto_canonicalize=s.auto_canonicalize,
        )
        import json
        return json.dumps(data)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Constraint authoring operators
# ---------------------------------------------------------------------------

_CONSTRAINT_COLORS_RGBA = {
    'root2d':      (0.9, 0.2, 0.2, 1.0),   # red
    'fullbody':    (0.9, 0.6, 0.1, 0.7),   # orange (semi-transparent — it's an armature)
    'left_hand':   (0.9, 0.9, 0.1, 1.0),   # yellow
    'right_hand':  (0.2, 0.8, 0.2, 1.0),   # green
    'left_foot':   (0.1, 0.7, 0.8, 1.0),   # cyan
    'right_foot':  (0.2, 0.3, 0.9, 1.0),   # blue
}


def _unique_name(base: str) -> str:
    """Return a name not already in bpy.data.objects."""
    existing = {o.name for o in bpy.data.objects}
    if base not in existing:
        return base
    i = 1
    while f"{base}_{i:02d}" in existing:
        i += 1
    return f"{base}_{i:02d}"


def _sample_curve_arc_length(curve_obj, n_samples, depsgraph):
    """Return n_samples evenly arc-length-spaced world-space positions along curve_obj."""
    evaluated = curve_obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    mw = curve_obj.matrix_world

    verts = [mw @ v.co for v in mesh.vertices]
    evaluated.to_mesh_clear()

    if len(verts) < 2:
        return verts

    # Build cumulative arc-length table
    lengths = [0.0]
    for i in range(len(verts) - 1):
        lengths.append(lengths[-1] + (verts[i + 1] - verts[i]).length)
    total = lengths[-1]

    if total == 0.0:
        return [verts[0]] * n_samples

    samples = []
    seg = 0
    for i in range(n_samples):
        t = (i / (n_samples - 1)) * total if n_samples > 1 else 0.0
        while seg < len(lengths) - 2 and lengths[seg + 1] < t:
            seg += 1
        seg_len = lengths[seg + 1] - lengths[seg]
        frac = (t - lengths[seg]) / seg_len if seg_len > 0 else 0.0
        samples.append(verts[seg].lerp(verts[seg + 1], frac))

    return samples


class KIMODO_OT_SampleCurveAsWaypoints(Operator):
    """Sample a curve into evenly-spaced Root XZ waypoint constraints"""
    bl_idname  = "kimodo.sample_curve_as_waypoints"
    bl_label   = "Sample Curve as Waypoints"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.kimodo

        if not s.path_curve:
            self.report({'ERROR'}, "Set a curve object in the Path Curve field.")
            return {'CANCELLED'}
        if s.path_start_frame >= s.path_end_frame:
            self.report({'ERROR'}, "Start Frame must be less than End Frame.")
            return {'CANCELLED'}

        n = s.path_waypoints
        depsgraph = context.evaluated_depsgraph_get()
        positions = _sample_curve_arc_length(s.path_curve, n, depsgraph)

        if not positions:
            self.report({'ERROR'}, "Could not sample any points from the curve.")
            return {'CANCELLED'}

        start_f = s.path_start_frame
        end_f   = s.path_end_frame
        color   = _CONSTRAINT_COLORS_RGBA['root2d']

        saved_active   = context.active_object
        saved_selected = list(context.selected_objects)
        bpy.ops.object.select_all(action='DESELECT')

        for i, pos in enumerate(positions):
            frame = round(start_f + (end_f - start_f) * i / (n - 1)) if n > 1 else start_f

            bpy.ops.object.empty_add(type='ARROWS', location=pos)
            empty = context.active_object
            empty.name = _unique_name(f"Kimodo_Path_{i + 1:02d}")
            empty.color = color
            empty.show_name = True
            empty.empty_display_size = 0.15
            empty["kimodo_constraint"] = True
            empty["kimodo_type"]       = 'root2d'

            item = s.motion_constraints.add()
            item.constraint_type = 'root2d'
            item.frame           = frame
            item.marker_object   = empty
            item.enabled         = True
            item.label           = empty.name

        s.constraint_index = len(s.motion_constraints) - 1

        bpy.ops.object.select_all(action='DESELECT')
        for o in saved_selected:
            o.select_set(True)
        if saved_active:
            context.view_layer.objects.active = saved_active

        self.report({'INFO'}, f"Added {n} path waypoints (frames {start_f}–{end_f})")
        return {'FINISHED'}


class KIMODO_OT_AddConstraint(Operator):
    """Add a Kimodo motion constraint marker at the 3D cursor"""
    bl_idname = "kimodo.add_constraint"
    bl_label = "Add Constraint Marker"

    constraint_type: bpy.props.StringProperty(default='root2d')

    def execute(self, context):
        s = context.scene.kimodo
        ctype = self.constraint_type
        cur_frame = context.scene.frame_current
        active = context.active_object

        # ---------------------------------------------------------------
        # fullbody needs an armature, not an Empty.
        # Priority: (1) active object is an armature → use it directly
        #           (2) source_armature exists → duplicate it as a reference
        #           (3) no armature available → create Empty but warn loudly
        # ---------------------------------------------------------------
        if ctype == 'fullbody':
            marker_obj = self._resolve_fullbody_armature(context, s, cur_frame)
            if marker_obj is None:
                return {'CANCELLED'}
        else:
            marker_obj = self._create_empty(context, ctype, cur_frame)

        item = s.motion_constraints.add()
        item.constraint_type = ctype
        item.frame = cur_frame
        item.marker_object = marker_obj
        item.enabled = True
        item.label = marker_obj.name
        s.constraint_index = len(s.motion_constraints) - 1

        self.report({'INFO'}, f"Added {ctype} constraint at frame {cur_frame} → '{marker_obj.name}'")
        return {'FINISHED'}

    # ------------------------------------------------------------------

    def _resolve_fullbody_armature(self, context, s, cur_frame):
        """
        Return the armature object to use for a fullbody constraint.
        Tries: active selection → source_armature duplicate → error.
        """
        active = context.active_object

        # Case 1: user has an armature selected — use it as-is
        if active and active.type == 'ARMATURE':
            self.report({'INFO'},
                f"Using selected armature '{active.name}' as full-body pose reference. "
                f"Pose it at frame {cur_frame} to define the keyframe.")
            return active

        # Case 2: source_armature exists → duplicate it for posing
        if s.source_armature:
            return self._duplicate_source_for_posing(context, s, cur_frame)

        # Case 3: nothing to work with
        self.report({'ERROR'},
            "Full-Body constraint needs an armature. "
            "Either: (a) select an armature first, or "
            "(b) generate a motion first so a source armature exists to duplicate.")
        return None

    def _duplicate_source_for_posing(self, context, s, cur_frame):
        """Duplicate source_armature and freeze its pose at cur_frame.

        Plain duplication inherits the source's BVH F-curves, so any bone the
        user rotates without explicitly keyframing gets reverted to the
        animated value the next time the depsgraph evaluates the frame —
        including during build_constraints_json, which silently undoes the
        user's posing.

        Fix: evaluate the source at cur_frame, copy that pose into the
        duplicate, then strip the duplicate's animation_data so its bone
        properties stick until the user changes them.
        """
        scene = context.scene
        saved_frame = scene.frame_current
        scene.frame_set(cur_frame)
        context.view_layer.update()

        bpy.ops.object.select_all(action='DESELECT')
        s.source_armature.select_set(True)
        context.view_layer.objects.active = s.source_armature
        bpy.ops.object.duplicate(linked=False)
        dup = context.active_object

        # Make sure the duplicate's pose reflects the source's frame-N state
        # before we strip animation data.
        context.view_layer.update()

        # Snapshot every pose bone's transform so we can re-apply after the
        # animation_data wipe (clearing the action can otherwise reset values).
        pose_snapshot = {}
        for pb in dup.pose.bones:
            pose_snapshot[pb.name] = (
                pb.rotation_mode,
                pb.location.copy(),
                pb.rotation_quaternion.copy(),
                pb.rotation_euler.copy(),
                tuple(pb.rotation_axis_angle),
                pb.scale.copy(),
            )

        # Strip both the object-level action (root motion / object xform) and
        # any data-level action (rare for BVH but possible). After this, the
        # bone properties are no longer overwritten on frame change.
        if dup.animation_data:
            dup.animation_data_clear()
        if dup.data.animation_data:
            dup.data.animation_data_clear()

        # Re-apply the snapshot to lock the pose in place.
        for pb in dup.pose.bones:
            snap = pose_snapshot.get(pb.name)
            if snap is None:
                continue
            mode, loc, qrot, erot, aarot, sc = snap
            pb.rotation_mode = mode
            pb.location = loc
            pb.rotation_quaternion = qrot
            pb.rotation_euler = erot
            pb.rotation_axis_angle = aarot
            pb.scale = sc

        name = _unique_name(f"Kimodo_PoseRef_{cur_frame:04d}")
        dup.name = name
        dup.data.name = name + "_data"
        # Visually distinguish — semi-transparent orange tint.
        dup.color = (0.9, 0.5, 0.1, 0.7)
        dup["kimodo_constraint"] = True
        dup["kimodo_type"] = 'fullbody'
        dup.show_name = True

        scene.frame_set(saved_frame)
        context.view_layer.update()

        self.report({'INFO'},
            f"Duplicated source armature as '{name}', frozen at frame {cur_frame}. "
            f"Enter Pose Mode and tweak — changes will stick (no F-curves to fight).")
        return dup

    def _create_empty(self, context, ctype, cur_frame):
        """Create a colour-coded Empty for non-fullbody constraint types."""
        empty_types = {
            'root2d':      'ARROWS',
            'left_hand':   'CUBE',
            'right_hand':  'CUBE',
            'left_foot':   'PLAIN_AXES',
            'right_foot':  'PLAIN_AXES',
        }
        bpy.ops.object.empty_add(
            type=empty_types.get(ctype, 'ARROWS'),
            location=context.scene.cursor.location,
        )
        empty = context.active_object

        type_labels = {
            'root2d':     'Kimodo_Waypoint',
            'left_hand':  'Kimodo_LHand',
            'right_hand': 'Kimodo_RHand',
            'left_foot':  'Kimodo_LFoot',
            'right_foot': 'Kimodo_RFoot',
        }
        base_name = type_labels.get(ctype, 'Kimodo_Marker')
        name = _unique_name(f"{base_name}_{cur_frame:04d}")
        empty.name = name
        empty.color = _CONSTRAINT_COLORS_RGBA.get(ctype, (1, 1, 1, 1))
        empty.show_name = True
        empty.empty_display_size = 0.15
        empty["kimodo_constraint"] = True
        empty["kimodo_type"] = ctype
        return empty


class KIMODO_OT_LinkExistingAsConstraint(Operator):
    """Link the currently selected object as a Kimodo constraint marker"""
    bl_idname = "kimodo.link_as_constraint"
    bl_label = "Link Selected as Constraint"

    constraint_type: bpy.props.StringProperty(default='root2d')

    def execute(self, context):
        s = context.scene.kimodo
        obj = context.active_object
        if not obj:
            self.report({'ERROR'}, "Select an object first.")
            return {'CANCELLED'}

        ctype = self.constraint_type
        item = s.motion_constraints.add()
        item.constraint_type = ctype
        item.frame = context.scene.frame_current
        item.marker_object = obj
        item.enabled = True
        item.label = obj.name
        s.constraint_index = len(s.motion_constraints) - 1

        self.report({'INFO'}, f"Linked '{obj.name}' as {ctype} constraint at frame {item.frame}")
        return {'FINISHED'}


class KIMODO_OT_RemoveConstraint(Operator):
    """Remove the selected constraint from the list (optionally delete the marker)"""
    bl_idname = "kimodo.remove_constraint"
    bl_label = "Remove Constraint"

    delete_object: BoolProperty(
        name="Delete Marker Object",
        description="Also delete the Empty/object from the scene",
        default=False,
    )

    def execute(self, context):
        s = context.scene.kimodo
        idx = s.constraint_index
        if not (0 <= idx < len(s.motion_constraints)):
            return {'CANCELLED'}

        item = s.motion_constraints[idx]
        if self.delete_object and item.marker_object:
            bpy.data.objects.remove(item.marker_object, do_unlink=True)

        s.motion_constraints.remove(idx)
        s.constraint_index = max(0, idx - 1)
        return {'FINISHED'}


class KIMODO_OT_GotoConstraintFrame(Operator):
    """Jump the timeline to this constraint's frame"""
    bl_idname = "kimodo.goto_constraint_frame"
    bl_label = "Go to Frame"

    frame: bpy.props.IntProperty()

    def execute(self, context):
        context.scene.frame_set(self.frame)
        return {'FINISHED'}


class KIMODO_OT_SelectConstraintObject(Operator):
    """Select and focus the constraint's marker object in the viewport"""
    bl_idname = "kimodo.select_constraint_object"
    bl_label = "Select Marker"

    index: bpy.props.IntProperty()

    def execute(self, context):
        s = context.scene.kimodo
        if not (0 <= self.index < len(s.motion_constraints)):
            return {'CANCELLED'}
        obj = s.motion_constraints[self.index].marker_object
        if not obj:
            self.report({'WARNING'}, "No object linked to this constraint.")
            return {'CANCELLED'}
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj
        # Frame the selection
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                region = next((r for r in area.regions if r.type == 'WINDOW'), None)
                if region:
                    with context.temp_override(area=area, region=region):
                        bpy.ops.view3d.view_selected()
                break
        return {'FINISHED'}


class KIMODO_OT_PreviewConstraintsJSON(Operator):
    """Build and show the constraints JSON in a Blender Text Editor block"""
    bl_idname = "kimodo.preview_constraints_json"
    bl_label = "Preview Constraints JSON"

    def execute(self, context):
        s = context.scene.kimodo
        try:
            json_str = cmod.constraints_to_json_string(
                s.motion_constraints,
                context.scene,
                kimodo_fps=s.kimodo_fps,
                auto_canonicalize=s.auto_canonicalize,
            )
        except Exception as e:
            self.report({'ERROR'}, f"Failed to build JSON: {e}")
            return {'CANCELLED'}

        s.constraint_json_preview = json_str

        # Write to a text block in the Text Editor
        block_name = "kimodo_constraints.json"
        if block_name in bpy.data.texts:
            bpy.data.texts.remove(bpy.data.texts[block_name])
        text_block = bpy.data.texts.new(block_name)
        text_block.write(json_str)

        self.report({'INFO'}, f"Constraints JSON written to Text Editor: '{block_name}'")
        return {'FINISHED'}


class KIMODO_OT_ClearConstraints(Operator):
    """Remove all Kimodo constraints from the list"""
    bl_idname = "kimodo.clear_constraints"
    bl_label = "Clear All Constraints"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        context.scene.kimodo.motion_constraints.clear()
        self.report({'INFO'}, "All constraints cleared.")
        return {'FINISHED'}


class KIMODO_OT_SetTo30FPS(Operator):
    """Set the scene frame rate to 30 FPS for Kimodo compatibility"""
    bl_idname = "kimodo.set_to_30fps"
    bl_label = "Set Scene to 30 FPS"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        context.scene.render.fps = 30
        context.scene.render.fps_base = 1.0
        self.report({'INFO'}, "Scene FPS set to 30.")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Armature picker utility
# ---------------------------------------------------------------------------

class KIMODO_OT_PickLatestKimodoArmature(Operator):
    """Auto-select the most recently generated Kimodo armature as the reuse target"""
    bl_idname = "kimodo.pick_latest_armature"
    bl_label  = "Pick Latest Generated Armature"

    def execute(self, context):
        s = context.scene.kimodo
        candidates = [
            obj for obj in context.scene.objects
            if obj.type == 'ARMATURE' and obj.get("kimodo_source")
        ]
        if not candidates:
            self.report({'WARNING'}, "No Kimodo-generated armatures found in scene")
            return {'CANCELLED'}

        latest = max(candidates, key=lambda obj: obj.get("kimodo_creation_time", 0.0))
        s.reuse_armature = latest
        self.report({'INFO'}, f"Reuse target set to '{latest.name}'")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Generation history operators
# ---------------------------------------------------------------------------

class KIMODO_OT_ReimportFromHistory(Operator):
    """Re-import the BVH file from a history entry"""
    bl_idname = "kimodo.reimport_from_history"
    bl_label  = "Re-import BVH"
    bl_options = {'REGISTER', 'UNDO'}

    index: IntProperty(default=0)

    def execute(self, context):
        s = context.scene.kimodo
        if not (0 <= self.index < len(s.generation_history)):
            self.report({'ERROR'}, "Invalid history index.")
            return {'CANCELLED'}
        entry = s.generation_history[self.index]
        if not os.path.isfile(entry.bvh_path):
            self.report({'ERROR'}, f"BVH file no longer exists: {entry.bvh_path}")
            return {'CANCELLED'}
        bpy.ops.kimodo.import_bvh('EXEC_DEFAULT', filepath=entry.bvh_path)
        self.report({'INFO'}, f"Re-imported '{os.path.basename(entry.bvh_path)}'")
        return {'FINISHED'}


class KIMODO_OT_ClearHistory(Operator):
    """Clear all generation history entries"""
    bl_idname = "kimodo.clear_history"
    bl_label  = "Clear History"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        context.scene.kimodo.generation_history.clear()
        self.report({'INFO'}, "Generation history cleared.")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Generate N Variations operator
# ---------------------------------------------------------------------------

class KIMODO_OT_GenerateVariations(Operator):
    """Generate N variations of the current prompt with different random seeds"""
    bl_idname = "kimodo.generate_variations"
    bl_label  = "Generate Variations"

    _timer       = None
    _thread      = None
    _seeds:  list = []
    _total:  int  = 0
    _current_idx: int = 0

    def invoke(self, context, event):
        s = context.scene.kimodo

        if not sc.is_running():
            self.report({'WARNING'}, "Kimodo is not running — click 'Start Kimodo' first.")
            return {'CANCELLED'}
        if s.is_generating:
            self.report({'WARNING'}, "Already generating — please wait.")
            return {'CANCELLED'}

        self._total       = s.num_variations
        self._seeds       = [random.randint(0, 2**31) for _ in range(self._total)]
        self._current_idx = 0

        s.is_generating = True
        _reset_state()
        _generation_state["running"] = True
        self._start_next(context, s)

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _start_next(self, context, s):
        var_num = self._current_idx + 1
        seed    = self._seeds[self._current_idx]
        s.generation_progress = f"Variation {var_num}/{self._total}…"

        _reset_state()
        _generation_state["running"] = True

        self._thread = threading.Thread(
            target=self._run_one,
            args=(s.prompt, s.duration, seed, s.output_format, s.bvh_standard_tpose),
            daemon=True,
        )
        self._thread.start()

    def _run_one(self, prompt, duration, seed, fmt, bvh_standard_tpose):
        def progress_cb(msg):
            _generation_state["progress"] = msg

        success, result = sc.generate_motion(
            prompt=prompt,
            duration=duration,
            seed=seed,
            output_format=fmt,
            constraints_json=None,
            bvh_standard_tpose=bvh_standard_tpose,
            progress_callback=progress_cb,
        )
        _generation_state["success"] = success
        _generation_state["result"]  = result
        _generation_state["done"]    = True
        _generation_state["running"] = False

    def modal(self, context, event):
        s = context.scene.kimodo

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        var_num = self._current_idx + 1
        s.generation_progress = (
            f"Variation {var_num}/{self._total}: "
            + _generation_state.get("progress", "")
        )
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        if not _generation_state["done"]:
            return {'RUNNING_MODAL'}

        # One variation finished
        seed      = self._seeds[self._current_idx]
        var_num   = self._current_idx + 1

        if not _generation_state["success"]:
            context.window_manager.event_timer_remove(self._timer)
            s.is_generating = False
            s.generation_progress = f"Variation {var_num} failed ✗"
            self.report({'ERROR'}, f"Variation {var_num} failed: {_generation_state['result']}")
            return {'FINISHED'}

        file_path = _generation_state["result"]
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".bvh":
            before = set(bpy.data.objects)
            bpy.ops.import_anim.bvh(
                filepath=file_path,
                axis_forward='-Z', axis_up='Y',
                target='ARMATURE',
                global_scale=0.01,
                frame_start=1,
                use_fps_scale=False,
                update_scene_fps=False,
                update_scene_duration=True,
                use_cyclic=False,
                rotate_mode='NATIVE',
            )
            after   = set(bpy.data.objects)
            new_arm = next((o for o in after - before if o.type == 'ARMATURE'), None)
            if new_arm:
                new_arm.name = f"Kimodo_Var_{var_num}"
                new_arm["kimodo_source"]        = True
                new_arm["kimodo_creation_time"] = time.time()

        _push_history(s, s.prompt, seed, s.duration, file_path)

        self._current_idx += 1
        if self._current_idx < self._total:
            self._start_next(context, s)
            return {'RUNNING_MODAL'}

        # All done
        context.window_manager.event_timer_remove(self._timer)
        s.is_generating = False
        s.generation_progress = f"All {self._total} variations done ✓"
        self.report({'INFO'}, f"Generated {self._total} variations ✓")
        return {'FINISHED'}

    def cancel(self, context):
        context.window_manager.event_timer_remove(self._timer)
        context.scene.kimodo.is_generating = False
        context.scene.kimodo.generation_progress = "Cancelled"
        _generation_state["running"] = False


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = [
    KIMODO_OT_StartKimodo,
    KIMODO_OT_StopKimodo,
    KIMODO_OT_Generate,
    KIMODO_OT_CancelGeneration,
    KIMODO_OT_ImportBVH,
    KIMODO_OT_ImportBVHAtFrame,
    KIMODO_OT_PickLatestKimodoArmature,
    KIMODO_OT_AutoMapBones,
    KIMODO_OT_AddBoneMapping,
    KIMODO_OT_RemoveBoneMapping,
    KIMODO_OT_ApplyRetargeting,
    KIMODO_OT_RemoveRetargeting,
    KIMODO_OT_BakeRetargeting,
    KIMODO_OT_SavePreset,
    KIMODO_OT_LoadPreset,
    KIMODO_OT_DeletePreset,
    KIMODO_OT_ExportPresetFile,
    KIMODO_OT_ImportPresetFile,
    # Segment operators
    KIMODO_OT_SelectSegment,
    KIMODO_OT_RemoveSegmentByIndex,
    KIMODO_OT_AddSegment,
    KIMODO_OT_RemoveSegment,
    KIMODO_OT_DuplicateSegment,
    KIMODO_OT_MoveSegmentUp,
    KIMODO_OT_MoveSegmentDown,
    KIMODO_OT_SyncSeeds,
    KIMODO_OT_GenerateSegment,
    KIMODO_OT_GenerateAllSegments,
    # History operators
    KIMODO_OT_ReimportFromHistory,
    KIMODO_OT_ClearHistory,
    KIMODO_OT_GenerateVariations,
    # Curve path operator
    KIMODO_OT_SampleCurveAsWaypoints,
    # Constraint operators
    KIMODO_OT_AddConstraint,
    KIMODO_OT_LinkExistingAsConstraint,
    KIMODO_OT_RemoveConstraint,
    KIMODO_OT_GotoConstraintFrame,
    KIMODO_OT_SelectConstraintObject,
    KIMODO_OT_PreviewConstraintsJSON,
    KIMODO_OT_ClearConstraints,
    KIMODO_OT_SetTo30FPS,
]


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
