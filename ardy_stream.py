"""Live ARDY streaming: generate motion while the timeline plays.

Press Start Stream, hit Play, and drag the target object around — the character
walks toward wherever it currently is. ARDY is autoregressive, so this is not a
trick: the bridge holds the motion state and each step regenerates the near
future against the target's *current* position. Moving the target rewrites what
has not been played yet, which is what makes it feel live.

How it fits together
--------------------
    modal timer (main thread)          worker thread            ARDY venv
    -------------------------          -------------            ---------
    playhead advances
    buffer running low?  ---------->   sc.stream_step()  ---->   one window
    keyframe the window  <----------   frames message    <----   of motion

Generation never runs on Blender's main thread — a diffusion pass would freeze
the viewport — so the timer only ever starts a request or drains a finished one,
following the same modal + background-thread pattern as
``KIMODO_OT_Generate`` in operators.py.

The result is an ordinary Action on an ordinary armature: when the stream ends
the keyframes are simply there, and the Retarget panel works on them unchanged.
"""

import threading
import time

import bpy
from bpy.types import Operator

from . import subprocess_client as sc
from . import ardy_steer


# ---------------------------------------------------------------------------
# Cross-thread state
# ---------------------------------------------------------------------------
# Same idiom as operators.py::_generation_state: the worker only ever writes
# these, the modal timer only ever reads them, and a single in-flight request is
# enforced by the client's own _busy flag.

_state = {
    "running": False,       # a stream_step is in flight
    "done": False,          # ...and it has finished
    "ok": False,
    "result": None,         # frames message, or an error string
    "step_seconds": 0.0,    # how long the last step took, for the UI
}


def _reset_state():
    _state.update(running=False, done=False, ok=False, result=None)


def _worker(frame_idx, target_xz, target_heading, prompt, timeout):
    started = time.monotonic()
    ok, result = sc.stream_step(
        frame_idx=frame_idx,
        target_xz=target_xz,
        target_heading=target_heading,
        prompt=prompt,
        timeout=timeout,
    )
    _state["step_seconds"] = time.monotonic() - started
    _state["ok"] = ok
    _state["result"] = result
    _state["done"] = True
    _state["running"] = False


# ---------------------------------------------------------------------------
# Armature
# ---------------------------------------------------------------------------

STREAM_ARMATURE_NAME = "ARDY_Stream"

# Fallback bone length for a joint with no children (hands, toes, head).
_LEAF_BONE_LENGTH = 0.08


def _build_armature(context, rest: dict):
    """Create the armature the stream animates, from the bridge's rest tables.

    Every bone is built pointing along +Y with zero roll, which makes each
    bone's rest rotation the identity. That matters: with an identity rest, a
    pose bone's quaternion *is* ARDY's local joint rotation, so applying a frame
    needs no per-bone change of basis. An oriented rest pose (bones aimed at
    their children, as a BVH import produces) would need every rotation
    conjugated by ``bone.matrix_local``, and a mistake there shows up as a
    subtly mangled rig rather than an exception — not a trade worth making for
    a rig whose job is to drive a retarget.

    The cost is cosmetic: the armature reads as parallel sticks rather than a
    figure. Bone lengths still follow the distance to the first child, so the
    proportions are recognisable.
    """
    names = rest["bone_names"]
    parents = rest["parents"]
    offsets = rest["rest_offsets"]
    root_idx = int(rest.get("root_idx", 0))

    # Rest joint positions, accumulated down the hierarchy, in Blender space.
    rest_pos = [None] * len(names)
    for i, parent in enumerate(parents):
        local = ardy_steer.ardy_to_blender_pos(offsets[i])
        if parent < 0:
            rest_pos[i] = local
        else:
            base = rest_pos[parent]
            rest_pos[i] = (base[0] + local[0], base[1] + local[1], base[2] + local[2])

    first_child = {}
    for i, parent in enumerate(parents):
        if parent >= 0 and parent not in first_child:
            first_child[parent] = i

    existing = bpy.data.objects.get(STREAM_ARMATURE_NAME)
    if existing is not None and existing.type == 'ARMATURE':
        bpy.data.objects.remove(existing, do_unlink=True)

    arm_data = bpy.data.armatures.new(STREAM_ARMATURE_NAME)
    arm_obj = bpy.data.objects.new(STREAM_ARMATURE_NAME, arm_data)
    context.scene.collection.objects.link(arm_obj)

    # Whatever the user was doing (posing another rig, editing a mesh), edit
    # bones can only be added from object mode on this armature.
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        for i, name in enumerate(names):
            bone = arm_data.edit_bones.new(name)
            head = rest_pos[i]
            child = first_child.get(i)
            if child is not None:
                c = rest_pos[child]
                length = max(
                    _LEAF_BONE_LENGTH,
                    ((c[0] - head[0]) ** 2 + (c[1] - head[1]) ** 2
                     + (c[2] - head[2]) ** 2) ** 0.5,
                )
            else:
                length = _LEAF_BONE_LENGTH
            bone.head = head
            bone.tail = (head[0], head[1] + length, head[2])
            bone.roll = 0.0
        for i, name in enumerate(names):
            if parents[i] >= 0:
                arm_data.edit_bones[name].parent = arm_data.edit_bones[names[parents[i]]]
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')

    for pb in arm_obj.pose.bones:
        pb.rotation_mode = 'QUATERNION'

    # Tag it the way the rest of the add-on recognises a source rig, so the
    # Retarget panel accepts the result without any special-casing.
    arm_obj["kimodo_source"] = True
    arm_obj["ardy_stream"] = True
    arm_obj["kimodo_creation_time"] = time.time()

    return arm_obj, names, root_idx, rest_pos[root_idx]


def action_fcurves(action, slot=None):
    """The F-curve collection of an Action, across Blender's three layouts.

    Blender 4.4 introduced slotted Actions and kept ``Action.fcurves`` as a
    shim; 5.0 removed it outright, and reading it there raises
    ``'Action' object has no attribute 'fcurves'``. On 5.x the curves live one
    level down, per slot:

        action.layers[] -> .strips[] -> .channelbag(slot) -> .fcurves

    Returns None when the action has no curve container yet, so callers can
    tell "no curves" apart from "wrong Blender version".
    """
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:                      # <= 4.3, and 4.4's shim
        return legacy

    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            bag = None
            getter = getattr(strip, "channelbag", None)
            if getter is not None and slot is not None:
                try:
                    bag = getter(slot, ensure=True)
                except TypeError:               # older signature, no ensure=
                    bag = getter(slot)
                except Exception:
                    bag = None
            if bag is None:
                bags = getattr(strip, "channelbags", None)
                if bags:
                    bag = bags[0]
            if bag is not None:
                return bag.fcurves
    return None


def _prepare_action(arm_obj, names, root_name, scaffold_frame: int):
    """Give every animated channel an f-curve, and return them indexed.

    Blender is left to create the action, its slot and the curves via one
    ``keyframe_insert`` per channel: the slotted-Action layout differs enough
    between 4.3, 4.4 and 5.x that letting Blender do the setup is far more
    robust than building layers by hand. After this, writes go straight to the
    curves.

    The scaffolding keys go on *scaffold_frame* — the frame the stream starts
    at — rather than being deleted here. Emptying every curve risks Blender
    collecting the now-keyless F-curves, which would leave the returned dict
    holding dangling references; instead the first window's own truncation
    overwrites them, since it starts at exactly this frame.
    """
    arm_obj.animation_data_create()
    for name in names:
        pb = arm_obj.pose.bones[name]
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pb.keyframe_insert(data_path="rotation_quaternion", frame=scaffold_frame)
    root = arm_obj.pose.bones[root_name]
    root.location = (0.0, 0.0, 0.0)
    root.keyframe_insert(data_path="location", frame=scaffold_frame)

    action = arm_obj.animation_data.action

    # Bind before collecting: on 5.x the curves are looked up per slot, so the
    # slot has to be settled first or we would index the wrong channelbag.
    try:
        from .operators import _bind_action_with_slot
        _bind_action_with_slot(arm_obj, action)
    except Exception:
        # Pre-4.4 Blender has no slots; the plain assignment already stands.
        pass

    slot = getattr(arm_obj.animation_data, "action_slot", None)
    fcurves = action_fcurves(action, slot)
    if fcurves is None:
        raise RuntimeError(
            "Blender created no F-curves for the stream armature "
            f"(Action {action.name!r}). This is a Blender-version difference in "
            "how Actions store curves — please report the Blender version."
        )

    curves = {}
    for fc in fcurves:
        curves[(fc.data_path, fc.array_index)] = fc

    return action, curves


def _truncate_from(fcurve, frame: float):
    """Drop every key at or after *frame* — the replan discarding stale future."""
    pts = fcurve.keyframe_points
    i = len(pts) - 1
    removed = False
    while i >= 0 and pts[i].co[0] >= frame:
        pts.remove(pts[i], fast=True)
        removed = True
        i -= 1
    return removed


def _write_channel(fcurve, start_frame: int, values):
    """Replace this curve's tail with *values*, one per consecutive frame."""
    _truncate_from(fcurve, start_frame)
    pts = fcurve.keyframe_points
    base = len(pts)
    pts.add(len(values))
    for offset, value in enumerate(values):
        kp = pts[base + offset]
        kp.co = (start_frame + offset, value)
        kp.interpolation = 'LINEAR'
    fcurve.update()


def _apply_window(curves, names, root_name, msg, scene_start_frame: int,
                  root_rest):
    """Keyframe one window of frames onto the armature.

    Writes go through the f-curves rather than ``keyframe_insert`` per bone per
    frame: at a joint count times four channels times a window every fraction of
    a second, per-key insertion is the one part of this loop that would not keep
    up with playback.
    """
    rows = msg["local_rot_aa"]
    roots = msg["root_positions"]
    start = scene_start_frame + int(msg["start_index"])
    if not rows:
        return start

    joints = len(names)
    quats = [[ardy_steer.ardy_axis_angle_to_blender_quat(row[j * 3:j * 3 + 3])
              for row in rows] for j in range(joints)]

    for j, name in enumerate(names):
        path = f'pose.bones["{name}"].rotation_quaternion'
        for axis in range(4):
            fc = curves.get((path, axis))
            if fc is not None:
                _write_channel(fc, start, [q[axis] for q in quats[j]])

    # A pose bone's location is an offset from its rest head, not a world
    # position — feeding ARDY's absolute root straight in would lift the whole
    # character by the height of its own rest hips.
    root_path = f'pose.bones["{root_name}"].location'
    positions = [ardy_steer.ardy_to_blender_pos(p) for p in roots]
    for axis in range(3):
        fc = curves.get((root_path, axis))
        if fc is not None:
            _write_channel(fc, start,
                           [p[axis] - root_rest[axis] for p in positions])

    return start + len(rows)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class KIMODO_OT_ArdyStreamStop(Operator):
    """Stop the running live stream"""
    bl_idname = "kimodo.ardy_stream_stop"
    bl_label = "Stop Stream"

    def execute(self, context):
        # The modal operator owns the teardown; this only asks it to finish, so
        # the bridge is closed from the same thread that opened it.
        context.scene.kimodo.is_streaming = False
        return {'FINISHED'}


class KIMODO_OT_ArdyStream(Operator):
    """Generate motion live while the timeline plays, following a target object"""
    bl_idname = "kimodo.ardy_stream"
    bl_label = "Start Live Stream"

    _timer = None
    _thread = None

    # -- lifecycle ----------------------------------------------------------
    def invoke(self, context, event):
        s = context.scene.kimodo
        if s.backend != 'ARDY':
            self.report({'ERROR'}, "Live streaming needs the ARDY backend.")
            return {'CANCELLED'}
        if not sc.is_running():
            self.report({'ERROR'}, "ARDY is not running — click 'Start ARDY' first.")
            return {'CANCELLED'}
        if s.is_streaming or s.is_generating:
            self.report({'WARNING'}, "Already busy — stop the current job first.")
            return {'CANCELLED'}

        import random
        seed = random.randint(0, 2 ** 31 - 1) if s.seed_mode == 'RANDOM' else s.seed

        ok, result = sc.stream_begin(
            prompt=s.prompt,
            seed=seed,
            diffusion_steps=s.ardy_stream_diffusion_steps,
            replan_buffer=s.ardy_stream_replan_buffer,
            cfg_text_weight=s.ardy_cfg_text_weight,
            cfg_constraint_weight=s.ardy_cfg_constraint_weight,
            history_frames=s.ardy_history_frames,
            max_speed=s.ardy_stream_max_speed,
        )
        if not ok:
            self.report({'ERROR'}, f"Could not start the stream: {result}")
            return {'CANCELLED'}

        self._rest = result
        self._fps = float(result.get("fps", 20.0))
        self._horizon = int(result.get("gen_horizon_len", 8))

        start_frame = context.scene.frame_current
        try:
            arm, names, root_idx, root_rest = _build_armature(context, result)
            root_name = names[root_idx] if root_idx < len(names) else names[0]
            self._action, self._curves = _prepare_action(
                arm, names, root_name, start_frame)
        except Exception as exc:
            sc.stream_end()
            self.report({'ERROR'}, f"Could not build the stream armature: {exc}")
            return {'CANCELLED'}

        self._arm = arm
        self._names = names
        self._root_name = root_name
        self._root_rest = root_rest
        self._start_frame = start_frame
        self._generated_until = self._start_frame     # exclusive
        self._last_prompt = s.prompt

        s.source_armature = arm
        s.is_streaming = True
        s.stream_status = "Starting…"
        _reset_state()

        if s.ardy_stream_autoplay and not context.screen.is_animation_playing:
            bpy.ops.screen.animation_play()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    # -- the loop -----------------------------------------------------------
    def modal(self, context, event):
        s = context.scene.kimodo

        if event.type in {'ESC'} or not s.is_streaming:
            return self._finish(context, "Stopped")

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        if not sc.is_running():
            return self._finish(context, "ARDY stopped")

        if _state["done"]:
            _state["done"] = False
            if not _state["ok"]:
                return self._finish(context, f"Failed: {_state['result']}")
            try:
                self._generated_until = _apply_window(
                    self._curves, self._names, self._root_name,
                    _state["result"], self._start_frame, self._root_rest,
                )
            except Exception as exc:
                return self._finish(context, f"Could not apply frames: {exc}")
            if context.scene.frame_end < self._generated_until:
                context.scene.frame_end = self._generated_until

        self._pump(context, s)
        self._redraw(context)
        return {'RUNNING_MODAL'}

    def _pump(self, context, s):
        """Start the next step when the buffer is running low."""
        playhead = context.scene.frame_current
        ahead = self._generated_until - playhead

        # Backpressure: the GPU is behind the playhead. Pausing is the only
        # honest option — letting playback run into un-keyframed frames would
        # freeze the character mid-stride and look like a bug.
        if ahead <= 0 and context.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
            s.stream_status = "Waiting for ARDY…"
        elif (s.ardy_stream_autoplay and ahead > self._horizon
                and not context.screen.is_animation_playing and s.is_streaming):
            bpy.ops.screen.animation_play()

        if _state["running"]:
            return
        if ahead >= s.ardy_stream_lead_frames:
            s.stream_status = self._status_line(ahead)
            return

        target_xz, heading = self._target(context, s)
        prompt = s.prompt if s.prompt != self._last_prompt else None
        self._last_prompt = s.prompt

        _state["running"] = True
        self._thread = threading.Thread(
            target=_worker,
            args=(max(0, playhead - self._start_frame), target_xz, heading,
                  prompt, 120.0),
            daemon=True,
        )
        self._thread.start()
        s.stream_status = self._status_line(ahead)

    def _target(self, context, s):
        """The follow target's ground position in ARDY space, or (None, None)."""
        obj = s.ardy_stream_target
        if obj is None:
            return None, None
        from . import constraints as cmod
        loc = obj.matrix_world.translation
        return list(cmod.blender_to_kimodo_2d(loc)), None

    def _status_line(self, ahead):
        step = _state.get("step_seconds", 0.0)
        budget = self._horizon / self._fps if self._fps else 0.0
        note = "" if step <= budget or not step else "  (GPU behind)"
        return f"Streaming — {max(0, ahead)} frames buffered, {step:.2f}s/step{note}"

    def _redraw(self, context):
        for area in context.screen.areas:
            if area.type in {'VIEW_3D', 'DOPESHEET_EDITOR', 'TIMELINE'}:
                area.tag_redraw()

    # -- teardown -----------------------------------------------------------
    def _finish(self, context, message: str):
        s = context.scene.kimodo
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if context.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)

        # A step may still be in flight. The client refuses a new request
        # while the pipe is busy, so wait briefly for it to land — otherwise
        # stream_end never reaches the bridge and it holds the motion tensor on
        # the GPU until the next stream_begin replaces it.
        deadline = time.monotonic() + 5.0
        while _state["running"] and time.monotonic() < deadline:
            time.sleep(0.05)
        try:
            sc.stream_end()
        except Exception:
            pass

        s.is_streaming = False
        s.stream_status = message
        frames = max(0, self._generated_until - self._start_frame)
        self.report({'INFO'}, f"{message} — {frames} frames on '{self._arm.name}'")
        return {'FINISHED'}

    def cancel(self, context):
        self._finish(context, "Cancelled")


CLASSES = (KIMODO_OT_ArdyStream, KIMODO_OT_ArdyStreamStop)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
