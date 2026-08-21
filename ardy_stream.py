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


def _worker(frame_idx, target_xz, target_heading, target_velocity, prompt,
            replan_buffer, max_speed, timeout):
    started = time.monotonic()
    ok, result = sc.stream_step(
        frame_idx=frame_idx,
        target_xz=target_xz,
        target_heading=target_heading,
        target_velocity=target_velocity,
        prompt=prompt,
        replan_buffer=replan_buffer,
        max_speed=max_speed,
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

# Blender's event names for the four arrows, in the demo's throttle/steer roles.
_ARROW_KEYS = {'UP_ARROW', 'DOWN_ARROW', 'LEFT_ARROW', 'RIGHT_ARROW'}

# Fallback bone length for a joint with no children (hands, toes, head).
_LEAF_BONE_LENGTH = 0.08


def _build_armature(context, rest: dict):
    """Create the armature the stream animates, from the bridge's rest tables.

    Bones point at their first child, the way a BVH import builds them, so the
    armature reads as a figure and retargeting has sensible bone axes to work
    with. Leaves continue the direction of the chain they end.

    That means rest rotations are *not* identity, so ARDY's local joint
    rotations cannot be written onto pose bones as-is: each needs rebasing
    through its own and its parent's rest orientation. The correction is
    computed here, once, from Blender's own ``bone.matrix_local`` — deriving it
    from Blender rather than from the offsets keeps it self-consistent with
    whatever Blender actually built.

    Returns each bone's rest rotation alongside the armature; see
    ``ardy_steer.rebase_local_rotation`` for how a joint rotation is mapped
    through it.
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
            bone.head = rest_pos[i]
            bone.tail = _rest_tail(rest_pos, parents, first_child, i)
            bone.roll = 0.0
        for i, name in enumerate(names):
            if parents[i] >= 0:
                arm_data.edit_bones[name].parent = arm_data.edit_bones[names[parents[i]]]
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')

    for pb in arm_obj.pose.bones:
        pb.rotation_mode = 'QUATERNION'

    # Each bone's rest rotation, read back from what Blender actually built.
    # ardy_steer.rebase_local_rotation turns an ARDY joint rotation into the
    # pose basis for a bone with this rest orientation; see its docstring for
    # the derivation.
    rest_rots = {}
    for name in names:
        q = arm_data.bones[name].matrix_local.to_quaternion()
        rest_rots[name] = (q.w, q.x, q.y, q.z)

    # Tag it the way the rest of the add-on recognises a source rig, so the
    # Retarget panel accepts the result without any special-casing.
    arm_obj["kimodo_source"] = True
    arm_obj["ardy_stream"] = True
    arm_obj["kimodo_creation_time"] = time.time()

    # The root's own rest rotation, needed to express its world translation as
    # a bone-space location (pose_bone.location lives in bone space, not world).
    root_rot_inv = ardy_steer.quat_inverse(rest_rots[names[root_idx]])

    return arm_obj, names, root_idx, rest_pos[root_idx], rest_rots, root_rot_inv


def _rest_tail(rest_pos, parents, first_child, i):
    """Where bone *i* points in the rest pose.

    At its first child when it has one; otherwise it continues the direction it
    arrived from, so a hand or toe sticks out along its chain rather than off
    at an arbitrary angle.
    """
    head = rest_pos[i]
    child = first_child.get(i)
    if child is not None:
        tail = rest_pos[child]
        if _distance(head, tail) > 1e-6:
            return tail

    parent = parents[i]
    if parent >= 0:
        dx = head[0] - rest_pos[parent][0]
        dy = head[1] - rest_pos[parent][1]
        dz = head[2] - rest_pos[parent][2]
        length = (dx * dx + dy * dy + dz * dz) ** 0.5
        if length > 1e-6:
            k = _LEAF_BONE_LENGTH / length
            return (head[0] + dx * k, head[1] + dy * k, head[2] + dz * k)

    # Nothing to derive a direction from: a zero-length bone is invalid, so
    # fall back to straight up rather than letting Blender drop the bone.
    return (head[0], head[1], head[2] + _LEAF_BONE_LENGTH)


def _distance(a, b) -> float:
    return ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2) ** 0.5


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
                  root_rest, rest_rots, root_rot_inv):
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

    # ARDY's local rotation, rebased into each bone's own rest frame.
    for j, name in enumerate(names):
        rest_rot = rest_rots[name]
        quats = []
        for row in rows:
            raw = ardy_steer.ardy_axis_angle_to_blender_quat(row[j * 3:j * 3 + 3])
            quats.append(ardy_steer.rebase_local_rotation(rest_rot, raw))

        path = f'pose.bones["{name}"].rotation_quaternion'
        for axis in range(4):
            fc = curves.get((path, axis))
            if fc is not None:
                _write_channel(fc, start, [q[axis] for q in quats])

    # A pose bone's location is an offset from its rest head expressed in *bone
    # space*, not a world position: feeding ARDY's absolute root straight in
    # would both lift the character by its own rest hip height and translate it
    # along the bone's axes rather than the world's.
    root_path = f'pose.bones["{root_name}"].location'
    locations = []
    for p in roots:
        world = ardy_steer.ardy_to_blender_pos(p)
        delta = (world[0] - root_rest[0],
                 world[1] - root_rest[1],
                 world[2] - root_rest[2])
        locations.append(ardy_steer.quat_rotate_vec(root_rot_inv, delta))
    for axis in range(3):
        fc = curves.get((root_path, axis))
        if fc is not None:
            _write_channel(fc, start, [loc[axis] for loc in locations])

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
            (arm, names, root_idx, root_rest,
             rest_rots, root_rot_inv) = _build_armature(context, result)
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
        self._rest_rots = rest_rots
        self._root_rot_inv = root_rot_inv
        self._start_frame = start_frame
        self._generated_until = self._start_frame     # exclusive
        self._last_prompt = s.prompt
        # Far enough back that the first tick always plans.
        self._last_plan_frame = start_frame - 10 ** 6

        s.source_armature = arm
        s.is_streaming = True
        s.stream_status = "Starting…"
        s.ardy_stream_velocity = (0.0, 0.0)
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

        # Arrow keys drive the character while the stream runs. They are
        # consumed rather than passed through: Blender binds them to frame
        # stepping, and stepping the playhead by hand mid-stream fights the
        # replan. They go back to normal the moment the stream ends.
        if (s.ardy_stream_control == 'KEYS' and event.value == 'PRESS'
                and event.type in _ARROW_KEYS):
            s.ardy_stream_velocity = ardy_steer.steer_velocity(
                tuple(s.ardy_stream_velocity), event.type,
                speed_step=s.ardy_stream_speed_step,
                turn_degrees=s.ardy_stream_turn_degrees,
            )
            # Re-plan on the next tick rather than waiting out the interval,
            # so a key press is felt immediately.
            self._last_plan_frame = -10 ** 6
            self._redraw(context)
            return {'RUNNING_MODAL'}

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
                    self._rest_rots, self._root_rot_inv,
                )
            except Exception as exc:
                return self._finish(context, f"Could not apply frames: {exc}")
            if context.scene.frame_end < self._generated_until:
                context.scene.frame_end = self._generated_until

        self._pump(context, s)
        self._redraw(context)
        return {'RUNNING_MODAL'}

    def _pump(self, context, s):
        """Start the next step when the plan is stale or the buffer is low."""
        playhead = context.scene.frame_current
        ahead = self._generated_until - playhead

        # Backpressure: the GPU is behind the playhead. Pausing is the only
        # honest option — letting playback run into un-keyframed frames would
        # freeze the character mid-stride and look like a bug.
        resume_at = min(self._horizon, max(1, s.ardy_stream_lead_frames // 2))
        if ahead <= 0 and context.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
            s.stream_status = "Waiting for ARDY…"
        elif (s.ardy_stream_autoplay and ahead > resume_at
                and not context.screen.is_animation_playing and s.is_streaming):
            bpy.ops.screen.animation_play()

        if _state["running"]:
            return

        # Two reasons to step, and the first is what makes following work.
        #
        # A step regenerates a whole horizon against the target's position at
        # that instant, so waiting for the buffer to drain before stepping
        # again meant the target was only re-read once per horizon — the
        # character committed to a two-second plan and ignored the empty until
        # it ran out. Re-planning on an interval decouples "how often do we
        # look at the target" from "how much motion is buffered".
        since = playhead - self._last_plan_frame
        stale = since >= max(1, s.ardy_stream_replan_interval)
        starving = ahead < s.ardy_stream_lead_frames
        if not (stale or starving):
            s.stream_status = self._status_line(ahead, s)
            return

        target_xz, heading, target_velocity = self._target(context, s)
        prompt = s.prompt if s.prompt != self._last_prompt else None
        self._last_prompt = s.prompt
        self._last_plan_frame = playhead

        _state["running"] = True
        self._thread = threading.Thread(
            target=_worker,
            args=(max(0, playhead - self._start_frame), target_xz, heading,
                  target_velocity, prompt, s.ardy_stream_replan_buffer,
                  s.ardy_stream_max_speed, 120.0),
            daemon=True,
        )
        self._thread.start()
        s.stream_status = self._status_line(ahead, s)

    def _target(self, context, s):
        """What to steer by this step: (position, heading, velocity).

        Exactly one of position or velocity is ever set — arrow-key driving and
        object-following are alternative controls, not layers.
        """
        if s.ardy_stream_control == 'KEYS':
            return None, None, list(s.ardy_stream_velocity)

        obj = s.ardy_stream_target
        if obj is None:
            return None, None, None
        from . import constraints as cmod
        loc = obj.matrix_world.translation
        return list(cmod.blender_to_kimodo_2d(loc)), None, None

    def _status_line(self, ahead, s=None):
        step = _state.get("step_seconds", 0.0)
        budget = self._horizon / self._fps if self._fps else 0.0
        note = "" if step <= budget or not step else "  (GPU behind)"
        drive = ""
        if s is not None and s.ardy_stream_control == 'KEYS':
            vx, vz = s.ardy_stream_velocity
            drive = f"  ·  {(vx * vx + vz * vz) ** 0.5:.1f} m/s"
        return (f"Streaming — {max(0, ahead)} frames buffered, "
                f"{step:.2f}s/step{note}{drive}")

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
