"""
Kimodo Blender Bridge — Operators
All bpy.ops.kimodo.* operators.
"""

import bpy
import addon_utils
import os
import math
import json
import threading
import random
import time
from bpy.types import Operator
from bpy.props import StringProperty, BoolProperty, IntProperty

from . import subprocess_client as sc
from . import remote_client as rc
from . import retarget as rt
from . import constraints as cmod
from . import setup_operator as so


# ---------------------------------------------------------------------------
# BVH importer compatibility (Blender 5.0+)
# ---------------------------------------------------------------------------

def _bvh_operator_registered() -> bool:
    """True only when import_anim.bvh is actually registered.

    Note: ``hasattr(bpy.ops.import_anim, "bvh")`` is useless for this — bpy.ops
    resolves lazily and always returns a wrapper, so it reports True even when
    the operator does not exist. A registered operator, however, gets a real
    type ``bpy.types.IMPORT_ANIM_OT_bvh``, so check for that instead.
    """
    return hasattr(bpy.types, "IMPORT_ANIM_OT_bvh")


def _ensure_bvh_importer() -> bool:
    """Make sure bpy.ops.import_anim.bvh is registered, enabling it if needed.

    Blender 5.0+ ships the legacy BVH importer (io_anim_bvh) disabled by
    default, so the operator is missing on a fresh install and calling it
    raises "could not be found". Enable it on demand. Returns True if the
    operator is available afterwards.
    """
    if _bvh_operator_registered():
        return True

    # Try the plain bundled module name first, then any extension-namespaced
    # variant (e.g. "bl_ext.blender_org.io_anim_bvh") discovered at runtime.
    candidates = ["io_anim_bvh"]
    try:
        candidates += [
            m.__name__ for m in addon_utils.modules()
            if m.__name__ != "io_anim_bvh" and m.__name__.endswith("io_anim_bvh")
        ]
    except Exception:
        pass

    for module_name in candidates:
        try:
            addon_utils.enable(module_name, default_set=True, persistent=True)
        except Exception:
            continue
        if _bvh_operator_registered():
            return True

    return _bvh_operator_registered()


def _uses_remote(_s=None) -> bool:
    from . import properties as props
    return props.uses_remote()


def _require_generator(operator, s) -> bool:
    """Report and return False when motion cannot be generated yet."""
    if _uses_remote(s):
        if not (s.is_connected and rc.is_ready()):
            operator.report(
                {'WARNING'},
                "Not connected — set the server URL and click Connect.",
            )
            return False
        return True
    if not sc.is_running():
        operator.report(
            {'WARNING'},
            "Kimodo is not running — click 'Start Kimodo' first.",
        )
        return False
    return True


def _remote_generation_args(context, scene_start_override=None):
    """
    Constraint payload for a remote job, built on the main thread.
    Returns (args, error). ``args`` is None when the backend is local.
    ``error`` is set when constraints could not be built; generation still
    proceeds without them, matching the local path.
    """
    s = context.scene.kimodo
    if not _uses_remote(s):
        return None, None
    caps = rc.capabilities()
    if not caps:
        # Still a remote job: an empty payload makes the thread report the
        # disconnect instead of falling through to the local bridge.
        return {
            "constraints": [],
            "transition_frames": s.num_transition_frames,
        }, "Not connected — set the server URL and click Connect."
    model = caps["models"][0]
    joints = [j["name"] for j in model["canonical_skeleton"]["joints"]]
    root = next(
        (j["name"] for j in model["canonical_skeleton"]["joints"] if not j.get("parent")),
        "Hips",
    )
    constraints = []
    err = None
    enabled = [c for c in s.motion_constraints if c.enabled and c.marker_object]
    if enabled:
        try:
            constraints = cmod.build_mmcp_constraints(
                s.motion_constraints,
                context.scene,
                kimodo_fps=s.kimodo_fps,
                auto_canonicalize=s.auto_canonicalize,
                scene_start_override=scene_start_override,
                skeleton_joints=joints,
                root_joint=root,
            )
        except Exception as exc:
            err = str(exc)
    return {
        "constraints": constraints,
        "transition_frames": s.num_transition_frames,
    }, err


def _generate_remote(prompts, durations, seeds, remote_args, progress_cb):
    """POST one MMCP generate and return (success, gltf_path_or_error)."""
    try:
        body = rc.build_generate_request(
            rc.capabilities(),
            prompts=list(prompts),
            durations=list(durations),
            seeds=list(seeds) if seeds is not None else None,
            constraints=(remote_args or {}).get("constraints"),
            transition_frames=(remote_args or {}).get("transition_frames", 5),
        )
    except rc.RemoteError as exc:
        return False, str(exc)
    return rc.generate_to_file(body, progress_callback=progress_cb)


def _import_bvh(**kwargs):
    """Import a BVH file, self-healing a missing/disabled importer add-on.

    Raises RuntimeError with an actionable message if the importer cannot be
    made available, so callers can report it cleanly instead of leaking a
    cryptic operator traceback.
    """
    if not _ensure_bvh_importer():
        raise RuntimeError(
            "Blender's BVH importer is not available. Enable "
            "'Import-Export: BioVision Motion Capture (BVH)' under "
            "Edit > Preferences > Add-ons (search 'BVH'). On Blender 5.0+ you "
            "may need to install it from Get Extensions first."
        )
    try:
        return bpy.ops.import_anim.bvh(**kwargs)
    except (AttributeError, RuntimeError) as e:
        # Operator vanished between the check and the call, or import failed.
        raise RuntimeError(f"BVH import failed: {e}")


def _is_gltf(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (".gltf", ".glb")


def _gltf_operator_registered() -> bool:
    return hasattr(bpy.types, "IMPORT_SCENE_OT_gltf")


def _ensure_gltf_importer() -> bool:
    if _gltf_operator_registered():
        return True
    for module_name in ("io_scene_gltf2", "bl_ext.blender_org.io_scene_gltf2"):
        try:
            addon_utils.enable(module_name, default_set=True, persistent=True)
        except Exception:
            continue
        if _gltf_operator_registered():
            return True
    return _gltf_operator_registered()


def _iter_action_fcurves(action):
    """Yield fcurves from legacy actions and from Blender 4.4+ layered actions."""
    if action is None:
        return
    seen = set()

    def _emit(curves):
        for fc in curves or []:
            ident = id(fc)
            if ident in seen:
                continue
            seen.add(ident)
            yield fc

    legacy = getattr(action, "fcurves", None)
    if legacy:
        yield from _emit(legacy)

    for layer in getattr(action, "layers", None) or []:
        for strip in getattr(layer, "strips", None) or []:
            bags = getattr(strip, "channelbags", None)
            if bags is not None:
                try:
                    bag_list = list(bags)
                except TypeError:
                    bag_list = []
                for bag in bag_list:
                    yield from _emit(getattr(bag, "fcurves", None))
            channelbag = getattr(strip, "channelbag", None)
            if not channelbag:
                continue
            for slot in getattr(action, "slots", None) or []:
                try:
                    bag = channelbag(slot)
                except Exception:
                    bag = None
                if bag is not None:
                    yield from _emit(getattr(bag, "fcurves", None))


def _action_frame_range(action):
    """(min, max) keyframe times, or (None, None) when the action has no keys."""
    lo = hi = None
    for fc in _iter_action_fcurves(action):
        for kp in fc.keyframe_points:
            x = kp.co.x
            lo = x if lo is None else min(lo, x)
            hi = x if hi is None else max(hi, x)
    return lo, hi


def _shift_action(action, delta: float) -> None:
    if not delta:
        return
    for fc in _iter_action_fcurves(action):
        for kp in fc.keyframe_points:
            kp.co.x += delta
            kp.handle_left.x += delta
            kp.handle_right.x += delta
        try:
            fc.update()
        except Exception:
            pass


def _place_gltf_animation(arm, start_frame: int, update_scene_duration: bool) -> None:
    """Slide the imported clip so its first key sits on ``start_frame``.

    glTF is Y-up metres. Blender's importer maps (x, y, z) to (x, -z, y),
    the same facing as the BVH path (axis_forward='-Z', axis_up='Y'):
    Kimodo +Z forward lands on Blender -Y, and +Y up lands on Blender +Z.
    The file is already in metres, so unlike BVH it is not scaled by 0.01.
    """
    action = arm.animation_data.action if arm.animation_data else None
    lo, hi = _action_frame_range(action)
    if lo is None:
        return
    delta = float(start_frame) - lo
    _shift_action(action, delta)
    if update_scene_duration and hi is not None:
        scene = bpy.context.scene
        end = int(math.ceil(hi + delta))
        if end > scene.frame_end:
            scene.frame_end = end


def _import_gltf(filepath: str, start_frame: int = 1, update_scene_duration: bool = True):
    """Import a glTF/GLB motion file. Returns the new armature, or None."""
    if not _ensure_gltf_importer():
        raise RuntimeError(
            "Blender's glTF importer is not available. Enable "
            "'Import-Export: glTF 2.0 format' under Edit > Preferences > Add-ons."
        )
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.gltf(filepath=filepath)
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(f"glTF import failed: {exc}") from exc
    new_arm = next(
        (obj for obj in set(bpy.data.objects) - before if obj.type == 'ARMATURE'),
        None,
    )
    if new_arm is not None:
        _place_gltf_animation(new_arm, start_frame, update_scene_duration)
    return new_arm


# ---------------------------------------------------------------------------
# Shared async state (thread → modal operator communication)
# ---------------------------------------------------------------------------

_generation_state = {
    "running": False,
    "done": False,
    "success": False,
    "cancelled": False, # user requested cancel; discard the result
    "result": "",       # file path on success, error message on failure
    "progress": "",
}


def _reset_state():
    _generation_state.update(running=False, done=False, success=False,
                              cancelled=False, result="", progress="")


def _discard_result_file():
    """Delete a finished output the user asked to throw away."""
    path = _generation_state.get("result") or ""
    if isinstance(path, str) and path and os.path.isfile(path):
        try:
            os.unlink(path)
        except OSError:
            pass


def _step_seed_after_generation(owner, resolved_seed: int = None) -> None:
    """Record/advance `owner.seed` per its seed mode after a successful generation.
    `owner` is the scene settings or a motion segment. In RANDOM mode the actual
    seed used is written back to seed so leaving RANDOM reproduces the last result.
    `last_used_seed` always receives the seed that was actually used so that
    switching to FIXED can lock the most recent result."""
    if owner.seed_mode == 'INCREMENT' and owner.seed >= 0:
        owner.seed = min(owner.seed + 1, 2**31 - 1)
    elif owner.seed_mode == 'DECREMENT' and owner.seed > 0:
        owner.seed -= 1
    elif owner.seed_mode == 'RANDOM' and resolved_seed is not None:
        owner.seed = resolved_seed
    if resolved_seed is not None:
        owner.last_used_seed = resolved_seed


_HISTORY_MAX = 20


def _push_history(s, prompt: str, seed: int, duration: float, bvh_path: str,
                  seeds: "list[int] | None" = None) -> None:
    """Prepend a new entry to generation_history, keeping newest-first, capped at _HISTORY_MAX.
    `seeds` carries the per-segment seeds of a multi-prompt generation."""
    import datetime
    entry = s.generation_history.add()
    entry.prompt    = prompt
    entry.seed      = seed
    entry.seeds     = ", ".join(str(x) for x in seeds) if seeds else ""
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

    def _run_start(self, python_exe: str, model_name: str, use_offload: bool,
                   remote_url=None):
        def progress(msg):
            _start_state["message"] = msg

        if remote_url is not None:
            success, msg = rc.connect(remote_url, progress_callback=progress)
        else:
            success, msg = sc.start(
                python_exe, model_name, use_offload=use_offload,
                progress_callback=progress,
            )
        _start_state["success"] = success
        _start_state["message"] = msg
        _start_state["done"]    = True
        _start_state["running"] = False

    def invoke(self, context, event):
        s = context.scene.kimodo

        remote = _uses_remote(s)
        if remote:
            if s.is_connected and rc.is_ready():
                self.report({'INFO'}, "Already connected to the Kimodo server.")
                return {'CANCELLED'}
        elif sc.is_running():
            self.report({'INFO'}, "Kimodo is already running.")
            return {'CANCELLED'}

        _reset_start_state()
        _start_state["running"] = True
        s.is_connected      = False
        s.connection_status = "Connecting…" if remote else "Starting…"

        # Resolve the Python hint on the main thread. When the scene has no
        # explicit path, fall back to the remembered managed-venv location
        # (addon preference) so a fresh scene still finds the install.
        python_hint = (s.python_executable or "").strip()
        if not python_hint and not remote:
            try:
                python_hint = so.managed_python()
            except Exception:
                python_hint = ""

        self._thread = threading.Thread(
            target=self._run_start,
            args=(python_hint, s.kimodo_model, s.use_offload),
            kwargs={"remote_url": s.server_url if remote else None},
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
            info = rc.session()
            if _uses_remote(s) and info:
                try:
                    s.kimodo_fps = float(info["fps"])
                except (TypeError, ValueError, KeyError):
                    pass
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
        s = context.scene.kimodo
        remote = _uses_remote(s)
        rc.disconnect()
        if sc.is_running():
            sc.stop()
        s.is_connected      = False
        s.connection_status = "Stopped"
        if remote:
            self.report({'INFO'}, "Disconnected from Kimodo server.")
        else:
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

    def _run_generation(self, prompt, duration, seed, fmt, constraints_json=None,
                        bvh_standard_tpose=False, remote_args=None):
        """Runs in background thread."""
        def progress_cb(msg):
            _generation_state["progress"] = msg

        if remote_args is not None:
            success, result = _generate_remote(
                [prompt], [duration], [seed], remote_args, progress_cb,
            )
        else:
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

        if not _require_generator(self, s):
            return {'CANCELLED'}
        if s.is_generating:
            self.report({'WARNING'}, "Already generating — please wait.")
            return {'CANCELLED'}

        # Resolve seed (upper bound must stay within a 32-bit IntProperty)
        seed = random.randint(0, 2**31 - 1) if s.seed_mode == 'RANDOM' else s.seed
        self._resolved_seed = seed
        print(f"[Kimodo] Generating with seed {seed}", flush=True)

        # Launch background thread
        _reset_state()
        _generation_state["running"] = True
        s.is_generating = True
        s.generation_progress = "Starting…"

        # Build constraints on the main thread (they read the scene).
        constraints_json = None
        remote_args = None
        if _uses_remote(s):
            remote_args, con_err = _remote_generation_args(context)
            if con_err:
                self.report({'WARNING'}, f"Constraint build failed (generating without): {con_err}")
        else:
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
                remote_args,
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

        if _generation_state["cancelled"]:
            _discard_result_file()
            s.generation_progress = "Cancelled"
            self.report({'INFO'}, "Generation cancelled — result discarded.")
        elif _generation_state["success"]:
            file_path = _generation_state["result"]
            s.last_bvh_path = file_path
            s.generation_progress = "Done ✓"
            _push_history(s, s.prompt, self._resolved_seed, s.duration, file_path)
            _step_seed_after_generation(s, self._resolved_seed)
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".bvh" or _is_gltf(file_path):
                result = bpy.ops.kimodo.import_bvh('EXEC_DEFAULT', filepath=file_path)
                kind = "glTF" if _is_gltf(file_path) else "BVH"
                if 'FINISHED' in result:
                    self.report({'INFO'}, f"Motion generated and imported from {os.path.basename(file_path)}")
                else:
                    self.report({'WARNING'}, f"Motion generated ({os.path.basename(file_path)}) but {kind} import failed — see error above.")
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
    """Cancel ongoing generation. The bridge cannot abort mid-diffusion, so the
    in-flight result is discarded when it arrives; a new generation can start
    once the bridge finishes the abandoned job."""
    bl_idname = "kimodo.cancel_generation"
    bl_label = "Cancel"

    def execute(self, context):
        s = context.scene.kimodo
        if not s.is_generating:
            self.report({'INFO'}, "Nothing is generating.")
            return {'CANCELLED'}
        # Stale state (#43): is_generating was saved into the .blend or
        # restored by undo, but no job is actually running — just unlock.
        if not _generation_state["running"] and not sc.is_busy() and not rc.is_busy():
            s.is_generating = False
            s.generating_segment_index = -1
            s.generation_progress = ""
            self.report({'INFO'}, "No generation was running — state reset.")
            return {'FINISHED'}
        _generation_state["cancelled"] = True
        sc.request_cancel()
        rc.request_cancel()
        s.generation_progress = "Cancelling…"
        self.report({'INFO'}, "Cancellation requested — result will be discarded.")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

# Stable slot name reused for every Kimodo action so the slot identifier stays
# constant across generations (see _bind_action_with_slot / issue #37).
_KIMODO_SLOT_NAME = "Kimodo"


def _bind_action_with_slot(obj: bpy.types.Object, action: bpy.types.Action) -> None:
    """Assign *action* to *obj* and bind its animation slot (Blender 4.4+).

    Blender 4.4 introduced "slotted" Actions: an Action carries one or more
    slots, and an object's ``animation_data`` must be bound to a specific slot
    for the keyframes to drive the rig.  Setting ``animation_data.action`` only
    auto-binds a slot whose identifier matches the previously used one
    (``last_slot_identifier``).  BVH import gives every freshly imported Action a
    slot with a randomized name, so when we swap actions on the reused source
    armature (i.e. generating motion a second time) the identifiers never match:
    no slot binds, and the armature stops animating until the slot is picked by
    hand (issue #37).

    To fix it we give the slot a stable name (so the identifier is constant
    across generations) and bind it explicitly.  On Blender <4.4, where slots
    don't exist, this degrades to a plain action assignment.
    """
    if obj.animation_data is None:
        obj.animation_data_create()
    ad = obj.animation_data

    slots = getattr(action, "slots", None)
    if slots is None or not hasattr(ad, "action_slot"):
        # Pre-4.4 Blender: no slot system, a plain assignment is all there is.
        ad.action = action
        return

    # Unify the slot name so the identifier stays constant across generations.
    if len(slots):
        try:
            slots[0].name_display = _KIMODO_SLOT_NAME
        except Exception:
            pass  # name_display may be read-only on some builds — binding still works.

    ad.action = action
    # Assigning the action may already auto-bind a slot; only force it otherwise.
    if len(slots) and ad.action_slot is None:
        ad.action_slot = slots[0]


def _apply_to_existing_source(s, new_arm: bpy.types.Object) -> bpy.types.Object:
    """
    If reuse_armature is set, transfer the action from new_arm to it and
    delete new_arm.  Returns the armature that should be used going forward.
    """
    existing = s.reuse_armature
    if not existing or existing.type != 'ARMATURE' or existing == new_arm:
        return new_arm  # nothing to reuse — keep the freshly imported one

    new_action = new_arm.animation_data.action if new_arm.animation_data else None
    old_action = existing.animation_data.action if existing.animation_data else None
    new_arm_data = new_arm.data

    # Transfer the action (the data-block survives object deletion).  Bind the
    # slot explicitly so Blender 4.4+ keeps animating after the swap (issue #37).
    if new_action:
        _bind_action_with_slot(existing, new_action)

    # Remove the temporary stand-in armature object and its now-orphan data, so
    # re-importing (e.g. from history) doesn't leave Kimodo_Source.001/.002 …
    bpy.data.objects.remove(new_arm, do_unlink=True)
    if new_arm_data and new_arm_data.users == 0:
        bpy.data.armatures.remove(new_arm_data)

    # Drop the action that used to be on the reused armature once nothing else
    # references it — otherwise every re-import piles up an orphan Action.001/.002.
    if old_action and old_action is not new_action and old_action.users == 0:
        bpy.data.actions.remove(old_action)

    return existing


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

        try:
            if _is_gltf(path):
                _import_gltf(path, start_frame=1, update_scene_duration=True)
            else:
                _import_bvh(
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
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        # Find newly added armature
        after = set(bpy.context.scene.objects)
        new_objs = after - before
        new_arm = next((o for o in new_objs if o.type == 'ARMATURE'), None)

        if new_arm:
            new_arm.name = "Kimodo_Source"
            new_arm["kimodo_source"] = True
            new_arm["kimodo_creation_time"] = time.time()
            new_arm = _apply_to_existing_source(s, new_arm)
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

        pairs = [(item.source_bone, item.target_bone, item.enabled,
                  item.retarget_mode, item.inherit_rotation)
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
                  "en": item.enabled, "mode": item.retarget_mode,
                  "inherit_rot": item.inherit_rotation}
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
            item.source_bone     = p.get("src", "")
            item.target_bone     = p.get("tgt", "")
            item.enabled         = p.get("en", True)
            item.retarget_mode   = p.get("mode", "COPY_ROTATION")
            item.inherit_rotation = p.get("inherit_rot", True)

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
                  "en": item.enabled, "mode": item.retarget_mode,
                  "inherit_rot": item.inherit_rotation}
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
            item.source_bone     = p.get("src", "")
            item.target_bone     = p.get("tgt", "")
            item.enabled         = p.get("en", True)
            item.retarget_mode   = p.get("mode", "COPY_ROTATION")
            item.inherit_rotation = p.get("inherit_rot", True)

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
        seg.seed_mode   = s.seed_mode
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
        new_seg.seed_mode   = src.seed_mode
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
            seg.seed        = global_seed
            seg.seed_mode   = s.seed_mode

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
        if not _require_generator(self, s):
            return {'CANCELLED'}
        if s.is_generating:
            self.report({'WARNING'}, "Already generating — please wait.")
            return {'CANCELLED'}

        seg = s.motion_segments[idx]
        self._target_segment_idx = idx
        return self._start_generation(context, s, seg)

    def _start_generation(self, context, s, seg):
        import random as _random

        seed = _random.randint(0, 2**31 - 1) if seg.seed_mode == 'RANDOM' else seg.seed
        self._resolved_seed = seed
        print(f"[Kimodo] Generating segment with seed {seed}", flush=True)
        fps  = context.scene.render.fps / context.scene.render.fps_base
        duration = (seg.end_frame - seg.start_frame + 1) / fps
        self._segment_duration = duration

        # Constraints are indexed from this segment's first frame so they
        # line up with the clip imported at seg.start_frame.
        remote_args = None
        if _uses_remote(s):
            remote_args, con_err = _remote_generation_args(
                context, scene_start_override=seg.start_frame,
            )
            constraints_json = None
        else:
            constraints_json, con_err = _build_segment_constraints(context, seg)
        if con_err:
            self.report({'WARNING'},
                        f"Constraint build failed (generating without): {con_err}")

        _reset_state()
        _generation_state["running"] = True
        s.is_generating = True
        s.generation_progress = f"Generating: {seg.prompt[:40]}…"
        s.generating_segment_index = self._target_segment_idx

        self._thread = threading.Thread(
            target=self._run_generation,
            args=(seg.prompt, duration, seed, s.output_format, constraints_json,
                  s.bvh_standard_tpose, remote_args),
            daemon=True,
        )
        self._thread.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _run_generation(self, prompt, duration, seed, fmt, constraints_json,
                        bvh_standard_tpose, remote_args=None):
        def progress_cb(msg):
            _generation_state["progress"] = msg

        if remote_args is not None:
            success, result = _generate_remote(
                [prompt], [duration], [seed], remote_args, progress_cb,
            )
        else:
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

        if _generation_state["cancelled"]:
            _discard_result_file()
            s.generation_progress = "Cancelled"
            self.report({'INFO'}, "Generation cancelled — result discarded.")
        elif _generation_state["success"]:
            file_path = _generation_state["result"]
            seg = s.motion_segments[self._target_segment_idx]
            seg.last_bvh_path = file_path
            seg.generated = True
            s.generation_progress = "Done ✓"
            _push_history(s, seg.prompt, self._resolved_seed, self._segment_duration, file_path)
            _step_seed_after_generation(seg, self._resolved_seed)

            if os.path.splitext(file_path)[1].lower() == ".bvh" or _is_gltf(file_path):
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


def _build_multi_prompt_constraints(context, first_start_frame: int) -> "tuple[str | None, str | None]":
    """Build scene constraints JSON for multi-prompt generation.

    Uses first_start_frame as the Kimodo sequence origin so constraint frame
    indices align with the combined BVH that starts at that Blender frame.
    Returns (constraints_json, error_message) — error_message is set when the
    build failed so the caller can warn instead of silently dropping them.
    """
    s = context.scene.kimodo
    enabled = [c for c in s.motion_constraints if c.enabled and c.marker_object]
    if not enabled:
        return None, None
    try:
        data = cmod.build_constraints_json(
            s.motion_constraints, context.scene,
            kimodo_fps=s.kimodo_fps,
            auto_canonicalize=s.auto_canonicalize,
            scene_start_override=first_start_frame,
        )
        return (json.dumps(data) if data else None), None
    except Exception as exc:
        return None, str(exc)


class KIMODO_OT_GenerateAllSegments(Operator):
    """Generate all enabled segments as a single multi-prompt sequence with smooth transitions"""
    bl_idname = "kimodo.generate_all_segments"
    bl_label  = "Generate All Segments"

    _timer           = None
    _thread          = None
    _segment_indices: list = []
    _start_frame: int = 1
    _resolved_seed: int = -1
    _resolved_seeds: list = []

    def invoke(self, context, event):
        s = context.scene.kimodo
        if not _require_generator(self, s):
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

        # Resolve seeds for all segments
        seeds = [random.randint(0, 2**31 - 1) if seg.seed_mode == 'RANDOM' else seg.seed
                 for _, seg in ordered]
        seed = seeds[0]
        self._resolved_seed = seed
        self._resolved_seeds = list(seeds)
        print(f"[Kimodo] Generating {len(seeds)} segments with seeds {seeds}", flush=True)

        # Build constraints relative to the start of the combined sequence
        remote_args = None
        if _uses_remote(s):
            remote_args, con_err = _remote_generation_args(
                context, scene_start_override=self._start_frame,
            )
            constraints_json = None
        else:
            constraints_json, con_err = _build_multi_prompt_constraints(
                context, self._start_frame,
            )
        if con_err:
            self.report({'WARNING'},
                        f"Constraint build failed (generating without): {con_err}")

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
                  constraints_json, s.bvh_standard_tpose, num_transition_frames, seeds,
                  remote_args),
            daemon=True,
        )
        self._thread.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _run_all(self, prompts, durations, seed, fmt, constraints_json, bvh_standard_tpose,
                 num_transition_frames=5, seeds=None, remote_args=None):
        def progress_cb(msg):
            _generation_state["progress"] = msg

        if remote_args is not None:
            success, result = _generate_remote(
                prompts, durations, seeds if seeds is not None else [seed],
                remote_args, progress_cb,
            )
            _generation_state["success"] = success
            _generation_state["result"]  = result
            _generation_state["done"]    = True
            _generation_state["running"] = False
            return

        success, result = sc.generate_motion_multi(
            prompts=prompts,
            durations=durations,
            seed=seed,
            output_format=fmt,
            constraints_json=constraints_json,
            bvh_standard_tpose=bvh_standard_tpose,
            num_transition_frames=num_transition_frames,
            progress_callback=progress_cb,
            seeds=seeds,
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

        if _generation_state["cancelled"]:
            _discard_result_file()
            s.generation_progress = "Cancelled"
            self.report({'INFO'}, "Generation cancelled — result discarded.")
        elif _generation_state["success"]:
            file_path = _generation_state["result"]

            # Mark all segments as generated and store the shared path
            generated_segments = [s.motion_segments[idx] for idx in self._segment_indices]
            for seg in generated_segments:
                seg.last_bvh_path = file_path
                seg.generated = True

            fps = context.scene.render.fps / context.scene.render.fps_base
            prompt = " | ".join(seg.prompt for seg in generated_segments)
            duration = sum((seg.end_frame - seg.start_frame + 1) / fps for seg in generated_segments)
            _push_history(s, prompt, self._resolved_seed, duration, file_path,
                          seeds=self._resolved_seeds)
            for i, seg in enumerate(generated_segments):
                _step_seed_after_generation(seg, self._resolved_seeds[i] if i < len(self._resolved_seeds) else None)

            if os.path.splitext(file_path)[1].lower() == ".bvh" or _is_gltf(file_path):
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

        try:
            if _is_gltf(self.filepath):
                _import_gltf(
                    self.filepath,
                    start_frame=self.start_frame,
                    update_scene_duration=False,
                )
            else:
                _import_bvh(
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
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        after   = set(bpy.data.objects)
        new_arm = next((o for o in after - before if o.type == 'ARMATURE'), None)

        if new_arm:
            name = f"Kimodo_{self.label}" if self.label else "Kimodo_Source"
            new_arm.name = name
            new_arm["kimodo_source"] = True
            new_arm["kimodo_creation_time"] = time.time()
            s = context.scene.kimodo
            new_arm = _apply_to_existing_source(s, new_arm)
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


def _build_segment_constraints(context, seg) -> "tuple[str | None, str | None]":
    """Build constraints JSON for a segment (reuses scene-level constraints for now).

    Returns (constraints_json, error_message) — error_message is set when the
    build failed so the caller can warn instead of silently dropping them.
    """
    s = context.scene.kimodo
    enabled = [c for c in s.motion_constraints if c.enabled and c.marker_object]
    if not enabled:
        return None, None
    try:
        data = cmod.build_constraints_json(
            s.motion_constraints, context.scene,
            kimodo_fps=s.kimodo_fps,
            auto_canonicalize=s.auto_canonicalize,
        )
        return json.dumps(data), None
    except Exception as exc:
        return None, str(exc)


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


class KIMODO_OT_DrawFreehandCurve(Operator):
    """Create a new curve and activate Blender's Draw tool to sketch a path freehand"""
    bl_idname  = "kimodo.draw_freehand_curve"
    bl_label   = "Draw Curve"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.kimodo

        # If we are already in Edit Mode, switch to Object Mode first
        if context.mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                pass

        # Create new Bezier Curve data block and object
        curve_data = bpy.data.curves.new(name="Kimodo_Draw_Path", type='CURVE')
        curve_data.dimensions = '3D'
        
        curve_obj = bpy.data.objects.new(name="Kimodo_Draw_Path", object_data=curve_data)
        context.scene.collection.objects.link(curve_obj)
        
        # Set it active & selected
        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = curve_obj
        curve_obj.select_set(True)
        
        # Assign to panel property
        s.path_curve = curve_obj
        
        # Enter edit mode
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Set tool to built-in Draw tool
        try:
            bpy.ops.wm.tool_set_by_id(name="builtin.draw")
            self.report({'INFO'}, "Draw tool active! Sketch a path in the 3D Viewport, then click 'Sample Curve' when done.")
        except Exception as e:
            self.report({'WARNING'}, f"Created curve, but could not set active tool: {e}")

        return {'FINISHED'}


class KIMODO_OT_SampleCurveAsWaypoints(Operator):
    """Sample a curve into evenly-spaced Root XZ waypoint constraints"""
    bl_idname  = "kimodo.sample_curve_as_waypoints"
    bl_label   = "Sample Curve as Waypoints"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.kimodo

        if context.mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                pass

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

        # Heading per sample: the forward direction of the curve on the
        # ground plane, so the character walks *along* the path instead of
        # always facing +Y. Use a forward difference (and the previous
        # segment for the final point); reuse the last valid angle across any
        # duplicate/zero-length samples.  angle = atan2(dx, -dy) gives the
        # heading_angle (consumed by build_constraints_json) that makes the
        # character face forward along the curve in Kimodo's frame.
        headings = []
        last_angle = 0.0
        n_pos = len(positions)
        for i in range(n_pos):
            if i + 1 < n_pos:
                d = positions[i + 1] - positions[i]      # forward difference
            elif i > 0:
                d = positions[i] - positions[i - 1]       # last point: previous segment
            else:
                d = None                                  # single point: no direction
            if d is not None and math.hypot(d.x, d.y) > 1e-6:
                last_angle = math.atan2(d.x, -d.y)
            headings.append(last_angle)

        start_f = s.path_start_frame
        end_f   = s.path_end_frame
        color   = _CONSTRAINT_COLORS_RGBA['root2d']

        saved_active   = context.active_object
        saved_selected = list(context.selected_objects)
        bpy.ops.object.select_all(action='DESELECT')

        for i, pos in enumerate(positions):
            frame = round(start_f + (end_f - start_f) * i / (n - 1)) if n > 1 else start_f

            angle = headings[i]

            bpy.ops.object.empty_add(type='ARROWS', location=pos)
            empty = context.active_object
            empty.name = _unique_name(f"Kimodo_Path_{i + 1:02d}")
            empty.color = color
            empty.show_name = True
            empty.empty_display_size = 0.15
            # Point the arrow empty along the curve so the heading is visible.
            empty.rotation_euler.z = angle
            empty["kimodo_constraint"] = True
            empty["kimodo_type"]       = 'root2d'

            item = s.motion_constraints.add()
            item.constraint_type = 'root2d'
            item.frame           = frame
            item.marker_object   = empty
            item.enabled         = True
            item.include_heading = True
            item.heading_angle   = angle
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

        Logic:
        - Case 1: an armature is selected AND active is an armature → use active armature
        - Case 2: no armature in the selection set → duplicate source_armature for posing
        - Case 3: otherwise → error
        """
        active = context.active_object
        has_selected_armature = any(obj.type == 'ARMATURE' for obj in context.selected_objects)

        # Case 1: an armature is selected and active is an armature → use active as-is
        if has_selected_armature and active and active.type == 'ARMATURE':
            self.report({'INFO'},
                f"Using selected armature '{active.name}' as full-body pose reference. "
                f"Pose it at frame {cur_frame} to define the keyframe.")
            return active

        # Case 2: no armature selected at all → duplicate source_armature for posing
        if not has_selected_armature and s.source_armature:
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

    index: IntProperty(
        name="Index",
        description="Index of the constraint to remove (-1 = active)",
        default=-1,
    )
    delete_object: BoolProperty(
        name="Delete Marker Object",
        description="Also delete the Empty/object from the scene",
        default=False,
    )

    def execute(self, context):
        s = context.scene.kimodo
        idx = self.index if self.index >= 0 else s.constraint_index
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
            if _uses_remote(s):
                caps = rc.capabilities()
                joints = None
                root = "Hips"
                if caps:
                    skel = caps["models"][0]["canonical_skeleton"]
                    joints = [j["name"] for j in skel["joints"]]
                    root = next(
                        (j["name"] for j in skel["joints"] if not j.get("parent")),
                        "Hips",
                    )
                data = cmod.build_mmcp_constraints(
                    s.motion_constraints,
                    context.scene,
                    kimodo_fps=s.kimodo_fps,
                    auto_canonicalize=s.auto_canonicalize,
                    skeleton_joints=joints,
                    root_joint=root,
                )
                json_str = json.dumps(data, indent=2)
            else:
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
        fps = int(round(context.scene.kimodo.kimodo_fps or 30.0))
        fps = max(1, fps)
        context.scene.render.fps = fps
        context.scene.render.fps_base = 1.0
        self.report({'INFO'}, f"Scene FPS set to {fps}.")
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
            self.report({'ERROR'}, f"Motion file no longer exists: {entry.bvh_path}")
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

        if not _require_generator(self, s):
            return {'CANCELLED'}
        if s.is_generating:
            self.report({'WARNING'}, "Already generating — please wait.")
            return {'CANCELLED'}

        self._remote      = _uses_remote(s)
        self._total       = s.num_variations
        self._seeds       = [random.randint(0, 2**31 - 1) for _ in range(self._total)]
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

        remote_args = None
        if self._remote:
            # Variations match the local path: prompt and seed only, no constraints.
            remote_args = {
                "constraints": [],
                "transition_frames": s.num_transition_frames,
            }
        self._thread = threading.Thread(
            target=self._run_one,
            args=(s.prompt, s.duration, seed, s.output_format, s.bvh_standard_tpose,
                  remote_args),
            daemon=True,
        )
        self._thread.start()

    def _run_one(self, prompt, duration, seed, fmt, bvh_standard_tpose, remote_args=None):
        def progress_cb(msg):
            _generation_state["progress"] = msg

        if remote_args is not None:
            success, result = _generate_remote(
                [prompt], [duration], [seed], remote_args, progress_cb,
            )
        else:
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

        if _generation_state["cancelled"]:
            _discard_result_file()
            context.window_manager.event_timer_remove(self._timer)
            s.is_generating = False
            s.generation_progress = "Cancelled"
            self.report({'INFO'},
                        f"Variations cancelled after {self._current_idx} of {self._total}.")
            return {'FINISHED'}

        if not _generation_state["success"]:
            context.window_manager.event_timer_remove(self._timer)
            s.is_generating = False
            s.generation_progress = f"Variation {var_num} failed ✗"
            self.report({'ERROR'}, f"Variation {var_num} failed: {_generation_state['result']}")
            return {'FINISHED'}

        file_path = _generation_state["result"]
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".bvh" or _is_gltf(file_path):
            before = set(bpy.data.objects)
            try:
                if _is_gltf(file_path):
                    _import_gltf(file_path, start_frame=1, update_scene_duration=True)
                else:
                    _import_bvh(
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
            except RuntimeError as e:
                context.window_manager.event_timer_remove(self._timer)
                s.is_generating = False
                s.generation_progress = f"Variation {var_num} failed ✗"
                self.report({'ERROR'}, str(e))
                return {'FINISHED'}
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
    KIMODO_OT_DrawFreehandCurve,
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
