"""
Kimodo Blender Bridge — Retargeting
Constraint-based motion retargeting from Kimodo armature → user's rig.

Three modes per bone pair
--------------------------
COPY_ROTATION   Copy Rotation in WORLD space.
                Root bone additionally gets Copy Location.
                Good for rigs with the same rest pose as Kimodo.

COPY_TRANSFORMS Copy Transforms in LOCAL space (loc + rot + scale together).
                Simpler than the split loc/rot approach; useful when the
                target rig's bone lengths don't match the source.

CHILD_OF        Full parent-child relationship via a Child Of constraint.
                The inverse matrix is set to identity so the target bone
                snaps to the source; set it manually in the UI if you need
                a rest-pose offset.  Best for floating / weapon bones or
                when you want exact world-space tracking.

Baking
-------
Blender's NLA bake (visual_keying=True) is used to push the constraint-
driven animation into actual keyframes; the constraints are cleared
afterwards so the rig is fully self-contained.
"""

import bpy
import json
import mathutils
import re


CONSTRAINT_PREFIX = "KIMODO_"   # prefix for all constraints we add


# ---------------------------------------------------------------------------
# Bone name auto-matching
# ---------------------------------------------------------------------------

# Common bone name pairs: (kimodo/bvh_name, common_rig_names...)
# Kimodo SOMA skeleton uses these bone names (based on BVH output conventions).
# These are heuristic — the user can always override in the UI.
_SOMA_BONE_MAP_HINTS = [
    # Kimodo name        # common alternatives in user rigs
    ("Hips",            ["hips", "pelvis", "root", "Hip", "Pelvis", "mixamorig:Hips"]),
    ("Spine",           ["spine", "Spine1", "mixamorig:Spine"]),
    ("Spine1",          ["spine1", "spine_01", "mixamorig:Spine1"]),
    ("Spine2",          ["spine2", "chest", "mixamorig:Spine2"]),
    ("Neck",            ["neck", "Neck1", "mixamorig:Neck"]),
    ("Head",            ["head", "Head", "mixamorig:Head"]),
    ("LeftShoulder",    ["l_shoulder", "shoulder.L", "mixamorig:LeftShoulder", "LeftShoulder"]),
    ("LeftArm",         ["upper_arm.L", "l_arm", "mixamorig:LeftArm", "LeftUpArm"]),
    ("LeftForeArm",     ["forearm.L", "l_forearm", "mixamorig:LeftForeArm"]),
    ("LeftHand",        ["hand.L", "l_hand", "mixamorig:LeftHand"]),
    ("RightShoulder",   ["r_shoulder", "shoulder.R", "mixamorig:RightShoulder", "RightShoulder"]),
    ("RightArm",        ["upper_arm.R", "r_arm", "mixamorig:RightArm", "RightUpArm"]),
    ("RightForeArm",    ["forearm.R", "r_forearm", "mixamorig:RightForeArm"]),
    ("RightHand",       ["hand.R", "r_hand", "mixamorig:RightHand"]),
    ("LeftUpLeg",       ["thigh.L", "l_thigh", "mixamorig:LeftUpLeg", "LeftThigh"]),
    ("LeftLeg",         ["shin.L", "l_shin", "mixamorig:LeftLeg", "LeftShin"]),
    ("LeftFoot",        ["foot.L", "l_foot", "mixamorig:LeftFoot"]),
    ("LeftToeBase",     ["toe.L", "l_toe", "mixamorig:LeftToeBase"]),
    ("RightUpLeg",      ["thigh.R", "r_thigh", "mixamorig:RightUpLeg", "RightThigh"]),
    ("RightLeg",        ["shin.R", "r_shin", "mixamorig:RightLeg", "RightShin"]),
    ("RightFoot",       ["foot.R", "r_foot", "mixamorig:RightFoot"]),
    ("RightToeBase",    ["toe.R", "r_toe", "mixamorig:RightToeBase"]),
]

_SMPLX_EXTRA_HINTS = [
    ("LeftHandIndex1",  ["f_index.01.L", "mixamorig:LeftHandIndex1"]),
    ("RightHandIndex1", ["f_index.01.R", "mixamorig:RightHandIndex1"]),
    ("LeftHandThumb1",  ["thumb.01.L",   "mixamorig:LeftHandThumb1"]),
    ("RightHandThumb1", ["thumb.01.R",   "mixamorig:RightHandThumb1"]),
    ("LeftHandMiddle1", ["f_middle.01.L","mixamorig:LeftHandMiddle1"]),
    ("RightHandMiddle1",["f_middle.01.R","mixamorig:RightHandMiddle1"]),
    ("LeftHandRing1",   ["f_ring.01.L",  "mixamorig:LeftHandRing1"]),
    ("RightHandRing1",  ["f_ring.01.R",  "mixamorig:RightHandRing1"]),
    ("LeftHandPinky1",  ["f_pinky.01.L", "mixamorig:LeftHandPinky1"]),
    ("RightHandPinky1", ["f_pinky.01.R", "mixamorig:RightHandPinky1"]),
]


# ---------------------------------------------------------------------------
# Built-in bone map presets
# ---------------------------------------------------------------------------
#
# Permanent presets that ship with the addon for common rigs.  Unlike user
# presets (stored as JSON in addon preferences), these live in code and cannot
# be overwritten or deleted from the UI.  Add new rigs here as new entries.

# Kimodo SOMA source bone -> Mixamo target bone.  Mixamo prefixes every bone
# with "mixamorig:" but otherwise uses the same names as our SOMA skeleton, so
# we derive the mapping straight from the SOMA hint table to avoid drift.
_MIXAMO_BONE_PAIRS = [
    (src_name, "mixamorig:" + src_name)
    for src_name, _alts in _SOMA_BONE_MAP_HINTS
]


def _build_mixamo_pairs(mode: str) -> list[dict]:
    """Build Mixamo preset bone pairs in the standard preset dict format."""
    return [
        {"src": src, "tgt": tgt, "en": True, "mode": mode}
        for src, tgt in _MIXAMO_BONE_PAIRS
    ]


# Registry of built-in presets: display name -> list of bone-pair dicts.
BUILTIN_PRESETS: "dict[str, list[dict]]" = {
    "Mixamo (Child Of)":      _build_mixamo_pairs("CHILD_OF"),
    "Mixamo (Copy Rotation)": _build_mixamo_pairs("COPY_ROTATION"),
}


def _normalize(name: str) -> str:
    """Lowercase, strip prefix up to ':', remove non-alphanumeric."""
    name = name.lower()
    if ":" in name:
        name = name.split(":")[-1]
    return re.sub(r"[^a-z0-9]", "", name)


def auto_build_mapping(source_arm: bpy.types.Object,
                       target_arm: bpy.types.Object,
                       model: str = "smpl") -> list[tuple[str, str]]:
    """
    Attempt to auto-match bones between source (Kimodo) and target (user) armatures.
    Returns list of (source_bone_name, target_bone_name) pairs.
    """
    hints = _SOMA_BONE_MAP_HINTS[:]
    if model == "smplx":
        hints += _SMPLX_EXTRA_HINTS

    src_bones = {b.name for b in source_arm.data.bones}
    tgt_bones = {b.name: _normalize(b.name) for b in target_arm.data.bones}

    result = []
    for src_name, alternatives in hints:
        if src_name not in src_bones:
            continue
        # Try exact match first
        matched = None
        for tgt_name in target_arm.data.bones.keys():
            if tgt_name == src_name:
                matched = tgt_name
                break
        # Try normalized match
        if not matched:
            src_norm = _normalize(src_name)
            for tgt_name, tgt_norm in tgt_bones.items():
                if src_norm == tgt_norm:
                    matched = tgt_name
                    break
        # Try alternatives
        if not matched:
            for alt in alternatives:
                if alt in tgt_bones:
                    matched = alt
                    break
                alt_norm = _normalize(alt)
                for tgt_name, tgt_norm in tgt_bones.items():
                    if alt_norm == tgt_norm:
                        matched = tgt_name
                        break
                if matched:
                    break
        if matched:
            result.append((src_name, matched))

    return result


# ---------------------------------------------------------------------------
# Constraint setup
# ---------------------------------------------------------------------------

def apply_retargeting_constraints(
    source_arm: bpy.types.Object,
    target_arm: bpy.types.Object,
    bone_pairs: "list[tuple[str, str, bool, str]]",  # (src, tgt, enabled, mode)
    root_bone: str = "",
) -> "tuple[int, list[str]]":
    """
    Add retargeting constraints to each enabled bone pair.
    Returns (n_applied, [warning_messages]).

    bone_pairs tuples: (source_bone, target_bone, enabled, retarget_mode)
    retarget_mode:  'COPY_ROTATION' | 'COPY_TRANSFORMS' | 'CHILD_OF'
    """
    source_arm.hide_viewport = False
    target_arm.hide_viewport = False

    tgt_pose = target_arm.pose
    applied  = 0
    warnings = []

    for entry in bone_pairs:
        # Accept both 3-tuples (legacy) and 4-tuples (with mode)
        if len(entry) == 4:
            src_name, tgt_name, enabled, mode = entry
        else:
            src_name, tgt_name, enabled = entry
            mode = "COPY_ROTATION"

        if not enabled:
            continue

        tgt_pbone = tgt_pose.bones.get(tgt_name)
        if not tgt_pbone:
            warnings.append(f"Target bone '{tgt_name}' not found — skipped.")
            continue
        if src_name not in source_arm.data.bones:
            warnings.append(f"Source bone '{src_name}' not in Kimodo armature — skipped.")
            continue

        # Clear previous Kimodo constraints on this bone
        for c in list(tgt_pbone.constraints):
            if c.name.startswith(CONSTRAINT_PREFIX):
                tgt_pbone.constraints.remove(c)

        is_root = (tgt_name == root_bone) or (src_name.lower() in ("hips", "pelvis", "root"))

        if mode == "COPY_ROTATION":
            _add_copy_rotation(tgt_pbone, source_arm, src_name, is_root)

        elif mode == "COPY_TRANSFORMS":
            _add_copy_transforms(tgt_pbone, source_arm, src_name)

        elif mode == "CHILD_OF":
            _add_child_of(tgt_pbone, source_arm, src_name)

        else:
            warnings.append(f"Unknown retarget mode '{mode}' for '{tgt_name}' — using Copy Rotation.")
            _add_copy_rotation(tgt_pbone, source_arm, src_name, is_root)

        applied += 1

    return applied, warnings


# ---------------------------------------------------------------------------
# Per-mode helpers
# ---------------------------------------------------------------------------

def _add_copy_rotation(pbone, source_arm, src_name: str, is_root: bool = False) -> None:
    """Copy Rotation in local space; root bone also gets Copy Location."""
    if is_root:
        loc = pbone.constraints.new("COPY_LOCATION")
        loc.name          = CONSTRAINT_PREFIX + "Location"
        loc.target        = source_arm
        loc.subtarget     = src_name
        loc.use_offset    = False

    rot = pbone.constraints.new("COPY_ROTATION")
    rot.name         = CONSTRAINT_PREFIX + "Rotation"
    rot.target       = source_arm
    rot.subtarget    = src_name
    rot.mix_mode     = 'REPLACE'
    rot.owner_space  = 'WORLD'
    rot.target_space = 'WORLD'


def _add_copy_transforms(pbone, source_arm, src_name: str) -> None:
    """Copy Transforms in local space (location + rotation + scale)."""
    ct = pbone.constraints.new("COPY_TRANSFORMS")
    ct.name         = CONSTRAINT_PREFIX + "CopyTransforms"
    ct.target       = source_arm
    ct.subtarget    = src_name
    ct.mix_mode     = 'REPLACE'
    ct.owner_space  = 'LOCAL'
    ct.target_space = 'LOCAL'


def _add_child_of(pbone, source_arm, src_name: str) -> None:
    """
    Child Of constraint with the inverse matrix set automatically.
    Equivalent to clicking 'Set Inverse' in the UI: stores the inverse of
    the source bone's current world matrix so the target bone doesn't jump
    when the constraint activates.
    """
    co = pbone.constraints.new("CHILD_OF")
    co.name           = CONSTRAINT_PREFIX + "ChildOf"
    co.target         = source_arm
    co.subtarget      = src_name
    co.use_location_x = True
    co.use_location_y = True
    co.use_location_z = True
    co.use_rotation_x = True
    co.use_rotation_y = True
    co.use_rotation_z = True
    co.use_scale_x    = False
    co.use_scale_y    = False
    co.use_scale_z    = False

    # Set Inverse: invert the source bone's current world matrix so the
    # target bone stays exactly where it is when the constraint first fires.
    src_pbone = source_arm.pose.bones.get(src_name)
    if src_pbone:
        co.inverse_matrix = (source_arm.matrix_world @ src_pbone.matrix).inverted()
    else:
        co.inverse_matrix = mathutils.Matrix.Identity(4)


def remove_retargeting_constraints(target_arm: bpy.types.Object) -> int:
    """Removes all Kimodo constraints from target armature. Returns count removed."""
    removed = 0
    for pbone in target_arm.pose.bones:
        for c in list(pbone.constraints):
            if c.name.startswith(CONSTRAINT_PREFIX):
                pbone.constraints.remove(c)
                removed += 1
    return removed


# ---------------------------------------------------------------------------
# Baking
# ---------------------------------------------------------------------------

def bake_retargeted_animation(
    target_arm: bpy.types.Object,
    frame_start: int,
    frame_end: int,
) -> bool:
    """
    Bakes the driven (constraint) animation into actual keyframes,
    then removes the retargeting constraints.
    Returns True on success.
    """
    try:
        # Select only target armature
        bpy.ops.object.select_all(action='DESELECT')
        target_arm.select_set(True)
        bpy.context.view_layer.objects.active = target_arm

        # Enter pose mode for baking
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='SELECT')

        bpy.ops.nla.bake(
            frame_start=frame_start,
            frame_end=frame_end,
            only_selected=False,
            visual_keying=True,
            clear_constraints=True,
            clear_parents=False,
            use_current_action=True,
            bake_types={'POSE'},
        )

        bpy.ops.object.mode_set(mode='OBJECT')
        return True

    except Exception as e:
        print(f"[Kimodo] Bake error: {e}")
        return False


# ---------------------------------------------------------------------------
# Preset save / load
# ---------------------------------------------------------------------------

def is_builtin_preset(preset_name: str) -> bool:
    """True if the name refers to a permanent, ship-with-the-addon preset."""
    return preset_name in BUILTIN_PRESETS


def save_preset(prefs, preset_name: str, bone_pairs: list[dict]) -> None:
    """Save bone mapping to addon preferences. Built-ins are never written."""
    import json
    if is_builtin_preset(preset_name):
        return
    try:
        presets = json.loads(prefs.saved_presets)
    except Exception:
        presets = {}
    presets[preset_name] = bone_pairs
    prefs.saved_presets = json.dumps(presets)


def load_preset(prefs, preset_name: str) -> list[dict] | None:
    """Load bone mapping. Checks built-ins first, then user presets.
    Returns None if not found."""
    import json
    if preset_name in BUILTIN_PRESETS:
        # Return a copy so callers can't mutate the registry.
        return [dict(p) for p in BUILTIN_PRESETS[preset_name]]
    try:
        presets = json.loads(prefs.saved_presets)
        return presets.get(preset_name)
    except Exception:
        return None


def list_presets(prefs) -> list[str]:
    """Returns preset names: built-ins first, then user presets."""
    import json
    try:
        user_names = list(json.loads(prefs.saved_presets).keys())
    except Exception:
        user_names = []
    # Built-ins always listed first; user presets can't shadow a built-in name.
    user_names = [n for n in user_names if n not in BUILTIN_PRESETS]
    return list(BUILTIN_PRESETS.keys()) + user_names
