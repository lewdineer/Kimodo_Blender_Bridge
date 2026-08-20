"""
Kimodo Blender Bridge — Constraints
====================================
Translates Blender scene objects (Empties, posed Armatures) into the
Kimodo constraints JSON format that gets sent alongside the text prompt.

Coordinate space conversion
----------------------------
Blender world space (Z-up)  →  Kimodo motion space (Y-up)
  X  (right)    →  X  (right)     same
  Y  (forward)  →  -Z (forward)   Y → -Z  (sign flip — see BVH import axis_forward='-Z')
  Z  (up)       →  Y  (up)        Z →  Y

The sign flip on Y is required because the BVH importer maps Kimodo +Z
back to Blender -Y (axis_forward='-Z' in operators.py).  Negating the
Blender-Y → Kimodo-Z conversion here keeps constraint positions consistent
with the imported animation.

All positions in meters (Blender scenes should use metric units).

Canonicalization
-----------------
Kimodo expects the root to start near (0,0) in XZ.  We auto-offset all
constraint positions by subtracting the XZ of the *earliest root2d* or
*earliest fullbody root* so the user can author constraints anywhere in
the scene.  If no frame-0 constraint exists the model still works — it
just starts from its learned distribution.

Frame mapping
-------------
  kimodo_frame = round((blender_frame - scene.frame_start) * kimodo_fps / blender_fps)
The Kimodo FPS defaults to 30; override it in the constraint settings.
"""

import bpy
import math
import json
import mathutils
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bone names for SOMASkeleton30, exactly matching Kimodo's bone_order_names_with_parents.
# The INDEX in this list is the slot index that Kimodo reads from local_joints_rot.
# All names exist in somaskel77 (the BVH skeleton) so pose_bones.get() will find them.
SOMA_JOINT_ORDER = [
    "Hips",              # 0  — root
    "Spine1",            # 1
    "Spine2",            # 2
    "Chest",             # 3
    "Neck1",             # 4
    "Neck2",             # 5
    "Head",              # 6
    "Jaw",               # 7
    "LeftEye",           # 8
    "RightEye",          # 9
    "LeftShoulder",      # 10
    "LeftArm",           # 11
    "LeftForeArm",       # 12
    "LeftHand",          # 13
    "LeftHandThumbEnd",  # 14
    "LeftHandMiddleEnd", # 15
    "RightShoulder",     # 16
    "RightArm",          # 17
    "RightForeArm",      # 18
    "RightHand",         # 19
    "RightHandThumbEnd", # 20
    "RightHandMiddleEnd",# 21
    "LeftLeg",           # 22 — hip (not "LeftUpLeg")
    "LeftShin",          # 23 — knee (not "LeftLeg")
    "LeftFoot",          # 24
    "LeftToeBase",       # 25
    "RightLeg",          # 26 — hip
    "RightShin",         # 27 — knee
    "RightFoot",         # 28
    "RightToeBase",      # 29
]

# Parent index for each joint in SOMA_JOINT_ORDER (mirrors SOMASkeleton30.bone_order_names_with_parents).
# -1 = root (no parent).  Used to compute local rotations from world rotations.
SOMA_JOINT_PARENTS = [
    -1,  # 0  Hips        (root)
     0,  # 1  Spine1
     1,  # 2  Spine2
     2,  # 3  Chest
     3,  # 4  Neck1
     4,  # 5  Neck2
     5,  # 6  Head
     6,  # 7  Jaw
     6,  # 8  LeftEye
     6,  # 9  RightEye
     3,  # 10 LeftShoulder
    10,  # 11 LeftArm
    11,  # 12 LeftForeArm
    12,  # 13 LeftHand
    13,  # 14 LeftHandThumbEnd
    13,  # 15 LeftHandMiddleEnd
     3,  # 16 RightShoulder
    16,  # 17 RightArm
    17,  # 18 RightForeArm
    18,  # 19 RightHand
    19,  # 20 RightHandThumbEnd
    19,  # 21 RightHandMiddleEnd
     0,  # 22 LeftLeg
    22,  # 23 LeftShin
    23,  # 24 LeftFoot
    24,  # 25 LeftToeBase
     0,  # 26 RightLeg
    26,  # 27 RightShin
    27,  # 28 RightFoot
    28,  # 29 RightToeBase
]

# ---------------------------------------------------------------------------
# ARDY CoreSkeleton27 (cskel27)
# ---------------------------------------------------------------------------
# Every released ARDY checkpoint uses this skeleton — the SOMA variant is
# announced but not published, so ardy/model/registry.py ships "core" and "g1"
# only. Order and parents mirror CoreSkeleton27.bone_order_names_with_parents.
# The names are Mixamo-style, which makes them easier to auto-map onto user
# rigs than SOMA's, and unlike SOMA30 there are no jaw/eye joints.

CORE_JOINT_ORDER = [
    "Hips",              # 0  — root
    "Spine",             # 1
    "Spine1",            # 2
    "Spine2",            # 3
    "Spine3",            # 4
    "Neck",              # 5
    "Head",              # 6
    "RightShoulder",     # 7
    "RightArm",          # 8
    "RightForeArm",      # 9
    "RightHand",         # 10
    "RightHandEnd",      # 11
    "RightHandThumb1",   # 12
    "LeftShoulder",      # 13
    "LeftArm",           # 14
    "LeftForeArm",       # 15
    "LeftHand",          # 16
    "LeftHandEnd",       # 17
    "LeftHandThumb1",    # 18
    "RightUpLeg",        # 19 — hip
    "RightLeg",          # 20 — knee
    "RightFoot",         # 21
    "RightToeBase",      # 22
    "LeftUpLeg",         # 23 — hip
    "LeftLeg",           # 24 — knee
    "LeftFoot",          # 25
    "LeftToeBase",       # 26
]

CORE_JOINT_PARENTS = [
    -1,  # 0  Hips            (root)
     0,  # 1  Spine
     1,  # 2  Spine1
     2,  # 3  Spine2
     3,  # 4  Spine3
     4,  # 5  Neck
     5,  # 6  Head
     4,  # 7  RightShoulder
     7,  # 8  RightArm
     8,  # 9  RightForeArm
     9,  # 10 RightHand
    10,  # 11 RightHandEnd
    10,  # 12 RightHandThumb1
     4,  # 13 LeftShoulder
    13,  # 14 LeftArm
    14,  # 15 LeftForeArm
    15,  # 16 LeftHand
    16,  # 17 LeftHandEnd
    16,  # 18 LeftHandThumb1
     0,  # 19 RightUpLeg
    19,  # 20 RightLeg
    20,  # 21 RightFoot
    21,  # 22 RightToeBase
     0,  # 23 LeftUpLeg
    23,  # 24 LeftLeg
    24,  # 25 LeftFoot
    25,  # 26 LeftToeBase
]

CORE_EFFECTOR_BONE = {
    'left_hand':  'LeftHand',
    'right_hand': 'RightHand',
    'left_foot':  'LeftFoot',
    'right_foot': 'RightFoot',
}
CORE_EFFECTOR_IDX = {
    'left_hand': 16, 'right_hand': 10, 'left_foot': 25, 'right_foot': 21,
}


# Coordinate-change matrix: v_kimodo = _M_BK @ v_blender
# Derived from blender_to_kimodo_pos: (Bx, By, Bz) → (Bx, Bz, -By)
# M_BK = [[1,0,0],[0,0,1],[0,-1,0]] — orthogonal, det=1
# For rotation matrices: R_kimodo = M_BK @ R_blender @ M_BK.T

# Bone names for each end-effector type (somaskel77 / SOMASkeleton30).
EFFECTOR_BONE = {
    'left_hand':  'LeftHand',
    'right_hand': 'RightHand',
    'left_foot':  'LeftFoot',
    'right_foot': 'RightFoot',
}
# Index of each end-effector in SOMA_JOINT_ORDER.
EFFECTOR_IDX = {
    'left_hand': 13, 'right_hand': 19, 'left_foot': 24, 'right_foot': 28,
}
# Fallback rest-pose offset from Hips to each effector in Kimodo Y-up meters
# (used when no source_armature is available to read true offsets from).
# Values approximate an adult SOMA T-pose: arms out sideways at shoulder height,
# feet straight below the hips at ground level (hips ~0.9 m up).
DEFAULT_TPOSE_OFFSETS = {
    'left_hand':  ( 0.75,  0.45, 0.0),
    'right_hand': (-0.75,  0.45, 0.0),
    'left_foot':  ( 0.10, -0.90, 0.0),
    'right_foot': (-0.10, -0.90, 0.0),
}


# ---------------------------------------------------------------------------
# Skeleton registry
# ---------------------------------------------------------------------------
# Constraint JSON is skeleton-shaped: `local_joints_rot` is indexed by the
# model's own joint order, so the tables above must match whichever backend is
# loaded. Everything else in this module — the coordinate conversion, the pose
# extraction, the JSON layout — is skeleton-agnostic and shared.

SKELETONS = {
    'soma': {
        "name":          "soma",
        "joint_order":   SOMA_JOINT_ORDER,
        "joint_parents": SOMA_JOINT_PARENTS,
        "effector_bone": EFFECTOR_BONE,
        "effector_idx":  EFFECTOR_IDX,
    },
    'core': {
        "name":          "core",
        "joint_order":   CORE_JOINT_ORDER,
        "joint_parents": CORE_JOINT_PARENTS,
        "effector_bone": CORE_EFFECTOR_BONE,
        "effector_idx":  CORE_EFFECTOR_IDX,
    },
}

# Names a bridge may report in its "ready" message, mapped onto the tables.
_SKELETON_ALIASES = {
    "somaskel30": 'soma', "somaskel77": 'soma', "soma": 'soma', "smpl": 'soma',
    "cskel27": 'core', "cskel29": 'core', "core": 'core',
}


def get_skeleton(name: "str | None") -> dict:
    """Return the joint tables for a skeleton name, defaulting to SOMA.

    Accepts what a bridge reports ("cskel27", "somaskel30") as well as the
    short keys used in the scene properties. An unknown name falls back to
    SOMA so an older .blend or a future model never hard-fails here.
    """
    key = _SKELETON_ALIASES.get((name or "").strip().lower(), 'soma')
    return SKELETONS[key]


def scene_skeleton(scene) -> dict:
    """Joint tables for whichever backend the scene is currently using."""
    return get_skeleton(getattr(getattr(scene, "kimodo", None),
                                "active_skeleton", "soma"))


# ---------------------------------------------------------------------------
# Coordinate conversion helpers
# ---------------------------------------------------------------------------

def blender_to_kimodo_pos(v: mathutils.Vector) -> list[float]:
    """Convert Blender world position (Z-up) to Kimodo Y-up position [x, y, z]."""
    return [v.x, v.z, -v.y]   # X same; Z→Y (up); -Y→Z (sign flip for forward axis)


def blender_to_kimodo_2d(v: mathutils.Vector) -> list[float]:
    """Extract 2D ground-plane position [x, z_kimodo] from Blender world pos."""
    return [v.x, -v.y]   # Blender X→Kimodo X; Blender -Y→Kimodo Z (sign flip)


def quat_to_axis_angle_vec(q: mathutils.Quaternion) -> list[float]:
    """Convert Blender Quaternion to axis-angle 3-vector [ax, ay, az] * angle."""
    q = q.normalized()
    angle = 2.0 * math.acos(max(-1.0, min(1.0, q.w)))
    s = math.sqrt(max(0.0, 1.0 - q.w * q.w))
    if s < 1e-6:
        return [0.0, 0.0, 0.0]
    # Convert axis from Blender to Kimodo space too
    ax = q.x / s
    ay = q.z / s    # Blender Z → Kimodo Y
    az = -q.y / s   # Blender -Y → Kimodo Z (sign flip matches position conversion)
    return [ax * angle, ay * angle, az * angle]


def euler_to_axis_angle_vec(e: mathutils.Euler) -> list[float]:
    """Convert Blender Euler to axis-angle 3-vector."""
    return quat_to_axis_angle_vec(e.to_quaternion())


def heading_from_angle(angle_rad: float) -> list[float]:
    """Kimodo expects heading as [cos(θ), sin(θ)]."""
    return [math.cos(angle_rad), math.sin(angle_rad)]


# ---------------------------------------------------------------------------
# Frame conversion
# ---------------------------------------------------------------------------

def blender_frame_to_kimodo(
    blender_frame: int,
    scene_start: int,
    blender_fps: float,
    kimodo_fps: float = 30.0,
) -> int:
    """Convert a Blender frame number to a 0-based Kimodo frame index."""
    elapsed_sec = (blender_frame - scene_start) / blender_fps
    return max(0, round(elapsed_sec * kimodo_fps))


# ---------------------------------------------------------------------------
# Pose reading helpers
# ---------------------------------------------------------------------------

def _rot3_to_axis_angle(m: mathutils.Matrix) -> list[float]:
    """3×3 rotation matrix (already in Kimodo Y-up space) → axis-angle 3-vector."""
    q = m.to_quaternion().normalized()
    angle = 2.0 * math.acos(max(-1.0, min(1.0, q.w)))
    s = math.sqrt(max(0.0, 1.0 - q.w * q.w))
    if s < 1e-6:
        return [0.0, 0.0, 0.0]
    return [q.x / s * angle, q.y / s * angle, q.z / s * angle]


def get_armature_joint_rots(
    armature_obj: bpy.types.Object,
    joint_order: list[str],
    joint_parents: "list[int] | None" = None,
) -> list[list[float]]:
    """Extract per-joint local rotations from a posed armature.

    Works for any skeleton: pass the model's ``joint_order`` and the matching
    ``joint_parents`` (defaults to SOMA's for backwards compatibility).

    SMPL/Kimodo defines per-joint local rotations via:
        G[i] = G[parent[i]] @ R[i]
    where G[i] is the bone's cumulative world rotation (relative to the
    canonical T-pose where every G is identity) and R[i] is the local rotation
    stored in ``local_joints_rot``.

    The previous implementation read ``pb.rotation_euler`` directly and assumed
    the resulting matrix was already R[i] in Kimodo's Y-up basis.  That holds
    only when each Blender bone's rest orientation is identity — but Blender's
    BVH importer builds bones whose rest matrices encode the head→tail
    direction, so most bones have non-trivial rests (e.g. shoulders at ~−82°
    around Z, hips flipped 156° around X).  Treating the bone-local Euler as a
    Kimodo-frame rotation produced axis-swapped output on those bones.

    This implementation is rest-orientation-invariant:

      1. For each bone, compute its pose-vs-rest delta in armature (Blender
         Z-up) space:  delta = pose_arm @ rest_arm⁻¹.  This is the physical
         rotation the bone underwent in world space and is independent of how
         Blender chose to orient the bone at rest.
      2. Conjugate the delta into Kimodo's Y-up basis (M_BK · delta · M_BK⁻¹)
         to get G[i] in Kimodo world coordinates.
      3. Recover the local rotation as R[i] = G[parent]⁻¹ @ G[i].
    """
    if joint_parents is None:
        joint_parents = SOMA_JOINT_PARENTS
    pose_bones = armature_obj.pose.bones

    # Blender Z-up → Kimodo Y-up basis change (see module header).
    # Vectors:  (x, y, z)_blender → (x, z, -y)_kimodo
    M_BK = mathutils.Matrix(((1, 0, 0), (0, 0, 1), (0, -1, 0)))
    M_KB = M_BK.transposed()

    # G[i]: bone's pose-relative-to-rest rotation, in Kimodo Y-up basis.
    G_kimodo: dict[str, mathutils.Matrix] = {}
    for name in joint_order:
        pb = pose_bones.get(name)
        if pb is None:
            G_kimodo[name] = mathutils.Matrix.Identity(3)
            continue
        rest_arm = pb.bone.matrix_local.to_3x3()
        pose_arm = pb.matrix.to_3x3()
        # rest is a pure rotation, so .transposed() == .inverted()
        delta_arm = pose_arm @ rest_arm.transposed()
        G_kimodo[name] = M_BK @ delta_arm @ M_KB

    result: list[list[float]] = []
    for i, name in enumerate(joint_order):
        G_i = G_kimodo.get(name, mathutils.Matrix.Identity(3))
        parent_idx = joint_parents[i] if i < len(joint_parents) else -1
        if parent_idx < 0:
            R_local = G_i
        else:
            parent_name = joint_order[parent_idx]
            G_parent = G_kimodo.get(parent_name, mathutils.Matrix.Identity(3))
            R_local = G_parent.transposed() @ G_i
        result.append(_rot3_to_axis_angle(R_local))

    return result


def get_root_position(armature_obj: bpy.types.Object) -> list[float]:
    """Get hip/root world position from armature → Kimodo [x,y,z].

    Reads the Hips bone head in world space so the result is correct
    regardless of where the armature object's origin sits (BVH imports
    always place the armature origin at the scene origin; the Hips bone
    moves via keyframes/pose, not via the object location).
    """
    for bone_name in ("Hips", "hips", "Hip", "pelvis", "Pelvis"):
        pb = armature_obj.pose.bones.get(bone_name)
        if pb:
            world_pos = armature_obj.matrix_world @ pb.head
            return blender_to_kimodo_pos(world_pos)
    return blender_to_kimodo_pos(armature_obj.location)


def get_bone_world_position(
    armature_obj: bpy.types.Object,
    bone_name: str,
) -> list[float] | None:
    """World position of a specific pose bone → Kimodo [x,y,z]."""
    pb = armature_obj.pose.bones.get(bone_name)
    if not pb:
        return None
    # head world position
    world_pos = armature_obj.matrix_world @ pb.head
    return blender_to_kimodo_pos(world_pos)


def get_effector_tpose_offset(
    scene: bpy.types.Scene,
    effector_type: str,
    skel: "dict | None" = None,
) -> tuple[float, float, float]:
    """Rest-pose offset from Hips to the named end-effector, in Kimodo Y-up meters.

    Reads bone rest positions from scene.kimodo.source_armature when present so
    the offset matches the actual character proportions. Falls back to a
    hard-coded adult SOMA T-pose when no source armature is set.
    """
    arm = getattr(getattr(scene, "kimodo", None), "source_armature", None)
    if arm and arm.type == 'ARMATURE':
        bones = arm.data.bones
        hips = (bones.get("Hips") or bones.get("hips")
                or bones.get("Hip") or bones.get("pelvis") or bones.get("Pelvis"))
        eff = bones.get((skel or SKELETONS['soma'])["effector_bone"][effector_type])
        if hips and eff:
            d = eff.head_local - hips.head_local   # Blender Z-up local space
            return (d.x, d.z, -d.y)                # → Kimodo Y-up
    return DEFAULT_TPOSE_OFFSETS[effector_type]


def get_bone_world_rotation(
    armature_obj: bpy.types.Object,
    bone_name: str,
) -> list[float]:
    """World rotation of a specific pose bone → axis-angle 3-vector (Kimodo space)."""
    pb = armature_obj.pose.bones.get(bone_name)
    if not pb:
        return [0.0, 0.0, 0.0]
    world_mat = armature_obj.matrix_world @ pb.matrix
    q = world_mat.to_quaternion()
    return quat_to_axis_angle_vec(q)


# ---------------------------------------------------------------------------
# Constraint JSON builder
# ---------------------------------------------------------------------------

def build_constraints_json(
    constraint_items,           # iterable of KIMODO_ConstraintItem
    scene: bpy.types.Scene,
    kimodo_fps: float = 30.0,
    auto_canonicalize: bool = True,
    scene_start_override: "int | None" = None,
    skeleton: "str | dict | None" = None,
) -> list[dict]:
    """
    Convert Blender constraint items to the model's constraints JSON list.

    The output format is shared by Kimodo and ARDY — both register the same
    constraint types and both accept 'smooth_root_2d' as an alias for
    'root_2d' — so only the joint indexing depends on the backend.

    Parameters
    ----------
    constraint_items : iterable of KIMODO_ConstraintItem PropertyGroup entries
    scene            : bpy.context.scene
    kimodo_fps       : the model's motion FPS (Kimodo 30, ARDY Core 20)
    auto_canonicalize: subtract XZ of earliest root waypoint so it lands at (0,0)
    skeleton         : skeleton name or table dict; defaults to the scene's
                       active backend skeleton

    Returns
    -------
    list of dicts ready to be json.dumps()-ed and sent to Kimodo
    """
    if isinstance(skeleton, dict):
        skel = skeleton
    elif skeleton:
        skel = get_skeleton(skeleton)
    else:
        skel = scene_skeleton(scene)
    joint_order   = skel["joint_order"]
    joint_parents = skel["joint_parents"]

    blender_fps = scene.render.fps / scene.render.fps_base
    scene_start = scene.frame_start if scene_start_override is None else scene_start_override

    # Collect enabled items sorted by frame
    items = sorted(
        [ci for ci in constraint_items if ci.enabled and ci.marker_object],
        key=lambda ci: ci.frame,
    )
    if not items:
        return []

    # Save scene frame — _evaluate_frame mutates it as a side effect.
    saved_frame = scene.frame_current

    try:
        # -----------------------------------------------------------------------
        # Auto-canonicalization: find the earliest XZ root position and use it
        # as the origin offset so the user can author constraints anywhere.
        # -----------------------------------------------------------------------
        origin_offset = [0.0, 0.0]  # [x_offset, z_offset] in Kimodo space
        if auto_canonicalize:
            for ci in items:
                ctype = ci.constraint_type
                if ctype == 'root2d':
                    pos2d = blender_to_kimodo_2d(ci.marker_object.location)
                    origin_offset = pos2d
                    break
                elif ctype == 'fullbody':
                    # Must evaluate the frame before reading the Hips bone position.
                    _evaluate_frame(scene, ci.frame)
                    pos3d = get_root_position(ci.marker_object)
                    origin_offset = [pos3d[0], pos3d[2]]
                    break

        def apply_offset_2d(pos2d):
            return [pos2d[0] - origin_offset[0], pos2d[1] - origin_offset[1]]

        def apply_offset_3d(pos3d):
            return [pos3d[0] - origin_offset[0], pos3d[1], pos3d[2] - origin_offset[1]]

        # -----------------------------------------------------------------------
        # Group ALL items of the same type into one block (not just consecutive).
        # Kimodo expects one block per constraint type with all frame_indices.
        # -----------------------------------------------------------------------
        type_order: list[str] = []
        type_to_items: dict[str, list] = {}
        for ci in items:
            ct = ci.constraint_type
            if ct not in type_to_items:
                type_order.append(ct)
                type_to_items[ct] = []
            type_to_items[ct].append(ci)

        out_constraints = []

        for ctype in type_order:
            group = type_to_items[ctype]
            frame_indices: list[int] = []
            smooth_root_2d: list = []
            root_positions: list = []
            local_joints_rot: list = []
            global_root_heading: list = []

            for ci in group:
                kframe = blender_frame_to_kimodo(ci.frame, scene_start, blender_fps, kimodo_fps)
                frame_indices.append(kframe)
                obj = ci.marker_object

                if ctype == 'root2d':
                    pos2d = blender_to_kimodo_2d(obj.location)
                    smooth_root_2d.append(apply_offset_2d(pos2d))
                    if ci.include_heading:
                        global_root_heading.append(heading_from_angle(ci.heading_angle))

                elif ctype == 'fullbody':
                    if obj.type == 'ARMATURE':
                        # Evaluate the frame FIRST so Hips bone is at the right
                        # world position (BVH armatures animate via bone keyframes,
                        # not via the object location, so position must be sampled
                        # at the correct frame).
                        _evaluate_frame(scene, ci.frame)
                        pos3d = get_root_position(obj)
                        root_positions.append(apply_offset_3d(pos3d))
                        jrot = get_armature_joint_rots(obj, joint_order, joint_parents)
                        local_joints_rot.append(jrot)
                        pos2d = [pos3d[0], pos3d[2]]
                        smooth_root_2d.append(apply_offset_2d(pos2d))
                    else:
                        # Empty used for fullbody — just root position, identity pose.
                        pos3d = blender_to_kimodo_pos(obj.location)
                        root_positions.append(apply_offset_3d(pos3d))
                        pos2d = [pos3d[0], pos3d[2]]
                        smooth_root_2d.append(apply_offset_2d(pos2d))
                        local_joints_rot.append([[0.0, 0.0, 0.0]] * len(joint_order))

                elif ctype in ('left_hand', 'right_hand', 'left_foot', 'right_foot'):
                    # Kimodo derives the effector target from FK on (root_positions,
                    # local_joints_rot) — root_positions is the HIPS, NOT the hand.
                    eff_idx = skel["effector_idx"][ctype]

                    if obj.type == 'ARMATURE':
                        # Armature marker: read hips + full pose like fullbody.
                        # FK of this pose naturally places the effector wherever
                        # the user posed it.
                        _evaluate_frame(scene, ci.frame)
                        hips_pos = get_root_position(obj)
                        jrot = get_armature_joint_rots(obj, joint_order, joint_parents)
                    else:
                        # Empty marker: its location is the *target* end-effector
                        # position. Pick hips such that a T-pose places the
                        # effector exactly there, and use a T-pose (all zero
                        # rotations) for the body. The Empty's rotation is used
                        # as the effector's orientation.
                        target = blender_to_kimodo_pos(obj.location)
                        ox, oy, oz = get_effector_tpose_offset(scene, ctype, skel)
                        hips_pos = [target[0] - ox, target[1] - oy, target[2] - oz]
                        jrot = [[0.0, 0.0, 0.0] for _ in joint_order]
                        # In T-pose all ancestors are identity, so the effector's
                        # local rotation equals its world rotation.
                        jrot[eff_idx] = quat_to_axis_angle_vec(
                            obj.matrix_world.to_quaternion()
                        )

                    pos3d = apply_offset_3d(hips_pos)
                    root_positions.append(pos3d)
                    smooth_root_2d.append([pos3d[0], pos3d[2]])
                    local_joints_rot.append(jrot)

            block: dict[str, Any] = {
                "type": ctype.replace("_", "-"),  # left_hand → left-hand
                "frame_indices": frame_indices,
            }

            if smooth_root_2d:
                block["smooth_root_2d"] = smooth_root_2d
            if root_positions:
                block["root_positions"] = root_positions
            if local_joints_rot:
                block["local_joints_rot"] = local_joints_rot
            if global_root_heading and len(global_root_heading) == len(frame_indices):
                block["global_root_heading"] = global_root_heading

            out_constraints.append(block)

        return out_constraints

    finally:
        # Restore the frame so constraint-building has no visible side effect.
        scene.frame_set(saved_frame)
        bpy.context.view_layer.update()


def _evaluate_frame(scene: bpy.types.Scene, frame: int):
    """Move timeline to a frame and update the depsgraph (for pose sampling)."""
    scene.frame_set(frame)
    bpy.context.view_layer.update()


def constraints_to_json_string(
    constraint_items,
    scene: bpy.types.Scene,
    kimodo_fps: float = 30.0,
    auto_canonicalize: bool = True,
    indent: int = 2,
) -> str:
    """Return the constraints as a formatted JSON string."""
    data = build_constraints_json(constraint_items, scene, kimodo_fps, auto_canonicalize)
    return json.dumps(data, indent=indent)
