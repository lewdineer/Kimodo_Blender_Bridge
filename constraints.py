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
) -> list[list[float]]:
    """Extract SOMASkeleton30 local joint rotations from a posed armature.

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
        parent_idx = SOMA_JOINT_PARENTS[i] if i < len(SOMA_JOINT_PARENTS) else -1
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
        eff = bones.get(EFFECTOR_BONE[effector_type])
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
) -> list[dict]:
    """
    Convert Blender constraint items to Kimodo constraints JSON list.

    Parameters
    ----------
    constraint_items : iterable of KIMODO_ConstraintItem PropertyGroup entries
    scene            : bpy.context.scene
    kimodo_fps       : Kimodo's motion FPS (default 30)
    auto_canonicalize: subtract XZ of earliest root waypoint so it lands at (0,0)

    Returns
    -------
    list of dicts ready to be json.dumps()-ed and sent to Kimodo
    """
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
                        jrot = get_armature_joint_rots(obj, SOMA_JOINT_ORDER)
                        local_joints_rot.append(jrot)
                        pos2d = [pos3d[0], pos3d[2]]
                        smooth_root_2d.append(apply_offset_2d(pos2d))
                    else:
                        # Empty used for fullbody — just root position, identity pose.
                        pos3d = blender_to_kimodo_pos(obj.location)
                        root_positions.append(apply_offset_3d(pos3d))
                        pos2d = [pos3d[0], pos3d[2]]
                        smooth_root_2d.append(apply_offset_2d(pos2d))
                        local_joints_rot.append([[0.0, 0.0, 0.0]] * len(SOMA_JOINT_ORDER))

                elif ctype in ('left_hand', 'right_hand', 'left_foot', 'right_foot'):
                    # Kimodo derives the effector target from FK on (root_positions,
                    # local_joints_rot) — root_positions is the HIPS, NOT the hand.
                    eff_idx = EFFECTOR_IDX[ctype]

                    if obj.type == 'ARMATURE':
                        # Armature marker: read hips + full pose like fullbody.
                        # FK of this pose naturally places the effector wherever
                        # the user posed it.
                        _evaluate_frame(scene, ci.frame)
                        hips_pos = get_root_position(obj)
                        jrot = get_armature_joint_rots(obj, SOMA_JOINT_ORDER)
                    else:
                        # Empty marker: its location is the *target* end-effector
                        # position. Pick hips such that a T-pose places the
                        # effector exactly there, and use a T-pose (all zero
                        # rotations) for the body. The Empty's rotation is used
                        # as the effector's orientation.
                        target = blender_to_kimodo_pos(obj.location)
                        ox, oy, oz = get_effector_tpose_offset(scene, ctype)
                        hips_pos = [target[0] - ox, target[1] - oy, target[2] - oz]
                        jrot = [[0.0, 0.0, 0.0] for _ in SOMA_JOINT_ORDER]
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


# ---------------------------------------------------------------------------
# MMCP constraint builder (remote motionmcp-kimodo server)
# ---------------------------------------------------------------------------
#
# The remote server does not accept the local bridge's constraint JSON.
# It wants MMCP constraints in the same Y-up metres this module already
# produces: root_path, effector_target, and pose_keyframe.

def _axis_angle_to_xyzw(aa: list[float]) -> list[float]:
    """Kimodo axis-angle 3-vector (axis * angle, radians) → quaternion xyzw."""
    ax, ay, az = aa
    angle = math.sqrt(ax * ax + ay * ay + az * az)
    if angle < 1e-8:
        return [0.0, 0.0, 0.0, 1.0]
    half = angle * 0.5
    scale = math.sin(half) / angle
    return [ax * scale, ay * scale, az * scale, math.cos(half)]


def _blender_quat_to_kimodo_xyzw(q: mathutils.Quaternion) -> list[float]:
    """Blender world quaternion → Kimodo Y-up quaternion xyzw."""
    # Same basis change as blender_to_kimodo_pos: (x, y, z) → (x, z, -y).
    basis = mathutils.Matrix(((1, 0, 0), (0, 0, 1), (0, -1, 0)))
    rotated = basis @ q.to_matrix().to_3x3() @ basis.transposed()
    out = rotated.to_quaternion()
    return [out.x, out.y, out.z, out.w]


def _bone_global_quat_xyzw(armature_obj: bpy.types.Object, bone_name: str):
    """Pose bone's world rotation in Kimodo space, or None if the bone is missing."""
    pb = armature_obj.pose.bones.get(bone_name)
    if pb is None:
        return None
    world = (armature_obj.matrix_world @ pb.matrix).to_3x3()
    return _blender_quat_to_kimodo_xyzw(world.to_quaternion())


def _dedupe_ascending(frames: list[int], *columns: list) -> tuple:
    """Keep the last sample at each frame and return strictly ascending frames."""
    order = sorted(range(len(frames)), key=lambda i: frames[i])
    kept_frames: list[int] = []
    kept_cols: list[list] = [[] for _ in columns]
    for i in order:
        if kept_frames and frames[i] <= kept_frames[-1]:
            if frames[i] == kept_frames[-1]:
                for col, src in zip(kept_cols, columns):
                    col[-1] = src[i]
            continue
        kept_frames.append(frames[i])
        for col, src in zip(kept_cols, columns):
            col.append(src[i])
    return (kept_frames, *kept_cols)


def build_mmcp_constraints(
    constraint_items,
    scene: bpy.types.Scene,
    kimodo_fps: float = 30.0,
    auto_canonicalize: bool = True,
    scene_start_override: "int | None" = None,
    skeleton_joints: "list[str] | None" = None,
    root_joint: str = "Hips",
) -> list[dict]:
    """
    Convert Blender constraint markers into MMCP constraint dicts.

    Positions are Kimodo/MMCP Y-up metres (see blender_to_kimodo_pos).
    ``skeleton_joints``, when given, is the server's canonical joint list;
    a marker whose joint is not in that list is skipped with an error so
    the caller can warn instead of sending a request the server will reject.
    """
    blender_fps = scene.render.fps / scene.render.fps_base
    scene_start = scene.frame_start if scene_start_override is None else scene_start_override
    known = set(skeleton_joints) if skeleton_joints else None

    items = sorted(
        [ci for ci in constraint_items if ci.enabled and ci.marker_object],
        key=lambda ci: ci.frame,
    )
    if not items:
        return []

    saved_frame = scene.frame_current
    try:
        origin_offset = [0.0, 0.0]
        if auto_canonicalize:
            for ci in items:
                if ci.constraint_type == 'root2d':
                    origin_offset = blender_to_kimodo_2d(ci.marker_object.location)
                    break
                if ci.constraint_type == 'fullbody':
                    _evaluate_frame(scene, ci.frame)
                    pos3d = get_root_position(ci.marker_object)
                    origin_offset = [pos3d[0], pos3d[2]]
                    break

        def apply_offset_2d(pos2d):
            return [pos2d[0] - origin_offset[0], pos2d[1] - origin_offset[1]]

        def apply_offset_3d(pos3d):
            return [pos3d[0] - origin_offset[0], pos3d[1], pos3d[2] - origin_offset[1]]

        def kframe(blender_frame: int) -> int:
            return blender_frame_to_kimodo(
                blender_frame, scene_start, blender_fps, kimodo_fps,
            )

        out: list[dict] = []

        root_frames: list[int] = []
        root_xz: list[list[float]] = []
        root_heading: list[float] = []
        root_heading_complete = True

        effector_samples: dict[str, dict] = {}

        for ci in items:
            ctype = ci.constraint_type
            obj = ci.marker_object
            frame = kframe(ci.frame)

            if ctype == 'root2d':
                root_frames.append(frame)
                root_xz.append(apply_offset_2d(blender_to_kimodo_2d(obj.location)))
                if ci.include_heading:
                    root_heading.append(float(ci.heading_angle))
                else:
                    root_heading_complete = False
                continue

            if ctype == 'fullbody':
                if obj.type == 'ARMATURE':
                    _evaluate_frame(scene, ci.frame)
                    root_pos = apply_offset_3d(get_root_position(obj))
                    rotations = get_armature_joint_rots(obj, SOMA_JOINT_ORDER)
                    joint_rotations = {}
                    for name, aa in zip(SOMA_JOINT_ORDER, rotations):
                        if known is not None and name not in known:
                            continue
                        if obj.pose.bones.get(name) is None:
                            continue
                        joint_rotations[name] = _axis_angle_to_xyzw(aa)
                    if not joint_rotations:
                        joint_rotations[root_joint] = [0.0, 0.0, 0.0, 1.0]
                else:
                    root_pos = apply_offset_3d(blender_to_kimodo_pos(obj.location))
                    joint_rotations = {root_joint: [0.0, 0.0, 0.0, 1.0]}
                if known is not None and root_joint not in known and root_joint in joint_rotations:
                    # Identity on an unknown root name would be rejected.
                    if len(joint_rotations) == 1:
                        raise ValueError(
                            f"Server skeleton has no joint '{root_joint}' for a full-body pose"
                        )
                    joint_rotations.pop(root_joint, None)
                # fill_mode "rest" asks the server for a full-body constraint.
                # "generate" collapses a posed body down to end effectors.
                out.append({
                    "type": "pose_keyframe",
                    "frame": frame,
                    "joint_rotations": joint_rotations,
                    "root_position": root_pos,
                    "fill_mode": "rest",
                })
                continue

            if ctype in EFFECTOR_BONE:
                joint = EFFECTOR_BONE[ctype]
                if known is not None and joint not in known:
                    raise ValueError(
                        f"Server skeleton has no joint '{joint}' for a {ctype} constraint"
                    )
                if obj.type == 'ARMATURE':
                    _evaluate_frame(scene, ci.frame)
                    pos = get_bone_world_position(obj, joint)
                    if pos is None:
                        pos = blender_to_kimodo_pos(obj.location)
                    rotation = _bone_global_quat_xyzw(obj, joint)
                else:
                    pos = blender_to_kimodo_pos(obj.location)
                    rotation = None
                    if obj.matrix_world.to_quaternion().angle > 1e-3:
                        rotation = _blender_quat_to_kimodo_xyzw(
                            obj.matrix_world.to_quaternion()
                        )
                bucket = effector_samples.setdefault(
                    joint, {"frames": [], "positions": [], "rotations": []},
                )
                bucket["frames"].append(frame)
                bucket["positions"].append(apply_offset_3d(pos))
                bucket["rotations"].append(rotation)

        if root_frames:
            if root_heading_complete and len(root_heading) == len(root_frames):
                frames, xz, heading = _dedupe_ascending(root_frames, root_xz, root_heading)
                block = {
                    "type": "root_path",
                    "frames": frames,
                    "positions_xz": xz,
                    "heading_radians": heading,
                }
            else:
                frames, xz = _dedupe_ascending(root_frames, root_xz)
                block = {
                    "type": "root_path",
                    "frames": frames,
                    "positions_xz": xz,
                }
            if frames:
                out.append(block)

        for joint, bucket in effector_samples.items():
            rotations = bucket["rotations"]
            if all(r is not None for r in rotations):
                frames, positions, rots = _dedupe_ascending(
                    bucket["frames"], bucket["positions"], rotations,
                )
                block = {
                    "type": "effector_target",
                    "joint": joint,
                    "frames": frames,
                    "positions": positions,
                    "rotations": rots,
                }
            else:
                frames, positions = _dedupe_ascending(
                    bucket["frames"], bucket["positions"],
                )
                block = {
                    "type": "effector_target",
                    "joint": joint,
                    "frames": frames,
                    "positions": positions,
                }
            if frames:
                out.append(block)

        return out
    finally:
        scene.frame_set(saved_frame)
        bpy.context.view_layer.update()
