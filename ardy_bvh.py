"""
BVH writer for ARDY motion output.

ARDY has no BVH exporter — ``ardy/exports/`` only ships a MuJoCo qpos writer,
and ``ardy/skeleton/bvh.py`` is a *parser*.  The add-on's entire import path is
``bpy.ops.import_anim.bvh``, so the bridge writes the BVH itself.

Everything needed is already on the ARDY skeleton object:

    skeleton.neutral_joints    [J, 3]  rest joint positions  -> BVH OFFSET
    skeleton.joint_parents     [J]     hierarchy             -> nesting
    skeleton.bone_order_names  [J]     names                 -> ROOT / JOINT
    skeleton.root_idx                  which joint is the root

plus ``local_rot_mats`` [T, J, 3, 3] and ``root_positions`` [T, 3] from a
generation.

Conventions
-----------
Units      ARDY works in metres; BVH is conventionally centimetres and the
           add-on imports with ``global_scale=0.01`` (see ``operators.py``
           ``KIMODO_OT_ImportBVH``).  Positions are therefore written x100.
Axes       ARDY motion space is already Y-up, which is what the importer
           expects (``axis_forward='-Z'``, ``axis_up='Y'``).  No conversion.
Channels   Root gets 6 channels, every other joint 3, in the order
           ``Zrotation Xrotation Yrotation`` — i.e. R = Rz @ Rx @ Ry, which is
           scipy's intrinsic ``"ZXY"`` Euler convention.

This module runs inside the ARDY venv (numpy + scipy are ARDY dependencies),
never inside Blender.
"""

import numpy as np
from scipy.spatial.transform import Rotation

# Metres -> centimetres. Matches the importer's global_scale=0.01.
_SCALE = 100.0

# End sites for leaf joints get a stub at least this long (cm) so Blender does
# not collapse the final bone to zero length and drop it on import.
_MIN_END_SITE = 1.0


def _to_numpy(x):
    """Accept a torch tensor or anything array-like and return a numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x, dtype=np.float64)


def _children_of(parents: "list[int]") -> "dict[int, list[int]]":
    kids: "dict[int, list[int]]" = {}
    for idx, parent in enumerate(parents):
        if parent is not None and parent >= 0:
            kids.setdefault(parent, []).append(idx)
    return kids


def _depth_first_order(root: int, kids: "dict[int, list[int]]") -> "list[int]":
    """Joint indices in the order BVH nests them (and channels are written)."""
    order: "list[int]" = []
    stack = [root]
    while stack:
        idx = stack.pop()
        order.append(idx)
        # reversed() so children keep their natural left-to-right order
        for child in reversed(kids.get(idx, [])):
            stack.append(child)
    return order


def rest_tables(skeleton):
    """(names, parents, offsets_m, root_idx) for an ARDY skeleton, in metres.

    Bone-local offsets: each joint's rest position relative to its parent, which
    is what both a BVH OFFSET and a Blender edit bone want. Metres is ARDY's own
    unit — ``_skeleton_tables`` scales to centimetres for BVH, while the live
    stream builds its armature straight from these and keeps the scene metric.
    """
    names = list(skeleton.bone_order_names)
    parents = [int(p) for p in _to_numpy(skeleton.joint_parents).reshape(-1)]
    neutral = _to_numpy(skeleton.neutral_joints).reshape(len(names), 3)

    offsets = np.zeros_like(neutral)
    for idx, parent in enumerate(parents):
        offsets[idx] = neutral[idx] if parent < 0 else neutral[idx] - neutral[parent]
    return names, parents, offsets, int(skeleton.root_idx)


def _skeleton_tables(skeleton):
    """Pull (names, parents, offsets_cm, root_idx) off an ARDY skeleton."""
    names, parents, offsets, root_idx = rest_tables(skeleton)
    return names, parents, offsets * _SCALE, root_idx


def _write_hierarchy(out, idx, names, parents, offsets, kids, root_idx, depth):
    """Recursively emit one joint's HIERARCHY block."""
    pad = "\t" * depth
    if idx == root_idx:
        out.append(f"{pad}ROOT {names[idx]}")
    else:
        out.append(f"{pad}JOINT {names[idx]}")
    out.append(f"{pad}{{")

    ox, oy, oz = offsets[idx]
    out.append(f"{pad}\tOFFSET {ox:.6f} {oy:.6f} {oz:.6f}")
    if idx == root_idx:
        out.append(f"{pad}\tCHANNELS 6 Xposition Yposition Zposition "
                   f"Zrotation Xrotation Yrotation")
    else:
        out.append(f"{pad}\tCHANNELS 3 Zrotation Xrotation Yrotation")

    children = kids.get(idx, [])
    if children:
        for child in children:
            _write_hierarchy(out, child, names, parents, offsets, kids,
                             root_idx, depth + 1)
    else:
        # Leaf: BVH needs an End Site or the last bone has no length. Extend in
        # the same direction the joint itself points away from its parent.
        stub = offsets[idx].copy()
        length = float(np.linalg.norm(stub))
        if length < 1e-6:
            stub = np.array([0.0, _MIN_END_SITE, 0.0])
        elif length < _MIN_END_SITE:
            stub = stub / length * _MIN_END_SITE
        out.append(f"{pad}\tEnd Site")
        out.append(f"{pad}\t{{")
        out.append(f"{pad}\t\tOFFSET {stub[0]:.6f} {stub[1]:.6f} {stub[2]:.6f}")
        out.append(f"{pad}\t}}")

    out.append(f"{pad}}}")


def save_ardy_bvh(
    path: str,
    local_rot_mats,
    root_positions,
    skeleton,
    fps: float,
) -> str:
    """Write an ARDY motion to *path* as BVH. Returns *path*.

    Parameters
    ----------
    path            destination .bvh file
    local_rot_mats  [T, J, 3, 3] per-frame local joint rotations
    root_positions  [T, 3] per-frame root translation, in metres
    skeleton        an ``ardy.skeleton.SkeletonBase`` instance
    fps             frame rate the motion was generated at (20 for Core,
                    25 for G1 — ARDY is not 30 FPS like Kimodo)
    """
    names, parents, offsets, root_idx = _skeleton_tables(skeleton)
    njoints = len(names)

    rots = _to_numpy(local_rot_mats)
    if rots.ndim == 5:          # [B, T, J, 3, 3] — take the first sample
        rots = rots[0]
    if rots.ndim != 4 or rots.shape[1:] != (njoints, 3, 3):
        raise ValueError(
            f"local_rot_mats must be [T, {njoints}, 3, 3]; got {rots.shape}"
        )

    root = _to_numpy(root_positions)
    if root.ndim == 3:          # [B, T, 3]
        root = root[0]
    if root.ndim != 2 or root.shape[1] != 3:
        raise ValueError(f"root_positions must be [T, 3]; got {root.shape}")

    nframes = min(rots.shape[0], root.shape[0])
    rots, root = rots[:nframes], root[:nframes] * _SCALE

    kids = _children_of(parents)
    order = _depth_first_order(root_idx, kids)
    if len(order) != njoints:
        # A joint unreachable from the root would silently lose its channels.
        missing = [names[i] for i in range(njoints) if i not in set(order)]
        raise ValueError(
            f"skeleton '{getattr(skeleton, 'name', '?')}' is not a single tree; "
            f"unreachable joints: {missing}"
        )

    # Euler angles for every joint at once: scipy's uppercase "ZXY" is the
    # intrinsic convention, giving R = Rz @ Rx @ Ry — exactly how a BVH reader
    # composes "Zrotation Xrotation Yrotation".
    eulers = Rotation.from_matrix(
        rots.reshape(-1, 3, 3)
    ).as_euler("ZXY", degrees=True).reshape(nframes, njoints, 3)

    lines: "list[str]" = ["HIERARCHY"]
    _write_hierarchy(lines, root_idx, names, parents, offsets, kids, root_idx, 0)
    lines.append("MOTION")
    lines.append(f"Frames: {nframes}")
    lines.append(f"Frame Time: {1.0 / float(fps):.8f}")

    for f in range(nframes):
        values: "list[str]" = []
        for idx in order:
            if idx == root_idx:
                values += [f"{v:.6f}" for v in root[f]]
            values += [f"{v:.6f}" for v in eulers[f, idx]]
        lines.append(" ".join(values))

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")
    return path


def supports_standard_tpose(skeleton) -> bool:
    """True when ``skeleton.to_standard_tpose()`` can actually run.

    Only skeletons shipping ``standard_t_pose_global_offsets_rots.p`` register a
    ``global_rot_offsets`` buffer. somaskel77 has it; cskel27 (the Core
    skeleton every released ARDY checkpoint uses today) does not, so the
    add-on's "Use Standard T-Pose" option has nothing to convert into.
    """
    return getattr(skeleton, "global_rot_offsets", None) is not None
