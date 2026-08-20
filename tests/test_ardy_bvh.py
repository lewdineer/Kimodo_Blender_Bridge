"""Round-trip check for ardy_bvh.save_ardy_bvh.

ARDY ships no BVH exporter, so the add-on writes its own — and a wrong Euler
convention or offset would only show up as a mangled armature inside Blender.
This builds a stand-in Core27 skeleton, generates random local rotations,
writes a BVH, parses it back with an independent reader, and compares the
resulting world joint positions against forward kinematics on the originals.

Needs numpy and scipy (both ARDY dependencies); skipped when absent.
"""
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Checker, Skip, load_standalone

try:
    import numpy as np
    from scipy.spatial.transform import Rotation
except ImportError as exc:                                    # pragma: no cover
    raise Skip(f"needs numpy + scipy ({exc})")

_bvh = load_standalone("ardy_bvh")
save_ardy_bvh, _SCALE = _bvh.save_ardy_bvh, _bvh._SCALE

CORE27 = [
    ("Hips", None), ("Spine", "Hips"), ("Spine1", "Spine"), ("Spine2", "Spine1"),
    ("Spine3", "Spine2"), ("Neck", "Spine3"), ("Head", "Neck"),
    ("RightShoulder", "Spine3"), ("RightArm", "RightShoulder"),
    ("RightForeArm", "RightArm"), ("RightHand", "RightForeArm"),
    ("RightHandEnd", "RightHand"), ("RightHandThumb1", "RightHand"),
    ("LeftShoulder", "Spine3"), ("LeftArm", "LeftShoulder"),
    ("LeftForeArm", "LeftArm"), ("LeftHand", "LeftForeArm"),
    ("LeftHandEnd", "LeftHand"), ("LeftHandThumb1", "LeftHand"),
    ("RightUpLeg", "Hips"), ("RightLeg", "RightUpLeg"), ("RightFoot", "RightLeg"),
    ("RightToeBase", "RightFoot"),
    ("LeftUpLeg", "Hips"), ("LeftLeg", "LeftUpLeg"), ("LeftFoot", "LeftLeg"),
    ("LeftToeBase", "LeftFoot"),
]


class FakeSkeleton:
    name = "cskel27"

    def __init__(self, rng):
        self.bone_order_names = [n for n, _ in CORE27]
        index = {n: i for i, n in enumerate(self.bone_order_names)}
        self.joint_parents = np.array(
            [-1 if p is None else index[p] for _, p in CORE27]
        )
        self.root_idx = 0
        # Plausible rest offsets; root at origin as ARDY's SkeletonBase asserts.
        offs = rng.normal(0, 0.15, (len(CORE27), 3))
        self.neutral_joints = np.zeros_like(offs)
        for i, p in enumerate(self.joint_parents):
            self.neutral_joints[i] = offs[i] if p < 0 else self.neutral_joints[p] + offs[i]
        self.neutral_joints[0] = 0.0


def fk(local_rots, root_pos, skel):
    """Reference FK: G[i] = G[parent] @ R[i]; p[i] = p[parent] + G[parent] @ offset[i]."""
    T, J = local_rots.shape[0], local_rots.shape[1]
    parents, neutral = skel.joint_parents, skel.neutral_joints
    offsets = np.zeros_like(neutral)
    for i, p in enumerate(parents):
        offsets[i] = neutral[i] if p < 0 else neutral[i] - neutral[p]
    G = np.zeros((T, J, 3, 3))
    P = np.zeros((T, J, 3))
    for i, p in enumerate(parents):
        if p < 0:
            G[:, i] = local_rots[:, i]
            P[:, i] = root_pos
        else:
            G[:, i] = G[:, p] @ local_rots[:, i]
            P[:, i] = P[:, p] + np.einsum("tab,b->ta", G[:, p], offsets[i])
    return P


def parse_bvh(path):
    """Minimal BVH reader -> (names, parents, offsets, channels, frames)."""
    text = open(path).read()
    head, motion = text.split("MOTION", 1)
    names, parents, offsets, channels = [], [], [], []
    stack = []
    tokens = head.replace("{", " { ").replace("}", " } ").split()
    i, pending = 0, None
    while i < len(tokens):
        t = tokens[i]
        if t in ("ROOT", "JOINT"):
            pending = tokens[i + 1]; i += 2
        elif t == "End":                      # "End Site"
            pending = None; i += 2
        elif t == "{":
            if pending is not None:
                names.append(pending)
                parents.append(stack[-1] if stack else -1)
                offsets.append(None); channels.append(None)
                stack.append(len(names) - 1)
            else:
                stack.append(None)            # End Site block
            pending = None; i += 1
        elif t == "}":
            stack.pop(); i += 1
        elif t == "OFFSET":
            vals = [float(v) for v in tokens[i + 1:i + 4]]
            if stack and stack[-1] is not None:
                offsets[stack[-1]] = vals
            i += 4
        elif t == "CHANNELS":
            n = int(tokens[i + 1])
            channels[stack[-1]] = tokens[i + 2:i + 2 + n]
            i += 2 + n
        else:
            i += 1
    lines = [l for l in motion.strip().splitlines()]
    nframes = int(re.search(r"Frames:\s*(\d+)", lines[0]).group(1))
    data = np.array([[float(v) for v in l.split()] for l in lines[2:2 + nframes]])
    return names, parents, np.array(offsets), channels, data


def eval_bvh(names, parents, offsets, channels, data):
    """Evaluate a parsed BVH to world joint positions, standard interpretation."""
    T, J = data.shape[0], len(names)
    P = np.zeros((T, J, 3)); G = np.zeros((T, J, 3, 3))
    col = 0
    for j in range(J):
        chans = channels[j]
        pos = np.zeros((T, 3))
        if len(chans) == 6:
            pos = data[:, col:col + 3]; col += 3
            rot_chans = chans[3:]
        else:
            rot_chans = chans
        order = "".join(c[0].upper() for c in rot_chans)   # e.g. "ZXY"
        ang = data[:, col:col + 3]; col += 3
        R = Rotation.from_euler(order, ang, degrees=True).as_matrix()
        p = parents[j]
        if p < 0:
            G[:, j] = R
            P[:, j] = offsets[j] + pos
        else:
            G[:, j] = G[:, p] @ R
            P[:, j] = P[:, p] + np.einsum("tab,b->ta", G[:, p], offsets[j])
    return P


def main(check):
    rng = np.random.default_rng(0)
    skel = FakeSkeleton(rng)
    J, T = len(CORE27), 24
    local = Rotation.from_rotvec(
        rng.normal(0, 0.6, (T * J, 3))
    ).as_matrix().reshape(T, J, 3, 3)
    root = rng.normal(0, 0.5, (T, 3)) + np.array([0.0, 0.9, 0.0])

    expected = fk(local, root, skel)

    path = os.path.join(tempfile.mkdtemp(), "t.bvh")
    save_ardy_bvh(path, local, root, skel, fps=20.0)

    names, parents, offsets, channels, data = parse_bvh(path)
    got = eval_bvh(names, parents, offsets, channels, data) / _SCALE

    # Reorder the BVH's depth-first joints back to the skeleton's array order.
    idx = [names.index(n) for n in skel.bone_order_names]
    got = got[:, idx]

    err = np.abs(got - expected)
    check(len(names) == J, f"BVH declares all {J} joints (got {len(names)})")
    check(err.max() < 1e-6,
          f"world joint positions match FK to {err.max():.2e} m "
          f"(mean {err.mean():.2e} m)")

    # Every leaf needs an End Site or Blender drops the final bone on import.
    text = open(path).read()
    leaves = J - len({p for p in parents if p >= 0})
    check(text.count("End Site") == leaves,
          f"every leaf has an End Site ({leaves} leaves)")

    # The importer is told global_scale=0.01, so the file must be centimetres.
    check(abs(_SCALE - 100.0) < 1e-9, "positions are written in centimetres")
    check("CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"
          in text, "root declares 6 channels in the expected order")
    check(text.count("CHANNELS 3 Zrotation Xrotation Yrotation") == J - 1,
          "every non-root joint declares 3 rotation channels")
    check("Frame Time: 0.05000000" in text, "20 FPS -> a 0.05 s frame time")


def run():
    check = Checker("ardy_bvh: BVH round-trips against forward kinematics")
    main(check)
    return check.done()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
