"""The root-path projection that makes a stream follow a moving target.

This is the one piece of the streaming path with no torch, numpy or bpy in it,
so it is also the only piece that can be checked properly without a GPU. Its
failure modes are all geometric — overshooting the target, snapping instantly
onto a new heading, pinning the character in place once it arrives — and each
one shows up as bad motion rather than an exception, so they are worth pinning
down here.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Checker, load_standalone

S = load_standalone("ardy_steer")

check = Checker("ardy_steer: root path projection")

FPS = 20.0
HORIZON = 8


def _dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


# --- shape -----------------------------------------------------------------
wp = S.project_root_waypoints((0.0, 0.0), (0.0, 0.0), (3.0, 0.0),
                              fps=FPS, num_frames=HORIZON)
check(len(wp) == HORIZON, "one waypoint per generated frame")
check([i for i, _ in wp] == list(range(HORIZON)),
      "  frame offsets are 0..n-1, relative to the first generated frame")

# --- arrival dead zone -----------------------------------------------------
check(S.project_root_waypoints((0.0, 0.0), (0.0, 0.0), (0.05, 0.0),
                               fps=FPS, num_frames=HORIZON) == [],
      "target already reached emits no constraint")
check(S.project_root_waypoints((1.0, 2.0), (0.0, 0.0), (1.0, 2.0),
                               fps=FPS, num_frames=HORIZON) == [],
      "  exactly on the target is also unconstrained")

# --- speed limit -----------------------------------------------------------
far = S.project_root_waypoints((0.0, 0.0), (0.0, 0.0), (100.0, 0.0),
                               fps=FPS, num_frames=HORIZON, max_speed=1.6)
steps = [_dist(far[i - 1][1], far[i][1]) for i in range(1, len(far))]
check(max(steps) <= 1.6 / FPS + 1e-9,
      "a distant target never exceeds max_speed")

# --- no overshoot ----------------------------------------------------------
# A target one slow step away, approached at speed: the path must stop on it.
near = S.project_root_waypoints((0.0, 0.0), (2.0, 0.0), (0.3, 0.0),
                                fps=FPS, num_frames=HORIZON, max_speed=1.6)
overshoot = max(_dist((0.0, 0.0), p) for _, p in near)
check(overshoot <= 0.3 + 1e-6,
      "an eased-into-target path never steps past the target")
check(_dist(near[-1][1], (0.3, 0.0)) < 1e-6,
      "  and settles exactly on it")

# --- easing ----------------------------------------------------------------
# Moving hard along +x, asked to go to +z: the first step must still be mostly
# +x. A snap would put the whole first step on +z and read as a teleport.
turn = S.project_root_waypoints((0.0, 0.0), (1.6, 0.0), (0.0, 5.0),
                                fps=FPS, num_frames=HORIZON,
                                transition_seconds=0.4)
first = turn[0][1]
check(abs(first[0]) > abs(first[1]),
      "a hard turn eases out of the current velocity instead of snapping")
last_step_z = turn[-1][1][1] - turn[-2][1][1]
first_step_z = turn[0][1][1]
check(last_step_z > first_step_z,
      "  and converges onto the new heading across the horizon")

# --- degenerate inputs -----------------------------------------------------
check(S.project_root_waypoints((0, 0), (0, 0), (5, 5), fps=FPS, num_frames=0) == [],
      "zero-length horizon emits nothing")
check(S.project_root_waypoints((0, 0), (0, 0), (5, 5), fps=0, num_frames=HORIZON) == [],
      "zero fps emits nothing rather than dividing by zero")

# --- heading ---------------------------------------------------------------
straight = S.project_root_waypoints((0.0, 0.0), (0.0, 0.0), (0.0, 5.0),
                                    fps=FPS, num_frames=HORIZON)
check(abs(S.heading_from_waypoints(straight) - 0.0) < 1e-3,
      "walking along +z reads as heading 0")
sideways = S.project_root_waypoints((0.0, 0.0), (0.0, 0.0), (5.0, 0.0),
                                    fps=FPS, num_frames=HORIZON)
check(abs(S.heading_from_waypoints(sideways) - math.pi / 2) < 1e-3,
      "walking along +x reads as heading pi/2")
check(S.heading_from_waypoints([], fallback=1.23) == 1.23,
      "an empty path falls back rather than raising")
check(S.heading_from_waypoints([(0, [1.0, 1.0]), (1, [1.0, 1.0])], fallback=0.5) == 0.5,
      "a stationary path falls back rather than reading noise as a heading")


# --- rebasing a joint rotation onto an oriented bone -----------------------
# This is the maths that decides whether the streamed rig animates correctly.
# It cannot be checked against Blender here, so instead simulate Blender's own
# pose composition and require that it reproduces ARDY's forward kinematics.
#
#     Blender:  P_i = P_parent . RL_parent^-1 . RL_i . B_i
#     ARDY:     G_i = G_parent . R_i,  and a posed bone is P_i = G_i . RL_i
#
# A wrong correction still yields a correct *rest* pose, which is exactly how
# the first attempt slipped through: only the animated pose is wrong.

def _axis_angle_quat(axis, angle):
    n = math.sqrt(sum(c * c for c in axis))
    ux, uy, uz = (c / n for c in axis)
    h = angle / 2.0
    sn = math.sin(h)
    return (math.cos(h), ux * sn, uy * sn, uz * sn)


def _close(a, b, tol=1e-9):
    """Quaternion equality up to sign — q and -q are the same rotation."""
    same = all(abs(x - y) < tol for x, y in zip(a, b))
    flipped = all(abs(x + y) < tol for x, y in zip(a, b))
    return same or flipped


# A three-bone chain with deliberately awkward, unequal rest orientations:
# an identity rest would make any formula look right.
REST = [
    _axis_angle_quat((0.0, 0.0, 1.0), 0.30),
    _axis_angle_quat((1.0, 0.2, 0.0), -0.80),
    _axis_angle_quat((0.3, 1.0, 0.5), 1.70),
]
PARENT = [-1, 0, 1]
LOCAL = [                                  # ARDY local joint rotations
    _axis_angle_quat((0.0, 1.0, 0.0), 0.45),
    _axis_angle_quat((1.0, 0.0, 0.0), -1.10),
    _axis_angle_quat((0.2, 0.3, 1.0), 0.65),
]

IDENT = (1.0, 0.0, 0.0, 0.0)

# ARDY forward kinematics: the global rotation of each joint.
G = []
for i, parent in enumerate(PARENT):
    G.append(S.quat_mul(G[parent] if parent >= 0 else IDENT, LOCAL[i]))

# Blender's composition, driven by the basis rotations the add-on computes.
P = []
for i, parent in enumerate(PARENT):
    basis = S.rebase_local_rotation(REST[i], LOCAL[i])
    parent_pose = P[parent] if parent >= 0 else IDENT
    parent_rest = REST[parent] if parent >= 0 else IDENT
    P.append(S.quat_mul(
        S.quat_mul(S.quat_mul(parent_pose, S.quat_inverse(parent_rest)), REST[i]),
        basis))

for i in range(len(PARENT)):
    check(_close(P[i], S.quat_mul(G[i], REST[i])),
          f"bone {i}: Blender's pose composition reproduces ARDY's FK")

# The rest pose must come out right for *any* correction, so a passing rest
# pose proves nothing — pin that the test above is the one doing the work.
P_rest = []
for i, parent in enumerate(PARENT):
    basis = S.rebase_local_rotation(REST[i], IDENT)
    parent_pose = P_rest[parent] if parent >= 0 else IDENT
    parent_rest = REST[parent] if parent >= 0 else IDENT
    P_rest.append(S.quat_mul(
        S.quat_mul(S.quat_mul(parent_pose, S.quat_inverse(parent_rest)), REST[i]),
        basis))
for i in range(len(PARENT)):
    check(_close(P_rest[i], REST[i]),
          f"  bone {i}: an unrotated joint leaves the bone at its rest pose")

check(_close(S.rebase_local_rotation(IDENT, LOCAL[0]), LOCAL[0]),
      "an identity rest needs no correction at all")

# --- quaternion primitives -------------------------------------------------
check(_close(S.quat_mul(LOCAL[0], IDENT), LOCAL[0]), "identity is the unit")
check(_close(S.quat_mul(LOCAL[0], S.quat_inverse(LOCAL[0])), IDENT),
      "a rotation composed with its inverse is identity")

spin = _axis_angle_quat((0.0, 0.0, 1.0), math.pi / 2)
rx, ry, rz = S.quat_rotate_vec(spin, (1.0, 0.0, 0.0))
check(abs(rx) < 1e-9 and abs(ry - 1.0) < 1e-9 and abs(rz) < 1e-9,
      "a quarter turn about +Z takes +X to +Y")


def run():
    return check.done()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
