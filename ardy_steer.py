"""Pure maths for live ARDY streaming: steering, and ARDY <-> Blender space.

Turning "the target Empty is over there" into something ARDY can be conditioned
on is not a matter of pinning one waypoint at the end of the horizon: that asks
the model to teleport and it fights the motion prior. ARDY's own demo instead
projects a short *path* of future root positions from a target velocity, easing
out of the character's current velocity so the turn is gradual — see
``_update_root_constraints_from_target_velocity`` in
``scripts/interactive_demo/gen_constraints.py``. This module is that idea,
reduced to the case the add-on needs: steer toward a point.

Deliberately dependency-free — no torch, no numpy, no bpy. It runs inside the
ARDY venv next to the bridge, and it is the one piece of the streaming path that
can be unit-tested anywhere.

Coordinates are ARDY's ground plane (x, z); ARDY motion space is Y-up, so "z"
here is ground-forward, not height.
"""

import math

# A brisk walk. Not a hard cap on what the model does — it is the speed the
# waypoints are laid out at, and the prompt still decides the gait.
DEFAULT_MAX_SPEED = 1.6          # metres / second
# How long the character takes to swing from its current velocity onto the new
# heading. Too short reads as a snap, too long overshoots the target.
DEFAULT_TRANSITION_SECONDS = 0.4
# Closer than this and the target counts as reached: emitting waypoints on top
# of the character pins it in place and stops it idling naturally.
ARRIVAL_RADIUS = 0.12            # metres


def _norm(x: float, z: float) -> float:
    return math.hypot(x, z)


def project_root_waypoints(
    current_xz,
    current_vel_xz,
    target_xz,
    fps: float,
    num_frames: int,
    max_speed: float = DEFAULT_MAX_SPEED,
    transition_seconds: float = DEFAULT_TRANSITION_SECONDS,
    arrival_radius: float = ARRIVAL_RADIUS,
):
    """Future root positions that walk from *current_xz* toward *target_xz*.

    Returns ``[(frame_offset, [x, z]), ...]`` with one entry per frame of the
    generation horizon, ``frame_offset`` counted from the first generated frame.
    Returns ``[]`` when the character is already within *arrival_radius*, which
    leaves the model unconstrained so it can idle, turn, or do whatever the
    prompt says instead of being nailed to the spot.

    The path eases from *current_vel_xz* onto the target heading over
    *transition_seconds*, and never steps past the target: speed is capped both
    by *max_speed* and by the distance remaining over the horizon, so a nearby
    target produces a slow approach rather than an overshoot-and-return.
    """
    if num_frames <= 0 or fps <= 0:
        return []

    cx, cz = float(current_xz[0]), float(current_xz[1])
    tx, tz = float(target_xz[0]), float(target_xz[1])
    vx, vz = float(current_vel_xz[0]), float(current_vel_xz[1])

    dx, dz = tx - cx, tz - cz
    distance = _norm(dx, dz)
    if distance <= arrival_radius:
        return []

    dt = 1.0 / fps
    horizon_seconds = num_frames * dt

    # Cap the cruise speed so the path arrives at the target rather than
    # sailing through it, then keeps going for the rest of the horizon.
    speed = min(max_speed, distance / horizon_seconds) if horizon_seconds > 0 else 0.0
    ux, uz = dx / distance, dz / distance
    target_vx, target_vz = ux * speed, uz * speed

    transition_frames = max(1, int(round(transition_seconds * fps)))

    waypoints = []
    px, pz = cx, cz
    travelled = 0.0
    for i in range(num_frames):
        alpha = min(1.0, (i + 1) / transition_frames)
        step_vx = (1.0 - alpha) * vx + alpha * target_vx
        step_vz = (1.0 - alpha) * vz + alpha * target_vz

        nx, nz = px + step_vx * dt, pz + step_vz * dt

        # Clamp onto the target: the eased velocity can carry past it while the
        # blend is still finishing, and a waypoint beyond the target would ask
        # the character to walk through it and come back.
        travelled += _norm(nx - px, nz - pz)
        if travelled >= distance:
            nx, nz = tx, tz

        px, pz = nx, nz
        waypoints.append((i, [px, pz]))

    return waypoints


def project_root_waypoints_from_velocity(
    current_xz,
    current_vel_xz,
    target_vel_xz,
    fps: float,
    num_frames: int,
    transition_seconds: float = DEFAULT_TRANSITION_SECONDS,
):
    """Future root positions for "keep walking at this velocity".

    The sibling of :func:`project_root_waypoints` for driving with keys rather
    than a target object: there is no destination to arrive at, so no distance
    cap and no arrival dead zone — just ease from the current velocity onto the
    requested one and integrate.

    A zero target velocity returns ``[]`` rather than a wall of
    stand-still waypoints: pinning the root to one spot every frame stops the
    character shifting its weight or turning on the spot, and reads as frozen.
    """
    if num_frames <= 0 or fps <= 0:
        return []

    tvx, tvz = float(target_vel_xz[0]), float(target_vel_xz[1])
    if _norm(tvx, tvz) < 1e-6:
        return []

    px, pz = float(current_xz[0]), float(current_xz[1])
    vx, vz = float(current_vel_xz[0]), float(current_vel_xz[1])

    dt = 1.0 / fps
    transition_frames = max(1, int(round(transition_seconds * fps)))

    waypoints = []
    for i in range(num_frames):
        alpha = min(1.0, (i + 1) / transition_frames)
        px += ((1.0 - alpha) * vx + alpha * tvx) * dt
        pz += ((1.0 - alpha) * vz + alpha * tvz) * dt
        waypoints.append((i, [px, pz]))
    return waypoints


def steer_velocity(velocity, key, speed_step: float = 0.2,
                   turn_degrees: float = 30.0, max_speed: float = 5.0):
    """One arrow-key press applied to a persistent target velocity.

    Mirrors ARDY's interactive demo (``CameraMixin.on_arrow_key_press``): the
    velocity persists between presses, up/down are a throttle and left/right a
    steering wheel. It is not hold-to-walk — one press nudges and the character
    keeps going, which is why no key-release tracking is needed.

    *key* is one of ``UP_ARROW`` / ``DOWN_ARROW`` / ``LEFT_ARROW`` /
    ``RIGHT_ARROW`` (Blender's own event type names). Anything else returns the
    velocity unchanged.
    """
    vx, vz = float(velocity[0]), float(velocity[1])
    speed = _norm(vx, vz)

    if key == 'UP_ARROW':
        if speed < 1e-6:
            # From a standstill there is no direction to scale, so start
            # forward: +z is ARDY's ground-forward axis.
            vx, vz = 0.0, speed_step
        else:
            scale = (speed + speed_step) / speed
            vx, vz = vx * scale, vz * scale

    elif key == 'DOWN_ARROW':
        slower = max(0.0, speed - speed_step)
        if speed > 1e-6:
            scale = slower / speed if slower > 1e-6 else 0.0
            vx, vz = vx * scale, vz * scale

    elif key in ('LEFT_ARROW', 'RIGHT_ARROW'):
        # Left turns anticlockwise about the up axis, right clockwise.
        angle = math.radians(turn_degrees if key == 'RIGHT_ARROW' else -turn_degrees)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        vx, vz = cos_a * vx - sin_a * vz, sin_a * vx + cos_a * vz

    else:
        return (vx, vz)

    # Clamp per axis, as the demo does, so a long press cannot run away.
    vx = max(-max_speed, min(max_speed, vx))
    vz = max(-max_speed, min(max_speed, vz))
    return (vx, vz)


# ---------------------------------------------------------------------------
# ARDY space -> Blender space
# ---------------------------------------------------------------------------
#
# The BVH path never needs this: it writes Y-up BVH and lets Blender's importer
# convert on the way in (axis_forward='-Z', axis_up='Y'). A live stream has no
# file and no importer, so the same change of basis has to happen here.
#
# constraints.py derives the Blender -> ARDY direction as
#     M = [[1,0,0],[0,0,1],[0,-1,0]]        (x, y, z) -> (x, z, -y)
# M is a proper rotation, so the inverse is its transpose and applies equally to
# positions and to rotation axes:
#     M^T = [[1,0,0],[0,0,-1],[0,1,0]]      (x, y, z) -> (x, -z, y)

def ardy_to_blender_pos(p):
    """A position in ARDY's Y-up space as Blender's Z-up (x, y, z), in metres."""
    return (float(p[0]), -float(p[2]), float(p[1]))


def ardy_axis_angle_to_blender_quat(aa):
    """An ARDY axis-angle rotation as a Blender (w, x, y, z) quaternion.

    The axis rotates with the same change of basis as a position; the angle —
    the vector's length — is basis-independent.
    """
    ax, ay, az = float(aa[0]), float(aa[1]), float(aa[2])
    angle = math.sqrt(ax * ax + ay * ay + az * az)
    if angle < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)

    # Normalise in ARDY space, then rebase the axis.
    ux, uy, uz = ax / angle, ay / angle, az / angle
    bx, by, bz = ux, -uz, uy

    half = angle * 0.5
    s = math.sin(half)
    return (math.cos(half), bx * s, by * s, bz * s)


# ---------------------------------------------------------------------------
# Quaternions, and rebasing a joint rotation onto a Blender bone
# ---------------------------------------------------------------------------
#
# Quaternions are plain (w, x, y, z) tuples so this stays testable outside
# Blender; the volume is small (joints x window, a few thousand multiplies a
# second) so pure Python costs nothing that matters.

def quat_mul(a, b):
    """Compose two rotations: apply *b*, then *a*."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_inverse(q):
    """Inverse of a unit quaternion — its conjugate."""
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_rotate_vec(q, v):
    """Rotate a 3-vector by a unit quaternion."""
    w, x, y, z = q
    vx, vy, vz = v
    # t = 2 * (q_vec x v); v' = v + w*t + q_vec x t
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def rebase_local_rotation(rest_rot, local_rot):
    """ARDY's local joint rotation as a Blender pose-bone basis rotation.

    Blender composes a pose as

        P_i = P_parent . RL_parent^-1 . RL_i . B_i

    and a bone's axes are its rest axes turned by the joint's global rotation,
    so P_i = G_i . RL_i.  With ARDY's FK being G_i = G_parent . R_i, those
    reduce to

        B_i = RL_i^-1 . R_i . RL_i

    — a conjugation by the bone's *own* rest rotation, with the parent's
    dropping out entirely.  Getting this wrong (an earlier version used
    RL_i^-1 . RL_parent . R_i) leaves every bone off by its own rest
    orientation: the rest pose looks right and the animation is scrambled.

    With an identity rest pose this collapses to B_i = R_i, which is why a rig
    whose bones all point the same way needs no correction at all.
    """
    return quat_mul(quat_mul(quat_inverse(rest_rot), local_rot), rest_rot)


def heading_from_waypoints(waypoints, fallback: float = 0.0):
    """Facing angle implied by the last meaningful step of a projected path.

    ARDY takes headings as an angle about the up axis in its own frame; this is
    the same ``atan2`` convention ``operators.py`` already uses when it derives
    waypoint headings from a sampled curve.
    """
    if len(waypoints) < 2:
        return fallback
    for i in range(len(waypoints) - 1, 0, -1):
        (_, a), (_, b) = waypoints[i - 1], waypoints[i]
        dx, dz = b[0] - a[0], b[1] - a[1]
        if _norm(dx, dz) > 1e-6:
            return math.atan2(dx, dz)
    return fallback
