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
