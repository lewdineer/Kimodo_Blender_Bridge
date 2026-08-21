# Tests

Plain-Python checks for the parts of the add-on that can be verified without
launching Blender or owning an NVIDIA GPU.

```bash
python3 tests/run_tests.py
```

Nothing here imports the real Blender: `bpy`, `mathutils` and `addon_utils`
are mocked in `tests/mocks/`, and the ARDY model itself is mocked too. Run the
suite from a normal Python interpreter, **not** from inside Blender — the mock
`bpy` would shadow the real one.

## What each module covers

| Module | Covers | Needs |
|---|---|---|
| `test_ardy_bvh.py` | The BVH writer ARDY does not ship. Builds a Core27 skeleton, writes a BVH, parses it back with an independent reader, and compares world joint positions against forward kinematics. Also checks channel order, End Sites on every leaf, centimetre units and frame time. | numpy, scipy |
| `test_ardy_bridge.py` | `ardy_bridge.py` driven as a real subprocess over its JSON protocol: startup handshake, single-clip and timeline generation, constraint pass-through, exact frame counts, malformed input, and that the request loop survives errors. Also the live-stream commands — replan arithmetic, history alignment, prompt-encoding cache, and that `stream_end` really drops the state. The mock model re-asserts the released model's own contract. | torch |
| `test_addon_registration.py` | Registering the add-on: every operator and property present, no duplicate `bl_idname`s, the Core skeleton tables matching ARDY's real `CoreSkeleton27`, the deletion guard refusing non-venv paths, and the PyTorch index picked per GPU. | — |
| `test_ardy_steer.py` | The streaming steering geometry: one waypoint per generated frame, the arrival dead zone, the speed cap, no overshoot, easing out of the current velocity instead of snapping, and the ARDY ↔ Blender change of basis. Pure maths, no dependencies. | — |
| `test_action_fcurves.py` | Locating an Action's F-curves across Blender's three Action layouts (pre-4.4, 4.4's shim, and 5.x's per-slot channelbags). Pins the lookup that broke Live Stream on Blender 5.2, including that the *bound* slot's curves are used rather than the first channelbag. | — |
| `test_armature_reuse.py` | Reusing a source armature across skeletons. Pins that Kimodo's somaskel77 and ARDY's cskel27 are never mixed — the bug behind a character landing in the wrong pose and sinking through the floor after a backend switch — while same-skeleton reuse still works. | — |
| `test_panels.py` | Every panel drawn across backend × mode × connected, asserting the backend-specific UI lands in the right view (FPS warning, model picker, T-pose option) and that `poll()` hides the ARDY-only Live Stream panel on Kimodo. | — |

A module whose dependency is missing is **skipped**, not failed, so the suite
still runs on a machine without the model stack.

## Checking against the real ARDY source

`test_addon_registration.py` cross-checks the Core27 joint order and parent
indices against ARDY's own `ardy/skeleton/definitions.py`, and verifies the
installer's `setup.py` rewrite still matches upstream. Point it at a clone:

```bash
ARDY_SOURCE=/path/to/nv-tlabs/ardy python3 tests/run_tests.py
```

It defaults to `~/nv-tlabs/ardy` and silently skips those two checks when no
clone is found.

## What is *not* covered

Everything that needs the real thing: model loading, generation quality,
whether the generated BVH imports cleanly into Blender, retarget results, and
the installer end to end. Those still need a manual pass on a machine with an
NVIDIA GPU — see `docs/ardy-feasibility.md`.
