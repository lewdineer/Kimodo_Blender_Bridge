"""Register the add-on against a mock bpy and check the ARDY wiring.

Catches the class of mistake that otherwise only surfaces when a user enables
the add-on: a missing property, a duplicate operator id, a skeleton table that
drifted from ARDY's own definitions, or a deletion guard that stopped guarding.
"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Checker, PACKAGE, load_addon

addon = load_addon()
import bpy   # the mock, now on sys.path

check = Checker("add-on: registration, properties and skeleton tables")

addon.register()
names = [c.__name__ for c in bpy.REGISTERED]
check(len(names) > 0, f"{len(names)} classes registered")

check("KIMODO_OT_InstallArdy" in names, "ARDY install operator registered")
check("KIMODO_OT_UseInstalledArdy" in names, "ARDY 'use installed' operator registered")
check("KIMODO_OT_ResetArdyVenv" in names, "ARDY reset-venv operator registered")
check("ARDY_OT_ConfirmDeleteDir" in names, "ARDY delete confirmation registered")
check("KIMODO_OT_OpenURL" in names, "URL opener registered")
check(len(names) == len(set(names)), "no duplicate class registrations")

# bl_idname collisions would make Blender refuse to register
idnames = [getattr(c, "bl_idname", "") for c in bpy.REGISTERED]
idnames = [i for i in idnames if i]
check(len(idnames) == len(set(idnames)),
      f"no duplicate bl_idnames ({len(idnames) - len(set(idnames))} dupes)")

# --- scene properties -------------------------------------------------------
P = sys.modules[PACKAGE + ".properties"]
ann = P.KIMODO_SceneSettings.__annotations__
for name in ("backend", "active_skeleton", "native_fps", "ardy_model",
             "ardy_python_executable", "ardy_text_encoder_device",
             "ardy_cfg_text_weight", "ardy_cfg_constraint_weight",
             "ardy_history_frames"):
    check(name in ann, f"scene property '{name}' defined")
check("ardy_install_location" in P.KIMODO_AddonPreferences.__annotations__,
      "preference 'ardy_install_location' defined")
check(ann["backend"].kw.get("default") == 'KIMODO',
      "backend defaults to Kimodo (no behaviour change for existing users)")

# --- skeleton tables --------------------------------------------------------
C = sys.modules[PACKAGE + ".constraints"]
core = C.get_skeleton("cskel27")
soma = C.get_skeleton("somaskel30")
check(core["name"] == "core", "cskel27 resolves to the Core tables")
check(soma["name"] == "soma", "somaskel30 resolves to the SOMA tables")
check(C.get_skeleton("something-new")["name"] == "soma",
      "an unknown skeleton name falls back to SOMA instead of raising")
check(len(C.CORE_JOINT_ORDER) == 27, "Core table has 27 joints")
check(len(C.CORE_JOINT_PARENTS) == 27, "Core parents table has 27 entries")
check(C.CORE_JOINT_PARENTS[0] == -1 and C.CORE_JOINT_PARENTS.count(-1) == 1,
      "exactly one root in the Core hierarchy")
check(all(p < i for i, p in enumerate(C.CORE_JOINT_PARENTS)),
      "Core parents are topologically ordered (parent before child)")
for eff, idx in C.CORE_EFFECTOR_IDX.items():
    check(C.CORE_JOINT_ORDER[idx] == C.CORE_EFFECTOR_BONE[eff],
          f"Core effector '{eff}' index {idx} points at {C.CORE_EFFECTOR_BONE[eff]}")
for eff, idx in C.EFFECTOR_IDX.items():
    check(C.SOMA_JOINT_ORDER[idx] == C.EFFECTOR_BONE[eff],
          f"SOMA effector '{eff}' index {idx} still points at {C.EFFECTOR_BONE[eff]}")

# Cross-check the Core table against the real ARDY source, if it is present.
# When a clone of nv-tlabs/ardy is available, check the tables against the
# real source instead of trusting the copy in this repo.
ardy_defs = os.environ.get(
    "ARDY_SOURCE",
    os.path.expanduser("~/nv-tlabs/ardy")) + "/ardy/skeleton/definitions.py"
if os.path.isfile(ardy_defs):
    import ast, re
    tree = ast.parse(open(ardy_defs).read())
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "CoreSkeleton27":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and any(
                        getattr(t, "id", "") == "bone_order_names_with_parents"
                        for t in stmt.targets):
                    found = ast.literal_eval(stmt.value)
    check(found is not None, "read CoreSkeleton27 from the real ARDY source")
    if found:
        names_real = [n for n, _ in found]
        index = {n: i for i, n in enumerate(names_real)}
        parents_real = [-1 if p is None else index[p] for _, p in found]
        check(names_real == C.CORE_JOINT_ORDER,
              "Core joint order matches ARDY's CoreSkeleton27 exactly")
        check(parents_real == C.CORE_JOINT_PARENTS,
              "Core parent indices match ARDY's CoreSkeleton27 exactly")

# --- subprocess client ------------------------------------------------------
sc = sys.modules[PACKAGE + ".subprocess_client"]
check(os.path.isfile(sc._bridge_path("kimodo")), "kimodo bridge script exists")
check(os.path.isfile(sc._bridge_path("ardy")), "ardy bridge script exists")
check(sc._bridge_path("ardy").endswith("ardy_bridge.py"), "ardy backend maps to ardy_bridge.py")
check(sc._bridge_path("nonsense").endswith("bridge_server.py"),
      "an unknown backend falls back to the Kimodo bridge")
check(sc.get_ready_info() == {}, "ready info starts empty")

# --- ardy_setup -------------------------------------------------------------
A = sys.modules[PACKAGE + ".ardy_setup"]
S = sys.modules[PACKAGE + ".setup_operator"]
check(A._default_venv().endswith(".ardy-venv"), "ARDY venv default is ~/.ardy-venv")
check(A._default_venv() != S._default_venv(), "ARDY and Kimodo venvs never collide")
check(S.install_target() == "kimodo", "install target defaults to kimodo")
check(S.torch_index_for(12, (12, 0))[1] == "12.8", "Blackwell -> cu128")
check(S.torch_index_for(13, (8, 6))[1] == "12.4", "py3.13 on Ampere -> cu124")
check(S.torch_index_for(12, (8, 6))[1] == "12.1", "py3.12 on Ampere -> cu121")

# Deletion guard must refuse anything that is not our own venv.
for bad in ("", os.path.expanduser("~"), "/", tempfile.mkdtemp()):
    try:
        A._safe_rmtree(bad)
        check(False, f"_safe_rmtree refused {bad!r}")
    except RuntimeError:
        check(True, f"_safe_rmtree refused {bad!r}")

# The ext_modules rewrite must actually match the real ARDY setup.py.
setup_py = os.path.dirname(os.path.dirname(ardy_defs)) + "/setup.py"
if os.path.isfile(setup_py):
    import re, shutil
    d = tempfile.mkdtemp()
    shutil.copy(setup_py, os.path.join(d, "setup.py"))
    A._disable_native_extension(d)
    out = open(os.path.join(d, "setup.py")).read()
    check("ext_modules=[]" in out, "native extension disabled in real ARDY setup.py")
    check("CMakeExtension(" not in out.split("setup(")[-1],
          "no CMakeExtension left in the setup() call")
    import ast as _ast
    _ast.parse(out)
    check(True, "patched setup.py still parses")

# --- retarget ---------------------------------------------------------------
R = sys.modules[PACKAGE + ".retarget"]
core_names = set(C.CORE_JOINT_ORDER)
hinted = {n for n, _ in R._SOMA_BONE_MAP_HINTS} | {n for n, _ in R._CORE_EXTRA_HINTS}
missing = sorted(core_names - hinted)
check(not missing, f"every Core joint has a retarget hint (missing: {missing})")

# --- generation arguments ---------------------------------------------------
# Every generate path funnels through _ardy_kwargs, so a key missing here is a
# setting the UI shows and the model never sees. diffusion_steps in particular
# used to be absent, leaving the client's default of 100 in force no matter
# what the panel said.
O = sys.modules[PACKAGE + ".operators"]


class _FakeGenSettings:
    diffusion_steps = 37
    ardy_cfg_text_weight = 2.5
    ardy_cfg_constraint_weight = 1.5
    ardy_history_frames = 0


_kwargs = O._ardy_kwargs(_FakeGenSettings())
check(_kwargs.get("diffusion_steps") == 37,
      "sample steps reach the bridge instead of the client default")
for _key in ("cfg_text_weight", "cfg_constraint_weight", "history_frames"):
    check(_key in _kwargs, f"  {_key} is still passed through")

# --- transient state (#43, and the same trap for live streaming) ------------
# is_generating / is_streaming are scene properties, so Blender saves them into
# the .blend. A loaded file can never have a live job, and a saved True leaves
# the UI stuck behind a button that can never finish.
class _FakeK:
    is_generating = True
    generation_progress = "Working…"
    generating_segment_index = 3
    is_streaming = True
    stream_status = "Streaming…"


_scene = type("S", (), {"kimodo": _FakeK()})()
_saved_scenes = bpy.data.scenes
bpy.data.scenes = [_scene]
try:
    P._reset_transient_generation_state()
finally:
    bpy.data.scenes = _saved_scenes

k = _scene.kimodo
check(k.is_generating is False, "a loaded file never stays 'generating'")
check(k.generation_progress == "", "  and its progress text is cleared")
check(k.generating_segment_index == -1, "  and its segment index is cleared")
check(k.is_streaming is False, "a loaded file never stays 'streaming'")
check(k.stream_status == "", "  and its stream status is cleared")

addon.unregister()
check(bpy.REGISTERED == [], "unregister removes every class")


def run():
    return check.done()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
