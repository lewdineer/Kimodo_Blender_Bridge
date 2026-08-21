"""Draw every panel in both backends against a recording fake layout.

Panel bugs — a property that no longer exists, an operator id that was renamed,
a branch that only runs when a backend is selected — are invisible until a user
opens the sidebar. This draws all panels across backend / mode / connected
combinations and asserts the backend-specific bits land in the right view.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Checker, PACKAGE, load_addon

addon = load_addon()
import bpy
P = sys.modules[PACKAGE + ".properties"]

check = Checker("panels: every panel draws in both backends")


class FakeLayout:
    """Records what a draw() emitted; every call returns a usable child."""
    def __init__(self, sink):
        self.sink = sink
        self.enabled = True
        self.alert = False
        self.alignment = 'LEFT'
        self.scale_y = 1.0
        self.scale_x = 1.0
        self.use_property_split = False
        self.use_property_decorate = False

    def _child(self, *a, **kw):
        return FakeLayout(self.sink)
    row = column = box = split = grid_flow = column_flow = _child

    def label(self, text="", icon='NONE', **kw):
        self.sink["labels"].append(text)

    def prop(self, data, name, **kw):
        # A property that does not exist would raise in Blender too.
        if isinstance(data, FakeSettings) and name not in data.__dict__:
            raise AttributeError(f"scene.kimodo has no property {name!r}")
        if isinstance(data, FakePrefs) and name not in data.__dict__:
            raise AttributeError(f"preferences has no property {name!r}")
        self.sink["props"].append(name)

    def operator(self, idname, **kw):
        if idname not in KNOWN_OPS:
            raise AttributeError(f"unknown operator {idname!r}")
        self.sink["ops"].append(idname)
        return types.SimpleNamespace(**{k: None for k in
                                        ("prompt_location", "confirmed", "url",
                                         "index", "filepath", "label", "direction")})

    def separator(self, **kw): pass
    def progress(self, **kw): self.sink["labels"].append(kw.get("text", ""))
    def template_list(self, *a, **kw): pass
    def menu(self, *a, **kw): pass
    def prop_search(self, *a, **kw): pass
    def popover(self, *a, **kw): pass
    def template_ID(self, *a, **kw): pass


def _default_for(prop):
    d = prop.kw.get("default")
    if d is not None:
        return d
    if prop.kind == "EnumProperty":
        items = prop.kw.get("items")
        return items[0][0] if isinstance(items, list) and items else ""
    if prop.kind in ("CollectionProperty",):
        return []
    if prop.kind == "PointerProperty":
        return None
    return {"StringProperty": "", "IntProperty": 0, "FloatProperty": 0.0,
            "BoolProperty": False}.get(prop.kind)


class FakeSettings:
    def __init__(self):
        for name, prop in P.KIMODO_SceneSettings.__annotations__.items():
            setattr(self, name, _default_for(prop))


class FakePrefs:
    def __init__(self):
        for name, prop in P.KIMODO_AddonPreferences.__annotations__.items():
            setattr(self, name, _default_for(prop))


class FakeRender:
    fps = 24
    fps_base = 1.0


class FakeScene:
    def __init__(self, settings):
        self.kimodo = settings
        self.render = FakeRender()
        self.frame_start = 1
        self.frame_end = 250
        self.frame_current = 1
        self.objects = []


class FakeContext:
    def __init__(self, scene, prefs):
        self.scene = scene
        self.region = types.SimpleNamespace(width=340)
        self.screen = types.SimpleNamespace(areas=[])
        self.preferences = types.SimpleNamespace(
            addons={PACKAGE: types.SimpleNamespace(preferences=prefs)},
            system=types.SimpleNamespace(ui_scale=1.0))
        self.window_manager = types.SimpleNamespace(windows=[])
        self.object = None


addon.register()
KNOWN_OPS = {c.bl_idname for c in bpy.REGISTERED if getattr(c, "bl_idname", "")}
PANELS = [c for c in bpy.REGISTERED if c.__name__.startswith("KIMODO_PT_")]
check(len(PANELS) > 0, f"{len(PANELS)} panels, {len(KNOWN_OPS)} operators")

for backend in ('KIMODO', 'ARDY'):
    for mode in ('SINGLE', 'TIMELINE'):
        for connected in (False, True):
            s = FakeSettings()
            s.backend = backend
            s.generate_mode = mode
            s.is_connected = connected
            s.show_advanced_connection = True     # exercise the hidden paths too
            s.native_fps = 20.0 if backend == 'ARDY' else 30.0
            s.active_skeleton = "cskel27" if backend == 'ARDY' else "somaskel30"
            prefs = FakePrefs()
            ctx = FakeContext(FakeScene(s), prefs)
            for cls in PANELS:
                # A panel with a poll() that rejects this context is never
                # drawn by Blender, so respect it here rather than exercising
                # a code path the user cannot reach.
                poll = getattr(cls, "poll", None)
                visible = True
                if poll is not None:
                    try:
                        visible = bool(poll(ctx))
                    except Exception as exc:
                        check(False, f"{cls.__name__}.poll raises: {exc}")
                        continue
                if cls.__name__ == "KIMODO_PT_LiveStream":
                    check(visible == (backend == 'ARDY'),
                          f"  Live Stream panel visible only for ARDY "
                          f"[{backend}]")
                if not visible:
                    continue

                panel = cls()
                sink = {"labels": [], "props": [], "ops": []}
                panel.layout = FakeLayout(sink)
                try:
                    panel.draw(ctx)
                except Exception as exc:
                    import traceback; traceback.print_exc()
                    check(False, f"{cls.__name__} draws "
                                 f"[{backend}/{mode}/connected={connected}]: {exc}")
                    continue
                check(True, f"{cls.__name__} draws "
                            f"[{backend}/{mode}/connected={connected}]")
                if cls.__name__ == "KIMODO_PT_Connection":
                    if backend == 'ARDY':
                        check("kimodo.install_ardy" in sink["ops"]
                              or "kimodo.use_installed_ardy" in sink["ops"]
                              or "ardy_python_executable" in sink["props"],
                              f"  ARDY connection offers an ARDY action "
                              f"[connected={connected}]")
                        check("kimodo_model" not in sink["props"],
                              "  ARDY view does not show the Kimodo model picker")
                    else:
                        check("kimodo_model" in sink["props"],
                              "  Kimodo view shows the Kimodo model picker")
                        check("ardy_model" not in sink["props"],
                              "  Kimodo view does not show the ARDY model picker")
                if cls.__name__ == "KIMODO_PT_LiveStream":
                    check("kimodo.ardy_stream" in sink["ops"],
                          "  Live Stream offers the start operator")
                    check("ardy_stream_target" in sink["props"],
                          "  Live Stream exposes the follow target")
                    check("ardy_stream_diffusion_steps" in sink["props"],
                          "  Live Stream exposes the latency dial")

                if cls.__name__ == "KIMODO_PT_Generate":
                    check("diffusion_steps" in sink["props"],
                          "  Generate exposes the sample-steps dial")
                    joined = " ".join(sink["labels"])
                    want = "20" if backend == 'ARDY' else "30"
                    check(f"needs {want}" in joined,
                          f"  FPS warning names {want} FPS for {backend}")
                    if backend == 'ARDY':
                        check("bvh_standard_tpose" not in sink["props"],
                              "  T-pose option hidden for ARDY (no T-pose asset)")
                    else:
                        check("bvh_standard_tpose" in sink["props"],
                              "  T-pose option shown for Kimodo")

addon.unregister()


def run():
    return check.done()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
