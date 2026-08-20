"""Minimal bpy stand-in: enough to import and register the add-on."""
import sys, types


class _Prop:
    """Marker returned by bpy.props.*; records its kwargs for inspection."""
    def __init__(self, kind, **kw):
        self.kind = kind
        self.kw = kw
    def __repr__(self):
        return f"<{self.kind} {self.kw.get('name','')}>"


def _mk(kind):
    def factory(*a, **kw):
        if a:
            raise TypeError(f"{kind} takes keyword arguments only, got {a!r}")
        return _Prop(kind, **kw)
    return factory


props = types.ModuleType("bpy.props")
for _n in ("StringProperty", "FloatProperty", "IntProperty", "BoolProperty",
           "EnumProperty", "CollectionProperty", "PointerProperty",
           "FloatVectorProperty", "IntVectorProperty", "BoolVectorProperty"):
    setattr(props, _n, _mk(_n))


class _Base:
    bl_idname = ""
    bl_label = ""


class PropertyGroup(_Base): pass
class Operator(_Base):
    def report(self, *a, **kw): pass
class Panel(_Base): pass
class AddonPreferences(_Base): pass
class UIList(_Base): pass
class Scene: pass
class Object: pass
class Armature: pass
class Action: pass
class IMPORT_ANIM_OT_bvh: pass


types_mod = types.ModuleType("bpy.types")
for _n, _v in list(globals().items()):
    if isinstance(_v, type) and _n[0].isupper():
        setattr(types_mod, _n, _v)

REGISTERED = []


class _Utils:
    @staticmethod
    def register_class(cls):
        # Blender turns bpy.props annotations into RNA properties and leaves
        # plain type annotations alone, so only the former are checked here.
        for name, ann in getattr(cls, "__annotations__", {}).items():
            if isinstance(ann, _Prop) and ann.kind == "EnumProperty":
                items = ann.kw.get("items")
                if isinstance(items, list):
                    keys = [i[0] for i in items]
                    if len(keys) != len(set(keys)):
                        raise TypeError(
                            f"{cls.__name__}.{name} has duplicate enum ids")
                    default = ann.kw.get("default")
                    if default is not None and default not in keys:
                        raise TypeError(
                            f"{cls.__name__}.{name} default {default!r} is not "
                            f"one of {keys}")
        REGISTERED.append(cls)

    @staticmethod
    def unregister_class(cls):
        if cls in REGISTERED:
            REGISTERED.remove(cls)


utils = _Utils()


class _Timers:
    def __init__(self):
        self._fns = []
    def register(self, fn, first_interval=0.0, persistent=False):
        self._fns.append(fn)
    def is_registered(self, fn):
        return fn in self._fns
    def unregister(self, fn):
        if fn in self._fns:
            self._fns.remove(fn)


class _Handlers:
    """bpy.app.handlers: lists of callbacks plus the @persistent decorator."""
    def __init__(self):
        for name in ("load_post", "load_pre", "save_post", "depsgraph_update_post",
                     "frame_change_post", "frame_change_pre", "undo_post", "redo_post"):
            setattr(self, name, [])

    @staticmethod
    def persistent(fn):
        fn._bpy_persistent = True
        return fn


class _App:
    timers = _Timers()
    handlers = _Handlers()
    version = (4, 5, 0)


app = _App()


class _Path:
    @staticmethod
    def abspath(p):
        return p


path = _Path()


class _Prefs:
    def __init__(self):
        self.addons = {}
        self.system = types.SimpleNamespace(ui_scale=1.0)


class _Context:
    def __init__(self):
        self.preferences = _Prefs()
        self.scene = None
        self.window_manager = types.SimpleNamespace(windows=[])
        self.region = None


context = _Context()


class _Ops:
    def __getattr__(self, name):
        return _Ops()
    def __call__(self, *a, **kw):
        return {'FINISHED'}


ops = _Ops()


class _Data:
    scenes = []
    objects = []
    actions = []
    armatures = []


data = _Data()

sys.modules["bpy.props"] = props
sys.modules["bpy.types"] = types_mod

# Must come last: `bpy.types` has to resolve to the mock, not the stdlib module
# this file imported at the top.
types = types_mod
