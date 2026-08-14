"""Minimal bpy stub so the add-on module can be imported outside Blender."""
from . import props, types, utils, path


class app:
    online_access = False


class context:
    preferences = None
