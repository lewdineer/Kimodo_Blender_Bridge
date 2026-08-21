"""Finding an Action's F-curves across Blender's three Action layouts.

Blender 4.4 introduced slotted Actions and kept `Action.fcurves` working as a
shim. Blender 5.0 removed it, so the live stream died on 5.2 with
`'Action' object has no attribute 'fcurves'` before a single frame was written.
There is no bpy here to test against, so these fakes model each layout's shape
— which is the part the lookup actually depends on.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Checker, PACKAGE, load_addon

load_addon()
S = sys.modules[PACKAGE + ".ardy_stream"]

check = Checker("ardy_stream: Action F-curve lookup across Blender versions")


class FakeFCurve:
    def __init__(self, path, index):
        self.data_path = path
        self.array_index = index


CURVES = [FakeFCurve('pose.bones["Hips"].rotation_quaternion', i) for i in range(4)]


class LegacyAction:
    """Blender <= 4.3, and 4.4's compatibility shim."""
    name = "legacy"
    fcurves = CURVES


class Channelbag:
    def __init__(self, slot, fcurves):
        self.slot = slot
        self.fcurves = fcurves


class Strip:
    """Blender 5.x keyframe strip: one channelbag per slot."""
    def __init__(self, bags):
        self._bags = bags
        self.channelbags = list(bags.values())
        self.ensure_calls = 0

    def channelbag(self, slot, ensure=False):
        if ensure:
            self.ensure_calls += 1
        return self._bags.get(slot)


class Layer:
    def __init__(self, strips):
        self.strips = strips


class SlottedAction:
    name = "slotted"

    def __init__(self, bags):
        self.strip = Strip(bags)
        self.layers = [Layer([self.strip])]
    # Deliberately no `fcurves`: that is exactly what 5.0 removed.


# --- legacy ----------------------------------------------------------------
check(S.action_fcurves(LegacyAction()) is CURVES,
      "Blender <= 4.4: Action.fcurves is used directly")
check(S.action_fcurves(LegacyAction(), slot="ignored") is CURVES,
      "  a slot argument does not disturb the legacy path")

# --- Blender 5.x, the reported failure -------------------------------------
slot = object()
other = object()
other_curves = [FakeFCurve("other", 0)]
act = SlottedAction({slot: Channelbag(slot, CURVES),
                     other: Channelbag(other, other_curves)})
check(S.action_fcurves(act, slot) is CURVES,
      "Blender 5.x: curves come from the channelbag of the bound slot")
check(act.strip.ensure_calls == 1,
      "  the channelbag is created if the slot has none yet (ensure=True)")

# Picking the wrong slot's curves would write onto another animation entirely.
act2 = SlottedAction({slot: Channelbag(slot, CURVES),
                      other: Channelbag(other, other_curves)})
check(S.action_fcurves(act2, other) is other_curves,
      "  a different slot yields that slot's own curves, not the first bag")

# --- older 5.x signature without ensure= -----------------------------------
class StripNoEnsure(Strip):
    def channelbag(self, slot):          # no ensure kwarg at all
        return self._bags.get(slot)


act3 = SlottedAction({slot: Channelbag(slot, CURVES)})
act3.strip = StripNoEnsure({slot: Channelbag(slot, CURVES)})
act3.layers = [Layer([act3.strip])]
check(S.action_fcurves(act3, slot).__class__ is list,
      "a channelbag() without an ensure= kwarg still resolves")

# --- no slot known ---------------------------------------------------------
act4 = SlottedAction({slot: Channelbag(slot, CURVES)})
check(S.action_fcurves(act4, None) is CURVES,
      "with no slot, the first channelbag is used as a fallback")

# --- nothing to find -------------------------------------------------------
class EmptyAction:
    name = "empty"
    layers = []


check(S.action_fcurves(EmptyAction()) is None,
      "an action with no curve container returns None rather than raising")


def run():
    return check.done()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
