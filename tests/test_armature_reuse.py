"""Reusing a source armature must never cross skeletons.

Kimodo emits somaskel77 and ARDY emits cskel27. Reusing one for the other
transfers an Action onto bones that mean something else — SOMA's LeftLeg is the
hip, Core27's is the knee — against a rest pose whose hips sit at a different
height, so the character lands in the wrong pose *and* sinks through the floor.
Generating on one backend, switching, and generating again did exactly that,
because the reuse pointer survives the switch.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Checker, PACKAGE, load_addon

load_addon()
O = sys.modules[PACKAGE + ".operators"]

check = Checker("operators: armature reuse is skeleton-safe")


class FakeBone:
    def __init__(self, name):
        self.name = name


class FakeArmData:
    # users mirrors Blender's ID user count: the reuse path deletes the
    # stand-in armature's data only once nothing references it.
    def __init__(self, names):
        self.bones = [FakeBone(n) for n in names]
        self.users = 1


class FakeAction:
    def __init__(self, name):
        self.name = name
        self.users = 1


class FakeAnimData:
    def __init__(self, action=None):
        self.action = action


class FakeObject:
    type = 'ARMATURE'

    def __init__(self, name, bones, action=None):
        self.name = name
        self.data = FakeArmData(bones)
        self.animation_data = FakeAnimData(action)


class FakeSettings:
    def __init__(self, reuse):
        self.reuse_armature = reuse


SOMA = ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Head",
        "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase"]
CORE = ["Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
        "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"]

# --- the reported bug ------------------------------------------------------
existing = FakeObject("Kimodo_Source", SOMA, FakeAction("old"))
incoming = FakeObject("Kimodo_Source.001", CORE, FakeAction("new"))
result = O._apply_to_existing_source(FakeSettings(existing), incoming)
check(result is incoming,
      "a different skeleton is not reused — the new armature is kept")

# The overlap is the trap: both skeletons really do share these names, so a
# check on 'do any bone names match' would have passed and still corrupted.
shared = set(SOMA) & set(CORE)
check(len(shared) >= 5,
      f"  the two skeletons genuinely share bone names ({len(shared)} of them)")
check("LeftLeg" in shared,
      "  including LeftLeg, which is the hip in one and the knee in the other")

# --- same skeleton still reuses -------------------------------------------
existing = FakeObject("Kimodo_Source", SOMA, FakeAction("old"))
incoming = FakeObject("Kimodo_Source.001", list(reversed(SOMA)), FakeAction("new"))
result = O._apply_to_existing_source(FakeSettings(existing), incoming)
check(result is existing,
      "the same skeleton is still reused (bone order does not matter)")

# --- degenerate inputs -----------------------------------------------------
incoming = FakeObject("Kimodo_Source", CORE, FakeAction("new"))
check(O._apply_to_existing_source(FakeSettings(None), incoming) is incoming,
      "no reuse target keeps the new armature")
check(O._apply_to_existing_source(FakeSettings(incoming), incoming) is incoming,
      "reusing itself is a no-op")

mesh = FakeObject("Cube", [])
mesh.type = 'MESH'
check(O._apply_to_existing_source(FakeSettings(mesh), incoming) is incoming,
      "a non-armature reuse target is ignored")

# --- signature -------------------------------------------------------------
check(O._skeleton_signature(FakeObject("a", SOMA))
      != O._skeleton_signature(FakeObject("b", CORE)),
      "the two skeletons have different signatures")
check(O._skeleton_signature(FakeObject("a", SOMA))
      == O._skeleton_signature(FakeObject("b", list(reversed(SOMA)))),
      "signature ignores bone order")


def run():
    return check.done()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
