"""Minimal mathutils.Vector stub (only what the add-on uses)."""


class Vector(tuple):
    def __new__(cls, values=(0.0, 0.0, 0.0)):
        return super().__new__(cls, tuple(float(v) for v in values))

    def dot(self, other):
        return sum(a * b for a, b in zip(self, other))

    def cross(self, other):
        ax, ay, az = self
        bx, by, bz = other
        return Vector((ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx))
