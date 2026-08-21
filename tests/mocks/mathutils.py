class Vector(tuple):
    def __new__(cls, v=(0, 0, 0)):
        return super().__new__(cls, tuple(v))
    x = property(lambda s: s[0]); y = property(lambda s: s[1]); z = property(lambda s: s[2])


class Matrix:
    def __init__(self, rows=None):
        self.rows = rows or ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    @classmethod
    def Identity(cls, n):
        return cls()
    def to_3x3(self): return self
    def to_quaternion(self): return Quaternion()
    def transposed(self): return self
    def inverted(self): return self
    def __matmul__(self, other): return self


class Quaternion:
    def __init__(self, *a):
        self.w, self.x, self.y, self.z = 1.0, 0.0, 0.0, 0.0
    def normalized(self): return self


class Euler:
    def __init__(self, *a): pass
