import math


class Vector2:
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y

    def set(self, x: float, y: float):
        self.x = x
        self.y = y
        return self

    def setScalar(self, scalar: float):
        self.x = scalar
        self.y = scalar
        return self

    def copy(self, v: "Vector2"):
        self.x = v.x
        self.y = v.y
        return self

    def clone(self):
        return Vector2(self.x, self.y)

    def add(self, v: "Vector2"):
        self.x += v.x
        self.y += v.y
        return self

    def addScalar(self, s: float):
        self.x += s
        self.y += s
        return self

    def addVectors(self, a: "Vector2", b: "Vector2"):
        self.x = a.x + b.x
        self.y = a.y + b.y
        return self

    def addScaledVector(self, v: "Vector2", s: float):
        self.x += v.x * s
        self.y += v.y * s
        return self

    def sub(self, v: "Vector2"):
        self.x -= v.x
        self.y -= v.y
        return self

    def subVectors(self, a: "Vector2", b: "Vector2"):
        self.x = a.x - b.x
        self.y = a.y - b.y
        return self

    def multiplyScalar(self, scalar: float):
        self.x *= scalar
        self.y *= scalar
        return self

    def divideScalar(self, scalar: float):
        if scalar != 0:
            inv_scalar = 1.0 / scalar
            self.x *= inv_scalar
            self.y *= inv_scalar
        else:
            self.x = 0
            self.y = 0
        return self

    def negate(self):
        self.x = -self.x
        self.y = -self.y
        return self

    def dot(self, v: "Vector2") -> float:
        return self.x * v.x + self.y * v.y

    def lengthSq(self) -> float:
        return self.x * self.x + self.y * self.y

    def length(self) -> float:
        return math.sqrt(self.lengthSq())

    def normalize(self):
        return self.divideScalar(self.length() or 1.0)

    def setLength(self, length: float):
        return self.normalize().multiplyScalar(length)

    def distanceTo(self, v: "Vector2") -> float:
        return math.sqrt(self.distanceToSquared(v))

    def distanceToSquared(self, v: "Vector2") -> float:
        dx = self.x - v.x
        dy = self.y - v.y
        return dx * dx + dy * dy

    def angle(self) -> float:
        return math.atan2(self.y, self.x)

    def angleTo(self, v: "Vector2") -> float:
        dot = self.dot(v)
        len_product = self.length() * v.length()
        if len_product == 0:
            return 0
        return math.acos(max(-1, min(1, dot / len_product)))

    def equals(self, v: "Vector2") -> bool:
        return self.x == v.x and self.y == v.y

    def toArray(self) -> tuple[float, float]:
        return (self.x, self.y)

    def fromArray(self, array: list[float]):
        self.x = array[0]
        self.y = array[1]
        return self

    def rotateAround(self, center: "Vector2", angle: float):
        # Translate to origin
        dx = self.x - center.x
        dy = self.y - center.y

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        rotated_x = dx * cos_a - dy * sin_a
        rotated_y = dx * sin_a + dy * cos_a

        # Translate back
        self.x = rotated_x + center.x
        self.y = rotated_y + center.y
        return self

    def __repr__(self):
        return f"Vector2({self.x}, {self.y})"
