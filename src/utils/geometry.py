import math
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Point:
    x: float
    y: float
    score: float = 1.0


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float = 1.0
    label: Optional[str] = None
    track_id: Optional[int] = None

    def center(self) -> Point:
        return Point((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0, self.score)

    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def intersects(self, other: "BBox") -> bool:
        return not (
            self.x2 < other.x1 or self.x1 > other.x2 or self.y2 < other.y1 or self.y1 > other.y2
        )

    def contains_point(self, p: Point) -> bool:
        return self.x1 <= p.x <= self.x2 and self.y1 <= p.y <= self.y2


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def l2_distance(p1: Point, p2: Point) -> float:
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def angle_between(p_shoulder: Point, p_elbow: Point, p_wrist: Point) -> float:
    # angle at elbow (shoulder-elbow-wrist)
    v1x = p_shoulder.x - p_elbow.x
    v1y = p_shoulder.y - p_elbow.y
    v2x = p_wrist.x - p_elbow.x
    v2y = p_wrist.y - p_elbow.y
    dot = v1x * v2x + v1y * v2y
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 == 0 or n2 == 0:
        return 180.0
    cosang = clamp(dot / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def rect_from_points(p1: Point, p2: Point) -> BBox:
    return BBox(int(min(p1.x, p2.x)), int(min(p1.y, p2.y)), int(max(p1.x, p2.x)), int(max(p1.y, p2.y)))


def expand_bbox(b: BBox, fx: float, fy: float, width: int, height: int) -> BBox:
    cx, cy = b.center().x, b.center().y
    w = (b.x2 - b.x1) * fx
    h = (b.y2 - b.y1) * fy
    x1 = int(clamp(cx - w / 2, 0, width - 1))
    y1 = int(clamp(cy - h / 2, 0, height - 1))
    x2 = int(clamp(cx + w / 2, 0, width - 1))
    y2 = int(clamp(cy + h / 2, 0, height - 1))
    return BBox(x1, y1, x2, y2, b.score, b.label)
