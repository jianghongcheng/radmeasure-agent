from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Dict, Mapping, Tuple


PointMap = Dict[str, Tuple[float, float]]


@dataclass(frozen=True)
class MovePoint:
    point_id: str
    dx: float
    dy: float

    def apply(self, points: Mapping[str, tuple[float, float]]) -> PointMap:
        if self.point_id not in points:
            raise ValueError(f"unknown point: {self.point_id}")
        updated = dict(points)
        x, y = updated[self.point_id]
        updated[self.point_id] = (
            min(1.0, max(0.0, x + self.dx)),
            min(1.0, max(0.0, y + self.dy)),
        )
        return updated


def gaussian_corruption(points: Mapping[str, tuple[float, float]], sigma: float,
                        rng: random.Random) -> tuple[PointMap, list[MovePoint]]:
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    current = dict(points)
    actions = []
    for point_id in points:
        action = MovePoint(point_id, rng.gauss(0, sigma), rng.gauss(0, sigma))
        current = action.apply(current)
        actions.append(action)
    return current, actions


def axis_rotation_corruption(points: Mapping[str, tuple[float, float]],
                             start: str, end: str, degrees: float) -> PointMap:
    """Rotate one endpoint pair around its midpoint in normalized coordinates."""
    if start not in points or end not in points:
        raise ValueError("axis endpoint is missing")
    p1, p2 = points[start], points[end]
    center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)

    def rotate(point):
        x, y = point[0] - center[0], point[1] - center[1]
        return (min(1.0, max(0.0, center[0] + cosine*x - sine*y)),
                min(1.0, max(0.0, center[1] + sine*x + cosine*y)))

    updated = dict(points)
    updated[start], updated[end] = rotate(p1), rotate(p2)
    return updated


def oracle_repair_step(current: Mapping[str, tuple[float, float]],
                       target: Mapping[str, tuple[float, float]]) -> tuple[PointMap, MovePoint]:
    """Upper bound: move the most displaced landmark exactly to its target."""
    if set(current) != set(target):
        raise ValueError("current and target must contain identical point IDs")
    point_id = max(current, key=lambda key: math.dist(current[key], target[key]))
    action = MovePoint(point_id, target[point_id][0] - current[point_id][0],
                       target[point_id][1] - current[point_id][1])
    return action.apply(current), action


def program_aware_oracle_step(
    current: Mapping[str, tuple[float, float]],
    target: Mapping[str, tuple[float, float]],
    score: Callable[[Mapping[str, tuple[float, float]]], float],
) -> tuple[PointMap, MovePoint | None]:
    """Choose the best single exact-point repair, including a safe no-op."""
    if set(current) != set(target):
        raise ValueError("current and target must contain identical point IDs")
    best_points, best_action, best_score = dict(current), None, float(score(current))
    for point_id in current:
        action = MovePoint(point_id, target[point_id][0] - current[point_id][0],
                           target[point_id][1] - current[point_id][1])
        candidate = action.apply(current)
        candidate_score = float(score(candidate))
        if candidate_score < best_score - 1e-12:
            best_points, best_action, best_score = candidate, action, candidate_score
    return best_points, best_action
