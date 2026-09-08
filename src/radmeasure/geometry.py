from __future__ import annotations

import math

from .models import Line, Measurement


def line_angle_degrees(line: Line) -> float:
    """Return an undirected line orientation in [0, 180) degrees."""
    dx = line.end.x - line.start.x
    dy = line.end.y - line.start.y
    if dx == 0 and dy == 0:
        raise ValueError("A measurement line must contain two distinct points")
    return math.degrees(math.atan2(dy, dx)) % 180.0


def acute_angle_degrees(first: Line, second: Line) -> float:
    difference = abs(line_angle_degrees(first) - line_angle_degrees(second))
    return min(difference, 180.0 - difference)


def verify_measurement(
    name: str,
    predicted_degrees: float,
    first: Line,
    second: Line,
    tolerance_degrees: float = 3.0,
) -> Measurement:
    analytical = acute_angle_degrees(first, second)
    discrepancy = abs(float(predicted_degrees) - analytical)
    return Measurement(
        name=name,
        predicted_degrees=round(float(predicted_degrees), 3),
        analytical_degrees=round(analytical, 3),
        discrepancy_degrees=round(discrepancy, 3),
        status="verified" if discrepancy <= tolerance_degrees else "review_required",
    )

