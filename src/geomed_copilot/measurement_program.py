from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class AxisSpec:
    name: str
    start: str
    end: str


@dataclass(frozen=True)
class MeasurementProgram:
    """A small typed DSL for executable radiographic angle protocols."""

    name: str
    axes: tuple[AxisSpec, ...]
    operation: str
    inputs: tuple[str, str]

    def validate(self) -> None:
        if self.operation != "acute_angle":
            raise ValueError(f"unsupported operation: {self.operation}")
        names = [axis.name for axis in self.axes]
        if len(names) != len(set(names)):
            raise ValueError("axis names must be unique")
        if any(name not in names for name in self.inputs):
            raise ValueError("program input references an unknown axis")

    @property
    def required_points(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            point for axis in self.axes for point in (axis.start, axis.end)
        ))

    def to_dict(self) -> dict:
        return asdict(self)


def denormalize_points(points: Mapping[str, tuple[float, float]], width: float,
                       height: float) -> dict[str, tuple[float, float]]:
    """Restore an anisotropically normalized image coordinate frame."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return {name: (point[0] * width, point[1] * height)
            for name, point in points.items()}


def _orientation(first: tuple[float, float], second: tuple[float, float]) -> float:
    dx, dy = second[0] - first[0], second[1] - first[1]
    if abs(dx) + abs(dy) < 1e-12:
        raise ValueError("axis endpoints must be distinct")
    return math.degrees(math.atan2(dy, dx)) % 180.0


def execute_program(program: MeasurementProgram,
                    points: Mapping[str, tuple[float, float]]) -> float:
    program.validate()
    missing = set(program.required_points) - set(points)
    if missing:
        raise ValueError(f"missing points: {sorted(missing)}")
    axes = {
        axis.name: _orientation(points[axis.start], points[axis.end])
        for axis in program.axes
    }
    difference = abs(axes[program.inputs[0]] - axes[program.inputs[1]])
    return min(difference, 180.0 - difference)


HV_AXES = (
    AxisSpec("great_toe_axis", "gt_proximal", "gt_distal"),
    AxisSpec("first_metatarsal_axis", "m1_proximal", "m1_distal"),
    AxisSpec("second_metatarsal_axis", "m2_proximal", "m2_distal"),
)

HVA_PROGRAM = MeasurementProgram(
    "HVA", HV_AXES, "acute_angle", ("great_toe_axis", "first_metatarsal_axis")
)
IMA_PROGRAM = MeasurementProgram(
    "IMA", HV_AXES, "acute_angle", ("first_metatarsal_axis", "second_metatarsal_axis")
)
