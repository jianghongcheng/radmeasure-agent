import pytest

from geomed_copilot.geometry_repair import (
    MovePoint, axis_rotation_corruption, oracle_repair_step, program_aware_oracle_step,
)
from geomed_copilot.measurement_program import (
    HVA_PROGRAM, IMA_PROGRAM, denormalize_points, execute_program,
)


POINTS = {
    "gt_proximal": (0.0, 0.0), "gt_distal": (1.0, 1.0),
    "m1_proximal": (0.0, 0.0), "m1_distal": (1.0, 0.0),
    "m2_proximal": (0.0, 0.0), "m2_distal": (0.0, 1.0),
}


def test_typed_program_executes_hva_and_ima():
    assert execute_program(HVA_PROGRAM, POINTS) == pytest.approx(45.0)
    assert execute_program(IMA_PROGRAM, POINTS) == pytest.approx(90.0)


def test_repair_actions_are_executable_and_bounded():
    moved = MovePoint("gt_distal", 2.0, -2.0).apply(POINTS)
    assert moved["gt_distal"] == (1.0, 0.0)
    rotated = axis_rotation_corruption(POINTS, "m1_proximal", "m1_distal", 45)
    assert execute_program(HVA_PROGRAM, rotated) < execute_program(HVA_PROGRAM, POINTS)


def test_oracle_step_repairs_most_displaced_point():
    current = dict(POINTS)
    current["gt_proximal"] = (.4, .3)
    repaired, action = oracle_repair_step(current, POINTS)
    assert action.point_id == "gt_proximal"
    assert repaired["gt_proximal"] == POINTS["gt_proximal"]


def test_denormalization_preserves_coordinate_frame_before_execution():
    points = dict(POINTS)
    points["gt_distal"] = (1.0, 0.5)
    normalized_angle = execute_program(HVA_PROGRAM, points)
    pixel_angle = execute_program(HVA_PROGRAM, denormalize_points(points, 100, 200))
    assert normalized_angle != pixel_angle
    assert pixel_angle == pytest.approx(45.0)


def test_program_aware_oracle_can_choose_safe_noop():
    repaired, action = program_aware_oracle_step(POINTS, POINTS, lambda points: 0.0)
    assert repaired == POINTS
    assert action is None
