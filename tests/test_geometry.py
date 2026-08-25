import pytest

from geomed_copilot.geometry import acute_angle_degrees, verify_measurement
from geomed_copilot.models import Line, Point


def test_acute_angle_is_orientation_invariant():
    horizontal = Line(Point(0, 0), Point(10, 0))
    vertical_reversed = Line(Point(0, 10), Point(0, 0))
    assert acute_angle_degrees(horizontal, vertical_reversed) == pytest.approx(90.0)


def test_verification_flags_large_disagreement():
    horizontal = Line(Point(0, 0), Point(10, 0))
    diagonal = Line(Point(0, 0), Point(10, 10))
    result = verify_measurement("HVA", 20.0, horizontal, diagonal, tolerance_degrees=3)
    assert result.analytical_degrees == pytest.approx(45.0)
    assert result.status == "review_required"

