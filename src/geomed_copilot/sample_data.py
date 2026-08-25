from __future__ import annotations

from .models import Evidence, Line, Point


EVIDENCE = [
    Evidence(
        "guideline-hva",
        "Hallux valgus radiographic measurement guidance",
        "HVA is measured between the first metatarsal and proximal phalanx axes. "
        "Measurements should be checked against landmark quality and acquisition conditions.",
        "https://example.org/guideline-hva",
        metadata={"measurement": "HVA"},
    ),
    Evidence(
        "method-geomed",
        "GeoMed relational geometry tokenization",
        "Predicted anatomical geometry can be represented explicitly and compared with "
        "analytical angle reconstruction to expose disagreement and support auditing.",
        "https://example.org/geomed",
        metadata={"measurement": "HVA"},
    ),
]

CASES = [
    Evidence(
        "case-001",
        "Reference HVA case 001",
        "De-identified public benchmark case with verified measurements.",
        "https://example.org/cases/001",
        "case",
        {"measurements": {"HVA": 15.0, "IMA": 8.0}},
    ),
    Evidence(
        "case-002",
        "Reference HVA case 002",
        "De-identified public benchmark case with verified measurements.",
        "https://example.org/cases/002",
        "case",
        {"measurements": {"HVA": 30.0, "IMA": 14.0}},
    ),
]

DEMO_LANDMARKS = {
    "first_metatarsal": Line(Point(0, 0), Point(10, 0)),
    "great_toe": Line(Point(0, 0), Point(10, 2.6795)),
    "second_metatarsal": Line(Point(0, 0), Point(10, 1.408)),
}
