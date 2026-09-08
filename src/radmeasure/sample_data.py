from __future__ import annotations

from .models import Evidence, Line, Point


EVIDENCE = [
    Evidence(
        "guideline-hva",
        "Synthetic HVA measurement note",
        "HVA is measured between the first metatarsal and proximal phalanx axes. "
        "Synthetic demonstration text, not a clinical guideline or citation.",
        "https://example.org/guideline-hva",
        metadata={"measurement": "HVA"},
    ),
    Evidence(
        "method-geomed",
        "Synthetic geometry verification note",
        "Predicted anatomical geometry can be represented explicitly and compared with "
        "analytical angle reconstruction. Synthetic demonstration text, not a published source.",
        "https://example.org/geomed",
        metadata={"measurement": "HVA"},
    ),
]

CASES = [
    Evidence(
        "case-001",
        "Synthetic HVA case 001",
        "Fabricated case for software demonstration; not a patient or benchmark record.",
        "https://example.org/cases/001",
        "case",
        {"measurements": {"HVA": 15.0, "IMA": 8.0}},
    ),
    Evidence(
        "case-002",
        "Synthetic HVA case 002",
        "Fabricated case for software demonstration; not a patient or benchmark record.",
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
