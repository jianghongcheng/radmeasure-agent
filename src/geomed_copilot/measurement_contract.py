"""Derive measurement output requirements from the existing protocol registry."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .execution_record import ContractSnapshot
from .protocols import MeasurementProtocol, ProtocolRegistry


@dataclass(frozen=True)
class MeasurementContract:
    protocols: tuple[MeasurementProtocol, ...]
    mandatory_review: bool = False

    @classmethod
    def from_registry(cls, registry: ProtocolRegistry, names: tuple[str, ...],
                      mandatory_review: bool = False) -> MeasurementContract:
        if len(set(names)) != len(names):
            raise ValueError("duplicate requested protocols")
        return cls(tuple(registry.get(name) for name in names), mandatory_review)

    def snapshot(self) -> ContractSnapshot:
        return ContractSnapshot.capture("radiographic_measurement", {
            "version": "1", "protocols": [p.to_dict() for p in self.protocols],
            "units": "degrees", "mandatory_review": self.mandatory_review,
            "checks": ["required_measurements", "unique_names", "finite_angles",
                       "geometry_fields_consistent_when_present"],
            "scope": "structural_and_geometric_checks_not_clinical_validity",
        })

    def validate(self, measurements: Any, *, geometry: bool) -> tuple[str, ...]:
        required = {p.name for p in self.protocols}
        if isinstance(measurements, dict) and not geometry:
            rows = [{"name": name, "predicted_degrees": value}
                    for name, value in measurements.items()]
        elif geometry and isinstance(measurements, list) and all(isinstance(row, dict) for row in measurements):
            rows = measurements
        else:
            return ("invalid_measurement_structure",)
        errors: list[str] = []
        selected = [row for row in rows if row.get("name") in required]
        names = [row["name"] for row in selected]
        if required - set(names):
            errors.append("required_measurement_not_produced")
        if len(names) != len(set(names)):
            errors.append("duplicate_measurements")
        for row in selected:
            keys = ("predicted_degrees", "analytical_degrees", "discrepancy_degrees") if geometry else ("predicted_degrees",)
            values = [row.get(key) for key in keys]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   for v in values):
                errors.append("invalid_measurement_numbers")
                continue
            if geometry:
                discrepancy = abs(row["predicted_degrees"] - row["analytical_degrees"])
                if row["discrepancy_degrees"] < 0 or not math.isclose(
                        discrepancy, row["discrepancy_degrees"], abs_tol=0.002):
                    errors.append("inconsistent_geometry_evidence")
                if row.get("status") not in {"verified", "review_required"}:
                    errors.append("invalid_measurement_status")
        return tuple(dict.fromkeys(errors))
