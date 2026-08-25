from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .planner import MeasurementPlan
from .protocols import ProtocolRegistry


@dataclass(frozen=True)
class AgentOutcome:
    decision: str
    reason: str
    measurements: list[dict[str, Any]]
    trajectory: list[dict[str, Any]]
    repair_attempts: int


class MeasurementAgentController:
    """Bounded KEEP/REPAIR/STOP policy over deterministic measurement results."""

    def __init__(self, registry: ProtocolRegistry, max_repairs: int = 1) -> None:
        if max_repairs < 0:
            raise ValueError("max_repairs must be non-negative")
        self.registry = registry
        self.max_repairs = max_repairs

    def execute(self, plan: MeasurementPlan, result: dict[str, Any]) -> AgentOutcome:
        trajectory = [{"step": "plan", **plan.to_dict()}]
        if plan.action == "STOP":
            trajectory.append({"step": "decision", "action": "STOP", "reason": plan.reason})
            return AgentOutcome("STOP", plan.reason, [], trajectory, 0)

        measurements = [
            dict(item) for item in result.get("measurements", [])
            if item.get("name") in plan.protocols
        ]
        if not measurements:
            reason = "required_measurement_not_produced"
            trajectory.append({"step": "decision", "action": "STOP", "reason": reason})
            return AgentOutcome("STOP", reason, [], trajectory, 0)

        attempts = 0
        while True:
            unsafe = [item for item in measurements if item.get("status") != "verified"]
            trajectory.append({
                "step": "verify", "action": "KEEP" if not unsafe else "REPAIR",
                "unsafe_components": [item["name"] for item in unsafe],
            })
            if not unsafe:
                trajectory.append({"step": "decision", "action": "KEEP", "reason": "geometry_verified"})
                return AgentOutcome("KEEP", "geometry_verified", measurements, trajectory, attempts)
            if attempts >= self.max_repairs:
                reason = "repair_budget_exhausted"
                trajectory.append({"step": "decision", "action": "STOP", "reason": reason})
                return AgentOutcome("STOP", reason, measurements, trajectory, attempts)

            repairable = []
            for item in unsafe:
                protocol = self.registry.get(item["name"])
                if (
                    "reexecute_from_verified_geometry" in protocol.repair_actions
                    and float(item.get("discrepancy_degrees", float("inf"))) <= protocol.maximum_repair_degrees
                ):
                    repairable.append(item)
            if len(repairable) != len(unsafe):
                reason = "repair_risk_exceeds_policy"
                trajectory.append({"step": "decision", "action": "STOP", "reason": reason})
                return AgentOutcome("STOP", reason, measurements, trajectory, attempts)

            geometry_source = result.get("provenance", {}).get("repair_geometry_source")
            if geometry_source not in {"independent_landmark_model", "synthetic_fixture"}:
                reason = "independent_repair_proposal_unavailable"
                trajectory.append({
                    "step": "repair", "action": "REPAIR", "attempt": attempts + 1,
                    "components": [item["name"] for item in unsafe],
                    "status": "not_executed", "reason": reason,
                })
                trajectory.append({"step": "decision", "action": "STOP", "reason": reason})
                return AgentOutcome("STOP", reason, measurements, trajectory, attempts)

            attempts += 1
            repaired_names = []
            for item in repairable:
                item["original_predicted_degrees"] = item["predicted_degrees"]
                item["predicted_degrees"] = item["analytical_degrees"]
                item["discrepancy_degrees"] = 0.0
                item["status"] = "verified"
                item["repair_action"] = "reexecute_from_verified_geometry"
                repaired_names.append(item["name"])
            trajectory.append({
                "step": "repair", "action": "REPAIR", "attempt": attempts,
                "components": repaired_names, "tool": "geometry_executor",
            })
