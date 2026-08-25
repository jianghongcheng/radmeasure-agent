#!/usr/bin/env python3
"""Frozen policy-unit benchmark for KEEP/REPAIR/STOP decision quality."""
from __future__ import annotations

import json
from pathlib import Path

from geomed_copilot.agent_controller import MeasurementAgentController
from geomed_copilot.planner import ConstrainedMeasurementPlanner
from geomed_copilot.protocols import ProtocolRegistry


def measurement(name, predicted, analytical, discrepancy, status):
    return {"name": name, "predicted_degrees": predicted,
            "analytical_degrees": analytical, "discrepancy_degrees": discrepancy,
            "status": status}


CASES = [
    ("keep-hva", "Measure HVA", [measurement("HVA", 16, 15, 1, "verified")], {}, "KEEP", False),
    ("keep-ima", "Measure IMA", [measurement("IMA", 9, 8, 1, "verified")], {}, "KEEP", False),
    ("keep-both", "Measure HVA and IMA", [measurement("HVA", 16, 15, 1, "verified"), measurement("IMA", 9, 8, 1, "verified")], {}, "KEEP", False),
    ("repair-hva", "Measure HVA", [measurement("HVA", 19, 16, 3, "review_required")], {"repair_geometry_source": "independent_landmark_model"}, "KEEP", True),
    ("repair-ima", "Measure IMA", [measurement("IMA", 11, 8, 3, "review_required")], {"repair_geometry_source": "independent_landmark_model"}, "KEEP", True),
    ("repair-both", "Measure HVA and IMA", [measurement("HVA", 19, 16, 3, "review_required"), measurement("IMA", 10, 8, 2, "review_required")], {"repair_geometry_source": "independent_landmark_model"}, "KEEP", True),
    ("stop-no-proposal-hva", "Measure HVA", [measurement("HVA", 19, 16, 3, "review_required")], {"repair_geometry_source": "released_annotation_ground_truth"}, "STOP", False),
    ("stop-no-proposal-ima", "Measure IMA", [measurement("IMA", 11, 8, 3, "review_required")], {}, "STOP", False),
    ("stop-risk-hva", "Measure HVA", [measurement("HVA", 35, 15, 20, "review_required")], {"repair_geometry_source": "independent_landmark_model"}, "STOP", False),
    ("stop-risk-ima", "Measure IMA", [measurement("IMA", 20, 8, 12, "review_required")], {"repair_geometry_source": "independent_landmark_model"}, "STOP", False),
    ("stop-unsupported", "Measure Cobb angle", [], {}, "STOP", False),
    ("stop-missing", "Measure HVA", [], {"repair_geometry_source": "independent_landmark_model"}, "STOP", False),
]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def main():
    registry = ProtocolRegistry()
    planner = ConstrainedMeasurementPlanner(registry)
    controller = MeasurementAgentController(registry)
    rows = []
    for case_id, request, measurements, provenance, expected, should_repair in CASES:
        plan = planner.plan(request)
        payload = {"measurements": measurements, "provenance": provenance}
        first = controller.execute(plan, payload)
        second = controller.execute(plan, payload)
        repaired = any(step.get("step") == "repair" and step.get("status") != "not_executed"
                       for step in first.trajectory)
        rows.append({
            "id": case_id, "expected_decision": expected, "decision": first.decision,
            "expected_repair": should_repair, "repair_executed": repaired,
            "task_success": first.decision == expected,
            "unsafe_action": first.decision == "KEEP" and expected == "STOP",
            "replay_consistent": first == second,
            "trajectory_steps": len(first.trajectory), "reason": first.reason,
        })
    expected_stop = sum(row["expected_decision"] == "STOP" for row in rows)
    predicted_stop = sum(row["decision"] == "STOP" for row in rows)
    true_stop = sum(row["expected_decision"] == row["decision"] == "STOP" for row in rows)
    expected_repair = sum(row["expected_repair"] for row in rows)
    executed_repair = sum(row["repair_executed"] for row in rows)
    correct_repair = sum(row["expected_repair"] and row["repair_executed"] for row in rows)
    summary = {
        "benchmark": "agent_decision_policy_v1", "n": len(rows),
        "scope": "policy-unit benchmark; not a clinical outcome benchmark",
        "task_success_rate": sum(row["task_success"] for row in rows) / len(rows),
        "unsafe_action_rate": sum(row["unsafe_action"] for row in rows) / len(rows),
        "repair_precision": ratio(correct_repair, executed_repair),
        "repair_recall": ratio(correct_repair, expected_repair),
        "stop_precision": ratio(true_stop, predicted_stop),
        "stop_recall": ratio(true_stop, expected_stop),
        "coverage": sum(row["decision"] == "KEEP" for row in rows) / len(rows),
        "mean_trajectory_steps": sum(row["trajectory_steps"] for row in rows) / len(rows),
        "replay_consistency": sum(row["replay_consistent"] for row in rows) / len(rows),
    }
    output = Path("outputs/portfolio/agent_decision_policy_v1.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "cases": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
