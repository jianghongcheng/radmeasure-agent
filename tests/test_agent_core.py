from geomed_copilot.agent_controller import MeasurementAgentController
from geomed_copilot.planner import ConstrainedMeasurementPlanner
from geomed_copilot.protocols import ProtocolRegistry


def test_constrained_planner_selects_registered_protocols():
    planner = ConstrainedMeasurementPlanner(ProtocolRegistry())
    plan = planner.plan("Measure hallux valgus angle and IMA")
    assert plan.action == "EXECUTE"
    assert plan.protocols == ("HVA", "IMA")
    assert "geometry_executor" in plan.tools


def test_unsupported_request_fails_closed():
    plan = ConstrainedMeasurementPlanner(ProtocolRegistry()).plan("Measure Cobb angle")
    assert plan.action == "STOP"
    assert plan.reason == "no_supported_protocol_requested"


def test_invalid_llm_tool_plan_fails_closed():
    class UnsafeModel:
        def complete(self, prompt):
            return '{"action":"EXECUTE","protocols":["HVA"],"tools":["shell"]}'

    plan = ConstrainedMeasurementPlanner(ProtocolRegistry(), UnsafeModel()).plan("Measure HVA")
    assert plan.action == "STOP"
    assert plan.reason == "invalid_or_unsafe_llm_plan"


def test_controller_repairs_once_then_keeps():
    registry = ProtocolRegistry()
    plan = ConstrainedMeasurementPlanner(registry).plan("Measure HVA")
    result = {"provenance": {"repair_geometry_source": "synthetic_fixture"}, "measurements": [{
        "name": "HVA", "predicted_degrees": 19.0, "analytical_degrees": 16.0,
        "discrepancy_degrees": 3.0, "status": "review_required",
    }]}
    outcome = MeasurementAgentController(registry).execute(plan, result)
    assert outcome.decision == "KEEP"
    assert outcome.repair_attempts == 1
    assert outcome.measurements[0]["predicted_degrees"] == 16.0
    assert [step["action"] for step in outcome.trajectory if "action" in step][-3:] == ["REPAIR", "KEEP", "KEEP"]


def test_controller_stops_when_repair_exceeds_policy():
    registry = ProtocolRegistry()
    plan = ConstrainedMeasurementPlanner(registry).plan("Measure HVA")
    result = {"measurements": [{
        "name": "HVA", "predicted_degrees": 30.0, "analytical_degrees": 10.0,
        "discrepancy_degrees": 20.0, "status": "review_required",
    }]}
    outcome = MeasurementAgentController(registry).execute(plan, result)
    assert outcome.decision == "STOP"
    assert outcome.reason == "repair_risk_exceeds_policy"


def test_controller_never_repairs_from_ground_truth_evaluation_geometry():
    registry = ProtocolRegistry()
    plan = ConstrainedMeasurementPlanner(registry).plan("Measure HVA")
    result = {
        "provenance": {"repair_geometry_source": "released_annotation_ground_truth"},
        "measurements": [{
            "name": "HVA", "predicted_degrees": 19.0, "analytical_degrees": 16.0,
            "discrepancy_degrees": 3.0, "status": "review_required",
        }],
    }
    outcome = MeasurementAgentController(registry).execute(plan, result)
    assert outcome.decision == "STOP"
    assert outcome.repair_attempts == 0
    assert outcome.reason == "independent_repair_proposal_unavailable"
    assert outcome.measurements[0]["predicted_degrees"] == 19.0
