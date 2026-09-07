import json
from dataclasses import replace

import pytest

from geomed_copilot.agent_controller import MeasurementAgentController
from geomed_copilot.jobs import SqliteJobRepository
from geomed_copilot.pipeline import JobPipeline
from geomed_copilot.planner import ConstrainedMeasurementPlanner
from geomed_copilot.production import DemoService
from geomed_copilot.protocols import ProtocolRegistry, DEFAULT_PROTOCOLS
from geomed_copilot.replay import build_replay_payload
from geomed_copilot.tools import GeoMedTools
from geomed_copilot.worker import Worker


def test_missing_requested_measurement_cannot_complete():
    registry = ProtocolRegistry()
    plan = ConstrainedMeasurementPlanner(registry).plan("Measure HVA and IMA")
    result = DemoService().analyze("demo-foot-001", "Measure HVA and IMA")
    result["measurements"] = [row for row in result["measurements"] if row["name"] == "HVA"]
    outcome = MeasurementAgentController(registry).execute(plan, result)
    assert outcome.decision == "STOP"
    assert outcome.reason == "required_measurement_not_produced"
    assert outcome.repair_attempts == 0


@pytest.mark.parametrize("corruption,reason", [
    ("duplicate", "duplicate_measurements"),
    ("nan", "invalid_measurement_numbers"),
    ("mismatch", "inconsistent_geometry_evidence"),
])
def test_measurement_contract_rejects_corrupt_evidence(corruption, reason):
    registry = ProtocolRegistry()
    plan = ConstrainedMeasurementPlanner(registry).plan("Measure HVA and IMA")
    result = DemoService().analyze("demo-foot-001", "Measure HVA and IMA")
    if corruption == "duplicate":
        result["measurements"].append(dict(result["measurements"][0]))
    elif corruption == "nan":
        result["measurements"][0]["predicted_degrees"] = float("nan")
    else:
        result["measurements"][0]["discrepancy_degrees"] = 500
    outcome = MeasurementAgentController(registry).execute(plan, result)
    assert outcome.decision == "STOP"
    assert outcome.reason == reason


def test_medical_record_persists_and_replay_checks_contract(tmp_path):
    repository = SqliteJobRepository(tmp_path / "jobs.db")
    job, _ = repository.submit("evaluation_analysis", {"image_id": "demo-foot-001"}, "first")
    tools = GeoMedTools(DemoService())
    Worker(repository, JobPipeline(tools)).run_once()
    original = repository.get(job.job_id)
    record = original.result["execution_record"]
    assert record["contract"]["domain"] == "radiographic_measurement"
    assert record["budgets"]["job_attempt"] == 1
    assert record["budgets"]["agent_repairs"] == 0
    assert {event["phase"] for event in record["events"]} == {"collect", "act", "verify", "decide"}
    payload = build_replay_payload(original, "new-trace", "tester")
    replay, _ = repository.submit("evaluation_analysis", payload, "replay")
    Worker(repository, JobPipeline(tools)).run_once()
    assert repository.get(replay.job_id).status == "completed"

    changed = ProtocolRegistry(tuple(replace(p, maximum_repair_degrees=2.0) for p in DEFAULT_PROTOCOLS))
    drift, _ = repository.submit("evaluation_analysis", payload, "drift")
    Worker(repository, JobPipeline(tools, controller=MeasurementAgentController(changed))).run_once()
    failed = repository.get(drift.job_id)
    assert failed.status == "failed"
    assert "contract differs" in failed.error["message"]


def test_non_medical_job_is_rejected_without_invoking_tools(tmp_path):
    repository = SqliteJobRepository(tmp_path / "jobs.db")
    job, _ = repository.submit("sql_analysis", {"question": "names"}, "sql", max_attempts=2)
    class UnusedTools:
        def analyze_radiograph(self, **kwargs):
            pytest.fail("a non-medical job must not call measurement tools")

    worker = Worker(repository, JobPipeline(UnusedTools()))
    worker.run_once()
    failed = repository.get(job.job_id)
    assert failed.status == "failed"
    assert "radiographic measurement jobs only" in failed.error["message"]
    assert failed.result is None


def test_live_invalid_numbers_are_reviewable_and_json_safe(tmp_path):
    class Client:
        def predict_artifact(self, **kwargs):
            return {"measurements": {"HVA": float("nan"), "IMA": 9.0},
                    "model": {"model_id": "fake"}, "quality": {"passed": True}}

    repository = SqliteJobRepository(tmp_path / "jobs.db")
    job, _ = repository.submit("uploaded_radiograph", {
        "artifact": {"sha256": "abc", "path": "s3://test/image"}}, "invalid")
    Worker(repository, JobPipeline(GeoMedTools(DemoService()), Client())).run_once()
    reviewed = repository.get(job.job_id)
    assert reviewed.status == "needs_review"
    assert reviewed.result["routing"]["reason"] == "invalid_measurement_numbers"
    json.dumps(reviewed.result, allow_nan=False)


def test_valid_live_repair_still_requires_review(tmp_path, monkeypatch):
    monkeypatch.delenv("GEOMED_REPAIR_MODEL_ID", raising=False)

    class Client:
        def predict_artifact(self, **kwargs):
            return {"measurements": {"HVA": 18.0, "IMA": 9.0}, "model": {"model_id": "fake"},
                    "quality": {"passed": True}, "repair_proposal": {
                        "accepted": True, "measurements": {"HVA": 17.0, "IMA": 8.0},
                        "confidence": .9, "threshold": .8, "action": "repair", "maximum_steps": 1}}

    repository = SqliteJobRepository(tmp_path / "jobs.db")
    job, _ = repository.submit("uploaded_radiograph", {
        "artifact": {"sha256": "abc", "path": "s3://test/image"}}, "repair")
    Worker(repository, JobPipeline(GeoMedTools(DemoService()), Client())).run_once()
    reviewed = repository.get(job.job_id)
    assert reviewed.status == "needs_review"
    record = reviewed.result["execution_record"]
    assert record["contract"]["specification"]["mandatory_review"] is True
    assert record["reason"] == "post_repair_human_review_required"
    assert record["budgets"]["agent_repairs"] == 1
    approved = repository.review(job.job_id, "reviewer", "approve")
    assert approved.result["execution_record"] == record
