from pathlib import Path

from geomed_copilot.jobs import SqliteJobRepository
from geomed_copilot.pipeline import JobPipeline
from geomed_copilot.production import DemoService
from geomed_copilot.tools import GeoMedTools
from geomed_copilot.worker import Worker


def test_worker_completes_verified_evaluation_job(tmp_path: Path):
    repository = SqliteJobRepository(tmp_path / "jobs.db")
    job, _ = repository.submit(
        "evaluation_analysis", {"image_id": "demo-foot-001", "top_k": 2}, "work-key"
    )
    worker = Worker(repository, JobPipeline(GeoMedTools(DemoService())))
    assert worker.run_once() is True
    completed = repository.get(job.job_id)
    assert completed.status == "completed"
    assert completed.result["routing"]["decision"] == "KEEP"
    assert completed.result["agent_plan"]["protocols"] == ["HVA", "IMA"]
    assert completed.result["agent_trajectory"][-1]["action"] == "KEEP"
    assert completed.result["trace_id"] is None


def test_uploaded_job_routes_to_human_review_without_fake_inference(tmp_path: Path):
    repository = SqliteJobRepository(tmp_path / "jobs.db")
    job, _ = repository.submit(
        "uploaded_radiograph", {"artifact": {"sha256": "abc"}}, "upload-key"
    )
    Worker(repository, JobPipeline(GeoMedTools(DemoService()))).run_once()
    reviewed = repository.get(job.job_id)
    assert reviewed.status == "needs_review"
    assert reviewed.result["routing"]["reason"] == "live_inference_adapter_unavailable"


def test_uploaded_job_runs_live_model_and_preserves_review_gate(tmp_path: Path):
    class FakeLiveClient:
        def predict_artifact(self, image_id, artifact_uri, media_type="image/jpeg"):
            assert image_id == "abc"
            assert artifact_uri == "s3://bucket/key.jpg"
            assert media_type == "image/jpeg"
            return {
                "measurements": {"HVA": 18.2, "IMA": 9.1},
                "model": {"model_id": "hvangle-resnet50", "live_image_inference": True},
                "quality": {"passed": True, "reasons": []},
                "image_metadata": {},
            }

    repository = SqliteJobRepository(tmp_path / "jobs.db")
    job, _ = repository.submit("uploaded_radiograph", {
        "artifact": {"sha256": "abc", "path": "s3://bucket/key.jpg"}
    }, "live-upload-key")
    pipeline = JobPipeline(GeoMedTools(DemoService()), FakeLiveClient())
    Worker(repository, pipeline).run_once()
    reviewed = repository.get(job.job_id)
    assert reviewed.status == "needs_review"
    assert reviewed.result["measurements"] == {"HVA": 18.2, "IMA": 9.1}
    assert reviewed.result["provenance"]["live_encoder_inference"] is True
    assert reviewed.result["routing"]["reason"] == "first_pass_live_model_requires_review"


def test_uploaded_job_rejects_independent_repair_on_cross_model_disagreement(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GEOMED_REPAIR_MODEL_ID", "hvangle-hrnet-repair")

    class FakeTwoModelClient:
        def predict_artifact(self, image_id, artifact_uri, media_type="image/jpeg", model_id=None):
            common = {
                "quality": {"passed": True, "reasons": []}, "image_metadata": {},
            }
            if model_id:
                return {
                    **common, "measurements": {"HVA": 90.0, "IMA": 90.0},
                    "model": {"model_id": model_id},
                    "repair_proposal": {
                        "accepted": True, "confidence": .82, "threshold": .8,
                        "measurements": {"HVA": 49.0, "IMA": 78.0},
                        "geometry_source": "independent_landmark_model",
                        "action": "learned_residual_landmark_repair", "maximum_steps": 1,
                    },
                }
            return {**common, "measurements": {"HVA": 30.0, "IMA": 10.0},
                    "model": {"model_id": "primary-resnet"}}

    repository = SqliteJobRepository(tmp_path / "jobs.db")
    job, _ = repository.submit("uploaded_radiograph", {
        "artifact": {"sha256": "abc", "path": "s3://bucket/key.jpg"}
    }, "independent-repair-key")
    Worker(repository, JobPipeline(GeoMedTools(DemoService()), FakeTwoModelClient())).run_once()
    reviewed = repository.get(job.job_id)
    assert reviewed.status == "needs_review"
    assert reviewed.result["routing"]["reason"] == "cross_model_disagreement"
    assert reviewed.result["measurements"] == {"HVA": 30.0, "IMA": 10.0}
    assert reviewed.result["repair_proposal"]["accepted"] is False
