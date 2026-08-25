from pathlib import Path
import sqlite3

from geomed_copilot.jobs import SqliteJobRepository
from geomed_copilot.replay import build_replay_payload, replay_guarantee


def test_submit_is_idempotent_and_claim_is_atomic(tmp_path: Path):
    repository = SqliteJobRepository(tmp_path / "jobs.db")
    first, created = repository.submit("evaluation_analysis", {"image_id": "x"}, "same-key")
    second, created_again = repository.submit("evaluation_analysis", {"image_id": "ignored"}, "same-key")
    assert created is True and created_again is False
    assert first.job_id == second.job_id
    claimed = repository.claim_next()
    assert claimed.job_id == first.job_id
    assert claimed.status == "running" and claimed.attempts == 1
    assert repository.claim_next() is None
    assert [event["event_type"] for event in repository.events(first.job_id)] == ["submitted", "claimed"]


def test_retry_budget_terminates_failures(tmp_path: Path):
    repository = SqliteJobRepository(tmp_path / "jobs.db")
    job, _ = repository.submit("x", {}, "retry-key", max_attempts=2)
    repository.claim_next()
    retried = repository.record_failure(job.job_id, "temporary", "first", retryable=True)
    assert retried.status == "queued"
    repository.claim_next()
    failed = repository.record_failure(job.job_id, "temporary", "second", retryable=True)
    assert failed.status == "failed" and failed.attempts == 2
    assert [event["event_type"] for event in repository.events(job.job_id)] == [
        "submitted", "claimed", "retry_scheduled", "claimed", "failed"
    ]


def test_only_running_jobs_can_finish(tmp_path: Path):
    repository = SqliteJobRepository(tmp_path / "jobs.db")
    job, _ = repository.submit("x", {}, "state-key")
    try:
        repository.finish(job.job_id, "completed", {})
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("queued job must not transition directly to completed")


def test_human_review_can_correct_and_approve_once(tmp_path: Path):
    repository = SqliteJobRepository(tmp_path / "jobs.db")
    job, _ = repository.submit("uploaded_radiograph", {"artifact": {}}, "review-key")
    claimed = repository.claim_next()
    repository.finish(claimed.job_id, "needs_review", {"measurements": {"HVA": 20.0, "IMA": 8.0}})
    reviewed = repository.review(job.job_id, "radiologist-1", "approve",
                                 {"HVA": 19.5, "IMA": 8.2}, "adjusted axes")
    assert reviewed.status == "review_approved"
    assert reviewed.result["measurements"]["HVA"] == 19.5
    assert reviewed.result["review"]["reviewer"] == "radiologist-1"
    assert repository.events(job.job_id)[-1]["event_type"] == "review_approved"
    try:
        repository.review(job.job_id, "radiologist-2", "reject")
    except RuntimeError as exc:
        assert "not awaiting review" in str(exc)
    else:
        raise AssertionError("review must be immutable after final decision")


def test_expired_worker_lease_returns_job_to_queue(tmp_path: Path):
    path = tmp_path / "jobs.db"
    repository = SqliteJobRepository(path)
    job, _ = repository.submit("x", {}, "lease-key", max_attempts=3)
    claimed = repository.claim_next("crashed-worker", lease_seconds=60)
    assert claimed.worker_id == "crashed-worker"
    with sqlite3.connect(str(path)) as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE job_id=?",
            (job.job_id,),
        )
    recovered = repository.claim_next("replacement-worker")
    assert recovered.job_id == job.job_id
    assert recovered.worker_id == "replacement-worker"
    assert recovered.attempts == 2
    assert [event["event_type"] for event in repository.events(job.job_id)] == [
        "submitted", "claimed", "lease_expired", "claimed"
    ]


def test_trace_lookup_includes_original_and_lineage_linked_replay(tmp_path: Path):
    repository = SqliteJobRepository(tmp_path / "jobs.db")
    original, _ = repository.submit(
        "evaluation_analysis", {"image_id": "x", "_trace_id": "trace-original"}, "original-key"
    )
    repository.claim_next()
    completed = repository.finish(original.job_id, "completed", {"measurements": {}})

    replay_payload = build_replay_payload(completed, "trace-replay", "operator")
    replay, _ = repository.submit(completed.job_type, replay_payload, "replay-key")

    traced = repository.find_by_trace_id("trace-original")
    assert [job.job_id for job in traced] == [original.job_id, replay.job_id]
    assert replay.payload["_replay_of_job_id"] == original.job_id
    assert replay.payload["_trace_id"] == "trace-replay"
    assert replay_guarantee(completed) == "locked_artifact_same_inputs"


def test_running_job_cannot_be_replayed(tmp_path: Path):
    repository = SqliteJobRepository(tmp_path / "jobs.db")
    job, _ = repository.submit("evaluation_analysis", {"_trace_id": "t"}, "key")
    repository.claim_next()
    try:
        build_replay_payload(repository.get(job.job_id), "new-trace", "operator")
    except RuntimeError as exc:
        assert "terminal" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a running job must not be replayable")
