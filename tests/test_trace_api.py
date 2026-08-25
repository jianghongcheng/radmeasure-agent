import json
import asyncio

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from geomed_copilot.api import create_app
from geomed_copilot.jobs import SqliteJobRepository


def request(app, method, path, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(send())


def test_trace_query_and_replay_endpoint_preserve_lineage(tmp_path, monkeypatch):
    database = tmp_path / "jobs.db"
    monkeypatch.setenv("GEOMED_DEMO_MODE", "1")
    monkeypatch.setenv("GEOMED_JOB_DB", str(database))
    monkeypatch.setenv(
        "GEOMED_API_KEYS",
        json.dumps({
            "viewer": {"name": "viewer", "role": "viewer"},
            "operator": {"name": "operator", "role": "operator"},
        }),
    )
    app = create_app()

    submitted = request(
        app,
        "POST",
        "/v1/jobs",
        json={"image_id": "demo-foot-001"},
        headers={
            "x-api-key": "operator",
            "idempotency-key": "original",
            "x-request-id": "trace-original",
        },
    )
    assert submitted.status_code == 202
    original_id = submitted.json()["job"]["job_id"]

    repository = SqliteJobRepository(database)
    claimed = repository.claim_next("test-worker")
    repository.finish(claimed.job_id, "completed", {"trace_id": "trace-original"})

    trace = request(
        app, "GET",
        "/v1/traces/trace-original", headers={"x-api-key": "viewer"}
    )
    assert trace.status_code == 200
    assert [run["job"]["job_id"] for run in trace.json()["runs"]] == [original_id]

    replayed = request(
        app, "POST",
        f"/v1/jobs/{original_id}/replay",
        headers={
            "x-api-key": "operator",
            "idempotency-key": "replay",
            "x-request-id": "trace-replay",
        },
    )
    assert replayed.status_code == 202
    assert replayed.json()["guarantee"] == "locked_artifact_same_inputs"

    lineage = request(
        app, "GET",
        "/v1/traces/trace-original", headers={"x-api-key": "viewer"}
    ).json()["runs"]
    assert len(lineage) == 2
    assert lineage[1]["job"]["payload"]["_replay_of_job_id"] == original_id


def test_nonterminal_job_replay_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOMED_DEMO_MODE", "1")
    monkeypatch.setenv("GEOMED_JOB_DB", str(tmp_path / "jobs.db"))
    monkeypatch.setenv(
        "GEOMED_API_KEYS",
        json.dumps({"operator": {"name": "operator", "role": "operator"}}),
    )
    app = create_app()
    submitted = request(
        app, "POST",
        "/v1/jobs",
        json={"image_id": "demo-foot-001"},
        headers={"x-api-key": "operator", "idempotency-key": "queued"},
    ).json()
    response = request(
        app, "POST",
        f"/v1/jobs/{submitted['job']['job_id']}/replay",
        headers={"x-api-key": "operator", "idempotency-key": "not-allowed"},
    )
    assert response.status_code == 409


def test_protocol_registry_and_constrained_plan_are_exposed(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOMED_DEMO_MODE", "1")
    monkeypatch.setenv("GEOMED_JOB_DB", str(tmp_path / "jobs.db"))
    monkeypatch.setenv(
        "GEOMED_API_KEYS",
        json.dumps({"viewer": {"name": "viewer", "role": "viewer"}}),
    )
    app = create_app()
    protocols = request(app, "GET", "/v1/protocols")
    assert protocols.status_code == 200
    assert {item["name"] for item in protocols.json()["protocols"]} == {"HVA", "IMA"}

    plan = request(
        app, "POST", "/v1/plan",
        headers={"x-api-key": "viewer"},
        json={"request": "Measure hallux valgus angle"},
    )
    assert plan.status_code == 200
    assert plan.json()["action"] == "EXECUTE"
    assert plan.json()["protocols"] == ["HVA"]
