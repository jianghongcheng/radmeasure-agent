import json

from test_trace_api import request
from radmeasure.api import create_app
from radmeasure.dashboard import render_dashboard


def test_api_rejects_sql_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("RADMEASURE_DEMO_MODE", "1")
    monkeypatch.delenv("RADMEASURE_EVAL_REPLAY", raising=False)
    monkeypatch.setenv("RADMEASURE_JOB_DB", str(tmp_path / "jobs.sqlite"))
    monkeypatch.setenv("RADMEASURE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RADMEASURE_API_KEYS", json.dumps({
        "operator": {"name": "test", "role": "operator"},
    }))
    app = create_app()
    response = request(app, "POST", "/v1/jobs",
        json={"image_id": "demo-foot-001", "job_type": "sql_analysis", "sql": "SELECT 1"},
        headers={"x-api-key": "operator", "idempotency-key": "sql-rejected"})
    assert response.status_code == 422
    capabilities = request(app, "GET", "/v1/capabilities").json()
    assert capabilities["domain"] == "medical_imaging_measurement"
    assert capabilities["measurements"] == ["HVA", "IMA"]


def test_dashboard_does_not_present_historical_accuracy_as_live():
    html = render_dashboard()
    assert "HVA + IMA" in html
    assert "Research only" in html
    assert "3.56°" not in html
    assert "persisted evaluation cases" not in html
