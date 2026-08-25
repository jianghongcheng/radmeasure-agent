from geomed_copilot.metrics import HttpMetrics, normalized_path


def test_metrics_normalize_job_ids_and_render_prometheus():
    assert normalized_path("/v1/jobs/123") == "/v1/jobs/{job_id}"
    assert normalized_path("/v1/jobs/123/events") == "/v1/jobs/{job_id}/events"
    metrics = HttpMetrics()
    metrics.observe("GET", "/v1/jobs/secret-id", 200, 0.012)
    output = metrics.render({"queued": 2})
    assert "secret-id" not in output
    assert 'path="/v1/jobs/{job_id}"' in output
    assert 'geomed_jobs{status="queued"} 2' in output
