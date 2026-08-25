import urllib.error
from pathlib import Path

from geomed_copilot.inference import LockedEvaluationAdapter, ModelRegistry
from geomed_copilot.inference_client import InferenceClient, InferenceUnavailable


def test_locked_adapter_exposes_version_hash_and_honest_capabilities():
    artifact = Path(__file__).parents[1] / "data/processed/hvangleest/medimageinsight_locked_test_eval.json"
    registry = ModelRegistry([LockedEvaluationAdapter(artifact)])
    model = registry.list_models()[0]
    assert model["artifact_sha256"]
    assert model["live_image_inference"] is False
    output = registry.predict(model["model_id"], "IMG000005.jpg")
    assert set(output["measurements"]) == {"HVA", "IMA"}


def test_inference_client_closes_circuit_after_success():
    expected = {"measurements": {"HVA": 1.0}, "model": {"version": "v1"}}
    client = InferenceClient("http://inference", "secret", transport=lambda request, timeout: expected)
    assert client.predict("case") == expected
    assert client.circuit_state == "closed"


def test_inference_client_sends_artifact_to_live_endpoint():
    seen = {}
    expected = {"measurements": {"HVA": 18.2, "IMA": 9.1}}

    def transport(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = request.data.decode()
        return expected

    client = InferenceClient("http://inference", "secret", transport=transport)
    assert client.predict_artifact("sha", "s3://bucket/key.jpg") == expected
    assert seen["url"].endswith("/v1/infer-artifact")
    assert '"model_id": "hvangle-resnet50"' in seen["body"]


def test_inference_client_opens_circuit_after_bounded_failures():
    calls = []

    def failing(request, timeout):
        calls.append(1)
        raise urllib.error.URLError("offline")

    client = InferenceClient(
        "http://inference", "secret", retries=1, failure_threshold=2,
        recovery_seconds=60, transport=failing,
    )
    for _ in range(2):
        try:
            client.predict("case")
        except InferenceUnavailable:
            pass
    assert len(calls) == 4
    assert client.circuit_state == "open"
    try:
        client.predict("case")
    except InferenceUnavailable as exc:
        assert "circuit is open" in str(exc)
    assert len(calls) == 4
