from pathlib import Path

from geomed_copilot.production import EvaluationReplayService
from geomed_copilot.tools import GeoMedTools


def _tools() -> GeoMedTools:
    root = Path(__file__).parents[1] / "data"
    processed = root / "processed" / "hvangleest"
    return GeoMedTools(EvaluationReplayService(
        processed / "medimageinsight_locked_test_eval.json",
        processed,
        processed / "train.jsonl",
        root / "evidence" / "catalog.json",
    ))


def test_portable_evaluation_replay_exposes_real_locked_cases():
    tools = _tools()
    capabilities = tools.capabilities()
    assert capabilities["mode"] == "portable_locked_evaluation_replay"
    assert capabilities["available_cases"] == 176
    image_id = tools.list_available_cases(1)["cases"][0]
    result = tools.analyze_radiograph(image_id, top_k=3)
    assert result["provenance"]["mode"] == "portable_locked_evaluation_replay"
    assert result["provenance"]["evaluation_artifact_sha256"]
    assert result["provenance"]["split_alignment_verified"] is False
    assert sum(result["provenance"]["current_manifest_split_distribution"].values()) == 176
    assert set(result["evaluation"]["absolute_error"]) == {"HVA", "IMA"}
    analytical = {item["name"]: item["analytical_degrees"] for item in result["measurements"]}
    for name, target in result["evaluation"]["target"].items():
        assert abs(analytical[name] - target) < 0.1
    assert result["status"] == "complete"
