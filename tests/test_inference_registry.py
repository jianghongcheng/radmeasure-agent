import pytest

from radmeasure import inference_api


def test_registry_without_artifacts_reports_missing_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("RADMEASURE_DATA_ROOT", str(tmp_path))
    for key in ("RADMEASURE_RESNET_CHECKPOINT", "RADMEASURE_LANDMARK_CHECKPOINT", "RADMEASURE_REPAIR_CHECKPOINT"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="No inference adapters configured"):
        inference_api.create_registry()


def test_live_adapter_does_not_require_historical_predictions(tmp_path, monkeypatch):
    monkeypatch.setenv("RADMEASURE_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("RADMEASURE_RESNET_CHECKPOINT", str(tmp_path / "model.pt"))
    monkeypatch.delenv("RADMEASURE_LANDMARK_CHECKPOINT", raising=False)
    monkeypatch.delenv("RADMEASURE_REPAIR_CHECKPOINT", raising=False)
    adapters = []
    sentinel = object()
    monkeypatch.setattr(inference_api, "ResNet50AngleAdapter", lambda path, device: sentinel)
    monkeypatch.setattr(inference_api, "ModelRegistry", lambda values: adapters.extend(values))
    inference_api.create_registry()
    assert adapters == [sentinel]


def test_present_replay_artifact_is_registered(tmp_path, monkeypatch):
    artifact = tmp_path / "processed/hvangleest/medimageinsight_locked_test_eval.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}")
    monkeypatch.setenv("RADMEASURE_DATA_ROOT", str(tmp_path))
    for key in ("RADMEASURE_RESNET_CHECKPOINT", "RADMEASURE_LANDMARK_CHECKPOINT", "RADMEASURE_REPAIR_CHECKPOINT"):
        monkeypatch.delenv(key, raising=False)
    seen = []
    monkeypatch.setattr(inference_api, "LockedEvaluationAdapter", lambda path: seen.append(path))
    monkeypatch.setattr(inference_api, "ModelRegistry", lambda values: values)
    inference_api.create_registry()
    assert seen == [artifact]
