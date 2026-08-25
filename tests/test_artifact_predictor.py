import csv

import pytest

from geomed_copilot.artifact_predictor import FrozenPredictionArtifact


def test_artifact_predictor_ensembles_three_seed_axes(tmp_path):
    path = tmp_path / "predictions.csv"
    fields = ["identifier", "seed"] + [f"{name}_axis_{axis}" for name in
             ("great_toe", "first_metatarsal", "second_metatarsal") for axis in ("dx", "dy")]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed in (17, 42, 73):
            writer.writerow({
                "identifier": "case.jpg", "seed": seed,
                "great_toe_axis_dx": 1, "great_toe_axis_dy": 0,
                "first_metatarsal_axis_dx": 0, "first_metatarsal_axis_dy": 1,
                "second_metatarsal_axis_dx": 1, "second_metatarsal_axis_dy": 1,
            })
    predictor = FrozenPredictionArtifact(path)
    result = predictor.predict("case.jpg")
    assert result["HVA"] == pytest.approx(90)
    assert result["IMA"] == pytest.approx(45)
    with pytest.raises(KeyError, match="live encoder inference is unavailable"):
        predictor.predict("unknown.jpg")

