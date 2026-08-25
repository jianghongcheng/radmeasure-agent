#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from geomed_copilot.artifact_predictor import FrozenPredictionArtifact


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def evaluate(predictions: Path, annotations: Path) -> dict:
    predictor = FrozenPredictionArtifact(predictions)
    truth = {}
    with annotations.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["filename"] in predictor.identifiers:
                if row["filename"] in truth:
                    raise ValueError("Locked artifact must select one foot per image")
                truth[row["filename"]] = {"HVA": float(row["HVA"]), "IMA": float(row["IMA"])}
    if set(truth) != predictor.identifiers:
        raise ValueError("Annotation table and prediction identifiers do not match")

    errors = {"HVA": [], "IMA": []}
    predictions_out = []
    for identifier in sorted(truth):
        predicted = predictor.predict(identifier)
        predictions_out.append({"image_id": identifier, "predicted": predicted, "target": truth[identifier]})
        for name in errors:
            errors[name].append(abs(predicted[name] - truth[identifier][name]))

    metrics = {}
    for name, values in errors.items():
        tolerance = 5.0 if name == "HVA" else 3.0
        metrics[name] = {
            "mae_degrees": statistics.mean(values),
            "median_ae_degrees": statistics.median(values),
            "p95_ae_degrees": percentile(values, .95),
            "within_tolerance_rate": sum(value <= tolerance for value in values) / len(values),
            "tolerance_degrees": tolerance,
        }
    return {
        "evaluation_type": "locked_patient_disjoint_test_artifact_replay",
        "model": "MedImageInsight frozen encoder + spatial line readout, 3-seed axis ensemble",
        "n_test": len(truth),
        "prediction_artifact_sha256": predictor.sha256,
        "metrics": metrics,
        "predictions": predictions_out,
        "limitations": [
            "Replays persisted model predictions; it is not live encoder inference.",
            "The locked benchmark contains unilateral images only.",
            "Research use only; no prospective clinical validation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.predictions, args.annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {**{key: value for key, value in result.items() if key != "predictions"}}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

