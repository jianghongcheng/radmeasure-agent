#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from geomed_copilot.artifact_predictor import FrozenPredictionArtifact, sha256_file


def normalize_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def ranking_metrics(ranking: list[str], relevant: set[str], k: int) -> dict[str, float]:
    top = ranking[:k]
    recall = len(set(top) & relevant) / len(relevant)
    reciprocal_rank = next((1.0 / (index + 1) for index, item in enumerate(ranking) if item in relevant), 0.0)
    dcg = sum((1.0 / math.log2(index + 2)) for index, item in enumerate(top) if item in relevant)
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(k, len(relevant))))
    return {f"recall@{k}": recall, "mrr": reciprocal_rank, f"ndcg@{k}": dcg / ideal}


def evaluate(
    predictions_path: Path,
    annotations_path: Path,
    split_manifest_path: Path,
    features_path: Path,
    k: int = 5,
) -> dict:
    predictor = FrozenPredictionArtifact(predictions_path)
    split_ids = json.loads(split_manifest_path.read_text())["split_ids"]
    ordered_ids = [identifier for split in ("train", "val", "test") for identifier in split_ids[split]]
    features = np.load(features_path, mmap_mode="r")
    if features.shape[0] != len(ordered_ids):
        raise ValueError("Feature array and locked split have different lengths")
    feature_by_id = dict(zip(ordered_ids, normalize_rows(np.asarray(features, np.float32))))

    annotation_by_id = {}
    with annotations_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    counts = {}
    for row in rows:
        counts[row["filename"]] = counts.get(row["filename"], 0) + 1
        if counts[row["filename"]] == 1:
            annotation_by_id[row["filename"]] = row
    locked = set(ordered_ids)
    if any(counts[identifier] != 1 for identifier in locked):
        raise ValueError("Locked benchmark must contain unilateral images only")

    scales = np.asarray([5.0, 3.0], np.float32)
    train_ids = split_ids["train"]
    train_targets = np.asarray([
        [float(annotation_by_id[item]["HVA"]), float(annotation_by_id[item]["IMA"])]
        for item in train_ids
    ], np.float32)
    train_features = np.stack([feature_by_id[item] for item in train_ids])
    all_metrics = {name: [] for name in ("predicted_geometry", "image_embedding", "hybrid")}
    cases = []
    for query_id in split_ids["test"]:
        target = np.asarray([
            float(annotation_by_id[query_id]["HVA"]),
            float(annotation_by_id[query_id]["IMA"]),
        ], np.float32)
        predicted_dict = predictor.predict(query_id)
        predicted = np.asarray([predicted_dict["HVA"], predicted_dict["IMA"]], np.float32)
        oracle_distance = np.linalg.norm((train_targets - target) / scales, axis=1)
        relevant_indices = np.argsort(oracle_distance, kind="stable")[:k]
        relevant = {train_ids[index] for index in relevant_indices}

        geometry_distance = np.linalg.norm((train_targets - predicted) / scales, axis=1)
        geometry_similarity = 1.0 / (1.0 + geometry_distance)
        image_similarity = train_features @ feature_by_id[query_id]
        # Fixed, predeclared equal weighting; no test-set tuning.
        image_unit = (image_similarity + 1.0) / 2.0
        hybrid_similarity = 0.5 * geometry_similarity + 0.5 * image_unit
        scores = {
            "predicted_geometry": geometry_similarity,
            "image_embedding": image_similarity,
            "hybrid": hybrid_similarity,
        }
        case_result = {"query_id": query_id, "relevant_ids": sorted(relevant)}
        for name, values in scores.items():
            order = np.argsort(-values, kind="stable")
            ranking = [train_ids[index] for index in order]
            metrics = ranking_metrics(ranking, relevant, k)
            all_metrics[name].append(metrics)
            case_result[name] = {"retrieved_ids": ranking[:k], **metrics}
        cases.append(case_result)

    summary = {}
    for name, rows in all_metrics.items():
        summary[name] = {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in rows[0]
        }
    return {
        "evaluation_type": "locked_patient_disjoint_case_retrieval",
        "n_queries": len(cases),
        "retrieval_pool": len(train_ids),
        "relevance_definition": f"oracle top-{k} training cases by normalized HVA/IMA target distance",
        "hybrid_weight": {"predicted_geometry": 0.5, "image_embedding": 0.5},
        "metrics": summary,
        "artifact_hashes": {
            "predictions": predictor.sha256,
            "features": sha256_file(features_path),
            "split_manifest": sha256_file(split_manifest_path),
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    result = evaluate(args.predictions, args.annotations, args.split_manifest, args.features, args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()

