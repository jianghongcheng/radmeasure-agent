#!/usr/bin/env python3
"""Cross-fitted information in observable features about edit advantage."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "research"


def fold_of(db_id, folds, seed):
    return int(hashlib.sha256(f"fidelity:{seed}:{db_id}".encode()).hexdigest()[:8], 16) % folds


def label(candidate):
    return 2 if candidate["advantage"] > 0 else 0 if candidate["advantage"] < 0 else 1


def text(record, candidate):
    return (
        f"question {record['question']} baseline {record['predicted_sql']} "
        f"action {candidate['action']} candidate {candidate['sql']} "
        f"executable {int(candidate['executable'])}"
    )


def flatten(records):
    return [
        (record["db_id"], text(record, candidate), label(candidate))
        for record in records for candidate in record["candidates"]
    ]


def probabilities_all_classes(model, matrix):
    raw = model.predict_proba(matrix)
    output = np.zeros((matrix.shape[0], 3), dtype=float)
    for column, value in enumerate(model.classes_):
        output[:, int(value)] = raw[:, column]
    output = np.clip(output, 1e-12, 1)
    return output / output.sum(axis=1, keepdims=True)


def crossfit_fidelity(records, folds=5, seed=7719):
    rows = flatten(records)
    fold_metrics = []
    all_y, all_model, all_null = [], [], []
    for outer in range(folds):
        train = [row for row in rows if fold_of(row[0], folds, seed) != outer]
        test = [row for row in rows if fold_of(row[0], folds, seed) == outer]
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            max_features=30000, sublinear_tf=True
        )
        x_train = vectorizer.fit_transform([row[1] for row in train])
        x_test = vectorizer.transform([row[1] for row in test])
        y_train = np.asarray([row[2] for row in train])
        y_test = np.asarray([row[2] for row in test])
        model = SGDClassifier(
            loss="log", alpha=1e-5, max_iter=2000, tol=1e-4,
            random_state=42
        ).fit(x_train, y_train)
        model_probability = probabilities_all_classes(model, x_test)
        counts = np.bincount(y_train, minlength=3).astype(float) + 1
        prior = counts / counts.sum()
        null_probability = np.tile(prior, (len(test), 1))
        model_ce = log_loss(y_test, model_probability, labels=[0, 1, 2])
        null_ce = log_loss(y_test, null_probability, labels=[0, 1, 2])
        fold_metrics.append({
            "fold": outer,
            "n_candidates": len(test),
            "n_databases": len({row[0] for row in test}),
            "model_cross_entropy": float(model_ce),
            "null_cross_entropy": float(null_ce),
            "normalized_information_gain": float((null_ce - model_ce) / null_ce),
        })
        all_y.append(y_test)
        all_model.append(model_probability)
        all_null.append(null_probability)
    y = np.concatenate(all_y)
    model_probability = np.vstack(all_model)
    null_probability = np.vstack(all_null)
    model_ce = log_loss(y, model_probability, labels=[0, 1, 2])
    null_ce = log_loss(y, null_probability, labels=[0, 1, 2])
    nonneutral = np.count_nonzero(y != 1)
    return {
        "n_candidates": len(y),
        "n_databases": len({row[0] for row in rows}),
        "class_counts": {
            "harm": int(np.count_nonzero(y == 0)),
            "neutral": int(np.count_nonzero(y == 1)),
            "benefit": int(np.count_nonzero(y == 2)),
        },
        "proposal_precision": float(np.count_nonzero(y == 2) / nonneutral) if nonneutral else None,
        "model_cross_entropy": float(model_ce),
        "null_cross_entropy": float(null_ce),
        "normalized_information_gain": float((null_ce - model_ce) / null_ce),
        "folds": fold_metrics,
    }


def intervention_records(source, manifest, target):
    by_index = {row["index"]: row for row in source}
    pair_by_index = {
        row["index"]: row
        for row in manifest["benefit_pairs"] + manifest["harm_pairs"]
    }
    allocation = manifest["allocations"][str(target)]
    active = set(allocation["benefit_active_indices"] + allocation["harm_active_indices"])
    records = []
    for index, pair in pair_by_index.items():
        original = by_index[index]
        candidate_index = (
            pair["signed_candidate_index"] if index in active
            else pair["neutral_candidate_index"]
        )
        row = {key: value for key, value in original.items() if key != "candidates"}
        row["candidates"] = [original["candidates"][candidate_index]]
        records.append(row)
    return records


def observed_gain(name):
    if name == "Spider weak":
        result = json.loads((OUT / "spider_advantage_selector_cv.json").read_text())
        return result["aggregate"]["learned_exact_advantage"]["net_benefit_minus_harm"]["mean"]
    if name == "Spider clean":
        result = json.loads((OUT / "spider_clean_base_quadruple_cv.json").read_text())
        return result["pooled"]["learned_selector_learned_proposal"]["absolute_gain"]
    size = name.split("-")[1].lower()
    result = json.loads((OUT / f"spider_codes_{size}_selector_cv.json").read_text())
    return result["aggregate"]["learned_exact_advantage"]["net_benefit_minus_harm"]["mean"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7719)
    parser.add_argument(
        "--output", type=Path,
        default=OUT / "feature_advantage_fidelity.json"
    )
    args = parser.parse_args()
    artifacts = {
        "Spider weak": "spider_executable_edits.json",
        "Spider clean": "spider_clean_base_executable_edits.json",
        "CodeS-1B": "spider_codes_1b_executable_edits.json",
        "CodeS-3B": "spider_codes_3b_executable_edits.json",
        "CodeS-7B": "spider_codes_7b_executable_edits.json",
        "CodeS-15B": "spider_codes_15b_executable_edits.json",
    }
    results = {}
    for name, filename in artifacts.items():
        records = json.loads((OUT / filename).read_text())["records"]
        fidelity = crossfit_fidelity(records, args.folds, args.seed)
        fidelity["observed_selector_gain"] = observed_gain(name)
        results[name] = fidelity
        print(json.dumps({"domain": name, **{k: v for k, v in fidelity.items() if k != "folds"}}, indent=2), flush=True)

    source = json.loads((OUT / "spider_clean_base_executable_edits.json").read_text())["records"]
    manifest = json.loads((OUT / "spider_precision_intervention_manifest.json").read_text())
    revealed = json.loads((OUT / "spider_precision_intervention_revealed.json").read_text())
    intervention = {}
    for target in [0.15, 0.25, 0.35, 0.45, 0.50]:
        records = intervention_records(source, manifest, target)
        fidelity = crossfit_fidelity(records, args.folds, args.seed)
        fidelity["observed_selector_gain"] = revealed["results"][str(target)]["pooled"]["learned"]["gain"]
        intervention[str(target)] = fidelity
        print(json.dumps({"intervention_precision": target, **{k: v for k, v in fidelity.items() if k != "folds"}}, indent=2), flush=True)

    names = list(results)
    gain = np.asarray([results[name]["observed_selector_gain"] for name in names])
    precision = np.asarray([results[name]["proposal_precision"] for name in names])
    fidelity = np.asarray([results[name]["normalized_information_gain"] for name in names])
    intervention_targets = [str(x) for x in [0.15, 0.25, 0.35, 0.45, 0.50]]
    intervention_gain = np.asarray([intervention[x]["observed_selector_gain"] for x in intervention_targets])
    intervention_precision = np.asarray([intervention[x]["proposal_precision"] for x in intervention_targets])
    intervention_fidelity = np.asarray([intervention[x]["normalized_information_gain"] for x in intervention_targets])
    result = {
        "definition": "(cross-fitted null CE - feature-model CE) / null CE for advantage sign {-1,0,+1}",
        "feature_space": "selector-visible char 3-5 gram TF-IDF over question, baseline SQL, action, candidate SQL, and executability",
        "database_grouped_crossfit": True,
        "domains": results,
        "precision_intervention": intervention,
        "correlations": {
            "six_bases": {
                "pearson_precision_vs_gain": float(np.corrcoef(precision, gain)[0, 1]),
                "pearson_fidelity_vs_gain": float(np.corrcoef(fidelity, gain)[0, 1]),
            },
            "precision_intervention": {
                "pearson_precision_vs_gain": float(np.corrcoef(intervention_precision, intervention_gain)[0, 1]),
                "pearson_fidelity_vs_gain": float(np.corrcoef(intervention_fidelity, intervention_gain)[0, 1]),
            },
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["correlations"], indent=2))


if __name__ == "__main__":
    main()
