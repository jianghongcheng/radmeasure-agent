#!/usr/bin/env python3
"""Qualification audit for harmful-candidate injection.

Pairs each existing cross-fitted proposal with an alternative harmful candidate
from the same question and operator type. A database-grouped classifier then
tests whether proposal versus injected origin is visible in the selector's text
feature space. No correction-policy gain is evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, filename):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fold_of(db_id, folds, seed):
    return int(hashlib.sha256(f"injection-audit:{seed}:{db_id}".encode()).hexdigest()[:8], 16) % folds


def operator_type(action):
    return action.split(":", 1)[0]


def candidate_text(record, candidate):
    return (
        f"question {record['question']} baseline {record['predicted_sql']} "
        f"action {candidate['action']} candidate {candidate['sql']} "
        f"executable {int(candidate['executable'])}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs/research/spider_clean_base_executable_edits.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/research/spider_injection_distinguishability_audit.json",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1931)
    args = parser.parse_args()

    records = json.loads(args.input.read_text())["records"]
    selector = load_script("spider_selector", "spider_advantage_selector_cv.py")
    pairs = []

    # Cross-fit the already defined proposal scorer. Only proposal identities,
    # not downstream selective gains, are retained.
    for outer in range(args.folds):
        test = [r for r in records if fold_of(r["db_id"], args.folds, args.seed) == outer]
        train = [r for r in records if fold_of(r["db_id"], args.folds, args.seed) != outer]
        models = selector.train(train)
        scores = selector.score(test, models)
        for record in test:
            if not record["candidates"]:
                continue
            original_index = int(np.argmax(scores[record["index"]]["benefit"]))
            original = record["candidates"][original_index]
            same_operator_harms = [
                candidate
                for index, candidate in enumerate(record["candidates"])
                if index != original_index
                and candidate["advantage"] < 0
                and operator_type(candidate["action"]) == operator_type(original["action"])
            ]
            if not same_operator_harms:
                continue
            # Frozen deterministic choice avoids looking at origin-classifier scores.
            injected = sorted(same_operator_harms, key=lambda row: (row["action"], row["sql"]))[0]
            pairs.append(
                {
                    "db_id": record["db_id"],
                    "original": candidate_text(record, original),
                    "injected": candidate_text(record, injected),
                }
            )

    fold_results = []
    for outer in range(args.folds):
        train_pairs = [p for p in pairs if fold_of(p["db_id"], args.folds, args.seed) != outer]
        test_pairs = [p for p in pairs if fold_of(p["db_id"], args.folds, args.seed) == outer]
        train_text = [p["original"] for p in train_pairs] + [p["injected"] for p in train_pairs]
        train_y = np.r_[np.zeros(len(train_pairs)), np.ones(len(train_pairs))]
        test_text = [p["original"] for p in test_pairs] + [p["injected"] for p in test_pairs]
        test_y = np.r_[np.zeros(len(test_pairs)), np.ones(len(test_pairs))]
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            max_features=30000, sublinear_tf=True
        )
        x_train = vectorizer.fit_transform(train_text)
        x_test = vectorizer.transform(test_text)
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        model.fit(x_train, train_y)
        probability = model.predict_proba(x_test)[:, 1]
        fold_results.append(
            {
                "fold": outer,
                "n_pairs": len(test_pairs),
                "auc": float(roc_auc_score(test_y, probability)),
            }
        )

    aucs = [row["auc"] for row in fold_results]
    result = {
        "status": "artifact qualification only; no correction gain evaluated",
        "pairing": "same question and operator type; injected item is an existing harmful candidate",
        "n_pairs": len(pairs),
        "database_grouped_cv": True,
        "folds": fold_results,
        "mean_auc": float(np.mean(aucs)),
        "sd_auc": float(np.std(aucs, ddof=1)),
        "qualification_rule": "auxiliary injection is rejected if mean AUC > 0.60",
        "qualified": bool(np.mean(aucs) <= 0.60),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
