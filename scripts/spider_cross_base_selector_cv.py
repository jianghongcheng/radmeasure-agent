#!/usr/bin/env python3
"""Leave-one-base-model-out advantage learning on unseen Spider databases."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/research"


def load_selector():
    path = ROOT / "scripts/spider_advantage_selector_cv.py"
    spec = importlib.util.spec_from_file_location("spider_selector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pooled(folds: list[dict], method: str) -> dict:
    cases = [case for fold in folds for case in fold["methods"][method]["cases"]]
    n = len(cases)
    benefit = sum(not case["before_correct"] and case["after_correct"] for case in cases)
    harm = sum(case["before_correct"] and not case["after_correct"] for case in cases)
    return {
        "n": n,
        "execution_accuracy": sum(case["after_correct"] for case in cases) / n,
        "coverage": sum(case["edited"] for case in cases) / n,
        "benefit_count": benefit,
        "harm_count": harm,
        "absolute_gain": (benefit - harm) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", type=Path, default=OUT / "spider_cross_base_selector_cv.json")
    args = parser.parse_args()
    selector = load_selector()
    files = {
        "weak": OUT / "spider_executable_edits.json",
        **{size: OUT / f"spider_codes_{size}_executable_edits.json" for size in ["1b", "3b", "7b", "15b"]},
    }
    datasets = {name: json.loads(path.read_text())["records"] for name, path in files.items()}
    db_order = {name: [row["db_id"] for row in rows] for name, rows in datasets.items()}
    assert all(order == next(iter(db_order.values())) for order in db_order.values())
    results = {}
    for target in ["1b", "3b", "7b", "15b"]:
        folds = []
        for outer in range(args.folds):
            test = [row for row in datasets[target] if selector.fold_of(row["db_id"], args.folds, args.seed) == outer]
            calibration = [row for row in datasets[target] if selector.fold_of(row["db_id"], args.folds, args.seed) == (outer + 1) % args.folds]
            training = [
                row
                for source, records in datasets.items()
                if source != target
                for row in records
                if selector.fold_of(row["db_id"], args.folds, args.seed) not in {outer, (outer + 1) % args.folds}
            ]
            models = selector.train(training)
            cal_scores = selector.score(calibration, models)
            test_scores = selector.score(test, models)
            policy = selector.calibrate(calibration, cal_scores)
            methods = {
                name: selector.evaluate(test, test_scores, policy["risk_weight"], policy["threshold"], mode)
                for name, mode in [("no_repair", "no_repair"), ("learned_cross_base", "learned"), ("oracle_candidate", "oracle")]
            }
            folds.append({
                "fold": outer,
                "target_base": target,
                "training_bases": sorted(set(datasets) - {target}),
                "n": {"training_candidate_records": len(training), "calibration": len(calibration), "test": len(test)},
                "policy": policy,
                "methods": methods,
            })
        results[target] = {"folds": folds, "pooled": {method: pooled(folds, method) for method in folds[0]["methods"]}}
        print(json.dumps({target: results[target]["pooled"]}, indent=2), flush=True)
    output = {
        "status": "exploratory Spider dev; leave-one-base-model-out and database-grouped",
        "targets": results,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
