#!/usr/bin/env python3
"""Patient-grouped cross-validation for learned selective axis repair."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch


def import_script(name, filename):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fold_for(patient_id, folds, seed):
    digest = hashlib.sha1(f"{seed}:{patient_id}".encode()).hexdigest()
    return int(digest[:8], 16) % folds


def assign(rows, test_fold, folds, seed):
    validation_fold = (test_fold + 1) % folds
    assigned = []
    for row in rows:
        copy = dict(row)
        fold = fold_for(row["patient_id"], folds, seed)
        copy["split"] = "test" if fold == test_fold else "val" if fold == validation_fold else "train"
        assigned.append(copy)
    return assigned


def evaluate_fold(base, selective, learned, rows, fold, folds, split_seed, epochs):
    assigned = assign(rows, fold, folds, split_seed)
    train, val, test = (base.tensors(assigned, name) for name in ["train", "val", "test"])
    torch.manual_seed(100 + fold); random.seed(100 + fold)
    proposal = base.train_refiner(train, val, True, epochs, image_conditioned=True)
    selector = learned.train_selector(base, selective, proposal, train, val, epochs)
    _, _, val_candidates, val_prediction = learned.predictions(
        base, selective, selector, proposal, val)
    threshold, validation, _ = learned.select_threshold(
        base, val, val_candidates, val_prediction)
    gain, proposed, candidates, predicted_gain = learned.predictions(
        base, selective, selector, proposal, test)
    initial = base.execute(test["directions"])
    learned_all = base.execute(proposed)
    learned_axis, learned_score = predicted_gain.argmax(1), predicted_gain.max(1).values
    learned_edit = learned_score > threshold
    oracle_axis, oracle_gain = gain.argmax(1), gain.max(1).values
    oracle_edit = oracle_gain > 0
    methods = {
        "no_repair": learned.metrics(base, test, initial,
            torch.zeros(len(initial), dtype=torch.bool)),
        "learned_repair_all": learned.metrics(base, test, learned_all),
        "learned_selector_learned_repair": learned.metrics(base, test,
            learned.choose(base, test, candidates, learned_axis, learned_edit), learned_edit),
        "oracle_selector_learned_repair": learned.metrics(base, test,
            learned.choose(base, test, candidates, oracle_axis, oracle_edit), oracle_edit),
    }
    return {
        "fold": fold, "n": {"train": len(train["targets"]), "val": len(val["targets"]),
                              "test": len(test["targets"])},
        "threshold": threshold, "validation_at_threshold": validation, "methods": methods,
    }


def aggregate(fold_results):
    output = {}
    for method in fold_results[0]["methods"]:
        rows = [fold["methods"][method] for fold in fold_results]
        output[method] = {}
        for metric in rows[0]:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            output[method][metric] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1))}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("/media/max/a/caxp (Copy 2)/outputs/benchmarks/controlled_cross_model_v2/hvangle_axis_geometry/results.json"))
    parser.add_argument("--annotations", type=Path, default=Path("/media/max/a/caxp/HVAngleEst/HVAngleEst/datasets.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("/media/max/a/caxp/HVAngleEst/HVAngleEst/images"))
    parser.add_argument("--output", type=Path, default=Path("outputs/research/learned_proposal_cv.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=2027)
    parser.add_argument("--epochs", type=int, default=250)
    args = parser.parse_args()
    base = import_script("decisive_cv", "decisive_structured_refinement_benchmark.py")
    selective = import_script("selective_cv", "selective_axis_verifier_benchmark.py")
    learned = import_script("learned_cv", "learned_proposal_selective_repair_benchmark.py")
    rows = base.load_real_errors(args.results, args.annotations, args.image_dir)
    results = []
    for fold in range(args.folds):
        result = evaluate_fold(base, selective, learned, rows, fold, args.folds,
                               args.split_seed, args.epochs)
        results.append(result)
        print(json.dumps({"completed_fold": fold, "methods": result["methods"]}), flush=True)
        partial = {"folds": results, "aggregate": aggregate(results)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(partial, indent=2) + "\n")
    output = {
        "patient_grouped": True, "fold_count": args.folds, "split_seed": args.split_seed,
        "epochs": args.epochs, "folds": results, "aggregate": aggregate(results),
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["aggregate"], indent=2))


if __name__ == "__main__":
    main()
