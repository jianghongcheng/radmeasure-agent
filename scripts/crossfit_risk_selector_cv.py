#!/usr/bin/env python3
"""Nested patient-level cross-fitting for risk-adjusted learned repair selection."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


def import_script(name, filename):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def hashed_fold(patient_id, folds, seed):
    return int(hashlib.sha1(f"{seed}:{patient_id}".encode()).hexdigest()[:8], 16) % folds


def select_rows(rows, predicate):
    chosen = [dict(row, split="chosen") for row in rows if predicate(row)]
    if not chosen:
        raise ValueError("empty patient partition")
    # Reuse the canonical tensorizer without allowing row-level splits.
    return chosen


def as_data(base, rows):
    return base.tensors(rows, "chosen")


def outer_partitions(rows, outer, folds, seed):
    test = select_rows(rows, lambda r: hashed_fold(r["patient_id"], folds, seed) == outer)
    calibration_fold = (outer + 1) % folds
    calibration = select_rows(rows, lambda r: hashed_fold(r["patient_id"], folds, seed) == calibration_fold)
    training = select_rows(rows, lambda r: hashed_fold(r["patient_id"], folds, seed)
                           not in {outer, calibration_fold})
    return training, calibration, test


def inner_fold(patient_id, folds, seed, outer):
    return hashed_fold(patient_id, folds, seed + 1009 * (outer + 1))


def concatenate(parts, key):
    return torch.cat([part[key] for part in parts], dim=0)


def build_oof_proposal_data(base, selective, learned, training_rows, outer, args):
    parts = []
    for held in range(args.inner_folds):
        fit_rows = select_rows(training_rows, lambda r: inner_fold(
            r["patient_id"], args.inner_folds, args.split_seed, outer) != held)
        held_rows = select_rows(training_rows, lambda r: inner_fold(
            r["patient_id"], args.inner_folds, args.split_seed, outer) == held)
        fit, held_data = as_data(base, fit_rows), as_data(base, held_rows)
        torch.manual_seed(3000 + outer * 10 + held); random.seed(3000 + outer * 10 + held)
        proposal = base.train_refiner(fit, fit, True, args.proposal_epochs,
                                      image_conditioned=True)
        gain, _, sensitivity, displacement = selective_inputs(
            base, selective, learned, proposal, held_data)
        parts.append({"geometry": base.features(held_data), "images": held_data["images"],
                      "gain": gain, "sensitivity": sensitivity,
                      "displacement": displacement, "centers": held_data["centers"],
                      "directions": held_data["directions"], "aspect": held_data["aspect"]})
    return {key: concatenate(parts, key) for key in parts[0]}


def selective_inputs(base, selective, learned, proposal, data):
    gain, proposed, candidates = learned.actual_gains(base, proposal, data)
    sensitivity = selective.sensitivities(base, data["directions"])
    return gain, candidates, sensitivity, proposed - data["directions"]


def selector_loss(output, gain, delta):
    positive, harmful = gain > delta, gain < -delta
    loss = nn.functional.cross_entropy(output["rank"], gain.argmax(1))
    loss += .5 * nn.functional.binary_cross_entropy_with_logits(
        output["benefit"], positive.float())
    loss += .5 * nn.functional.binary_cross_entropy_with_logits(
        output["harm"], harmful.float())
    if positive.any():
        loss += nn.functional.smooth_l1_loss(output["gain"][positive], gain[positive], beta=.1)
    if harmful.any():
        loss += nn.functional.smooth_l1_loss(
            output["harm_magnitude"][harmful], -gain[harmful], beta=.1)
    distribution = torch.softmax(gain / .15, dim=1)
    loss += .25 * -(distribution * torch.log_softmax(output["rank"], dim=1)).sum(1).mean()
    return loss


def train_crossfit_selector(risk, oof, epochs, delta):
    model = risk.RiskAdjustedSelector()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        output = model(oof["geometry"], oof["images"], oof["sensitivity"],
                       oof["displacement"])
        loss = selector_loss(output, oof["gain"], delta)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model


def detailed(base, learned, risk, data, prediction, edit, gain, axis, delta):
    return risk.detailed_metrics(base, learned, data, prediction, edit, gain, axis, delta)


def evaluate_outer(base, selective, learned, risk, rows, outer, args):
    train_rows, calibration_rows, test_rows = outer_partitions(
        rows, outer, args.outer_folds, args.split_seed)
    train, calibration, test = map(lambda x: as_data(base, x),
                                   [train_rows, calibration_rows, test_rows])
    oof = build_oof_proposal_data(base, selective, learned, train_rows, outer, args)
    torch.manual_seed(5000 + outer); random.seed(5000 + outer)
    selector = train_crossfit_selector(risk, oof, args.selector_epochs, args.delta)
    # The final proposal sees only outer-training patients. No calibration/test labels.
    torch.manual_seed(6000 + outer); random.seed(6000 + outer)
    proposal = base.train_refiner(train, train, True, args.proposal_epochs,
                                  image_conditioned=True)

    cal_gain, cal_candidates, cal_sens, cal_disp = selective_inputs(
        base, selective, learned, proposal, calibration)
    with torch.no_grad():
        cal_output = selector(base.features(calibration), calibration["images"],
                              cal_sens, cal_disp)
    benefit_t, harm_t, policy, _ = risk.select_policy(
        base, learned, calibration, cal_gain, cal_candidates, cal_output,
        args.delta, args.risk_limit)

    gain, candidates, sensitivity, displacement = selective_inputs(
        base, selective, learned, proposal, test)
    with torch.no_grad():
        output = selector(base.features(test), test["images"], sensitivity, displacement)
    values = risk.utility(output, benefit_t, harm_t, policy["risk_weight"])
    axis = output["rank"].argmax(1)
    edit = values[torch.arange(len(axis)), axis] > policy["threshold"]
    selected = learned.choose(base, test, candidates, axis, edit)
    oracle_axis, oracle_gain = gain.argmax(1), gain.max(1).values
    oracle_edit = oracle_gain > 0
    oracle = learned.choose(base, test, candidates, oracle_axis, oracle_edit)
    initial = base.execute(test["directions"])
    return {
        "outer_fold": outer,
        "n": {"selector_train": len(train["targets"]),
              "calibration": len(calibration["targets"]), "test": len(test["targets"])},
        "policy": policy, "benefit_temperature": benefit_t, "harm_temperature": harm_t,
        "methods": {
            "no_repair": detailed(base, learned, risk, test, initial,
                torch.zeros(len(initial), dtype=torch.bool), gain, axis, args.delta),
            "crossfit_risk_selector": detailed(base, learned, risk, test, selected,
                edit, gain, axis, args.delta),
            "oracle_selector_learned_repair": detailed(base, learned, risk, test, oracle,
                oracle_edit, gain, oracle_axis, args.delta),
        },
    }


def aggregate(folds):
    result = {}
    for method in folds[0]["methods"]:
        result[method] = {}
        for metric in folds[0]["methods"][method]:
            values = np.asarray([fold["methods"][method][metric] for fold in folds])
            result[method][metric] = {"mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("/media/max/a/caxp (Copy 2)/outputs/benchmarks/controlled_cross_model_v2/hvangle_axis_geometry/results.json"))
    parser.add_argument("--annotations", type=Path, default=Path("/media/max/a/caxp/HVAngleEst/HVAngleEst/datasets.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("/media/max/a/caxp/HVAngleEst/HVAngleEst/images"))
    parser.add_argument("--output", type=Path, default=Path("outputs/research/crossfit_risk_selector_cv.json"))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=2027)
    parser.add_argument("--proposal-epochs", type=int, default=180)
    parser.add_argument("--selector-epochs", type=int, default=250)
    parser.add_argument("--delta", type=float, default=.02)
    parser.add_argument("--risk-limit", type=float, default=.10)
    args = parser.parse_args()
    base = import_script("cf_base", "decisive_structured_refinement_benchmark.py")
    selective = import_script("cf_selective", "selective_axis_verifier_benchmark.py")
    learned = import_script("cf_learned", "learned_proposal_selective_repair_benchmark.py")
    risk = import_script("cf_risk", "risk_adjusted_selector_cv.py")
    rows = base.load_real_errors(args.results, args.annotations, args.image_dir)
    folds = []
    for outer in range(args.outer_folds):
        folds.append(evaluate_outer(base, selective, learned, risk, rows, outer, args))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"folds": folds, "aggregate": aggregate(folds)}, indent=2)+"\n")
        print(json.dumps({"completed_outer_fold": outer,
                          "methods": folds[-1]["methods"]}), flush=True)
    output = {"nested_patient_crossfit": True, "outer_folds": args.outer_folds,
              "inner_folds": args.inner_folds, "folds": folds,
              "aggregate": aggregate(folds)}
    args.output.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output["aggregate"], indent=2))


if __name__ == "__main__": main()
