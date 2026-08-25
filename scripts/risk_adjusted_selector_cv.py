#!/usr/bin/env python3
"""Cross-validated risk-adjusted selection over learned axis proposals."""
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


def assign_folds(rows, test_fold, folds, seed):
    validation_fold = (test_fold + 1) % folds
    output = []
    for row in rows:
        fold = int(hashlib.sha1(f"{seed}:{row['patient_id']}".encode()).hexdigest()[:8], 16) % folds
        item = dict(row)
        item["split"] = "test" if fold == test_fold else "val" if fold == validation_fold else "train"
        output.append(item)
    return output


class RiskAdjustedSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2), nn.GELU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())
        self.trunk = nn.Sequential(nn.Linear(256 + 15 + 3 + 6, 192), nn.GELU(),
                                   nn.LayerNorm(192), nn.Linear(192, 96), nn.GELU())
        self.rank = nn.Linear(96, 3)
        self.benefit_logit = nn.Linear(96, 3)
        self.harm_logit = nn.Linear(96, 3)
        self.positive_gain = nn.Linear(96, 3)
        self.harm_magnitude = nn.Linear(96, 3)

    def forward(self, geometry, images, sensitivity, displacement):
        hidden = self.trunk(torch.cat([geometry, self.encoder(images),
            torch.log1p(sensitivity) / 5, displacement.flatten(1)], dim=-1))
        return {"rank": self.rank(hidden), "benefit": self.benefit_logit(hidden),
                "harm": self.harm_logit(hidden),
                "gain": nn.functional.softplus(self.positive_gain(hidden)),
                "harm_magnitude": nn.functional.softplus(self.harm_magnitude(hidden))}


def inputs(base, selective, learned, proposal, data):
    gain, proposed, candidates = learned.actual_gains(base, proposal, data)
    sensitivity = selective.sensitivities(base, data["directions"])
    displacement = proposed - data["directions"]
    return gain, candidates, sensitivity, displacement


def train_selector(base, selective, learned, proposal, train, val, epochs, delta):
    train_gain, _, train_sens, train_disp = inputs(base, selective, learned, proposal, train)
    val_gain, _, val_sens, val_disp = inputs(base, selective, learned, proposal, val)
    model = RiskAdjustedSelector()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best, state = float("inf"), None
    for epoch in range(epochs):
        out = model(base.features(train), train["images"], train_sens, train_disp)
        positive, harmful = train_gain > delta, train_gain < -delta
        loss = nn.functional.cross_entropy(out["rank"], train_gain.argmax(1))
        loss += .5 * nn.functional.binary_cross_entropy_with_logits(out["benefit"], positive.float())
        loss += .5 * nn.functional.binary_cross_entropy_with_logits(out["harm"], harmful.float())
        if positive.any():
            loss += nn.functional.smooth_l1_loss(out["gain"][positive], train_gain[positive], beta=.1)
        if harmful.any():
            loss += nn.functional.smooth_l1_loss(out["harm_magnitude"][harmful], -train_gain[harmful], beta=.1)
        # Listwise utility ranking is separate from accept/stop supervision.
        target_order = torch.softmax(train_gain / .15, dim=1)
        loss += .25 * -(target_order * torch.log_softmax(out["rank"], dim=1)).sum(1).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                out = model(base.features(val), val["images"], val_sens, val_disp)
                score = nn.functional.cross_entropy(out["rank"], val_gain.argmax(1))
                score += nn.functional.binary_cross_entropy_with_logits(
                    out["benefit"], (val_gain > delta).float())
                score += nn.functional.binary_cross_entropy_with_logits(
                    out["harm"], (val_gain < -delta).float())
            if score.item() < best:
                best, state = score.item(), {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model


def calibrate_temperature(logits, labels):
    best = (float("inf"), 1.)
    for temperature in torch.linspace(.4, 4., 37):
        loss = nn.functional.binary_cross_entropy_with_logits(logits / temperature, labels.float()).item()
        if loss < best[0]: best = (loss, float(temperature))
    return best[1]


def predict(base, selective, learned, proposal, selector, data):
    gain, candidates, sensitivity, displacement = inputs(base, selective, learned, proposal, data)
    with torch.no_grad():
        output = selector(base.features(data), data["images"], sensitivity, displacement)
    return gain, candidates, output


def utility(output, benefit_temperature, harm_temperature, risk_weight):
    benefit = torch.sigmoid(output["benefit"] / benefit_temperature)
    harm = torch.sigmoid(output["harm"] / harm_temperature)
    return benefit * output["gain"] - risk_weight * harm * output["harm_magnitude"]


def execute_selection(base, learned, data, candidates, axis, edit):
    return learned.choose(base, data, candidates, axis, edit)


def detailed_metrics(base, learned, data, prediction, edit, gain, selected_axis, delta):
    row = learned.metrics(base, data, prediction, edit)
    best_gain = gain.max(1).values
    opportunity = best_gain > delta
    selected_gain = gain[torch.arange(len(gain)), selected_axis]
    harmful_edit = edit & (selected_gain < -delta)
    row.update({
        "opportunity_prevalence": opportunity.float().mean().item(),
        "opportunity_recall": (edit & opportunity).float().sum().div(opportunity.float().sum().clamp_min(1)).item(),
        "conditional_harm": harmful_edit.float().sum().div(edit.float().sum().clamp_min(1)).item(),
        "selected_positive_rate": (edit & (selected_gain > delta)).float().sum().div(edit.float().sum().clamp_min(1)).item(),
    })
    return row


def select_policy(base, learned, val, gain, candidates, output, delta, risk_limit):
    benefit_t = calibrate_temperature(output["benefit"], gain > delta)
    harm_t = calibrate_temperature(output["harm"], gain < -delta)
    policies = []
    for risk_weight in [.5, 1., 2., 4., 8.]:
        values = utility(output, benefit_t, harm_t, risk_weight)
        # Ranking head answers which component; utility only decides intervention.
        axis = output["rank"].argmax(1)
        score = values[torch.arange(len(axis)), axis]
        thresholds = torch.cat([torch.tensor([score.min()-1e-3, score.max()+1e-3]),
                                torch.quantile(score, torch.linspace(0, 1, 41))]).unique()
        for threshold in thresholds:
            edit = score > threshold
            prediction = execute_selection(base, learned, val, candidates, axis, edit)
            row = detailed_metrics(base, learned, val, prediction, edit, gain, axis, delta)
            policies.append({"risk_weight": risk_weight, "threshold": float(threshold), **row})
    feasible = [p for p in policies if p["joint_harm_rate"] <= risk_limit]
    best = max(feasible, key=lambda p: (p["mean_protocol_gain"], p["opportunity_recall"]))
    return benefit_t, harm_t, best, policies


def evaluate_fold(base, selective, learned, rows, fold, args):
    assigned = assign_folds(rows, fold, args.folds, args.split_seed)
    train, val, test = (base.tensors(assigned, split) for split in ["train", "val", "test"])
    torch.manual_seed(700 + fold); random.seed(700 + fold)
    proposal = base.train_refiner(train, val, True, args.epochs, image_conditioned=True)
    selector = train_selector(base, selective, learned, proposal, train, val, args.epochs, args.delta)
    val_gain, val_candidates, val_output = predict(base, selective, learned, proposal, selector, val)
    benefit_t, harm_t, policy, _ = select_policy(
        base, learned, val, val_gain, val_candidates, val_output, args.delta, args.risk_limit)
    gain, candidates, output = predict(base, selective, learned, proposal, selector, test)
    values = utility(output, benefit_t, harm_t, policy["risk_weight"])
    axis = output["rank"].argmax(1)
    edit = values[torch.arange(len(axis)), axis] > policy["threshold"]
    prediction = execute_selection(base, learned, test, candidates, axis, edit)
    oracle_axis, oracle_gain = gain.argmax(1), gain.max(1).values
    oracle_edit = oracle_gain > 0
    initial = base.execute(test["directions"])
    return {"fold": fold, "n": len(test["targets"]), "benefit_temperature": benefit_t,
            "harm_temperature": harm_t, "policy": policy,
            "methods": {
                "no_repair": detailed_metrics(base, learned, test, initial,
                    torch.zeros(len(initial), dtype=torch.bool), gain, axis, args.delta),
                "risk_adjusted_selector": detailed_metrics(base, learned, test, prediction,
                    edit, gain, axis, args.delta),
                "oracle_selector_learned_repair": detailed_metrics(base, learned, test,
                    execute_selection(base, learned, test, candidates, oracle_axis, oracle_edit),
                    oracle_edit, gain, oracle_axis, args.delta)}}


def aggregate(folds):
    result = {}
    for method in folds[0]["methods"]:
        result[method] = {}
        for metric in folds[0]["methods"][method]:
            values = np.array([fold["methods"][method][metric] for fold in folds])
            result[method][metric] = {"mean": float(values.mean()),
                                      "sd": float(values.std(ddof=1)) if len(values)>1 else 0.}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("/media/max/a/caxp (Copy 2)/outputs/benchmarks/controlled_cross_model_v2/hvangle_axis_geometry/results.json"))
    parser.add_argument("--annotations", type=Path, default=Path("/media/max/a/caxp/HVAngleEst/HVAngleEst/datasets.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("/media/max/a/caxp/HVAngleEst/HVAngleEst/images"))
    parser.add_argument("--output", type=Path, default=Path("outputs/research/risk_adjusted_selector_cv.json"))
    parser.add_argument("--folds", type=int, default=5); parser.add_argument("--split-seed", type=int, default=2027)
    parser.add_argument("--epochs", type=int, default=250); parser.add_argument("--delta", type=float, default=.02)
    parser.add_argument("--risk-limit", type=float, default=.10)
    args = parser.parse_args()
    base = import_script("risk_base", "decisive_structured_refinement_benchmark.py")
    selective = import_script("risk_selective", "selective_axis_verifier_benchmark.py")
    learned = import_script("risk_learned", "learned_proposal_selective_repair_benchmark.py")
    rows = base.load_real_errors(args.results, args.annotations, args.image_dir)
    folds = []
    for fold in range(args.folds):
        folds.append(evaluate_fold(base, selective, learned, rows, fold, args))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"folds": folds, "aggregate": aggregate(folds)}, indent=2)+"\n")
        print(json.dumps({"completed_fold": fold, "methods": folds[-1]["methods"]}), flush=True)
    output = {"method": "risk-adjusted hurdle selector", "patient_grouped": True,
              "folds": folds, "aggregate": aggregate(folds)}
    args.output.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output["aggregate"], indent=2))


if __name__ == "__main__": main()
