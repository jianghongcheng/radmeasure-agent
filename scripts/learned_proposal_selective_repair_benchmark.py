#!/usr/bin/env python3
"""End-to-end one-edit selective repair with learned component proposals."""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

import torch
from torch import nn


def import_script(name, filename):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def protocol_error(measurements, targets):
    error = (measurements - targets).abs()
    return error[..., 0] / 5 + error[..., 1] / 3


def proposed_candidates(base, proposal, data):
    """Execute each of the three component-specific learned proposals."""
    with torch.no_grad():
        _, proposed = base.refined_state(proposal, data)
    candidates = []
    for axis in range(3):
        state = data["directions"].clone()
        state[:, axis] = proposed[:, axis]
        candidates.append(base.execute(state))
    return proposed, torch.stack(candidates, dim=1)


def actual_gains(base, proposal, data):
    proposed, candidates = proposed_candidates(base, proposal, data)
    initial = base.execute(data["directions"])
    gain = protocol_error(initial, data["targets"])[:, None] - protocol_error(
        candidates, data["targets"][:, None, :])
    return gain, proposed, candidates


class ProposalGainSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
        )
        # Geometry, image evidence, sensitivity, and proposed component displacement.
        self.head = nn.Sequential(nn.Linear(256 + 15 + 3 + 6, 192), nn.GELU(),
                                  nn.LayerNorm(192), nn.Linear(192, 96), nn.GELU(),
                                  nn.Linear(96, 3))

    def forward(self, geometry, images, sensitivity, displacement):
        return self.head(torch.cat([geometry, self.encoder(images),
            torch.log1p(sensitivity) / 5, displacement.flatten(1)], dim=-1))


def selector_inputs(base, selective, proposal, data):
    gain, proposed, candidates = actual_gains(base, proposal, data)
    sensitivity = selective.sensitivities(base, data["directions"])
    displacement = proposed - data["directions"]
    return gain, proposed, candidates, sensitivity, displacement


def train_selector(base, selective, proposal, train, val, epochs):
    train_gain, _, _, train_sensitivity, train_displacement = selector_inputs(
        base, selective, proposal, train)
    val_gain, _, _, val_sensitivity, val_displacement = selector_inputs(
        base, selective, proposal, val)
    model = ProposalGainSelector()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best, best_state = float("inf"), None
    for epoch in range(epochs):
        prediction = model(base.features(train), train["images"], train_sensitivity,
                           train_displacement)
        regression = nn.functional.smooth_l1_loss(prediction, train_gain, beta=.1)
        ranking = nn.functional.cross_entropy(prediction, train_gain.argmax(1))
        useful = (train_gain.max(1).values > .02).float()
        stop_logit = prediction.max(1).values * 4
        stopping = nn.functional.binary_cross_entropy_with_logits(stop_logit, useful)
        loss = regression + .2 * ranking + .1 * stopping
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                val_prediction = model(base.features(val), val["images"], val_sensitivity,
                                       val_displacement)
                val_loss = nn.functional.smooth_l1_loss(val_prediction, val_gain, beta=.1).item()
            if val_loss < best:
                best = val_loss
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def choose(base, data, candidates, axis, edit):
    initial = base.execute(data["directions"])
    selected = candidates[torch.arange(len(axis)), axis]
    return torch.where(edit[:, None], selected, initial)


def metrics(base, data, prediction, edit=None):
    initial = base.execute(data["directions"])
    before, after = protocol_error(initial, data["targets"]), protocol_error(prediction, data["targets"])
    absolute = (prediction - data["targets"]).abs()
    return {
        "HVA_MAE": absolute[:, 0].mean().item(), "IMA_MAE": absolute[:, 1].mean().item(),
        "mean_MAE": absolute.mean().item(),
        "coverage": float(edit.float().mean()) if edit is not None else 1.,
        "joint_harm_rate": (after > before).float().mean().item(),
        "success_rate": (after < before - .05).float().mean().item(),
        "mean_protocol_gain": (before - after).mean().item(),
    }


def predictions(base, selective, selector, proposal, data):
    gain, proposed, candidates, sensitivity, displacement = selector_inputs(
        base, selective, proposal, data)
    with torch.no_grad():
        predicted_gain = selector(base.features(data), data["images"], sensitivity, displacement)
    return gain, proposed, candidates, predicted_gain


def select_threshold(base, val, candidates, predicted_gain, risk_limit=.20):
    axis, score = predicted_gain.argmax(1), predicted_gain.max(1).values
    thresholds = torch.cat([torch.tensor([score.min() - 1e-3, score.max() + 1e-3]),
                            torch.quantile(score, torch.linspace(0, 1, 41))]).unique().sort().values
    rows = []
    for threshold in thresholds:
        edit = score > threshold
        row = metrics(base, val, choose(base, val, candidates, axis, edit), edit)
        rows.append({"threshold": float(threshold), **row})
    feasible = [row for row in rows if row["joint_harm_rate"] <= risk_limit]
    best = max(feasible, key=lambda row: (row["mean_protocol_gain"], row["coverage"]))
    return best["threshold"], best, rows


def coverage_curve(base, data, candidates, predicted_gain):
    axis, score = predicted_gain.argmax(1), predicted_gain.max(1).values
    rows = []
    for target in [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]:
        if target == 0:
            edit = torch.zeros(len(score), dtype=torch.bool)
        elif target == 1:
            edit = torch.ones(len(score), dtype=torch.bool)
        else:
            edit = score >= torch.quantile(score, 1 - target)
        rows.append(metrics(base, data, choose(base, data, candidates, axis, edit), edit))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("/media/max/a/caxp (Copy 2)/outputs/benchmarks/controlled_cross_model_v2/hvangle_axis_geometry/results.json"))
    parser.add_argument("--annotations", type=Path, default=Path("/media/max/a/caxp/HVAngleEst/HVAngleEst/datasets.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("/media/max/a/caxp/HVAngleEst/HVAngleEst/images"))
    parser.add_argument("--output", type=Path, default=Path("outputs/research/learned_proposal_selective_repair.json"))
    parser.add_argument("--epochs", type=int, default=400)
    args = parser.parse_args()
    random.seed(42); torch.manual_seed(42)
    base = import_script("decisive", "decisive_structured_refinement_benchmark.py")
    selective = import_script("selective", "selective_axis_verifier_benchmark.py")
    rows = base.load_real_errors(args.results, args.annotations, args.image_dir)
    train, val, test = (base.tensors(rows, split) for split in ["train", "val", "test"])

    proposal = base.train_refiner(train, val, True, args.epochs, image_conditioned=True)
    selector = train_selector(base, selective, proposal, train, val, args.epochs)
    val_gain, _, val_candidates, val_prediction = predictions(base, selective, selector, proposal, val)
    threshold, validation, validation_curve = select_threshold(
        base, val, val_candidates, val_prediction)
    gain, proposed, candidates, predicted_gain = predictions(
        base, selective, selector, proposal, test)

    initial = base.execute(test["directions"])
    learned_all = base.execute(proposed)
    learned_axis, learned_score = predicted_gain.argmax(1), predicted_gain.max(1).values
    learned_edit = learned_score > threshold
    oracle_axis, oracle_gain = gain.argmax(1), gain.max(1).values
    oracle_edit = oracle_gain > 0
    result = {
        "proposal": "image-conditioned measurement-aware component direction proposal",
        "selector_target": "actual executable gain after applying learned proposal",
        "split": {"train": len(train["targets"]), "val": len(val["targets"]), "test": len(test["targets"])},
        "risk_limit_selected_on_validation": .20, "threshold": threshold,
        "validation_at_threshold": validation, "validation_coverage_risk": validation_curve,
        "test_coverage_risk": coverage_curve(base, test, candidates, predicted_gain),
        "methods": {
            "no_repair": metrics(base, test, initial, torch.zeros(len(initial), dtype=torch.bool)),
            "learned_repair_all": metrics(base, test, learned_all),
            "learned_selector_learned_repair": metrics(base, test,
                choose(base, test, candidates, learned_axis, learned_edit), learned_edit),
            "oracle_selector_learned_repair": metrics(base, test,
                choose(base, test, candidates, oracle_axis, oracle_edit), oracle_edit),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
