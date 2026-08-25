#!/usr/bin/env python3
"""Benchmark selective diagnosis of measurement-critical axes.

The correction operator replaces one selected axis by its ground-truth direction.
This deliberately isolates selection from correction quality. It is an analysis
benchmark and oracle-action upper bound, not a deployable repair system.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

import torch
from torch import nn


def load_base_module():
    path = Path(__file__).with_name("decisive_structured_refinement_benchmark.py")
    spec = importlib.util.spec_from_file_location("decisive_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def candidate_measurements(base, data):
    candidates = []
    for axis in range(3):
        directions = data["directions"].clone()
        directions[:, axis] = data["gt_directions"][:, axis]
        candidates.append(base.execute(directions))
    return torch.stack(candidates, dim=1)


def protocol_error(measurements, targets):
    error = (measurements - targets).abs()
    return error[..., 0] / 5 + error[..., 1] / 3


def gain_labels(base, data):
    initial = base.execute(data["directions"])
    candidates = candidate_measurements(base, data)
    before = protocol_error(initial, data["targets"])
    after = protocol_error(candidates, data["targets"][:, None, :])
    return before[:, None] - after, candidates


def sensitivities(base, directions):
    state = directions.detach().clone().requires_grad_(True)
    measurements = base.execute(state)
    scores = torch.zeros(len(state), 3)
    for measurement, tolerance in [(0, 5.), (1, 3.)]:
        gradient = torch.autograd.grad(
            (measurements[:, measurement] / tolerance).sum(), state,
            retain_graph=True)[0]
        scores += gradient.norm(dim=-1).detach().square()
    return scores.sqrt()


class ExpectedGainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(256 + 15 + 3, 192), nn.GELU(),
                                  nn.LayerNorm(192), nn.Linear(192, 96), nn.GELU(),
                                  nn.Linear(96, 3))

    def forward(self, geometry, image, sensitivity):
        return self.head(torch.cat([geometry, self.encoder(image),
                                    torch.log1p(sensitivity) / 5], dim=-1))


def train_model(base, train, val, epochs):
    train_gain, _ = gain_labels(base, train)
    val_gain, _ = gain_labels(base, val)
    train_sensitivity = sensitivities(base, train["directions"])
    val_sensitivity = sensitivities(base, val["directions"])
    model = ExpectedGainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best_loss, best_state = float("inf"), None
    for epoch in range(epochs):
        prediction = model(base.features(train), train["images"], train_sensitivity)
        regression = nn.functional.smooth_l1_loss(prediction, train_gain, beta=.2)
        true_best = train_gain.argmax(1)
        ranking = nn.functional.cross_entropy(prediction, true_best)
        positive = (train_gain.max(1).values > .05).float()
        stop_score = prediction.max(1).values
        stopping = nn.functional.binary_cross_entropy_with_logits(stop_score, positive)
        loss = regression + .2 * ranking + .1 * stopping
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                val_prediction = model(base.features(val), val["images"], val_sensitivity)
                val_loss = nn.functional.smooth_l1_loss(val_prediction, val_gain, beta=.2).item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def apply_selection(base, data, candidate_values, axis, edit):
    initial = base.execute(data["directions"])
    selected = candidate_values[torch.arange(len(axis)), axis]
    return torch.where(edit[:, None], selected, initial)


def metrics(base, data, prediction, edited, selected_axis, oracle_axis):
    initial = base.execute(data["directions"])
    before, after = protocol_error(initial, data["targets"]), protocol_error(prediction, data["targets"])
    absolute = (prediction - data["targets"]).abs()
    return {
        "HVA_MAE": absolute[:, 0].mean().item(), "IMA_MAE": absolute[:, 1].mean().item(),
        "mean_MAE": absolute.mean().item(), "coverage": edited.float().mean().item(),
        "selection_accuracy_when_edited": (selected_axis[edited] == oracle_axis[edited]).float().mean().item()
            if edited.any() else 0.,
        "joint_harm_rate": (after > before).float().mean().item(),
        "success_rate": (after < before - .05).float().mean().item(),
        "mean_protocol_gain": (before - after).mean().item(),
    }


def choose_threshold(base, model, val):
    sensitivity = sensitivities(base, val["directions"])
    with torch.no_grad():
        predicted = model(base.features(val), val["images"], sensitivity)
    gains, candidates = gain_labels(base, val)
    axis, score = predicted.argmax(1), predicted.max(1).values
    thresholds = torch.cat([torch.tensor([score.min() - 1e-3, score.max() + 1e-3]),
                            torch.quantile(score, torch.linspace(0, 1, 41))]).unique().sort().values
    curve = []
    for threshold in thresholds:
        edit = score > threshold
        output = apply_selection(base, val, candidates, axis, edit)
        row = metrics(base, val, output, edit, axis, gains.argmax(1))
        curve.append({"threshold": float(threshold), **row})
    feasible = [row for row in curve if row["joint_harm_rate"] <= .20]
    best = max(feasible, key=lambda row: (row["mean_protocol_gain"], row["coverage"]))
    return best["threshold"], best, curve


def risk_curve(base, data, predicted_gain):
    gains, candidates = gain_labels(base, data)
    axis, score = predicted_gain.argmax(1), predicted_gain.max(1).values
    rows = []
    for coverage in [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]:
        if coverage == 0:
            edit = torch.zeros(len(score), dtype=torch.bool)
        elif coverage == 1:
            edit = torch.ones(len(score), dtype=torch.bool)
        else:
            edit = score >= torch.quantile(score, 1 - coverage)
        output = apply_selection(base, data, candidates, axis, edit)
        rows.append(metrics(base, data, output, edit, axis, gains.argmax(1)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("/media/max/a/caxp (Copy 2)/outputs/benchmarks/controlled_cross_model_v2/hvangle_axis_geometry/results.json"))
    parser.add_argument("--annotations", type=Path, default=Path("/media/max/a/caxp/HVAngleEst/HVAngleEst/datasets.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("/media/max/a/caxp/HVAngleEst/HVAngleEst/images"))
    parser.add_argument("--output", type=Path, default=Path("outputs/research/selective_axis_verifier.json"))
    parser.add_argument("--epochs", type=int, default=400)
    args = parser.parse_args()
    random.seed(42); torch.manual_seed(42)
    base = load_base_module()
    rows = base.load_real_errors(args.results, args.annotations, args.image_dir)
    train, val, test = (base.tensors(rows, name) for name in ["train", "val", "test"])
    model = train_model(base, train, val, args.epochs)
    threshold, val_selection, validation_curve = choose_threshold(base, model, val)

    true_gain, candidates = gain_labels(base, test)
    oracle_axis = true_gain.argmax(1)
    initial = base.execute(test["directions"])
    sensitivity = sensitivities(base, test["directions"])
    geometric_error = torch.rad2deg(torch.acos((test["directions"] *
        test["gt_directions"]).sum(-1).abs().clamp(0, 1 - 1e-7)))
    with torch.no_grad():
        learned_gain = model(base.features(test), test["images"], sensitivity)

    random_axis = torch.randint(0, 3, (len(test["targets"]),), generator=torch.Generator().manual_seed(7))
    geometry_axis = geometric_error.argmax(1)
    sensitivity_axis = sensitivity.argmax(1)
    learned_axis, learned_score = learned_gain.argmax(1), learned_gain.max(1).values
    all_edit = torch.ones(len(random_axis), dtype=torch.bool)
    learned_edit = learned_score > threshold
    strategies = {}
    for name, axis, edit in [
        ("no_repair", random_axis, torch.zeros_like(all_edit)),
        ("random_axis", random_axis, all_edit),
        ("largest_geometric_error_oracle_selector", geometry_axis, all_edit),
        ("largest_sensitivity", sensitivity_axis, all_edit),
        ("learned_expected_gain_with_stop", learned_axis, learned_edit),
        ("oracle_expected_gain_with_stop", oracle_axis, true_gain.max(1).values > 0),
    ]:
        output = apply_selection(base, test, candidates, axis, edit)
        strategies[name] = metrics(base, test, output, edit, axis, oracle_axis)
    result = {
        "analysis_only_oracle_action": True,
        "note": "Selection is evaluated with ground-truth axis replacement to isolate diagnosis.",
        "split": {"train": len(train["targets"]), "val": len(val["targets"]), "test": len(test["targets"])},
        "selected_threshold": threshold, "validation_at_threshold": val_selection,
        "validation_risk_curve": validation_curve,
        "test_coverage_risk_curve": risk_curve(base, test, learned_gain),
        "strategies": strategies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
