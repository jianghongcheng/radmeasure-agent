#!/usr/bin/env python3
"""Train a geometry-only residual repair baseline on patient-disjoint splits."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn


def load_split(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    points, dimensions, targets = [], [], []
    for row in rows:
        flat = row["great_toe"] + row["first_metatarsal"] + row["second_metatarsal"]
        points.append(flat)
        dimensions.append([row["image_width"], row["image_height"]])
        targets.append([row["HVA"], row["IMA"]])
    return (torch.tensor(points, dtype=torch.float32).reshape(-1, 6, 2),
            torch.tensor(dimensions, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32))


def angle_between(points, dimensions, first_axis, second_axis):
    scaled = points * dimensions[:, None, :]
    first = scaled[:, first_axis[1]] - scaled[:, first_axis[0]]
    second = scaled[:, second_axis[1]] - scaled[:, second_axis[0]]
    cosine = (first * second).sum(-1).abs() / (
        first.norm(dim=-1) * second.norm(dim=-1)).clamp_min(1e-6)
    return torch.rad2deg(torch.acos(cosine.clamp(0, 1 - 1e-7)))


def execute(points, dimensions):
    return torch.stack([
        angle_between(points, dimensions, (0, 1), (2, 3)),
        angle_between(points, dimensions, (2, 3), (4, 5)),
    ], dim=-1)


def relational_features(points):
    pairs = []
    for first in range(6):
        for second in range(first + 1, 6):
            delta = points[:, second] - points[:, first]
            distance = delta.norm(dim=-1, keepdim=True)
            direction = delta / distance.clamp_min(1e-6)
            pairs.append(torch.cat([distance, direction], dim=-1))
    return torch.cat(pairs, dim=-1)


class RepairMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(12 + 45 + 1, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(.1),
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 12),
        )

    def forward(self, points, dimensions):
        aspect = torch.log(dimensions[:, :1] / dimensions[:, 1:].clamp_min(1))
        features = torch.cat([points.flatten(1), relational_features(points), aspect], dim=-1)
        return self.network(features).reshape(-1, 6, 2)


class VerifierMLP(nn.Module):
    """Predict whether a proposed edit improves executable protocol error."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(12 + 12 + 1 + 4, 128), nn.GELU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, points, delta, dimensions):
        proposed = (points + delta).clamp(0, 1)
        aspect = torch.log(dimensions[:, :1] / dimensions[:, 1:].clamp_min(1))
        angles = torch.cat([execute(points, dimensions), execute(proposed, dimensions)], dim=-1)
        features = torch.cat([points.flatten(1), delta.flatten(1), aspect, angles], dim=-1)
        return self.network(features).squeeze(-1)


def corrupt(points, min_sigma, max_sigma, identity_fraction=.25):
    sigma = torch.empty((len(points), 1, 1), device=points.device).uniform_(min_sigma, max_sigma)
    identity_count = int(len(points) * identity_fraction)
    sigma[:identity_count] = 0
    return (points + torch.randn_like(points) * sigma).clamp(0, 1)


def metrics(model, data, sigmas, steps):
    clean, dimensions, targets = data
    output = {}
    model.eval()
    with torch.no_grad():
        for sigma in sigmas:
            generator = torch.Generator().manual_seed(1000 + int(sigma * 10000))
            noisy = (clean + torch.randn(clean.shape, generator=generator) * sigma).clamp(0, 1)
            initial_angles = execute(noisy, dimensions)
            current = noisy
            for _ in range(steps):
                current = (current + model(current, dimensions)).clamp(0, 1)
            repaired_angles = execute(current, dimensions)
            initial_error = (initial_angles - targets).abs()
            repaired_error = (repaired_angles - targets).abs()
            initial_lm = (noisy - clean).norm(dim=-1).mean(-1)
            repaired_lm = (current - clean).norm(dim=-1).mean(-1)
            initial_protocol = initial_error[:, 0] / 5 + initial_error[:, 1] / 3
            repaired_protocol = repaired_error[:, 0] / 5 + repaired_error[:, 1] / 3
            output[str(sigma)] = {
                "n": len(clean), "steps": steps,
                "landmark_MRE_before": initial_lm.mean().item(),
                "landmark_MRE_after": repaired_lm.mean().item(),
                "HVA_MAE_before": initial_error[:, 0].mean().item(),
                "HVA_MAE_after": repaired_error[:, 0].mean().item(),
                "IMA_MAE_before": initial_error[:, 1].mean().item(),
                "IMA_MAE_after": repaired_error[:, 1].mean().item(),
                "joint_protocol_harm_rate": (repaired_protocol > initial_protocol).float().mean().item(),
                "any_measurement_harm_rate": (repaired_error > initial_error).any(dim=1).float().mean().item(),
            }
    return output


def gated_metrics(model, verifier, threshold, data, sigmas, max_steps):
    clean, dimensions, targets = data
    output = {}
    model.eval(); verifier.eval()
    with torch.no_grad():
        for sigma in sigmas:
            generator = torch.Generator().manual_seed(1000 + int(sigma * 10000))
            noisy = (clean + torch.randn(clean.shape, generator=generator) * sigma).clamp(0, 1)
            initial_angles = execute(noisy, dimensions)
            current = noisy
            edit_count = torch.zeros(len(clean))
            for _ in range(max_steps):
                delta = model(current, dimensions)
                probability = torch.sigmoid(verifier(current, delta, dimensions))
                gate = probability >= threshold
                current = torch.where(gate[:, None, None], (current + delta).clamp(0, 1), current)
                edit_count += gate.float()
            repaired_angles = execute(current, dimensions)
            initial_error, repaired_error = (initial_angles-targets).abs(), (repaired_angles-targets).abs()
            initial_protocol = initial_error[:, 0] / 5 + initial_error[:, 1] / 3
            repaired_protocol = repaired_error[:, 0] / 5 + repaired_error[:, 1] / 3
            output[str(sigma)] = {
                "n": len(clean), "max_steps": max_steps, "threshold": threshold,
                "coverage": (edit_count > 0).float().mean().item(),
                "mean_edits": edit_count.mean().item(),
                "HVA_MAE_before": initial_error[:, 0].mean().item(),
                "HVA_MAE_after": repaired_error[:, 0].mean().item(),
                "IMA_MAE_before": initial_error[:, 1].mean().item(),
                "IMA_MAE_after": repaired_error[:, 1].mean().item(),
                "joint_protocol_harm_rate": (repaired_protocol > initial_protocol).float().mean().item(),
                "any_measurement_harm_rate": (repaired_error > initial_error).any(dim=1).float().mean().item(),
            }
    return output


def train_verifier(model, train, epochs=200):
    verifier = VerifierMLP()
    optimizer = torch.optim.AdamW(verifier.parameters(), lr=2e-3, weight_decay=1e-4)
    model.eval()
    for _ in range(epochs):
        noisy = corrupt(train[0], 0, .04, identity_fraction=.25)
        with torch.no_grad():
            delta = model(noisy, train[1])
            proposed = (noisy + delta).clamp(0, 1)
            before = (execute(noisy, train[1]) - train[2]).abs()
            after = (execute(proposed, train[1]) - train[2]).abs()
            improvement = (before[:, 0] / 5 + before[:, 1] / 3) - (after[:, 0] / 5 + after[:, 1] / 3)
            label = (improvement > 0).float()
        logits = verifier(noisy, delta, train[1])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, label)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return verifier


def select_threshold(model, verifier, val):
    best_threshold, best_score = .5, float("inf")
    for threshold in [.3, .4, .5, .6, .7, .8, .9]:
        rows = gated_metrics(model, verifier, threshold, val, [.005, .01, .02, .04], 1)
        score = sum(row["HVA_MAE_after"] / 5 + row["IMA_MAE_after"] / 3 +
                    8 * row["joint_protocol_harm_rate"] for row in rows.values())
        if score < best_score:
            best_threshold, best_score = threshold, score
    return best_threshold, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/hvangleest"))
    parser.add_argument("--output", type=Path, default=Path("outputs/research/repair_mlp"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    train = load_split(args.data_root / "train.jsonl")
    val = load_split(args.data_root / "val.jsonl")
    test = load_split(args.data_root / "test.jsonl")
    model = RepairMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_state, best_score = None, float("inf")
    for epoch in range(args.epochs):
        model.train(); noisy = corrupt(train[0], .002, .04)
        delta = model(noisy, train[1]); repaired = (noisy + delta).clamp(0, 1)
        coordinate_loss = nn.functional.smooth_l1_loss(repaired, train[0], beta=.01)
        angle_loss = nn.functional.smooth_l1_loss(execute(repaired, train[1]), train[2], beta=2.0)
        step_loss = delta.abs().mean()
        loss = coordinate_loss + .001 * angle_loss + .01 * step_loss
        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if (epoch + 1) % 5 == 0:
            val_metrics = metrics(model, val, [.01, .02], 1)
            score = sum(
                row["HVA_MAE_after"] + row["IMA_MAE_after"] +
                10.0 * row["joint_protocol_harm_rate"] + 100.0 * row["landmark_MRE_after"]
                for row in val_metrics.values()
            )
            if score < best_score:
                best_score = score
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    verifier = train_verifier(model, train)
    threshold, verifier_val_score = select_threshold(model, verifier, val)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"repair_state_dict": model.state_dict(),
                "verifier_state_dict": verifier.state_dict(),
                "threshold": threshold, "seed": args.seed}, args.output / f"seed{args.seed}.pt")
    result = {
        "method": "one_shot_geometry_only_residual_mlp",
        "seed": args.seed, "patient_disjoint_split": True,
        "validation_selection_score": best_score,
        "verifier_threshold": threshold,
        "verifier_validation_score": verifier_val_score,
        "test_one_step": metrics(model, test, [.005, .01, .02, .04], 1),
        "test_three_steps_same_policy": metrics(model, test, [.005, .01, .02, .04], 3),
        "test_gated_one_step": gated_metrics(model, verifier, threshold, test, [.005, .01, .02, .04], 1),
        "test_gated_three_steps": gated_metrics(model, verifier, threshold, test, [.005, .01, .02, .04], 3),
    }
    (args.output / f"seed{args.seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
