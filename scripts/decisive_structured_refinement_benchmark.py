#!/usr/bin/env python3
"""Go/no-go comparison on real weighted-PCA detector geometry.

Compares vanilla state refinement, measurement-aware state refinement, and scalar
residual correction on identical patient-grouped splits. This first experiment is
geometry-only by design; image conditioning is justified only if this gate passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import pandas as pd
import torch
from torch import nn
import cv2


def canonical(direction):
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-7)
    sign = torch.where(direction[..., 1:2] < 0, -1.0, 1.0)
    return direction * sign


def execute(directions):
    directions = canonical(directions)
    def angle(a, b):
        cosine = (a * b).sum(-1).abs().clamp(0, 1 - 1e-7)
        return torch.rad2deg(torch.acos(cosine))
    return torch.stack([angle(directions[:, 0], directions[:, 1]),
                        angle(directions[:, 1], directions[:, 2])], dim=-1)


def parse_axis(text, width, height):
    values = [float(value) for value in str(text).split(",")]
    points = torch.tensor([[values[0] * width, values[1] * height],
                           [values[2] * width, values[3] * height]])
    center = points.mean(0) / torch.tensor([width, height])
    direction = canonical((points[1] - points[0]).reshape(1, 2))[0]
    return center, direction


def patient_partition(patient_id, seed=2027):
    value = int(hashlib.sha1(f"{seed}:{patient_id}".encode()).hexdigest()[:8], 16) % 10
    return "train" if value < 6 else "val" if value < 8 else "test"


def load_real_errors(results_path: Path, csv_path: Path, image_dir: Path):
    annotations = pd.read_csv(csv_path).set_index("filename")
    runs = json.loads(results_path.read_text())
    rows = []
    for run in runs:
        test = run["test"]
        targets = torch.tensor([
            test["measurements"]["hva"]["ground_truth"],
            test["measurements"]["ima"]["ground_truth"],
        ]).T
        for index, (identifier, geometry) in enumerate(zip(
                test["identifiers"], test["predicted_axis_directions"])):
            annotation = annotations.loc[identifier]
            width, height = float(annotation.image_width), float(annotation.image_height)
            gt = [parse_axis(annotation[name], width, height)
                  for name in ["great_toe", "first_metatarsal", "second_metatarsal"]]
            predicted_centers = torch.tensor(geometry["centers_px"], dtype=torch.float32)
            predicted_centers /= torch.tensor([width, height])
            rows.append({
                "identifier": identifier, "patient_id": int(annotation.patient_id),
                "seed": run["seed"], "split": patient_partition(int(annotation.patient_id)),
                "predicted_centers": predicted_centers,
                "predicted_directions": canonical(torch.tensor(geometry["directions"], dtype=torch.float32)),
                "gt_centers": torch.stack([item[0] for item in gt]),
                "gt_directions": torch.stack([item[1] for item in gt]),
                "target": targets[index].float(),
                "aspect": torch.tensor([math.log(width / height)]),
                "image": torch.tensor(cv2.resize(
                    cv2.imread(str(image_dir / identifier), cv2.IMREAD_GRAYSCALE), (64, 64)
                ) / 255., dtype=torch.float32).unsqueeze(0),
            })
    by_identifier = {}
    for row in rows:
        by_identifier.setdefault(row["identifier"], []).append(row)
    for group in by_identifier.values():
        stacked = torch.stack([row["predicted_directions"] for row in group])
        consensus = canonical(stacked.mean(0))
        deviations = torch.rad2deg(torch.acos(
            (stacked * consensus[None]).sum(-1).abs().clamp(0, 1 - 1e-7)))
        dispersion = deviations.mean(0)
        for index, row in enumerate(group):
            row["ensemble_consensus_directions"] = consensus
            row["ensemble_axis_features"] = torch.stack(
                [deviations[index], dispersion], dim=-1)
    return rows


def tensors(rows, split):
    chosen = [row for row in rows if row["split"] == split]
    stack = lambda key: torch.stack([row[key] for row in chosen])
    return {
        "centers": stack("predicted_centers"), "directions": stack("predicted_directions"),
        "gt_centers": stack("gt_centers"), "gt_directions": stack("gt_directions"),
        "targets": stack("target"), "aspect": stack("aspect"),
        "images": stack("image"),
        "ensemble_axis_features": stack("ensemble_axis_features"),
        "ensemble_consensus_directions": stack("ensemble_consensus_directions"),
        "identifiers": [row["identifier"] for row in chosen],
    }


def features(data):
    angles = execute(data["directions"])
    return torch.cat([data["centers"].flatten(1), data["directions"].flatten(1),
                      angles / 90, data["aspect"]], dim=-1)


class MLP(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(15, 128), nn.GELU(), nn.LayerNorm(128),
                                 nn.Linear(128, 128), nn.GELU(), nn.Linear(128, output))

    def forward(self, inputs):
        return self.net(inputs)


class ImageGeometryModel(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(256 + 15, 192), nn.GELU(), nn.LayerNorm(192),
                                  nn.Linear(192, 96), nn.GELU(), nn.Linear(96, output))

    def forward(self, geometry_features, images):
        return self.head(torch.cat([geometry_features, self.encoder(images)], dim=-1))


def refined_state(model, data):
    delta = (model(features(data), data["images"]) if isinstance(model, ImageGeometryModel)
             else model(features(data))).reshape(-1, 3, 4)
    centers = (data["centers"] + .1 * torch.tanh(delta[..., :2])).clamp(0, 1)
    directions = canonical(data["directions"] + .25 * torch.tanh(delta[..., 2:]))
    return centers, directions


def train_refiner(train, val, measurement_aware, epochs=600, image_conditioned=False):
    model, optimizer = (ImageGeometryModel(12) if image_conditioned else MLP(12)), None
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best, best_state = float("inf"), None
    for epoch in range(epochs):
        centers, directions = refined_state(model, train)
        center_loss = nn.functional.smooth_l1_loss(centers, train["gt_centers"], beta=.01)
        direction_loss = (1 - (directions * train["gt_directions"]).sum(-1).abs()).mean()
        loss = center_loss + direction_loss
        if measurement_aware:
            angle_loss = nn.functional.smooth_l1_loss(execute(directions), train["targets"], beta=2.)
            initial = train["directions"].detach().requires_grad_(True)
            sensitivity = torch.autograd.grad(execute(initial).sum(), initial)[0].norm(dim=-1).detach()
            component_error = 1 - (directions * train["gt_directions"]).sum(-1).abs()
            loss = loss + .05 * angle_loss + .1 * (sensitivity * component_error).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                _, val_directions = refined_state(model, val)
                score = (execute(val_directions) - val["targets"]).abs().mean().item()
            if score < best:
                best = score
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def train_scalar(train, val, epochs=600, image_conditioned=False):
    model = ImageGeometryModel(2) if image_conditioned else MLP(2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best, best_state = float("inf"), None
    for epoch in range(epochs):
        residual = (model(features(train), train["images"]) if image_conditioned else model(features(train)))
        prediction = execute(train["directions"]) + residual
        loss = nn.functional.smooth_l1_loss(prediction, train["targets"], beta=2.)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                residual = (model(features(val), val["images"]) if image_conditioned else model(features(val)))
                score = (execute(val["directions"]) + residual - val["targets"]).abs().mean().item()
            if score < best:
                best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def local_oracle(data):
    initial = execute(data["directions"])
    candidates = [initial]
    for axis in range(3):
        directions = data["directions"].clone()
        directions[:, axis] = data["gt_directions"][:, axis]
        candidates.append(execute(directions))
    candidates = torch.stack(candidates, dim=1)
    score = (candidates - data["targets"][:, None]).abs()
    score = score[..., 0] / 5 + score[..., 1] / 3
    index = score.argmin(1)
    return candidates[torch.arange(len(index)), index], index


def report(name, prediction, data, initial=None):
    error = (prediction - data["targets"]).abs()
    result = {"HVA_MAE": error[:, 0].mean().item(), "IMA_MAE": error[:, 1].mean().item(),
              "mean_MAE": error.mean().item(), "P90_joint_abs_error": error.mean(1).quantile(.9).item()}
    if initial is not None:
        before = (initial - data["targets"]).abs()
        result["joint_harm_rate"] = ((error[:, 0] / 5 + error[:, 1] / 3) >
                                     (before[:, 0] / 5 + before[:, 1] / 3)).float().mean().item()
    return name, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("/media/max/a/caxp (Copy 2)/outputs/benchmarks/controlled_cross_model_v2/hvangle_axis_geometry/results.json"))
    parser.add_argument("--annotations", type=Path, default=Path("/media/max/a/caxp/HVAngleEst/HVAngleEst/datasets.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("/media/max/a/caxp/HVAngleEst/HVAngleEst/images"))
    parser.add_argument("--output", type=Path, default=Path("outputs/research/decisive_structured_refinement.json"))
    parser.add_argument("--epochs", type=int, default=600)
    args = parser.parse_args()
    random.seed(42); torch.manual_seed(42)
    rows = load_real_errors(args.results, args.annotations, args.image_dir)
    train, val, test = (tensors(rows, split) for split in ["train", "val", "test"])
    vanilla = train_refiner(train, val, False, args.epochs)
    aware = train_refiner(train, val, True, args.epochs)
    scalar = train_scalar(train, val, args.epochs)
    image_aware = train_refiner(train, val, True, args.epochs, image_conditioned=True)
    image_scalar = train_scalar(train, val, args.epochs, image_conditioned=True)
    initial = execute(test["directions"])
    with torch.no_grad():
        vanilla_centers, vanilla_directions = refined_state(vanilla, test)
        aware_centers, aware_directions = refined_state(aware, test)
        scalar_prediction = initial + scalar(features(test))
        _, image_aware_directions = refined_state(image_aware, test)
        image_scalar_prediction = initial + image_scalar(features(test), test["images"])
        oracle_prediction, oracle_edit = local_oracle(test)
    results = dict([
        report("analytic_detector", initial, test),
        report("local_one_axis_oracle", oracle_prediction, test, initial),
        report("vanilla_geometry_refinement", execute(vanilla_directions), test, initial),
        report("measurement_aware_refinement", execute(aware_directions), test, initial),
        report("scalar_residual_correction", scalar_prediction, test, initial),
        report("image_conditioned_measurement_aware_refinement", execute(image_aware_directions), test, initial),
        report("image_conditioned_scalar_correction", image_scalar_prediction, test, initial),
    ])
    results["local_one_axis_oracle"]["edit_rate"] = (oracle_edit > 0).float().mean().item()
    for name, centers, directions in [("vanilla_geometry_refinement", vanilla_centers, vanilla_directions),
                                      ("measurement_aware_refinement", aware_centers, aware_directions)]:
        results[name]["center_MRE"] = (centers - test["gt_centers"]).norm(dim=-1).mean().item()
        results[name]["axis_angular_MAE"] = torch.rad2deg(torch.acos(
            (directions * test["gt_directions"]).sum(-1).abs().clamp(0, 1 - 1e-7))).mean().item()
    output = {"split": {key: len(value["targets"]) for key, value in
                         [("train", train), ("val", val), ("test", test)]},
              "grouped_by_patient": True,
              "experiment": "geometry-only and lightweight image-conditioned go-no-go",
              "image_encoder": "three-layer CNN trained from scratch at 64x64",
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
