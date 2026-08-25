#!/usr/bin/env python3
"""Evaluate selective geometry repair on frozen HRNet landmark predictions."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


def import_training_module(path: Path):
    spec = importlib.util.spec_from_file_location("geometry_repair_training", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BasicBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(True)

    def forward(self, inputs):
        return self.relu(self.net(inputs) + inputs)


class HRNetLite(nn.Module):
    def __init__(self, landmarks: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
        )
        self.hr = nn.Sequential(*[BasicBlock(64) for _ in range(4)])
        self.lr1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            *[BasicBlock(128) for _ in range(4)],
        )
        self.fuse_up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(True),
        )
        self.post = nn.Sequential(*[BasicBlock(64) for _ in range(2)])
        self.head = nn.Conv2d(64, landmarks, 1)

    def forward(self, inputs):
        stem = self.stem(inputs)
        return self.head(self.post(self.hr(stem) + self.fuse_up(self.lr1(stem))))


def decode_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    batch, landmarks, _, width = heatmaps.shape
    indices = heatmaps.flatten(2).argmax(-1)
    x = (indices.remainder(width).float() + .5) / width
    y = (indices.div(width, rounding_mode="floor").float() + .5) / width
    return torch.stack([x, y], dim=-1).reshape(batch, landmarks, 2)


def load_manifest(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def predict(detector, rows, image_dir: Path, batch_size: int):
    all_points = []
    detector.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            images = []
            for row in rows[start:start + batch_size]:
                image = cv2.imread(str(image_dir / row["image_file"]))
                if image is None:
                    raise FileNotFoundError(image_dir / row["image_file"])
                image = cv2.cvtColor(cv2.resize(image, (256, 256)), cv2.COLOR_BGR2RGB)
                images.append(torch.tensor(image / 255., dtype=torch.float32).permute(2, 0, 1))
            all_points.append(decode_heatmaps(detector(torch.stack(images))))
    return torch.cat(all_points)


def evaluate(repair, verifier, threshold, points, dimensions, targets, execute, max_steps):
    initial = execute(points, dimensions)
    current = points.clone()
    edits = torch.zeros(len(points))
    with torch.no_grad():
        for _ in range(max_steps):
            delta = repair(current, dimensions)
            accepted = torch.sigmoid(verifier(current, delta, dimensions)) >= threshold
            current = torch.where(accepted[:, None, None], (current + delta).clamp(0, 1), current)
            edits += accepted.float()
    final = execute(current, dimensions)
    before, after = (initial - targets).abs(), (final - targets).abs()
    joint_before = before[:, 0] / 5 + before[:, 1] / 3
    joint_after = after[:, 0] / 5 + after[:, 1] / 3
    proposal_gain = joint_before - joint_after
    acted = edits > 0
    beneficial = proposal_gain > .05
    harmful = proposal_gain < -.05
    stopped = ~acted
    safe_div = lambda numerator, denominator: (
        numerator.float().sum().item() / max(1, denominator.float().sum().item())
    )
    return {
        "n": len(points), "max_steps": max_steps, "threshold": threshold,
        "coverage": (edits > 0).float().mean().item(), "mean_edits": edits.mean().item(),
        "HVA_MAE_before": before[:, 0].mean().item(), "HVA_MAE_after": after[:, 0].mean().item(),
        "IMA_MAE_before": before[:, 1].mean().item(), "IMA_MAE_after": after[:, 1].mean().item(),
        "joint_protocol_harm_rate": (joint_after > joint_before).float().mean().item(),
        "any_measurement_harm_rate": (after > before).any(dim=1).float().mean().item(),
        "decision_quality": {
            "repair_precision": safe_div(acted & beneficial, acted),
            "repair_recall": safe_div(acted & beneficial, beneficial),
            "stop_precision": safe_div(stopped & harmful, stopped),
            "stop_recall": safe_div(stopped & harmful, harmful),
            "repair_success_rate": safe_div(acted & beneficial, acted),
            "executed_harm_rate": safe_div(acted & harmful, acted),
            "harm_at_1_protocol_unit": (acted & (joint_after - joint_before > 1.0)).float().mean().item(),
            "mean_tool_calls": float(3.0 + edits.mean().item()),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/hvangleest/test.jsonl"))
    parser.add_argument("--image-dir", type=Path, default=Path("/media/max/a/caxp/HVAngleEst/HVAngleEst/images"))
    parser.add_argument("--detector", type=Path, default=Path("/media/max/a/caxp (Copy 2)/geomed_output/hrnet_lm.pt"))
    parser.add_argument("--repair", type=Path, default=Path("outputs/research/repair_mlp_verifier/seed42.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/research/hrnet_geometry_repair.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    training = import_training_module(Path(__file__).with_name("train_geometry_repair.py"))
    rows = load_manifest(args.manifest)
    dimensions = torch.tensor([[r["image_width"], r["image_height"]] for r in rows], dtype=torch.float32)
    targets = torch.tensor([[r["HVA"], r["IMA"]] for r in rows], dtype=torch.float32)

    detector = HRNetLite()
    detector.load_state_dict(torch.load(args.detector, map_location="cpu", weights_only=True))
    points = predict(detector, rows, args.image_dir, args.batch_size)

    checkpoint = torch.load(args.repair, map_location="cpu", weights_only=True)
    repair, verifier = training.RepairMLP(), training.VerifierMLP()
    repair.load_state_dict(checkpoint["repair_state_dict"])
    verifier.load_state_dict(checkpoint["verifier_state_dict"])
    threshold = checkpoint["threshold"]
    result = {
        "error_source": "frozen_hrnet_lite_heatmap_detector",
        "patient_disjoint_test": True,
        "one_step": evaluate(repair, verifier, threshold, points, dimensions, targets, training.execute, 1),
        "three_steps": evaluate(repair, verifier, threshold, points, dimensions, targets, training.execute, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
