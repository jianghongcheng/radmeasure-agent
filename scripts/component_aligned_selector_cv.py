#!/usr/bin/env python3
"""Nested cross-fit selector using axis-aligned local image evidence."""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geomed_copilot.component_patches import extract_component_patches


def import_script(name, filename):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ComponentAlignedSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2), nn.GELU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(256, 64), nn.GELU())
        self.axis_embedding = nn.Embedding(3, 8)
        self.shared = nn.Sequential(nn.Linear(64 + 8 + 15 + 1 + 2, 128), nn.GELU(),
                                    nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU())
        self.output = nn.Linear(64, 5)

    def forward(self, geometry, patches, sensitivity, displacement):
        batch, components = patches.shape[:2]
        local = self.patch_encoder(patches.reshape(batch * components, *patches.shape[2:]))
        local = local.reshape(batch, components, -1)
        indices = torch.arange(components, device=patches.device)
        identity = self.axis_embedding(indices)[None].expand(batch, -1, -1)
        context = geometry[:, None].expand(-1, components, -1)
        hidden = self.shared(torch.cat([local, identity, context, sensitivity[..., None],
                                        displacement], dim=-1))
        raw = self.output(hidden)
        return {"rank": raw[..., 0], "benefit": raw[..., 1], "harm": raw[..., 2],
                "gain": nn.functional.softplus(raw[..., 3]),
                "harm_magnitude": nn.functional.softplus(raw[..., 4])}


def patches(images, centers, directions, aspect):
    return extract_component_patches(images, centers, directions, aspect,
                                     output_size=32, half_width=.14, half_length=.30)


def train_selector(cf, model, oof, epochs, delta):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    local = patches(oof["images"], oof["centers"], oof["directions"], oof["aspect"])
    for _ in range(epochs):
        output = model(oof["geometry"], local, oof["sensitivity"], oof["displacement"])
        loss = cf.selector_loss(output, oof["gain"], delta)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model


def run_model(base, selective, learned, proposal, model, data):
    gain, candidates, sensitivity, displacement = cf_selective_inputs(
        base, selective, learned, proposal, data)
    local = patches(data["images"], data["centers"], data["directions"], data["aspect"])
    with torch.no_grad(): output = model(base.features(data), local, sensitivity, displacement)
    return gain, candidates, output


def cf_selective_inputs(base, selective, learned, proposal, data):
    gain, proposed, candidates = learned.actual_gains(base, proposal, data)
    sensitivity = selective.sensitivities(base, data["directions"])
    return gain, candidates, sensitivity, proposed - data["directions"]


def evaluate(base, selective, learned, risk, cf, rows, outer, args):
    train_rows, cal_rows, test_rows = cf.outer_partitions(
        rows, outer, args.outer_folds, args.split_seed)
    train, calibration, test = [cf.as_data(base, part) for part in [train_rows, cal_rows, test_rows]]
    oof = cf.build_oof_proposal_data(base, selective, learned, train_rows, outer, args)
    torch.manual_seed(8000 + outer); random.seed(8000 + outer)
    selector = train_selector(cf, ComponentAlignedSelector(), oof, args.selector_epochs, args.delta)
    torch.manual_seed(9000 + outer); random.seed(9000 + outer)
    proposal = base.train_refiner(train, train, True, args.proposal_epochs, image_conditioned=True)
    cal_gain, cal_candidates, cal_output = run_model(
        base, selective, learned, proposal, selector, calibration)
    benefit_t, harm_t, policy, _ = risk.select_policy(
        base, learned, calibration, cal_gain, cal_candidates, cal_output,
        args.delta, args.risk_limit)
    gain, candidates, output = run_model(base, selective, learned, proposal, selector, test)
    values = risk.utility(output, benefit_t, harm_t, policy["risk_weight"])
    axis = output["rank"].argmax(1)
    edit = values[torch.arange(len(axis)), axis] > policy["threshold"]
    selected = learned.choose(base, test, candidates, axis, edit)
    oracle_axis, oracle_gain = gain.argmax(1), gain.max(1).values
    oracle_edit = oracle_gain > 0
    initial = base.execute(test["directions"])
    metric = lambda prediction, accepted, chosen: risk.detailed_metrics(
        base, learned, test, prediction, accepted, gain, chosen, args.delta)
    return {"outer_fold": outer, "policy": policy, "methods": {
        "no_repair": metric(initial, torch.zeros(len(axis), dtype=torch.bool), axis),
        "component_aligned_selector": metric(selected, edit, axis),
        "oracle_selector_learned_repair": metric(learned.choose(
            base, test, candidates, oracle_axis, oracle_edit), oracle_edit, oracle_axis)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("/media/max/a/caxp (Copy 2)/outputs/benchmarks/controlled_cross_model_v2/hvangle_axis_geometry/results.json"))
    parser.add_argument("--annotations", type=Path, default=Path("/media/max/a/caxp/HVAngleEst/HVAngleEst/datasets.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("/media/max/a/caxp/HVAngleEst/HVAngleEst/images"))
    parser.add_argument("--output", type=Path, default=Path("outputs/research/component_aligned_selector_cv.json"))
    parser.add_argument("--outer-folds", type=int, default=5); parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=2027); parser.add_argument("--proposal-epochs", type=int, default=180)
    parser.add_argument("--selector-epochs", type=int, default=250); parser.add_argument("--delta", type=float, default=.02)
    parser.add_argument("--risk-limit", type=float, default=.10)
    args = parser.parse_args()
    base = import_script("ca_base", "decisive_structured_refinement_benchmark.py")
    selective = import_script("ca_sel", "selective_axis_verifier_benchmark.py")
    learned = import_script("ca_learn", "learned_proposal_selective_repair_benchmark.py")
    risk = import_script("ca_risk", "risk_adjusted_selector_cv.py")
    cf = import_script("ca_cf", "crossfit_risk_selector_cv.py")
    rows = base.load_real_errors(args.results, args.annotations, args.image_dir)
    folds = []
    for outer in range(args.outer_folds):
        folds.append(evaluate(base, selective, learned, risk, cf, rows, outer, args))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"folds": folds, "aggregate": cf.aggregate(folds)}, indent=2)+"\n")
        print(json.dumps({"completed_outer_fold": outer, "methods": folds[-1]["methods"]}), flush=True)
    output = {"component_aligned": True, "nested_patient_crossfit": True,
              "folds": folds, "aggregate": cf.aggregate(folds)}
    args.output.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output["aggregate"], indent=2))


if __name__ == "__main__": main()
