#!/usr/bin/env python3
"""Go/no-go benchmark for recoverability through explicit geometry repair."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from geomed_copilot.geometry_repair import gaussian_corruption, program_aware_oracle_step
from geomed_copilot.measurement_program import (
    HVA_PROGRAM, IMA_PROGRAM, denormalize_points, execute_program,
)


POINT_NAMES = (
    "gt_proximal", "gt_distal", "m1_proximal", "m1_distal",
    "m2_proximal", "m2_distal",
)


def parse_points(row: dict[str, str]) -> dict[str, tuple[float, float]]:
    values = []
    for field in ("great_toe", "first_metatarsal", "second_metatarsal"):
        raw = [float(value) for value in row[field].split(",")]
        values.extend([(raw[0], raw[1]), (raw[2], raw[3])])
    return dict(zip(POINT_NAMES, values))


def mean(values):
    return sum(values) / max(len(values), 1)


def run(dataset: Path, sigmas: list[float], repeats: int, seed: int) -> dict:
    with dataset.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rng = random.Random(seed)
    result = {"dataset": str(dataset), "samples": len(rows), "seed": seed, "conditions": []}
    for sigma in sigmas:
        initial_errors = {"HVA": [], "IMA": []}
        repaired_errors = {"HVA": [], "IMA": []}
        landmark_before, landmark_after, harms, edited = [], [], [], []
        for row in rows:
            target = parse_points(row)
            width, height = float(row["image_width"]), float(row["image_height"])
            targets = {"HVA": float(row["HVA"]), "IMA": float(row["IMA"])}
            for _ in range(repeats):
                corrupted, _ = gaussian_corruption(target, sigma, rng)
                def protocol_score(candidate):
                    pixels = denormalize_points(candidate, width, height)
                    hva_error = abs(execute_program(HVA_PROGRAM, pixels) - targets["HVA"])
                    ima_error = abs(execute_program(IMA_PROGRAM, pixels) - targets["IMA"])
                    return (hva_error / 5.0) + (ima_error / 3.0)

                repaired, action = program_aware_oracle_step(corrupted, target, protocol_score)
                edited.append(action is not None)
                before_lm = mean([((corrupted[k][0]-target[k][0])**2 + (corrupted[k][1]-target[k][1])**2)**.5 for k in target])
                after_lm = mean([((repaired[k][0]-target[k][0])**2 + (repaired[k][1]-target[k][1])**2)**.5 for k in target])
                landmark_before.append(before_lm); landmark_after.append(after_lm)
                got_worse = False
                for name, program in (("HVA", HVA_PROGRAM), ("IMA", IMA_PROGRAM)):
                    before = abs(execute_program(
                        program, denormalize_points(corrupted, width, height)) - targets[name])
                    after = abs(execute_program(
                        program, denormalize_points(repaired, width, height)) - targets[name])
                    initial_errors[name].append(before); repaired_errors[name].append(after)
                    got_worse = got_worse or after > before + 1e-9
                harms.append(got_worse)
        result["conditions"].append({
            "sigma_normalized": sigma,
            "trials": len(landmark_before),
            "landmark_mre_before": mean(landmark_before),
            "landmark_mre_after_one_oracle_edit": mean(landmark_after),
            "HVA_MAE_before": mean(initial_errors["HVA"]),
            "HVA_MAE_after_one_oracle_edit": mean(repaired_errors["HVA"]),
            "IMA_MAE_before": mean(initial_errors["IMA"]),
            "IMA_MAE_after_one_oracle_edit": mean(repaired_errors["IMA"]),
            "harm_rate": mean(harms),
            "edit_rate": mean(edited),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[.005, .01, .02, .04])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = run(args.dataset, args.sigmas, args.repeats, args.seed)
    rendered = json.dumps(output, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
