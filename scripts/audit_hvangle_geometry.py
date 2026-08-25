#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from geomed_copilot.geometry import acute_angle_degrees
from geomed_copilot.models import Line, Point


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def audit(manifest_dir: Path) -> dict:
    errors = {"HVA": [], "IMA": []}
    split_counts = {}
    for split in ("train", "val", "test"):
        records = [json.loads(line) for line in (manifest_dir / f"{split}.jsonl").read_text().splitlines()]
        split_counts[split] = len(records)
        for record in records:
            width, height = record["image_width"], record["image_height"]

            def line(name: str) -> Line:
                x1, y1, x2, y2 = record[name]
                return Line(Point(x1 * width, y1 * height), Point(x2 * width, y2 * height))

            calculated = {
                "HVA": acute_angle_degrees(line("great_toe"), line("first_metatarsal")),
                "IMA": acute_angle_degrees(line("first_metatarsal"), line("second_metatarsal")),
            }
            for name in errors:
                errors[name].append(abs(calculated[name] - record[name]))

    metrics = {}
    for name, values in errors.items():
        metrics[name] = {
            "samples": len(values),
            "reconstruction_mae_degrees": statistics.mean(values),
            "reconstruction_p95_error_degrees": percentile(values, 0.95),
            "reconstruction_max_error_degrees": max(values),
            "within_0_1_degree_rate": sum(value <= 0.1 for value in values) / len(values),
        }
    return {
        "audit": "annotation_geometry_reconstruction",
        "coordinate_system": "normalized coordinates converted to source-image pixels",
        "split_counts": split_counts,
        "metrics": metrics,
        "interpretation": "Checks manifest conversion only; this is not model performance.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct HVAngleEst angles from manifest landmarks")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.manifest_dir)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

