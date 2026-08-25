#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


LANDMARK_COLUMNS = ("great_toe", "first_metatarsal", "second_metatarsal")
OUTPUT_COLUMNS = (
    "sample_id", "image_file", "side", "source", "image_width", "image_height",
    "truncated", "box", "box_clipped", *LANDMARK_COLUMNS, "HVA", "IMA",
)


def opaque_id(filename: str, side: str, occurrence: int) -> str:
    raw = f"geomed-copilot:v1:{filename}:{side}:{occurrence}".encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def parse_vector(value: str, expected: int) -> list[float]:
    values = [float(item.strip()) for item in value.split(",")]
    if len(values) != expected:
        raise ValueError(f"Expected {expected} coordinates, received {len(values)}")
    if not all(0.0 <= item <= 1.0 for item in values):
        raise ValueError("Expected normalized coordinates in [0, 1]")
    return values


def parse_box(value: str) -> tuple[list[float], bool]:
    values = [float(item.strip()) for item in value.split(",")]
    if len(values) != 4:
        raise ValueError(f"Expected 4 box coordinates, received {len(values)}")
    clipped = [min(1.0, max(0.0, item)) for item in values]
    return clipped, clipped != values


def assign_patients(patient_ids: list[str], seed: int) -> dict[str, str]:
    unique = sorted(set(patient_ids))
    random.Random(seed).shuffle(unique)
    train_end = round(len(unique) * 0.70)
    val_end = train_end + round(len(unique) * 0.15)
    assignments = {}
    for index, patient_id in enumerate(unique):
        assignments[patient_id] = (
            "train" if index < train_end else "val" if index < val_end else "test"
        )
    return assignments


def prepare(source_csv: Path, image_dir: Path, output_dir: Path, seed: int) -> dict:
    with source_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Source CSV is empty")
    required = {"patient_id", "filename", "labels", "source", "image_width", "image_height",
                "boxes", "properties", "HVA", "IMA", *LANDMARK_COLUMNS}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    missing_images = sorted({row["filename"] for row in rows if not (image_dir / row["filename"]).is_file()})
    if missing_images:
        raise FileNotFoundError(f"{len(missing_images)} referenced images are missing; first={missing_images[0]}")

    assignments = assign_patients([row["patient_id"] for row in rows], seed)
    counts = Counter()
    patient_sets = defaultdict(set)
    seen = Counter()
    prepared = []
    clipped_boxes = 0
    # Validate and transform every row before opening output files. A malformed
    # source row therefore cannot leave a plausible-looking partial manifest.
    for row in rows:
        key = (row["filename"], row["labels"].strip().lower())
        occurrence = seen[key]
        seen[key] += 1
        split = assignments[row["patient_id"]]
        patient_sets[split].add(row["patient_id"])
        box, box_clipped = parse_box(row["boxes"])
        clipped_boxes += int(box_clipped)
        record = {
                "sample_id": opaque_id(row["filename"], key[1], occurrence),
                "image_file": row["filename"],
                "side": key[1],
                "source": row["source"].strip().lower(),
                "image_width": int(float(row["image_width"])),
                "image_height": int(float(row["image_height"])),
                "truncated": row["properties"].strip().lower() == "truncated",
                "box": box,
                "box_clipped": box_clipped,
                "great_toe": parse_vector(row["great_toe"], 4),
                "first_metatarsal": parse_vector(row["first_metatarsal"], 4),
                "second_metatarsal": parse_vector(row["second_metatarsal"], 4),
                "HVA": float(row["HVA"]),
                "IMA": float(row["IMA"]),
        }
        assert tuple(record) == OUTPUT_COLUMNS
        prepared.append((split, record))
        counts[split] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_paths = {split: output_dir / f".{split}.jsonl.tmp" for split in ("train", "val", "test")}
    handles = {split: path.open("w", encoding="utf-8") for split, path in temp_paths.items()}
    try:
        for split, record in prepared:
            handles[split].write(json.dumps(record, separators=(",", ":")) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    for split, temp_path in temp_paths.items():
        temp_path.replace(output_dir / f"{split}.jsonl")

    patient_overlap = {
        "train_val": len(patient_sets["train"] & patient_sets["val"]),
        "train_test": len(patient_sets["train"] & patient_sets["test"]),
        "val_test": len(patient_sets["val"] & patient_sets["test"]),
    }
    audit = {
        "dataset": "HVAngleEst",
        "preparation_version": 1,
        "seed": seed,
        "source_rows": len(rows),
        "unique_images": len({row["filename"] for row in rows}),
        "unique_patients_used_only_for_splitting": len(assignments),
        "records": dict(counts),
        "patients": {split: len(values) for split, values in patient_sets.items()},
        "patient_overlap": patient_overlap,
        "patient_id_in_output": False,
        "image_bytes_copied": False,
        "boxes_clipped_to_image_bounds": clipped_boxes,
        "license_review": "Do not redistribute until dataset terms are confirmed.",
    }
    if any(patient_overlap.values()):
        raise AssertionError(f"Patient leakage detected: {patient_overlap}")
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare patient-disjoint HVAngleEst manifests")
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_csv, args.image_dir, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
