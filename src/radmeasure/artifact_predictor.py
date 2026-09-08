from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path


AXES = ("great_toe", "first_metatarsal", "second_metatarsal")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(dx: float, dy: float) -> tuple[float, float]:
    norm = math.hypot(dx, dy)
    if norm <= 1e-12:
        raise ValueError("Predicted axis has zero length")
    dx, dy = dx / norm, dy / norm
    if dx < 0 or (abs(dx) < 1e-12 and dy < 0):
        dx, dy = -dx, -dy
    return dx, dy


def _mean_axis(axes: list[tuple[float, float]]) -> tuple[float, float]:
    # Align undirected axes before averaging across seeds.
    reference = _canonical(*axes[0])
    aligned = []
    for axis in axes:
        current = _canonical(*axis)
        if reference[0] * current[0] + reference[1] * current[1] < 0:
            current = (-current[0], -current[1])
        aligned.append(current)
    return _canonical(
        sum(axis[0] for axis in aligned) / len(aligned),
        sum(axis[1] for axis in aligned) / len(aligned),
    )


def _acute(first: tuple[float, float], second: tuple[float, float]) -> float:
    cosine = abs(first[0] * second[0] + first[1] * second[1])
    return math.degrees(math.acos(min(1.0, max(0.0, cosine))))


class FrozenPredictionArtifact:
    """Replay versioned image-model predictions without claiming live inference."""

    def __init__(self, csv_path: Path, expected_sha256: str | None = None) -> None:
        self.csv_path = csv_path
        self.sha256 = sha256_file(csv_path)
        if expected_sha256 and self.sha256 != expected_sha256:
            raise ValueError(f"Prediction artifact hash mismatch: {self.sha256}")
        grouped: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        seeds: dict[str, set[int]] = defaultdict(set)
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                identifier = row["identifier"]
                seeds[identifier].add(int(row["seed"]))
                for name in AXES:
                    grouped[identifier][name].append((
                        float(row[f"{name}_axis_dx"]),
                        float(row[f"{name}_axis_dy"]),
                    ))
        if not grouped:
            raise ValueError("Prediction artifact is empty")
        seed_counts = {len(value) for value in seeds.values()}
        if seed_counts != {3}:
            raise ValueError(f"Expected three seeds per case, got {sorted(seed_counts)}")
        self._axes = {
            identifier: {name: _mean_axis(values) for name, values in axes.items()}
            for identifier, axes in grouped.items()
        }

    @property
    def identifiers(self) -> set[str]:
        return set(self._axes)

    def predict(self, image_id: str, landmarks: dict | None = None) -> dict[str, float]:
        if image_id not in self._axes:
            raise KeyError(
                f"{image_id!r} is not in the locked prediction artifact; "
                "live encoder inference is unavailable"
            )
        axes = self._axes[image_id]
        return {
            "HVA": _acute(axes["great_toe"], axes["first_metatarsal"]),
            "IMA": _acute(axes["first_metatarsal"], axes["second_metatarsal"]),
        }

    def predict_axes(self, image_id: str) -> dict[str, tuple[float, float]]:
        if image_id not in self._axes:
            raise KeyError(
                f"{image_id!r} is not in the locked prediction artifact; "
                "live encoder inference is unavailable"
            )
        return dict(self._axes[image_id])
