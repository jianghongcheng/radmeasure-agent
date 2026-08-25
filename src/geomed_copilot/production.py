from __future__ import annotations

import json
from pathlib import Path

from .artifact_predictor import FrozenPredictionArtifact, sha256_file
from .evidence import load_evidence_catalog
from .models import CopilotRequest, Line, Point
from .orchestrator import GeoMedCopilot
from .real_cases import cases_from_locked_split, cases_from_manifest
from .retrieval import CaseRetriever, HybridRetriever
from .sample_data import CASES, DEMO_LANDMARKS, EVIDENCE


class LockedArtifactService:
    """Auditable application service backed by locked model artifacts."""

    def __init__(self, predictions: Path, annotations: Path,
                 split_manifest: Path, evidence_catalog: Path) -> None:
        self.predictor = FrozenPredictionArtifact(predictions)
        evidence = load_evidence_catalog(evidence_catalog)
        cases = cases_from_locked_split(annotations, split_manifest, "train")
        self.copilot = GeoMedCopilot(HybridRetriever(evidence), CaseRetriever(cases))

    def analyze(self, image_id: str, question: str, top_k: int = 3) -> dict:
        predicted = self.predictor.predict(image_id)
        axes = self.predictor.predict_axes(image_id)
        lines = {name: Line(Point(0.0, 0.0), Point(dx, dy))
                 for name, (dx, dy) in axes.items()}
        response = self.copilot.run(CopilotRequest(
            question=question, image_id=image_id, landmarks=lines,
            predicted_angles=predicted, top_k=top_k))
        result = response.to_dict()
        result["provenance"] = {
            "mode": "locked_prediction_artifact_replay",
            "repair_geometry_source": "same_prediction_artifact",
            "automatic_repair_allowed": False,
            "prediction_artifact_sha256": self.predictor.sha256,
            "live_encoder_inference": False,
            "clinical_use": False,
        }
        return result


class DemoService:
    """Self-contained deterministic demo; never presented as model inference."""

    identifiers = {"demo-foot-001"}

    def __init__(self) -> None:
        self.copilot = GeoMedCopilot(HybridRetriever(EVIDENCE), CaseRetriever(CASES))

    def analyze(self, image_id: str, question: str, top_k: int = 3) -> dict:
        if image_id not in self.identifiers:
            raise KeyError(f"Unknown demo image_id: {image_id}")
        response = self.copilot.run(CopilotRequest(
            question=question,
            image_id=image_id,
            landmarks=DEMO_LANDMARKS,
            predicted_angles={"HVA": 15.0, "IMA": 8.0},
            top_k=top_k,
        ))
        result = response.to_dict()
        result["provenance"] = {
            "mode": "deterministic_synthetic_demo",
            "repair_geometry_source": "synthetic_fixture",
            "automatic_repair_allowed": True,
            "prediction_artifact_sha256": None,
            "live_encoder_inference": False,
            "clinical_use": False,
        }
        return result


class EvaluationReplayService:
    """Portable replay of persisted test predictions against public annotations."""

    def __init__(self, evaluation: Path, manifests_dir: Path,
                 train_manifest: Path, evidence_catalog: Path) -> None:
        artifact = json.loads(evaluation.read_text(encoding="utf-8"))
        self.metrics = artifact["metrics"]
        self.artifact_sha256 = sha256_file(evaluation)
        self.model = artifact["model"]
        self._predictions = {row["image_id"]: row for row in artifact["predictions"]}
        candidates: dict[str, list[tuple[str, dict]]] = {}
        for split in ("train", "val", "test"):
            with (manifests_dir / f"{split}.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    candidates.setdefault(record["image_file"], []).append((split, record))
        self._records = {}
        self.split_distribution = {"train": 0, "val": 0, "test": 0}
        for image_id, prediction in self._predictions.items():
            matches = [
                (split, record) for split, record in candidates.get(image_id, [])
                if all(abs(record[name] - prediction["target"][name]) < 1e-8 for name in ("HVA", "IMA"))
            ]
            if len(matches) == 1:
                split, record = matches[0]
                self._records[image_id] = record
                self.split_distribution[split] += 1
        missing = set(self._predictions) - set(self._records)
        if missing:
            raise ValueError(f"Evaluation cases without a unique annotation match: {sorted(missing)[:3]}")
        self.identifiers = set(self._predictions)
        evidence = load_evidence_catalog(evidence_catalog)
        query_ids = {record["sample_id"] for record in self._records.values()}
        cases = [case for case in cases_from_manifest(train_manifest) if case.evidence_id not in query_ids]
        self.copilot = GeoMedCopilot(HybridRetriever(evidence), CaseRetriever(cases))

    @staticmethod
    def _line(values: list[float], width: int, height: int) -> Line:
        """Restore pixel aspect ratio before reconstructing an image-space angle."""
        return Line(
            Point(values[0] * width, values[1] * height),
            Point(values[2] * width, values[3] * height),
        )

    def analyze(self, image_id: str, question: str, top_k: int = 3) -> dict:
        if image_id not in self._predictions:
            raise KeyError(f"{image_id!r} is not in the portable evaluation replay")
        prediction = self._predictions[image_id]
        record = self._records[image_id]
        landmarks = {
            name: self._line(record[name], record["image_width"], record["image_height"])
            for name in ("great_toe", "first_metatarsal", "second_metatarsal")
        }
        response = self.copilot.run(CopilotRequest(
            question=question,
            image_id=image_id,
            landmarks=landmarks,
            predicted_angles=prediction["predicted"],
            top_k=top_k,
        ))
        result = response.to_dict()
        result["evaluation"] = {
            "target": prediction["target"],
            "absolute_error": {
                name: round(abs(value - prediction["target"][name]), 6)
                for name, value in prediction["predicted"].items()
            },
            "locked_test_metrics": self.metrics,
        }
        result["provenance"] = {
            "mode": "portable_locked_evaluation_replay",
            "repair_geometry_source": "released_annotation_ground_truth",
            "automatic_repair_allowed": False,
            "evaluation_artifact_sha256": self.artifact_sha256,
            "model": self.model,
            "current_manifest_split_distribution": self.split_distribution,
            "split_alignment_verified": self.split_distribution == {"train": 0, "val": 0, "test": 176},
            "live_encoder_inference": False,
            "clinical_use": False,
        }
        return result
