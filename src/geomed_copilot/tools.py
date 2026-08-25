from __future__ import annotations

from typing import Any

from .production import DemoService, EvaluationReplayService, LockedArtifactService


class GeoMedTools:
    """Typed application boundary shared by HTTP, MCP, and local evaluation.

    The current backend replays hash-locked predictions. Keeping this boundary
    explicit prevents clients from mistaking artifact replay for live inference.
    """

    def __init__(self, service: LockedArtifactService | DemoService | EvaluationReplayService) -> None:
        self.service = service

    def capabilities(self) -> dict[str, Any]:
        demo = isinstance(self.service, DemoService)
        replay = isinstance(self.service, EvaluationReplayService)
        identifiers = self.service.identifiers if (demo or replay) else self.service.predictor.identifiers
        mode = "deterministic_synthetic_demo" if demo else (
            "portable_locked_evaluation_replay" if replay else "locked_prediction_artifact_replay"
        )
        limitations = [
            "Only identifiers exposed by the configured backend are accepted.",
            "No image bytes are processed by the current backend.",
            "Outputs are for research and portfolio demonstration only.",
        ]
        if replay and not self.service.split_distribution == {"train": 0, "val": 0, "test": 176}:
            limitations.append(
                "The persisted evaluation artifact does not align with the current processed split manifests; patient-disjoint status is not claimed."
            )
        return {
            "service": "radmeasure",
            "mode": mode,
            "live_encoder_inference": False,
            "accepted_input": "image_id",
            "measurements": ["HVA", "IMA"],
            "agent": {
                "planner": "registry_constrained",
                "decisions": ["KEEP", "REPAIR", "STOP"],
                "maximum_repair_attempts": 1,
                "deterministic_executor": True,
            },
            "tools": [
                "list_geomed_capabilities",
                "analyze_radiograph",
                "list_available_cases",
            ],
            "available_cases": len(identifiers),
            "clinical_use": False,
            "limitations": limitations,
        }

    def list_available_cases(self, limit: int = 20) -> dict[str, Any]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if hasattr(self.service, "identifiers"):
            identifiers = self.service.identifiers
        else:
            identifiers = self.service.predictor.identifiers
        ordered = sorted(identifiers)
        return {"cases": ordered[:limit], "returned": min(limit, len(ordered)), "total": len(ordered)}

    def analyze_radiograph(
        self,
        image_id: str,
        question: str = "Measure HVA and IMA and retrieve supporting evidence.",
        top_k: int = 3,
    ) -> dict[str, Any]:
        image_id = image_id.strip()
        question = question.strip()
        if not image_id:
            raise ValueError("image_id must not be empty")
        if not question:
            raise ValueError("question must not be empty")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        return self.service.analyze(image_id, question, top_k)
