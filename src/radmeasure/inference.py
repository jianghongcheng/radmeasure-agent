from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .artifact_predictor import sha256_file
from .imaging import decode_medical_image


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    version: str
    backend: str
    artifact_sha256: str
    measurements: list[str]
    accepted_input: str
    live_image_inference: bool
    ready: bool
    evaluation: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class InferenceOutput:
    image_id: str
    measurements: dict[str, float]
    model: ModelInfo
    quality: dict | None = None
    image_metadata: dict | None = None

    def to_dict(self) -> dict:
        output = {"image_id": self.image_id, "measurements": self.measurements, "model": self.model.to_dict()}
        if self.quality is not None:
            output["quality"] = self.quality
        if self.image_metadata is not None:
            output["image_metadata"] = self.image_metadata
        return output


class InferenceAdapter(Protocol):
    @property
    def info(self) -> ModelInfo: ...
    def predict(self, image_id: str) -> InferenceOutput: ...


class ResNet50AngleAdapter:
    """Live HVA/IMA regression from radiograph bytes using a locked checkpoint."""

    def __init__(self, checkpoint: Path, device: str | None = None) -> None:
        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models
            import torchvision.transforms as transforms
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - exercised by ML image
            raise RuntimeError("Install the ML inference dependencies") from exc

        class ResNet50Head(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                backbone = models.resnet50(weights=None)
                backbone.fc = nn.Sequential(
                    nn.Linear(2048, 512), nn.GELU(), nn.Dropout(0.3),
                    nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.2),
                    nn.Linear(128, 2),
                )
                self.model = backbone

            def forward(self, inputs):
                return self.model(inputs)

        requested = device or "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self._torch, self._image = torch, Image
        self._device = torch.device(requested)
        self._transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._model = ResNet50Head().to(self._device)
        state = torch.load(checkpoint, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state, strict=True)
        self._model.eval()
        self._info = ModelInfo(
            model_id="hvangle-resnet50",
            version="hvangleest-seed42-v1",
            backend=f"pytorch_{self._device.type}",
            artifact_sha256=sha256_file(checkpoint),
            measurements=["HVA", "IMA"],
            accepted_input="jpeg_png_or_single_frame_dicom_bytes",
            live_image_inference=True,
            ready=True,
            evaluation={
                "dataset": "HVAngleEst unilateral test split",
                "n": 120,
                "HVA_MAE_degrees": 3.1205,
                "IMA_MAE_degrees": 1.5037,
                "research_only": True,
            },
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    def predict_bytes(self, image_id: str, content: bytes,
                      media_type: str = "image/jpeg") -> InferenceOutput:
        try:
            image, quality, metadata = decode_medical_image(content, media_type)
        except Exception as exc:
            raise ValueError(f"artifact cannot be decoded safely: {exc}") from exc
        inputs = self._transform(image).unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            values = self._model(inputs)[0].detach().cpu().tolist()
        return InferenceOutput(
            image_id=image_id,
            measurements={"HVA": float(values[0]), "IMA": float(values[1])},
            model=self.info,
            quality=quality.to_dict(),
            image_metadata=metadata,
        )


class LockedEvaluationAdapter:
    """Versioned adapter for persisted predictions; not live image inference."""

    def __init__(self, artifact: Path) -> None:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        self._predictions = {row["image_id"]: row["predicted"] for row in payload["predictions"]}
        self._info = ModelInfo(
            model_id="medimageinsight-spatial-readout",
            version="locked-eval-v1",
            backend="persisted_prediction_replay",
            artifact_sha256=sha256_file(artifact),
            measurements=["HVA", "IMA"],
            accepted_input="image_id",
            live_image_inference=False,
            ready=True,
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    def predict(self, image_id: str) -> InferenceOutput:
        if image_id not in self._predictions:
            raise KeyError(f"unknown locked inference image_id: {image_id}")
        return InferenceOutput(image_id, dict(self._predictions[image_id]), self.info)


class ModelRegistry:
    def __init__(self, adapters: list[InferenceAdapter]) -> None:
        self._adapters = {adapter.info.model_id: adapter for adapter in adapters}
        if not self._adapters:
            raise ValueError("at least one inference adapter is required")

    def list_models(self) -> list[dict]:
        return [adapter.info.to_dict() for adapter in self._adapters.values()]

    def predict(self, model_id: str, image_id: str) -> dict:
        if model_id not in self._adapters:
            raise KeyError(f"unknown model_id: {model_id}")
        return self._adapters[model_id].predict(image_id).to_dict()

    def predict_bytes(self, model_id: str, image_id: str, content: bytes,
                      media_type: str = "image/jpeg") -> dict:
        if model_id not in self._adapters:
            raise KeyError(f"unknown model_id: {model_id}")
        adapter = self._adapters[model_id]
        if not hasattr(adapter, "predict_bytes"):
            raise ValueError(f"model {model_id} does not accept image bytes")
        output = adapter.predict_bytes(image_id, content, media_type)
        return output if isinstance(output, dict) else output.to_dict()
