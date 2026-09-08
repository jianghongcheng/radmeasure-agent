from __future__ import annotations

import math
from pathlib import Path

from .artifact_predictor import sha256_file
from .imaging import decode_medical_image
from .inference import InferenceOutput, ModelInfo


def _angle(points, dimensions, first: tuple[int, int], second: tuple[int, int]):
    import torch
    scaled = points * dimensions[:, None, :]
    a = scaled[:, first[1]] - scaled[:, first[0]]
    b = scaled[:, second[1]] - scaled[:, second[0]]
    cosine = (a * b).sum(-1).abs() / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-6)
    return torch.rad2deg(torch.acos(cosine.clamp(0, 1 - 1e-7)))


def execute_geometry(points, dimensions):
    import torch
    return torch.stack([
        _angle(points, dimensions, (0, 1), (2, 3)),
        _angle(points, dimensions, (2, 3), (4, 5)),
    ], dim=-1)


class IndependentGeometryRepairAdapter:
    """HRNet landmark proposal followed by a separately trained repair/verifier."""

    def __init__(self, detector_checkpoint: Path, repair_checkpoint: Path,
                 device: str | None = None) -> None:
        import torch
        from torch import nn

        class Block(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels), nn.ReLU(True),
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                )
                self.relu = nn.ReLU(True)

            def forward(self, value):
                return self.relu(self.net(value) + value)

        class HRNetLite(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(64), nn.ReLU(True),
                    nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(64), nn.ReLU(True),
                )
                self.hr = nn.Sequential(*[Block(64) for _ in range(4)])
                self.lr1 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(128), nn.ReLU(True),
                    *[Block(128) for _ in range(4)],
                )
                self.fuse_up = nn.Sequential(
                    nn.ConvTranspose2d(128, 64, 2, stride=2),
                    nn.BatchNorm2d(64), nn.ReLU(True),
                )
                self.post = nn.Sequential(*[Block(64) for _ in range(2)])
                self.head = nn.Conv2d(64, 6, 1)

            def forward(self, value):
                stem = self.stem(value)
                return self.head(self.post(self.hr(stem) + self.fuse_up(self.lr1(stem))))

        class RepairMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(58, 256), nn.GELU(), nn.LayerNorm(256),
                    nn.Linear(256, 256), nn.GELU(), nn.Dropout(.1),
                    nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 12),
                )

            def forward(self, points, dimensions):
                pairs = []
                for first in range(6):
                    for second in range(first + 1, 6):
                        delta = points[:, second] - points[:, first]
                        distance = delta.norm(dim=-1, keepdim=True)
                        pairs.append(torch.cat([distance, delta / distance.clamp_min(1e-6)], -1))
                aspect = torch.log(dimensions[:, :1] / dimensions[:, 1:].clamp_min(1))
                return self.network(torch.cat([points.flatten(1), *pairs, aspect], -1)).reshape(-1, 6, 2)

        class VerifierMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(29, 128), nn.GELU(), nn.LayerNorm(128),
                    nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1),
                )

            def forward(self, points, delta, dimensions):
                proposed = (points + delta).clamp(0, 1)
                aspect = torch.log(dimensions[:, :1] / dimensions[:, 1:].clamp_min(1))
                angles = torch.cat([execute_geometry(points, dimensions),
                                    execute_geometry(proposed, dimensions)], -1)
                return self.network(torch.cat([points.flatten(1), delta.flatten(1), aspect, angles], -1)).squeeze(-1)

        requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self._device = torch.device(requested)
        self._detector = HRNetLite().to(self._device)
        self._repair = RepairMLP().to(self._device)
        self._verifier = VerifierMLP().to(self._device)
        self._detector.load_state_dict(torch.load(detector_checkpoint, map_location=self._device, weights_only=True))
        checkpoint = torch.load(repair_checkpoint, map_location=self._device, weights_only=True)
        self._repair.load_state_dict(checkpoint["repair_state_dict"])
        self._verifier.load_state_dict(checkpoint["verifier_state_dict"])
        self._threshold = float(checkpoint["threshold"])
        self._detector.eval(); self._repair.eval(); self._verifier.eval()
        self._info = ModelInfo(
            model_id="hvangle-hrnet-repair", version="hrnet-repair-seed42-v1",
            backend=f"pytorch_{self._device.type}",
            artifact_sha256=f"detector:{sha256_file(detector_checkpoint)};repair:{sha256_file(repair_checkpoint)}",
            measurements=["HVA", "IMA"], accepted_input="jpeg_png_or_single_frame_dicom_bytes",
            live_image_inference=True, ready=True,
            evaluation={
                "dataset": "HVAngleEst patient-disjoint test split", "n": 243,
                "role": "independent_repair_proposal_not_primary_measurement_model",
                "verifier_threshold": self._threshold, "research_only": True,
            },
        )

    @property
    def info(self):
        return self._info

    @staticmethod
    def _decode_heatmaps(heatmaps):
        import torch
        batch, landmarks, _, width = heatmaps.shape
        indices = heatmaps.flatten(2).argmax(-1)
        x = (indices.remainder(width).float() + .5) / width
        y = (indices.div(width, rounding_mode="floor").float() + .5) / width
        return torch.stack([x, y], -1).reshape(batch, landmarks, 2)

    def predict_bytes(self, image_id: str, content: bytes,
                      media_type: str = "image/jpeg") -> InferenceOutput:
        import numpy as np
        torch = self._torch
        image, quality, metadata = decode_medical_image(content, media_type)
        width, height = image.size
        resized = image.convert("RGB").resize((256, 256))
        inputs = torch.tensor(np.asarray(resized).copy(), dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
        inputs = inputs.to(self._device)
        dimensions = torch.tensor([[width, height]], dtype=torch.float32, device=self._device)
        with torch.inference_mode():
            points = self._decode_heatmaps(self._detector(inputs))
            initial = execute_geometry(points, dimensions)
            delta = self._repair(points, dimensions)
            proposed_points = (points + delta).clamp(0, 1)
            proposed = execute_geometry(proposed_points, dimensions)
            confidence = torch.sigmoid(self._verifier(points, delta, dimensions)).item()
        accepted = confidence >= self._threshold
        before = initial[0].cpu().tolist()
        after = proposed[0].cpu().tolist()
        output = InferenceOutput(
            image_id=image_id,
            measurements={"HVA": float(before[0]), "IMA": float(before[1])},
            model=self.info, quality=quality.to_dict(), image_metadata=metadata,
        ).to_dict()
        output["repair_proposal"] = {
            "accepted": accepted, "confidence": confidence, "threshold": self._threshold,
            "measurements": {"HVA": float(after[0]), "IMA": float(after[1])},
            "geometry_source": "independent_landmark_model",
            "action": "learned_residual_landmark_repair",
            "maximum_steps": 1,
        }
        return output
