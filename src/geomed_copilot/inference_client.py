from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable


class InferenceUnavailable(RuntimeError):
    pass


class InferenceClient:
    """Synchronous worker client with bounded retries and a local circuit breaker."""

    def __init__(self, base_url: str, token: str, timeout_seconds: float = 3.0,
                 retries: int = 2, failure_threshold: int = 3,
                 recovery_seconds: float = 15.0,
                 transport: Callable[[urllib.request.Request, float], dict] | None = None) -> None:
        self.base_url, self.token = base_url.rstrip("/"), token
        self.timeout_seconds, self.retries = timeout_seconds, retries
        self.failure_threshold, self.recovery_seconds = failure_threshold, recovery_seconds
        self._failures = 0
        self._opened_at: float | None = None
        self._transport = transport or self._urlopen

    @staticmethod
    def _urlopen(request: urllib.request.Request, timeout: float) -> dict:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    @property
    def circuit_state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at >= self.recovery_seconds:
            return "half_open"
        return "open"

    def predict(self, image_id: str,
                model_id: str = "medimageinsight-spatial-readout") -> dict:
        if self.circuit_state == "open":
            raise InferenceUnavailable("inference circuit is open")
        payload = json.dumps({"model_id": model_id, "image_id": image_id}).encode()
        request = urllib.request.Request(
            self.base_url + "/v1/infer", data=payload, method="POST",
            headers={"content-type": "application/json", "x-inference-token": self.token},
        )
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                output = self._transport(request, self.timeout_seconds)
                self._failures, self._opened_at = 0, None
                return output
            except (OSError, TimeoutError, urllib.error.URLError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.05 * (2 ** attempt))
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()
        raise InferenceUnavailable(f"inference request failed: {last_error}")

    def predict_artifact(self, image_id: str, artifact_uri: str,
                         media_type: str = "image/jpeg",
                         model_id: str | None = None) -> dict:
        if self.circuit_state == "open":
            raise InferenceUnavailable("inference circuit is open")
        selected_model = model_id or os.environ.get("GEOMED_ARTIFACT_MODEL_ID", "hvangle-resnet50")
        payload = json.dumps({"model_id": selected_model, "image_id": image_id,
                              "artifact_uri": artifact_uri, "media_type": media_type}).encode()
        request = urllib.request.Request(
            self.base_url + "/v1/infer-artifact", data=payload, method="POST",
            headers={"content-type": "application/json", "x-inference-token": self.token},
        )
        try:
            output = self._transport(request, self.timeout_seconds)
            self._failures, self._opened_at = 0, None
            return output
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = time.monotonic()
            raise InferenceUnavailable(f"inference request failed: {exc}") from exc


def inference_client_from_env() -> InferenceClient | None:
    url = os.environ.get("GEOMED_INFERENCE_URL")
    if not url:
        return None
    token = os.environ.get("GEOMED_INFERENCE_TOKEN")
    if not token:
        raise RuntimeError("GEOMED_INFERENCE_TOKEN is required when inference URL is configured")
    return InferenceClient(url, token)
