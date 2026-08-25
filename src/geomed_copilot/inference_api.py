import hmac
import os
from pathlib import Path

from .inference import LockedEvaluationAdapter, ModelRegistry, ResNet50AngleAdapter
from .repair_inference import IndependentGeometryRepairAdapter
from .logging_config import configure_json_logging


def create_registry() -> ModelRegistry:
    root = Path(os.environ.get("GEOMED_DATA_ROOT", "data"))
    artifact = root / "processed" / "hvangleest" / "medimageinsight_locked_test_eval.json"
    adapters = [LockedEvaluationAdapter(artifact)]
    checkpoint = os.environ.get("GEOMED_RESNET_CHECKPOINT")
    if checkpoint:
        adapters.append(ResNet50AngleAdapter(Path(checkpoint), os.environ.get("GEOMED_MODEL_DEVICE")))
    landmark_checkpoint = os.environ.get("GEOMED_LANDMARK_CHECKPOINT")
    repair_checkpoint = os.environ.get("GEOMED_REPAIR_CHECKPOINT")
    if landmark_checkpoint and repair_checkpoint:
        adapters.append(IndependentGeometryRepairAdapter(
            Path(landmark_checkpoint), Path(repair_checkpoint), os.environ.get("GEOMED_MODEL_DEVICE")
        ))
    return ModelRegistry(adapters)


def load_artifact(uri: str) -> bytes:
    if uri.startswith("s3://"):
        import boto3
        from botocore.config import Config
        bucket, key = uri[5:].split("/", 1)
        client = boto3.client(
            "s3", endpoint_url=os.environ["GEOMED_S3_ENDPOINT"],
            aws_access_key_id=os.environ["GEOMED_S3_ACCESS_KEY"],
            aws_secret_access_key=os.environ["GEOMED_S3_SECRET_KEY"],
            config=Config(signature_version="s3v4"), region_name="us-east-1",
        )
        return client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return Path(uri).read_bytes()


def create_app():
    try:
        from fastapi import FastAPI, Header, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the API extra: pip install -e '.[api]'") from exc
    configure_json_logging()
    registry = create_registry()
    expected_token = os.environ.get("GEOMED_INFERENCE_TOKEN")
    if not expected_token:
        raise RuntimeError("GEOMED_INFERENCE_TOKEN is required")

    class InferenceRequest(BaseModel):
        model_id: str = Field(min_length=1, max_length=128)
        image_id: str = Field(min_length=1, max_length=512)

    class ArtifactInferenceRequest(BaseModel):
        model_id: str = Field(default="hvangle-resnet50", min_length=1, max_length=128)
        image_id: str = Field(min_length=1, max_length=512)
        artifact_uri: str = Field(min_length=1, max_length=2048)
        media_type: str = Field(default="image/jpeg", min_length=1, max_length=128)

    def authorize(token: str | None) -> None:
        if not token or not hmac.compare_digest(token, expected_token):
            raise HTTPException(status_code=401, detail="invalid inference service token")

    app = FastAPI(title="GeoMed Inference Service", version="1.0.0")

    @app.get("/health")
    async def health() -> dict:
        models = registry.list_models()
        return {"status": "ok", "ready_models": sum(item["ready"] for item in models), "gpu_live": any(item["live_image_inference"] for item in models)}

    @app.get("/v1/models")
    async def models(x_inference_token: str | None = Header(default=None)) -> dict:
        authorize(x_inference_token)
        return {"models": registry.list_models()}

    @app.post("/v1/infer")
    async def infer(payload: InferenceRequest, x_inference_token: str | None = Header(default=None)) -> dict:
        authorize(x_inference_token)
        try:
            return registry.predict(payload.model_id, payload.image_id)
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/infer-artifact")
    async def infer_artifact(payload: ArtifactInferenceRequest, x_inference_token: str | None = Header(default=None)) -> dict:
        authorize(x_inference_token)
        try:
            content = load_artifact(payload.artifact_uri)
            return registry.predict_bytes(payload.model_id, payload.image_id, content, payload.media_type)
        except (KeyError, ValueError, OSError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
