import logging
import os
import base64
import urllib.request
import time
import uuid

from .backends import artifact_store_from_env, job_repository_from_env
from .dashboard import render_dashboard
from .factory import create_tools_from_env
from .logging_config import configure_json_logging
from .metrics import HttpMetrics
from .planner import planner_from_env
from .protocols import ProtocolRegistry
from .replay import build_replay_payload, replay_guarantee
from .security import ApiKeyAuthorizer, Principal


def create_app():
    try:
        from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
        from fastapi.responses import HTMLResponse, PlainTextResponse
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the API extra: pip install -e '.[api]'") from exc

    configure_json_logging()
    logger = logging.getLogger("geomed.api")
    tools = create_tools_from_env()
    jobs = job_repository_from_env()
    artifacts = artifact_store_from_env()
    authorizer = ApiKeyAuthorizer.from_env()
    metrics = HttpMetrics()
    protocol_registry = ProtocolRegistry()
    measurement_planner = planner_from_env(protocol_registry)

    def authorize(api_key: str | None, role: str) -> Principal:
        try:
            return authorizer.authenticate(api_key, role)
        except PermissionError as exc:
            status = 401 if "invalid or missing" in str(exc) else 403
            raise HTTPException(status_code=status, detail=str(exc)) from exc

    class AnalysisPayload(BaseModel):
        request_id: str = Field(min_length=1, max_length=128)
        image_id: str = Field(min_length=1, max_length=512)
        question: str = Field(min_length=1, max_length=4000)
        top_k: int = Field(default=3, ge=1, le=20)

    class JobPayload(BaseModel):
        image_id: str = Field(min_length=1, max_length=512)
        question: str = Field(default="Measure HVA and IMA with supporting evidence.", min_length=1, max_length=4000)
        top_k: int = Field(default=3, ge=1, le=20)

    class PlanPayload(BaseModel):
        request: str = Field(min_length=1, max_length=4000)

    class ReviewPayload(BaseModel):
        decision: str = Field(pattern="^(approve|reject)$")
        corrected_measurements: dict[str, float] | None = None
        notes: str = Field(default="", max_length=2000)

    def fetch_orthanc_instance(instance_id: str) -> bytes:
        if not instance_id or any(ch not in "0123456789abcdefABCDEF-" for ch in instance_id):
            raise ValueError("invalid Orthanc instance ID")
        base = os.environ.get("GEOMED_ORTHANC_URL")
        if not base:
            raise RuntimeError("Orthanc integration is not configured")
        request = urllib.request.Request(base.rstrip("/") + f"/instances/{instance_id}/file")
        user = os.environ.get("GEOMED_ORTHANC_USERNAME", "")
        password = os.environ.get("GEOMED_ORTHANC_PASSWORD", "")
        if user:
            token = base64.b64encode(f"{user}:{password}".encode()).decode()
            request.add_header("Authorization", f"Basic {token}")
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.read(artifacts.max_bytes + 1)

    def sanitize_upload(content: bytes, media_type: str) -> tuple[bytes, list[str]]:
        if media_type != "application/dicom":
            return content, []
        from .imaging import deidentify_dicom
        return deidentify_dicom(content)

    app = FastAPI(
        title="RadMeasure API",
        version="0.4.0",
        description="Verifiable multimodal execution for protocol-driven radiographic measurements",
    )
    cache: dict[str, dict] = {}

    @app.middleware("http")
    async def request_observability(request: Request, call_next):
        started = time.perf_counter()
        supplied = request.headers.get("x-request-id", "")
        trace_id = supplied if supplied and len(supplied) <= 128 else str(uuid.uuid4())
        request.state.trace_id = trace_id
        try:
            response = await call_next(request)
        except Exception:
            metrics.observe(request.method, request.url.path, 500, time.perf_counter() - started)
            logger.exception("request_failed", extra={"trace_id": trace_id})
            raise
        duration = time.perf_counter() - started
        metrics.observe(request.method, request.url.path, response.status_code, duration)
        response.headers["x-request-id"] = trace_id
        logger.info("request_completed", extra={"trace_id": trace_id, "event_type": f"http_{response.status_code}"})
        return response

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        return render_dashboard()

    @app.get("/health")
    async def health() -> dict:
        capabilities = tools.capabilities()
        return {"status": "ok", "mode": capabilities["mode"],
                "available_cases": capabilities["available_cases"]}

    @app.get("/v1/capabilities")
    async def capabilities() -> dict:
        return tools.capabilities()

    @app.get("/v1/protocols")
    async def protocols() -> dict:
        return {"protocols": protocol_registry.describe()}

    @app.post("/v1/plan")
    async def plan_measurement(payload: PlanPayload,
                               x_api_key: str | None = Header(default=None)) -> dict:
        authorize(x_api_key, "viewer")
        return measurement_planner.plan(payload.request).to_dict()

    @app.get("/v1/cases")
    async def cases(limit: int = 20) -> dict:
        try:
            return tools.list_available_cases(limit)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/jobs", status_code=202)
    async def submit_job(request: Request, payload: JobPayload,
                         idempotency_key: str = Header(min_length=1, max_length=256),
                         x_api_key: str | None = Header(default=None)) -> dict:
        principal = authorize(x_api_key, "operator")
        job_payload = payload.model_dump()
        job_payload["_trace_id"] = request.state.trace_id
        job_payload["_submitted_by"] = principal.name
        job, created = jobs.submit(
            "evaluation_analysis", job_payload, idempotency_key=idempotency_key
        )
        return {"job": job.to_dict(), "created": created}

    @app.post("/v1/uploads", status_code=202)
    async def upload_radiograph(
        request: Request,
        file: UploadFile = File(),
        idempotency_key: str = Header(min_length=1, max_length=256),
        x_api_key: str | None = Header(default=None),
    ) -> dict:
        principal = authorize(x_api_key, "operator")
        content = await file.read(artifacts.max_bytes + 1)
        try:
            content, removed = sanitize_upload(content, file.content_type or "")
            artifact, stored = artifacts.put(content, file.content_type or "")
            job, created = jobs.submit(
                "uploaded_radiograph", {
                    "artifact": artifact.__dict__, "_trace_id": request.state.trace_id,
                    "_submitted_by": principal.name, "deidentification": {
                        "applied": file.content_type == "application/dicom",
                        "removed_fields": removed,
                    },
                }, idempotency_key
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"job": job.to_dict(), "created": created, "artifact_stored": stored}

    @app.post("/v1/orthanc/instances/{instance_id}/jobs", status_code=202)
    async def submit_orthanc_instance(
        instance_id: str, request: Request,
        idempotency_key: str = Header(min_length=1, max_length=256),
        x_api_key: str | None = Header(default=None),
    ) -> dict:
        principal = authorize(x_api_key, "operator")
        try:
            content = fetch_orthanc_instance(instance_id)
            content, removed = sanitize_upload(content, "application/dicom")
            artifact, stored = artifacts.put(content, "application/dicom")
            job, created = jobs.submit("uploaded_radiograph", {
                "artifact": artifact.__dict__, "orthanc_instance_id": instance_id,
                "deidentification": {"applied": True, "removed_fields": removed},
                "_trace_id": request.state.trace_id, "_submitted_by": principal.name,
            }, idempotency_key)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"job": job.to_dict(), "created": created, "artifact_stored": stored}

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str, x_api_key: str | None = Header(default=None)) -> dict:
        authorize(x_api_key, "viewer")
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job.to_dict()

    @app.get("/v1/jobs/{job_id}/events")
    async def get_job_events(job_id: str, x_api_key: str | None = Header(default=None)) -> dict:
        authorize(x_api_key, "viewer")
        if jobs.get(job_id) is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {"job_id": job_id, "events": jobs.events(job_id)}

    @app.get("/v1/traces/{trace_id}")
    async def get_trace(trace_id: str,
                        x_api_key: str | None = Header(default=None)) -> dict:
        """Return the complete job/event trajectory associated with a trace."""
        authorize(x_api_key, "viewer")
        if not trace_id or len(trace_id) > 128:
            raise HTTPException(status_code=422, detail="invalid trace_id")
        traced_jobs = jobs.find_by_trace_id(trace_id)
        if not traced_jobs:
            raise HTTPException(status_code=404, detail="trace not found")
        return {
            "trace_id": trace_id,
            "runs": [
                {"job": job.to_dict(), "events": jobs.events(job.job_id)}
                for job in traced_jobs
            ],
        }

    @app.post("/v1/jobs/{job_id}/replay", status_code=202)
    async def replay_job(
        job_id: str,
        request: Request,
        idempotency_key: str = Header(min_length=1, max_length=256),
        x_api_key: str | None = Header(default=None),
    ) -> dict:
        """Create a lineage-linked replay using the original persisted inputs."""
        principal = authorize(x_api_key, "operator")
        original = jobs.get(job_id)
        if original is None:
            raise HTTPException(status_code=404, detail="job not found")
        try:
            payload = build_replay_payload(original, request.state.trace_id, principal.name)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        replay, created = jobs.submit(
            original.job_type,
            payload,
            idempotency_key=idempotency_key,
            max_attempts=original.max_attempts,
        )
        return {
            "job": replay.to_dict(),
            "created": created,
            "replay_of_job_id": original.job_id,
            "replay_of_trace_id": original.payload.get("_trace_id"),
            "guarantee": replay_guarantee(original),
        }

    @app.post("/v1/jobs/{job_id}/review")
    async def review_job(job_id: str, payload: ReviewPayload,
                         x_api_key: str | None = Header(default=None)) -> dict:
        principal = authorize(x_api_key, "admin")
        if payload.corrected_measurements is not None:
            if set(payload.corrected_measurements) != {"HVA", "IMA"}:
                raise HTTPException(status_code=422, detail="corrected measurements must contain HVA and IMA")
            if any(not -20 <= value <= 100 for value in payload.corrected_measurements.values()):
                raise HTTPException(status_code=422, detail="corrected measurement outside safety bounds")
        try:
            reviewed = jobs.review(job_id, principal.name, payload.decision,
                                   payload.corrected_measurements, payload.notes)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return reviewed.to_dict()

    @app.get("/v1/jobs/{job_id}/report")
    async def structured_report(job_id: str,
                                x_api_key: str | None = Header(default=None)) -> dict:
        authorize(x_api_key, "viewer")
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status != "review_approved":
            raise HTTPException(status_code=409, detail="report requires approved human review")
        result = job.result or {}
        return {
            "schema": "geomed.measurement-report.v1",
            "job_id": job.job_id,
            "status": "final",
            "measurements": result.get("measurements", {}),
            "model": result.get("model", {}),
            "quality": result.get("quality", {}),
            "review": result.get("review", {}),
            "provenance": result.get("provenance", {}),
            "disclaimer": "Research use only; not for diagnosis or patient care.",
        }

    @app.get("/v1/operations")
    async def operations(x_api_key: str | None = Header(default=None)) -> dict:
        authorize(x_api_key, "viewer")
        return {"job_status_counts": jobs.status_counts()}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics(x_api_key: str | None = Header(default=None)) -> str:
        authorize(x_api_key, "viewer")
        return metrics.render(jobs.status_counts())

    @app.post("/v1/analyze")
    async def analyze(payload: AnalysisPayload, x_api_key: str | None = Header(default=None)) -> dict:
        authorize(x_api_key, "operator")
        if payload.request_id in cache:
            return cache[payload.request_id]
        try:
            output = tools.analyze_radiograph(
                payload.image_id, payload.question, payload.top_k
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        cache[payload.request_id] = output
        return output

    @app.get("/v1/result/{request_id}")
    async def result(request_id: str, x_api_key: str | None = Header(default=None)) -> dict:
        authorize(x_api_key, "viewer")
        if request_id not in cache:
            raise HTTPException(status_code=404, detail="request_id not found")
        return cache[request_id]

    return app
