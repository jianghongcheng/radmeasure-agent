from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from .jobs import Job, SqliteJobRepository
from .storage import LocalArtifactStore, StoredArtifact


class JobRepository(Protocol):
    def submit(self, job_type: str, payload: dict, idempotency_key: str,
               max_attempts: int = 3) -> tuple[Job, bool]: ...
    def get(self, job_id: str) -> Job | None: ...
    def find_by_trace_id(self, trace_id: str) -> list[Job]: ...
    def claim_next(self, worker_id: str = "worker", lease_seconds: int = 60) -> Job | None: ...
    def finish(self, job_id: str, status: str, result: dict) -> Job: ...
    def record_failure(self, job_id: str, code: str, message: str,
                       retryable: bool = True) -> Job: ...
    def status_counts(self) -> dict[str, int]: ...
    def events(self, job_id: str) -> list[dict]: ...
    def review(self, job_id: str, reviewer: str, decision: str,
               corrected_measurements: dict[str, float] | None = None,
               notes: str = "") -> Job: ...


class ArtifactStore(Protocol):
    max_bytes: int
    def put(self, content: bytes, media_type: str) -> tuple[StoredArtifact, bool]: ...


def job_repository_from_env() -> JobRepository:
    database_url = os.environ.get("GEOMED_DATABASE_URL")
    if database_url:
        from .postgres_jobs import PostgresJobRepository
        return PostgresJobRepository(database_url)
    return SqliteJobRepository(Path(os.environ.get("GEOMED_JOB_DB", "runtime/jobs.db")))


def artifact_store_from_env() -> ArtifactStore:
    endpoint = os.environ.get("GEOMED_S3_ENDPOINT")
    if endpoint:
        from .s3_storage import S3ArtifactStore
        return S3ArtifactStore(
            endpoint_url=endpoint,
            bucket=os.environ.get("GEOMED_S3_BUCKET", "geomed-artifacts"),
            access_key=os.environ.get("GEOMED_S3_ACCESS_KEY", "minioadmin"),
            secret_key=os.environ.get("GEOMED_S3_SECRET_KEY", "minioadmin"),
        )
    return LocalArtifactStore(Path(os.environ.get("GEOMED_ARTIFACT_ROOT", "runtime/artifacts")))
