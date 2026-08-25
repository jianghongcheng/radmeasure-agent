from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator


TERMINAL_STATUSES = {"completed", "needs_review", "review_approved", "review_rejected", "failed"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Job:
    job_id: str
    job_type: str
    status: str
    payload: dict[str, Any]
    result: dict[str, Any] | None
    error: dict[str, Any] | None
    attempts: int
    max_attempts: int
    idempotency_key: str
    created_at: str
    updated_at: str
    worker_id: str | None = None
    lease_expires_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SqliteJobRepository:
    """Durable local job repository with atomic worker claims.

    SQLite is the reference deployment adapter. The API depends only on this
    boundary so a PostgreSQL adapter can replace it without changing workflows.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.path), timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(jobs)")}
            if "worker_id" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN worker_id TEXT")
            if "lease_expires_at" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN lease_expires_at TEXT")
            connection.execute("CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status, created_at)")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS job_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    details TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

    @staticmethod
    def _event(connection: sqlite3.Connection, job_id: str,
               event_type: str, details: dict[str, Any]) -> None:
        connection.execute(
            "INSERT INTO job_events(job_id,event_type,details,created_at) VALUES (?,?,?,?)",
            (job_id, event_type, json.dumps(details), _now()),
        )

    @staticmethod
    def _job(row: sqlite3.Row) -> Job:
        return Job(
            job_id=row["job_id"], job_type=row["job_type"], status=row["status"],
            payload=json.loads(row["payload"]),
            result=json.loads(row["result"]) if row["result"] else None,
            error=json.loads(row["error"]) if row["error"] else None,
            attempts=row["attempts"], max_attempts=row["max_attempts"],
            idempotency_key=row["idempotency_key"], created_at=row["created_at"],
            updated_at=row["updated_at"],
            worker_id=row["worker_id"], lease_expires_at=row["lease_expires_at"],
        )

    def submit(self, job_type: str, payload: dict[str, Any],
               idempotency_key: str, max_attempts: int = 3) -> tuple[Job, bool]:
        if not job_type or not idempotency_key:
            raise ValueError("job_type and idempotency_key are required")
        if not 1 <= max_attempts <= 10:
            raise ValueError("max_attempts must be between 1 and 10")
        timestamp, job_id = _now(), str(uuid.uuid4())
        with self._connect() as connection:
            try:
                connection.execute(
                    """INSERT INTO jobs
                    (job_id,job_type,status,payload,result,error,attempts,max_attempts,idempotency_key,created_at,updated_at,worker_id,lease_expires_at)
                    VALUES (?, ?, 'queued', ?, NULL, NULL, 0, ?, ?, ?, ?, NULL, NULL)""",
                    (job_id, job_type, json.dumps(payload), max_attempts,
                     idempotency_key, timestamp, timestamp),
                )
                self._event(connection, job_id, "submitted", {
                    "job_type": job_type,
                    "trace_id": payload.get("_trace_id"),
                    "submitted_by": payload.get("_submitted_by"),
                })
                created = True
            except sqlite3.IntegrityError:
                created = False
            row = connection.execute(
                "SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
        return self._job(row), created

    def get(self, job_id: str) -> Job | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._job(row) if row else None

    def find_by_trace_id(self, trace_id: str) -> list[Job]:
        """Return the original run and any replays linked to a trace."""
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM jobs
                WHERE json_extract(payload, '$._trace_id') = ?
                   OR json_extract(payload, '$._replay_of_trace_id') = ?
                ORDER BY created_at""",
                (trace_id, trace_id),
            ).fetchall()
        return [self._job(row) for row in rows]

    def status_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute("SELECT status, COUNT(*) AS count FROM jobs GROUP BY status").fetchall()
        return {row["status"]: row["count"] for row in rows}

    def events(self, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT event_type,details,created_at FROM job_events WHERE job_id=? ORDER BY event_id",
                (job_id,),
            ).fetchall()
        return [{"event_type": row["event_type"], "details": json.loads(row["details"]), "created_at": row["created_at"]} for row in rows]

    def claim_next(self, worker_id: str = "worker", lease_seconds: int = 60) -> Job | None:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = _now()
            expired = connection.execute(
                "SELECT job_id,worker_id FROM jobs WHERE status='running' AND lease_expires_at < ?", (now,)
            ).fetchall()
            for stale in expired:
                connection.execute(
                    "UPDATE jobs SET status='queued',worker_id=NULL,lease_expires_at=NULL,updated_at=? WHERE job_id=?",
                    (now, stale["job_id"]),
                )
                self._event(connection, stale["job_id"], "lease_expired", {"worker_id": stale["worker_id"]})
            row = connection.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            lease_expires = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
            connection.execute(
                "UPDATE jobs SET status='running', attempts=attempts+1, worker_id=?, lease_expires_at=?, updated_at=? WHERE job_id=? AND status='queued'",
                (worker_id, lease_expires, now, row["job_id"]),
            )
            self._event(connection, row["job_id"], "claimed", {"worker_id": worker_id, "lease_seconds": lease_seconds})
            connection.execute("COMMIT")
        return self.get(row["job_id"])

    def finish(self, job_id: str, status: str, result: dict[str, Any]) -> Job:
        if status not in {"completed", "needs_review"}:
            raise ValueError("finish status must be completed or needs_review")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET status=?, result=?, error=NULL, worker_id=NULL, lease_expires_at=NULL, updated_at=? WHERE job_id=? AND status='running'",
                (status, json.dumps(result), _now(), job_id),
            )
            if cursor.rowcount == 1:
                self._event(connection, job_id, status, {})
        if cursor.rowcount != 1:
            raise RuntimeError("job is not running")
        return self.get(job_id)

    def record_failure(self, job_id: str, code: str, message: str,
                       retryable: bool = True) -> Job:
        job = self.get(job_id)
        if job is None or job.status != "running":
            raise RuntimeError("job is not running")
        status = "queued" if retryable and job.attempts < job.max_attempts else "failed"
        error = {"code": code, "message": message, "retryable": retryable}
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET status=?, error=?, worker_id=NULL, lease_expires_at=NULL, updated_at=? WHERE job_id=? AND status='running'",
                (status, json.dumps(error), _now(), job_id),
            )
            self._event(connection, job_id, "retry_scheduled" if status == "queued" else "failed", error)
        return self.get(job_id)

    def review(self, job_id: str, reviewer: str, decision: str,
               corrected_measurements: dict[str, float] | None = None,
               notes: str = "") -> Job:
        if decision not in {"approve", "reject"}:
            raise ValueError("decision must be approve or reject")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id=? AND status='needs_review'", (job_id,)
            ).fetchone()
            if row is None:
                connection.execute("ROLLBACK")
                raise RuntimeError("job is not awaiting review")
            result = json.loads(row["result"])
            review = {"reviewer": reviewer, "decision": decision, "notes": notes,
                      "reviewed_at": _now()}
            if corrected_measurements is not None:
                review["corrected_measurements"] = corrected_measurements
                result["measurements"] = corrected_measurements
            result["review"] = review
            status = "review_approved" if decision == "approve" else "review_rejected"
            connection.execute(
                "UPDATE jobs SET status=?,result=?,updated_at=? WHERE job_id=?",
                (status, json.dumps(result), _now(), job_id),
            )
            self._event(connection, job_id, status, review)
            connection.execute("COMMIT")
        return self.get(job_id)
