from __future__ import annotations

from typing import Any

from .jobs import Job, TERMINAL_STATUSES


def build_replay_payload(job: Job, trace_id: str, submitted_by: str) -> dict[str, Any]:
    """Copy replay-safe inputs while recording an immutable lineage link."""
    if job.status not in TERMINAL_STATUSES:
        raise RuntimeError("only terminal jobs can be replayed")
    payload = dict(job.payload)
    original_trace = payload.get("_replay_of_trace_id") or payload.get("_trace_id")
    payload.update({
        "_trace_id": trace_id,
        "_submitted_by": submitted_by,
        "_replay_of_job_id": job.job_id,
        "_replay_root_job_id": payload.get("_replay_root_job_id") or job.job_id,
        "_replay_of_trace_id": original_trace,
    })
    return payload


def replay_guarantee(job: Job) -> str:
    """Describe honestly which parts of a replay are held fixed."""
    if job.job_type == "evaluation_analysis":
        return "locked_artifact_same_inputs"
    return "content_addressed_input_current_model"
