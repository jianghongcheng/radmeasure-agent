from __future__ import annotations

import argparse
import logging
import os
import socket
import time

from .backends import JobRepository, job_repository_from_env
from .factory import create_tools_from_env
from .logging_config import configure_json_logging
from .inference_client import inference_client_from_env
from .pipeline import JobPipeline


class Worker:
    def __init__(self, repository: JobRepository, pipeline: JobPipeline,
                 worker_id: str | None = None) -> None:
        self.repository, self.pipeline = repository, pipeline
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"

    def run_once(self) -> bool:
        job = self.repository.claim_next(self.worker_id)
        if job is None:
            return False
        logger = logging.getLogger("geomed.worker")
        extra = {"job_id": job.job_id, "trace_id": job.payload.get("_trace_id"), "worker_id": self.worker_id}
        logger.info("job_claimed", extra={**extra, "event_type": "claimed"})
        try:
            outcome = self.pipeline.run(job)
            self.repository.finish(job.job_id, outcome.status, outcome.result)
            logger.info("job_finished", extra={**extra, "event_type": outcome.status})
        except (KeyError, TypeError, ValueError) as exc:
            self.repository.record_failure(job.job_id, "invalid_job", str(exc), retryable=False)
            logger.warning("job_invalid", extra={**extra, "event_type": "failed"})
        except Exception as exc:  # operational failures are retried within the job budget
            self.repository.record_failure(job.job_id, "pipeline_failure", str(exc), retryable=True)
            logger.exception("job_pipeline_failure", extra={**extra, "event_type": "retry_scheduled"})
        return True

    def run_forever(self, poll_seconds: float = 0.5) -> None:
        while True:
            if not self.run_once():
                time.sleep(poll_seconds)


def main() -> None:
    configure_json_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    args = parser.parse_args()
    worker = Worker(
        job_repository_from_env(),
        JobPipeline(create_tools_from_env(), inference_client_from_env()),
    )
    if args.once:
        worker.run_once()
    else:
        worker.run_forever(args.poll_seconds)


if __name__ == "__main__":
    main()
