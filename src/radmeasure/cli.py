from __future__ import annotations

import argparse
import json

from .planner import ConstrainedMeasurementPlanner
from .protocols import ProtocolRegistry
from .pipeline import JobPipeline
from .production import DemoService
from .tools import RadMeasureTools
from .orchestrator import RadMeasureOrchestrator
from .retrieval import CaseRetriever, HybridRetriever
from .sample_data import CASES, DEMO_LANDMARKS, EVIDENCE


def build_demo() -> RadMeasureOrchestrator:
    return RadMeasureOrchestrator(HybridRetriever(EVIDENCE), CaseRetriever(CASES))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline RadMeasure medical measurement demo")
    parser.add_argument("--question", default="How should HVA be measured and verified?")
    args = parser.parse_args()
    from tempfile import TemporaryDirectory
    from pathlib import Path
    from .jobs import SqliteJobRepository
    from .worker import Worker

    with TemporaryDirectory(prefix="radmeasure-demo-") as directory:
        jobs = SqliteJobRepository(Path(directory) / "jobs.sqlite")
        job, _ = jobs.submit("evaluation_analysis", {
            "image_id": "demo-foot-001", "question": args.question,
        }, "medical-demo")
        Worker(jobs, JobPipeline(
            RadMeasureTools(DemoService()),
            planner=ConstrainedMeasurementPlanner(ProtocolRegistry()),
        )).run_once()
        print(json.dumps(jobs.get(job.job_id).to_dict(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
