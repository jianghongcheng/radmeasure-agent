from __future__ import annotations

import os
from pathlib import Path

from .config import LockedArtifactConfig
from .production import DemoService, EvaluationReplayService, LockedArtifactService
from .tools import RadMeasureTools


def create_tools_from_env() -> RadMeasureTools:
    if os.environ.get("RADMEASURE_EVAL_REPLAY", "").lower() in {"1", "true", "yes"}:
        root = Path(os.environ.get("RADMEASURE_DATA_ROOT", "data")).expanduser().resolve()
        processed = root / "processed" / "hvangleest"
        return RadMeasureTools(EvaluationReplayService(
            evaluation=processed / "medimageinsight_locked_test_eval.json",
            manifests_dir=processed,
            train_manifest=processed / "train.jsonl",
            evidence_catalog=root / "evidence" / "catalog.json",
        ))
    if os.environ.get("RADMEASURE_DEMO_MODE", "").lower() in {"1", "true", "yes"}:
        return RadMeasureTools(DemoService())
    config = LockedArtifactConfig.from_env()
    return RadMeasureTools(LockedArtifactService(
        predictions=config.predictions,
        annotations=config.annotations,
        split_manifest=config.split_manifest,
        evidence_catalog=config.evidence_catalog,
    ))
