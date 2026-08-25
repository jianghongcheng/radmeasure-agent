from __future__ import annotations

import os
from pathlib import Path

from .config import LockedArtifactConfig
from .production import DemoService, EvaluationReplayService, LockedArtifactService
from .tools import GeoMedTools


def create_tools_from_env() -> GeoMedTools:
    if os.environ.get("GEOMED_EVAL_REPLAY", "").lower() in {"1", "true", "yes"}:
        root = Path(os.environ.get("GEOMED_DATA_ROOT", "data")).expanduser().resolve()
        processed = root / "processed" / "hvangleest"
        return GeoMedTools(EvaluationReplayService(
            evaluation=processed / "medimageinsight_locked_test_eval.json",
            manifests_dir=processed,
            train_manifest=processed / "train.jsonl",
            evidence_catalog=root / "evidence" / "catalog.json",
        ))
    if os.environ.get("GEOMED_DEMO_MODE", "").lower() in {"1", "true", "yes"}:
        return GeoMedTools(DemoService())
    config = LockedArtifactConfig.from_env()
    return GeoMedTools(LockedArtifactService(
        predictions=config.predictions,
        annotations=config.annotations,
        split_manifest=config.split_manifest,
        evidence_catalog=config.evidence_catalog,
    ))
