from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LockedArtifactConfig:
    predictions: Path
    annotations: Path
    split_manifest: Path
    evidence_catalog: Path

    @classmethod
    def from_env(cls) -> "LockedArtifactConfig":
        names = {
            "predictions": "GEOMED_PREDICTIONS",
            "annotations": "GEOMED_ANNOTATIONS",
            "split_manifest": "GEOMED_SPLIT_MANIFEST",
            "evidence_catalog": "GEOMED_EVIDENCE_CATALOG",
        }
        missing = [env for env in names.values() if not os.environ.get(env)]
        if missing:
            raise RuntimeError(
                "Missing required environment variables: " + ", ".join(missing)
            )
        return cls(**{
            field: Path(os.environ[env]).expanduser().resolve()
            for field, env in names.items()
        })

