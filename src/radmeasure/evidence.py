from __future__ import annotations

import json
from pathlib import Path

from .models import Evidence


def load_evidence_catalog(path: Path) -> list[Evidence]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    required = {"evidence_id", "title", "text", "source_url"}
    seen = set()
    evidence = []
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Evidence row {index} missing {sorted(missing)}")
        if row["evidence_id"] in seen:
            raise ValueError(f"Duplicate evidence ID: {row['evidence_id']}")
        seen.add(row["evidence_id"])
        evidence.append(Evidence(
            evidence_id=row["evidence_id"],
            title=row["title"],
            text=row["text"],
            source_url=row["source_url"],
            evidence_type=row.get("evidence_type", "literature"),
            metadata=row.get("metadata", {}),
        ))
    return evidence

