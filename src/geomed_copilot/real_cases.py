from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import Evidence


def cases_from_manifest(path: Path) -> list[Evidence]:
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            cases.append(Evidence(
                evidence_id=row["sample_id"],
                title=f"HVAngleEst reference case {row['sample_id'][:8]}",
                text="De-identified public research case indexed by radiographic measurements.",
                source_url="https://doi.org/10.1038/s41597-025-05261-9",
                evidence_type="case",
                metadata={
                    "measurements": {"HVA": row["HVA"], "IMA": row["IMA"]},
                    "side": row["side"],
                    "source": row["source"],
                },
            ))
    return cases


def cases_from_locked_split(annotations: Path, split_manifest: Path, split: str = "train") -> list[Evidence]:
    identifiers = json.loads(split_manifest.read_text(encoding="utf-8"))["split_ids"][split]
    with annotations.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    by_name = {}
    counts = {}
    for row in rows:
        counts[row["filename"]] = counts.get(row["filename"], 0) + 1
        by_name[row["filename"]] = row
    cases = []
    for identifier in identifiers:
        if counts.get(identifier) != 1:
            raise ValueError(f"Locked case is not unilateral: {identifier}")
        row = by_name[identifier]
        cases.append(Evidence(
            evidence_id=f"locked-{identifier.removesuffix('.jpg')}",
            title=f"HVAngleEst locked reference case {identifier}",
            text="De-identified public research case from the locked patient-disjoint training pool.",
            source_url="https://doi.org/10.1038/s41597-025-05261-9",
            evidence_type="case",
            metadata={
                "measurements": {"HVA": float(row["HVA"]), "IMA": float(row["IMA"])},
                "side": row["labels"].strip().lower(),
                "source": row["source"].strip().lower(),
            },
        ))
    return cases
