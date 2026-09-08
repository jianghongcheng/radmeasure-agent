from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from .models import Evidence, SearchHit


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    numerator = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


class HybridRetriever:
    """Small offline baseline with lexical and structured-metadata scoring.

    The interface is intentionally backend-neutral so Qdrant/Elasticsearch and
    learned embeddings can replace this implementation without changing the
    orchestration or evaluation layers.
    """

    def __init__(self, evidence: Iterable[Evidence]) -> None:
        self._evidence = list(evidence)
        self._vectors = {
            item.evidence_id: Counter(_tokens(f"{item.title} {item.text}"))
            for item in self._evidence
        }

    @staticmethod
    def _metadata_score(filters: dict[str, str], evidence: Evidence) -> float:
        if not filters:
            return 0.0
        matches = sum(str(evidence.metadata.get(key, "")).lower() == str(value).lower()
                      for key, value in filters.items())
        return matches / len(filters)

    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: dict[str, str] | None = None,
    ) -> list[SearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_vector = Counter(_tokens(query))
        filters = filters or {}
        hits: list[SearchHit] = []
        for item in self._evidence:
            lexical = _cosine(query_vector, self._vectors[item.evidence_id])
            metadata = self._metadata_score(filters, item)
            score = 0.8 * lexical + 0.2 * metadata
            hits.append(SearchHit(item, score, lexical, metadata))
        return sorted(hits, key=lambda hit: (-hit.score, hit.evidence.evidence_id))[:top_k]


class CaseRetriever:
    """Retrieve cases by normalized measurement-vector distance."""

    def __init__(self, cases: Iterable[Evidence]) -> None:
        self._cases = [case for case in cases if case.evidence_type == "case"]

    def search(self, measurements: dict[str, float], top_k: int = 3) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for case in self._cases:
            case_measurements = case.metadata.get("measurements", {})
            shared = sorted(set(measurements) & set(case_measurements))
            if not shared:
                continue
            mae = sum(abs(measurements[key] - float(case_measurements[key])) for key in shared) / len(shared)
            score = 1.0 / (1.0 + mae)
            hits.append(SearchHit(case, score, 0.0, score))
        return sorted(hits, key=lambda hit: (-hit.score, hit.evidence.evidence_id))[:top_k]

