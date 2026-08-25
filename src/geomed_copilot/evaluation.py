from __future__ import annotations

import statistics

from .models import CopilotResponse, SearchHit


def recall_at_k(hits: list[SearchHit], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        raise ValueError("relevant_ids cannot be empty")
    retrieved = {hit.evidence.evidence_id for hit in hits[:k]}
    return len(retrieved & relevant_ids) / len(relevant_ids)


def citation_correctness(response: CopilotResponse, relevant_ids: set[str]) -> float:
    cited = {item.evidence_id for item in response.citations}
    if not cited:
        return 0.0
    return len(cited & relevant_ids) / len(cited)


def tool_success_rate(response: CopilotResponse) -> float:
    if not response.traces:
        return 0.0
    return sum(trace.ok for trace in response.traces) / len(response.traces)


def measurement_mae(response: CopilotResponse, targets: dict[str, float]) -> float:
    errors = [
        abs(item.predicted_degrees - targets[item.name])
        for item in response.measurements
        if item.name in targets
    ]
    return statistics.mean(errors) if errors else float("nan")

