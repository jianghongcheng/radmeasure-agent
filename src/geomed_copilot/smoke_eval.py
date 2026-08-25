from __future__ import annotations

import json
from dataclasses import dataclass

from .cli import build_demo
from .evaluation import citation_correctness, measurement_mae, tool_success_rate
from .models import CopilotRequest
from .sample_data import DEMO_LANDMARKS


@dataclass(frozen=True)
class GoldenItem:
    item_id: str
    question: str
    predicted_angles: dict[str, float]
    target_angles: dict[str, float]
    relevant_evidence: set[str]


SMOKE_SET = [
    GoldenItem(
        "smoke-hva-001",
        "How is HVA measured using the first metatarsal and proximal phalanx?",
        {"HVA": 15.2, "IMA": 8.1},
        {"HVA": 15.0, "IMA": 8.0},
        {"guideline-hva", "method-geomed"},
    ),
    GoldenItem(
        "smoke-hva-002",
        "Why compare a predicted angle with analytical geometry?",
        {"HVA": 15.4},
        {"HVA": 15.0},
        {"method-geomed"},
    ),
]


def run_smoke_eval() -> dict:
    copilot = build_demo()
    results = []
    for item in SMOKE_SET:
        response = copilot.run(CopilotRequest(
            question=item.question,
            image_id=item.item_id,
            landmarks=DEMO_LANDMARKS,
            predicted_angles=item.predicted_angles,
            top_k=2,
        ))
        results.append({
            "item_id": item.item_id,
            "measurement_mae": measurement_mae(response, item.target_angles),
            "citation_correctness": citation_correctness(response, item.relevant_evidence),
            "tool_success_rate": tool_success_rate(response),
            "latency_ms": response.total_latency_ms,
        })
    return {
        "evaluation_type": "synthetic_smoke_test_not_a_benchmark",
        "items": len(results),
        "measurement_mae": sum(row["measurement_mae"] for row in results) / len(results),
        "citation_correctness": sum(row["citation_correctness"] for row in results) / len(results),
        "tool_success_rate": sum(row["tool_success_rate"] for row in results) / len(results),
        "max_latency_ms": max(row["latency_ms"] for row in results),
        "details": results,
    }


def main() -> None:
    print(json.dumps(run_smoke_eval(), indent=2))


if __name__ == "__main__":
    main()

