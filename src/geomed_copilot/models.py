from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Line:
    start: Point
    end: Point


@dataclass(frozen=True)
class Measurement:
    name: str
    predicted_degrees: float
    analytical_degrees: float
    discrepancy_degrees: float
    status: str


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    title: str
    text: str
    source_url: str
    evidence_type: str = "literature"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    evidence: Evidence
    score: float
    lexical_score: float
    metadata_score: float


@dataclass(frozen=True)
class ToolTrace:
    tool: str
    ok: bool
    latency_ms: float
    output_summary: str
    error: str | None = None


@dataclass
class CopilotRequest:
    question: str
    image_id: str
    landmarks: dict[str, Line]
    predicted_angles: dict[str, float] = field(default_factory=dict)
    top_k: int = 3


@dataclass
class CopilotResponse:
    answer: str
    measurements: list[Measurement]
    citations: list[Evidence]
    similar_cases: list[SearchHit]
    traces: list[ToolTrace]
    total_latency_ms: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

