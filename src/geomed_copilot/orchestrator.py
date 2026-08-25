from __future__ import annotations

import time
from collections.abc import Callable

from .geometry import verify_measurement
from .models import (
    CopilotRequest,
    CopilotResponse,
    Evidence,
    Measurement,
    ToolTrace,
)
from .retrieval import CaseRetriever, HybridRetriever


ANGLE_LINE_PAIRS = {
    "HVA": ("great_toe", "first_metatarsal"),
    "IMA": ("first_metatarsal", "second_metatarsal"),
    "Cobb": ("upper_endplate", "lower_endplate"),
}


class GeoMedCopilot:
    """Deterministic tool orchestrator with auditable traces and fallbacks."""

    def __init__(
        self,
        evidence_retriever: HybridRetriever,
        case_retriever: CaseRetriever,
        predictor: Callable[[str, dict], dict[str, float]] | None = None,
    ) -> None:
        self.evidence_retriever = evidence_retriever
        self.case_retriever = case_retriever
        self.predictor = predictor

    @staticmethod
    def _run_tool(name: str, fn: Callable[[], object]) -> tuple[object | None, ToolTrace]:
        started = time.perf_counter()
        try:
            output = fn()
            latency = (time.perf_counter() - started) * 1000
            size = len(output) if hasattr(output, "__len__") else 1
            return output, ToolTrace(name, True, round(latency, 3), f"returned {size} item(s)")
        except Exception as exc:  # tool boundary intentionally captures failures
            latency = (time.perf_counter() - started) * 1000
            return None, ToolTrace(name, False, round(latency, 3), "tool failed", str(exc))

    def run(self, request: CopilotRequest) -> CopilotResponse:
        started = time.perf_counter()
        traces: list[ToolTrace] = []

        predicted = dict(request.predicted_angles)
        if not predicted and self.predictor:
            result, trace = self._run_tool(
                "predict_measurements",
                lambda: self.predictor(request.image_id, request.landmarks),
            )
            traces.append(trace)
            predicted = dict(result or {})

        measurements: list[Measurement] = []
        for name, prediction in predicted.items():
            if name not in ANGLE_LINE_PAIRS:
                continue
            first_name, second_name = ANGLE_LINE_PAIRS[name]
            if first_name not in request.landmarks or second_name not in request.landmarks:
                continue
            result, trace = self._run_tool(
                f"verify_{name.lower()}",
                lambda n=name, p=prediction, a=first_name, b=second_name: verify_measurement(
                    n, p, request.landmarks[a], request.landmarks[b]
                ),
            )
            traces.append(trace)
            if isinstance(result, Measurement):
                measurements.append(result)

        measurement_map = {item.name: item.predicted_degrees for item in measurements}
        case_hits, case_trace = self._run_tool(
            "retrieve_similar_cases",
            lambda: self.case_retriever.search(measurement_map, request.top_k),
        )
        traces.append(case_trace)
        case_hits = list(case_hits or [])

        evidence_hits, evidence_trace = self._run_tool(
            "retrieve_evidence",
            lambda: self.evidence_retriever.search(request.question, request.top_k),
        )
        traces.append(evidence_trace)
        evidence_hits = list(evidence_hits or [])
        citations = [hit.evidence for hit in evidence_hits if hit.score > 0]

        answer = self._compose_answer(measurements, citations, case_hits)
        total_latency = (time.perf_counter() - started) * 1000
        required_tools_ok = all(trace.ok for trace in traces)
        has_measurement = bool(measurements)
        status = "complete" if required_tools_ok and has_measurement else "partial"
        return CopilotResponse(
            answer=answer,
            measurements=measurements,
            citations=citations,
            similar_cases=case_hits,
            traces=traces,
            total_latency_ms=round(total_latency, 3),
            status=status,
        )

    @staticmethod
    def _compose_answer(measurements, citations, cases) -> str:
        if not measurements:
            return "No verifiable measurement was produced; human review is required."
        measurement_text = "; ".join(
            f"{item.name}={item.predicted_degrees:.2f}° "
            f"(analytical {item.analytical_degrees:.2f}°, {item.status})"
            for item in measurements
        )
        citation_text = ", ".join(f"[{item.evidence_id}]" for item in citations) or "no supporting citation"
        case_text = f"{len(cases)} similar case(s) retrieved"
        return f"Measurements: {measurement_text}. Evidence: {citation_text}; {case_text}."
