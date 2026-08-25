from geomed_copilot.evaluation import citation_correctness, tool_success_rate
from geomed_copilot.models import CopilotRequest
from geomed_copilot.orchestrator import GeoMedCopilot
from geomed_copilot.retrieval import CaseRetriever, HybridRetriever
from geomed_copilot.sample_data import CASES, DEMO_LANDMARKS, EVIDENCE


def test_end_to_end_offline_workflow_is_traceable():
    copilot = GeoMedCopilot(HybridRetriever(EVIDENCE), CaseRetriever(CASES))
    response = copilot.run(CopilotRequest(
        question="How is HVA measured using the first metatarsal?",
        image_id="test-001",
        landmarks=DEMO_LANDMARKS,
        predicted_angles={"HVA": 15.1, "IMA": 8.0},
        top_k=2,
    ))
    assert response.status == "complete"
    assert len(response.measurements) == 2
    assert response.similar_cases[0].evidence.evidence_id == "case-001"
    assert tool_success_rate(response) == 1.0
    assert citation_correctness(response, {"guideline-hva", "method-geomed"}) == 1.0
    assert all(trace.latency_ms >= 0 for trace in response.traces)


def test_missing_landmarks_returns_partial_without_inventing_measurement():
    copilot = GeoMedCopilot(HybridRetriever(EVIDENCE), CaseRetriever(CASES))
    response = copilot.run(CopilotRequest(
        question="Measure HVA",
        image_id="test-002",
        landmarks={},
        predicted_angles={"HVA": 15.0},
    ))
    assert response.status == "partial"
    assert not response.measurements
    assert "human review" in response.answer

