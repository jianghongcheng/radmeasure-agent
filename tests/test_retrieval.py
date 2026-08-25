from geomed_copilot.models import Evidence
from geomed_copilot.retrieval import CaseRetriever, HybridRetriever


def test_hybrid_retriever_returns_relevant_measurement_document_first():
    docs = [
        Evidence("hva", "HVA measurement", "first metatarsal proximal phalanx angle", "a"),
        Evidence("cobb", "Cobb measurement", "upper and lower vertebral endplates", "b"),
    ]
    hits = HybridRetriever(docs).search("How is HVA first metatarsal measured?", top_k=2)
    assert hits[0].evidence.evidence_id == "hva"


def test_case_retriever_uses_measurement_distance():
    cases = [
        Evidence("near", "near", "", "", "case", {"measurements": {"HVA": 15}}),
        Evidence("far", "far", "", "", "case", {"measurements": {"HVA": 40}}),
    ]
    hits = CaseRetriever(cases).search({"HVA": 16}, top_k=2)
    assert [hit.evidence.evidence_id for hit in hits] == ["near", "far"]

