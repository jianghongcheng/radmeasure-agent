from pathlib import Path

from geomed_copilot.evidence import load_evidence_catalog
from geomed_copilot.retrieval import HybridRetriever


def test_real_evidence_catalog_has_traceable_sources():
    path = Path(__file__).parents[1] / "data" / "evidence" / "catalog.json"
    evidence = load_evidence_catalog(path)
    assert len(evidence) >= 4
    assert len({item.evidence_id for item in evidence}) == len(evidence)
    hits = HybridRetriever(evidence).search("HVA IMA anatomical line measurement", top_k=2)
    assert "hvangleest-2025-measurement" in {hit.evidence.evidence_id for hit in hits}

