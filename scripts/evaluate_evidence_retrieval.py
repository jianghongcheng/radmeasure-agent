#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from geomed_copilot.evidence import load_evidence_catalog
from geomed_copilot.retrieval import HybridRetriever


GOLDEN = [
    ("How are HVA and IMA reconstructed from anatomical lines?", {"hvangleest-2025-measurement"}),
    ("How many images and annotated feet are in HVAngleEst?", {"hvangleest-2025-data"}),
    ("Is the GeoMed result live inference or an artifact replay?", {"geomed-artifact-audit"}),
    ("Which benchmark evaluates Cobb angle on spine radiographs?", {"aasce-2019-challenge"}),
    ("What patient split is used to audit the GeoMed predictions?", {"geomed-artifact-audit"}),
]


def evaluate(catalog: Path, k: int = 3) -> dict:
    retriever = HybridRetriever(load_evidence_catalog(catalog))
    rows = []
    for question, relevant in GOLDEN:
        hits = retriever.search(question, top_k=k)
        ids = [hit.evidence.evidence_id for hit in hits]
        rank = next((index + 1 for index, item in enumerate(ids) if item in relevant), None)
        rows.append({
            "question": question,
            "relevant_ids": sorted(relevant),
            "retrieved_ids": ids,
            f"hit@{k}": float(rank is not None),
            "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
        })
    return {
        "evaluation_type": "manually_labeled_curated_evidence_retrieval",
        "n_questions": len(rows),
        f"hit@{k}": sum(row[f"hit@{k}"] for row in rows) / len(rows),
        "mrr": sum(row["reciprocal_rank"] for row in rows) / len(rows),
        "limitations": "Small transparent golden set; expand before production claims.",
        "details": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

