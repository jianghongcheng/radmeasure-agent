"""Summarize an archived selector result without publishing per-image records.

Usage: python scripts/summarize_repair_study.py /path/to/selector_baselines_fixed20.json
Prints aggregate JSON to stdout; does not run a model or rewrite source artifacts.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean


def summarize(path: Path) -> dict:
    content = path.read_bytes()
    artifact = json.loads(content)
    rows = [row for fold in artifact["folds"]
            for row in fold["cases"]["learned_expected_gain"]]
    edited = [row for row in rows if row["edited"]]
    counts = Counter(row["identifier"] for row in rows)
    return {
        "study": "Historical learned selective repair, fixed 20% coverage budget",
        "method": "learned_expected_gain",
        "source_file": path.name,
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "folds": len(artifact["folds"]),
        "split_seed": artifact["split_seed"],
        "coverage_budget": artifact["coverage_budget"],
        "case_records": len(rows),
        "unique_case_identifiers": len(counts),
        "records_per_identifier": sorted(set(counts.values())),
        "intervened_records": len(edited),
        "intervention_rate": len(edited) / len(rows),
        "mean_error_before_deg": mean(row["mean_before"] for row in rows),
        "mean_error_after_deg": mean(row["mean_after"] for row in rows),
        "mean_improvement_intervened_deg": mean(
            row["mean_before"] - row["mean_after"] for row in edited
        ) if edited else None,
        "records_with_increased_error": sum(
            row["mean_after"] > row["mean_before"] for row in rows
        ),
        "records_with_error_increase_over_half_degree": sum(
            row["mean_after"] - row["mean_before"] > 0.5 for row in rows
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    print(json.dumps(summarize(parser.parse_args().artifact), indent=2))
