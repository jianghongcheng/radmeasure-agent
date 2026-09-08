import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "repair_summary", Path(__file__).parents[1] / "scripts/summarize_repair_study.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_summary_counts_repeated_cases_and_net_intervention_gain(tmp_path):
    artifact = tmp_path / "study.json"
    artifact.write_text(json.dumps({
        "split_seed": 2027, "coverage_budget": 0.2,
        "folds": [{"cases": {"learned_expected_gain": [
            {"identifier": "a", "mean_before": 3, "mean_after": 1, "edited": True},
            {"identifier": "a", "mean_before": 1, "mean_after": 2, "edited": True},
            {"identifier": "b", "mean_before": 2, "mean_after": 2, "edited": False},
        ]}}],
    }))
    result = module.summarize(artifact)
    assert result["case_records"] == 3
    assert result["unique_case_identifiers"] == 2
    assert result["intervened_records"] == 2
    assert result["mean_improvement_intervened_deg"] == pytest.approx(0.5)
    assert result["records_with_increased_error"] == 1
