import json
from pathlib import Path

from geomed_copilot.harbor_export import frozen_sql_submission


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_sql_submission_preserves_all_36_case_ids():
    rows = frozen_sql_submission(
        ROOT / "data/benchmarks/sql_repair_v1.json",
        ROOT / "outputs/portfolio/sql_harness_ablation_qwen3_8b.json",
    )
    assert len(rows) == 36
    assert len({row["id"] for row in rows}) == 36
    assert all(row["source"] == "qwen3:8b-frozen" for row in rows)


def test_harbor_agent_environment_does_not_contain_hidden_labels():
    public = json.loads(
        (ROOT / "harbor/tasks/radmeasure_sql_repair_v1/environment/cases.json").read_text()
    )
    assert len(public) == 36
    assert all("expected_action" not in row and "gold_sql" not in row for row in public)
    hidden = json.loads(
        (ROOT / "harbor/tasks/radmeasure_sql_repair_v1/tests/expected.json").read_text()
    )
    assert len(hidden) == 36
    assert all("expected_action" in row and "gold_sql" in row for row in hidden)


def test_harbor_task_uses_separate_offline_verifier_and_declared_artifact():
    config = (ROOT / "harbor/tasks/radmeasure_sql_repair_v1/task.toml").read_text()
    assert 'artifacts = ["/app/submission.json"]' in config
    assert 'environment_mode = "separate"' in config
    assert config.count('network_mode = "no-network"') == 3
