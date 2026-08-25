from geomed_copilot.smoke_eval import run_smoke_eval


def test_smoke_evaluation_exercises_metrics_without_claiming_benchmark_status():
    result = run_smoke_eval()
    assert result["evaluation_type"] == "synthetic_smoke_test_not_a_benchmark"
    assert result["items"] == 2
    assert result["tool_success_rate"] == 1.0
    assert 0.0 <= result["citation_correctness"] <= 1.0

