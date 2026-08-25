import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "portfolio_benchmark.py"
    spec = importlib.util.spec_from_file_location("portfolio_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_portfolio_benchmark_reports_reliability_metrics():
    report = _module().run(5)
    assert report["runs"] == report["successful_runs"] == 5
    assert report["tool_success_rate"] == 1.0
    assert report["citation_presence_rate"] == 1.0
    assert report["provenance_mode"] == "deterministic_synthetic_demo"
