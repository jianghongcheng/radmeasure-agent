import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "reveal_spider_precision_intervention.py"
    spec = importlib.util.spec_from_file_location("reveal_precision", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fold_is_deterministic():
    module = load_module()
    assert module.fold_of("academic", 5, 260825) == module.fold_of("academic", 5, 260825)


def test_metric_counts_benefit_and_harm():
    module = load_module()
    cases = [
        {"before_correct": False, "after_correct": True, "edited": True},
        {"before_correct": True, "after_correct": False, "edited": True},
        {"before_correct": True, "after_correct": True, "edited": False},
    ]
    result = module.metric(cases)
    assert result["benefit_count"] == 1
    assert result["harm_count"] == 1
    assert result["gain"] == 0
