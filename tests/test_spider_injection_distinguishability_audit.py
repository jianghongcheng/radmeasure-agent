import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "spider_injection_distinguishability_audit.py"
    spec = importlib.util.spec_from_file_location("injection_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_operator_type():
    module = load_module()
    assert module.operator_type("aggregation:count->max") == "aggregation"


def test_database_fold_is_deterministic():
    module = load_module()
    assert module.fold_of("concert_singer", 5, 1931) == module.fold_of("concert_singer", 5, 1931)
