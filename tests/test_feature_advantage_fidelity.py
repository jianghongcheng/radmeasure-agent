import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "feature_advantage_fidelity.py"
    spec = importlib.util.spec_from_file_location("fidelity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_label_mapping():
    module = load_module()
    assert module.label({"advantage": -1}) == 0
    assert module.label({"advantage": 0}) == 1
    assert module.label({"advantage": 1}) == 2


def test_fold_deterministic():
    module = load_module()
    assert module.fold_of("db", 5, 7719) == module.fold_of("db", 5, 7719)
