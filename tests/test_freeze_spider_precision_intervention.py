import importlib.util
from pathlib import Path
import pytest


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "freeze_spider_precision_intervention.py"
    spec = importlib.util.spec_from_file_location("freeze_precision", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bracket_exact_and_interpolated():
    module = load_module()
    assert module.bracket([0.1, 0.2, 0.3], 0.2) == (0.2, 0.2, 0.0)
    low, high, weight = module.bracket([0.1, 0.2, 0.3], 0.15)
    assert (low, high) == (0.1, 0.2)
    assert weight == pytest.approx(0.5)


def test_operator_type():
    module = load_module()
    assert module.operator_type("logic:and->or") == "logic"
