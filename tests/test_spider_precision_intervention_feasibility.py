import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "spider_precision_intervention_feasibility.py"
    spec = importlib.util.spec_from_file_location("precision_feasibility", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_advantage_sign():
    module = load_module()
    assert module.sign(1) == "benefit"
    assert module.sign(0) == "neutral"
    assert module.sign(-1) == "harm"


def test_difficulty_proxy_orders_simple_before_nested_join():
    module = load_module()
    assert module.difficulty_proxy("select a from t") == "easy"
    assert module.difficulty_proxy(
        "select a from t join u on t.id=u.id where a in (select a from v) group by a"
    ) in {"medium", "hard", "extra"}


def test_operator_type_removes_edit_payload():
    module = load_module()
    assert module.operator_type("column:a->b") == "column"
