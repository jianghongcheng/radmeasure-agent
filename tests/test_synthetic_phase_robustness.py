import importlib.util
import sys
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "scripts" / "synthetic_phase_robustness.py"
SPEC = importlib.util.spec_from_file_location("phase_robustness", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_best_threshold_can_choose_stop():
    scores = MODULE.np.array([0.9, 0.8, 0.7])
    advantages = MODULE.np.array([-1.0, -1.0, -1.0])
    threshold = MODULE.best_threshold(scores, advantages)
    assert not (scores >= threshold).any()


def test_magnitude_condition_is_deterministic_and_finite():
    first = MODULE.magnitude_condition(0.5, 0.2, 10, 0, "meaningful", seed=19)
    second = MODULE.magnitude_condition(0.5, 0.2, 10, 0, "meaningful", seed=19)
    assert first == second
    assert MODULE.np.isfinite(first["learned_gain"])


def test_candidate_condition_supports_both_dependence_models():
    for dependence in ("independent", "shared_case"):
        row = MODULE.candidate_condition(0.5, 0.2, 10, 3, dependence, 0, seed=23)
        assert -1 <= row["learned_gain"] <= 1
        assert row["candidate_count"] == 3
