import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "synthetic_selective_correction_phase.py"
SPEC = importlib.util.spec_from_file_location("synthetic_phase", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_generator_controls_marginal_precision():
    condition = MODULE.Condition(0.7, 0.0, 300, 0)
    data = MODULE.generate_condition(condition, samples_per_cluster=100, seed=19)
    assert abs(float(data["beneficial"].mean()) - 0.7) < 0.02


def test_cluster_splits_are_disjoint_and_exhaustive():
    split = MODULE.cluster_split(30, seed=5, repeat=2)
    sets = [set(values.tolist()) for values in split]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == set(range(30))


def test_evaluation_is_reproducible_and_finite():
    condition = MODULE.Condition(0.6, 0.1, 30, 4)
    first = MODULE.evaluate_condition(condition, samples_per_cluster=10, seed=41)
    second = MODULE.evaluate_condition(condition, samples_per_cluster=10, seed=41)
    assert first == second
    assert np.isfinite(first["learned_gain"])
    assert 0.0 <= first["learned_coverage"] <= 1.0
