import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "synthetic_correlated_noise_extension.py"
SPEC = importlib.util.spec_from_file_location("correlated_noise", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_disagreement_formula_can_exceed_half():
    assert MOD.implied_disagreement(.4, -.35) > .5


def test_joint_sampler_matches_target():
    rng = np.random.default_rng(7)
    e1, e2 = MOD.joint_errors(rng, 200_000, .4, -.35)
    assert abs(e1.mean() - .4) < .005
    assert abs(e2.mean() - .4) < .005
    assert abs(np.mean(e1 != e2) - MOD.implied_disagreement(.4, -.35)) < .005
