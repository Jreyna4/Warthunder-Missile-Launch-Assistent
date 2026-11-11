from pathlib import Path

import numpy as np

from wtmla.advisor import advise
from wtmla.envelope import envelope
from wtmla.missile import IR, MissileSpec
from wtmla.nn import Regressor, gradient_check
from wtmla.sheets import load_any
from wtmla.simulator import simulate

FIX = Path(__file__).parent / "fixtures"
AAM4 = load_any(FIX / "AAM-4.tsv")[0]


def test_backprop_matches_finite_differences():
    assert gradient_check() < 1e-6


def test_network_learns_a_curve():
    rng = np.random.default_rng(1)
    X = rng.uniform(-3, 3, size=(400, 1))
    Y = np.sin(X)
    reg = Regressor(1, 1, hidden=(16, 16)).fit(X, Y, epochs=600, verbose=False)
    assert reg.meta["val_mae"][0] < 0.05


def test_short_head_on_shot_hits():
    r = simulate(AAM4, 300, 8000, 300, 8000, 0, [5000.0, 10000.0])
    assert r["hit"].all()


def test_envelope_trends():
    env = envelope(AAM4, [300, 300], [2000, 10000], 300, [2000, 10000], 0)
    assert (env["r_ne"] <= env["r_max"] + 1).all()
    assert env["r_max"][1] > env["r_max"][0]  # thinner air -> longer range


def test_rear_aspect_ir_cannot_shoot_head_on():
    ir = MissileSpec.from_dict({**AAM4.to_dict(), "name": "IR test", "category": IR,
                                "lock_range_all_aspect": None, "lock_range_rear_aspect": 4.0})
    adv = advise(ir, 280, 3000, aspect_deg=0)
    assert adv.recommended_km is None
    adv = advise(ir, 280, 3000, aspect_deg=180)
    assert adv.recommended_km is not None and adv.recommended_km <= 4.0
