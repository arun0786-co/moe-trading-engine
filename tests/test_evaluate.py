import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from evaluate import strategy  # noqa: E402


def test_costs_paid_only_when_position_changes():
    probs = np.array([0.9, 0.9, 0.1, 0.9])
    ret = np.array([0.01, 0.02, 0.03, -0.01])
    daily, pos = strategy(probs, ret, buy_cost=0.001, sell_cost=0.002)
    assert pos.tolist() == [1, 1, 0, 1]
    want = [0.01 - 0.001, 0.02, -0.002, -0.01 - 0.001]
    assert np.allclose(daily, want)


def test_flat_earns_nothing():
    daily, pos = strategy(np.zeros(5), np.ones(5) * 0.01, 0.001, 0.001)
    assert (daily == 0).all() and (pos == 0).all()
