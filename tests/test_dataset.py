import numpy as np
import pandas as pd

from engine.data.dataset import TradingDataset


def test_target_is_next_day_return_after_window():
    # close goes 100, 101, 102, ...; forward return at day i is close[i+1]/close[i] - 1
    n, lookback = 50, 5
    close = 100.0 + np.arange(n)
    fwd = pd.Series(close).pct_change().shift(-1).fillna(0.0).values
    feats = np.arange(n, dtype=np.float32)[:, None]
    ds = TradingDataset(feats, feats, fwd, np.arange(n), lookback=lookback, horizon=1)
    item = ds[0]
    last_day = int(item["expert_input"][-1, 0])
    assert last_day == lookback - 1
    want = close[last_day + 1] / close[last_day] - 1
    assert abs(float(item["target_return"]) - want) < 1e-6


def test_no_index_past_the_end():
    n, lookback = 20, 5
    fwd = np.ones(n) * 0.01
    feats = np.zeros((n, 3), np.float32)
    ds = TradingDataset(feats, feats, fwd, np.arange(n), lookback=lookback, horizon=1)
    ds[len(ds) - 1]
