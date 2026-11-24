import pandas as pd

import engine.data.downloader as dl


def test_cached_data_used_when_gap_is_a_weekend(tmp_path, monkeypatch):
    # cache ends on friday 20 nov, the run asks for data up to sunday 22 nov
    idx = pd.date_range("2026-11-16", "2026-11-20", freq="B", name="Date")
    cached = pd.DataFrame({"Close": 1.0, "High": 1.0, "Low": 1.0, "Open": 1.0, "Volume": 1.0}, index=idx)
    cached.to_csv(tmp_path / "IDX_NSEI_2026-11-16_2026-11-22.csv")
    monkeypatch.setattr(dl.yf, "download", lambda *a, **k: pd.DataFrame())

    df = dl.download_ticker("^NSEI", start="2026-11-16", end="2026-11-22", raw_dir=tmp_path,
                            max_retries=2, initial_backoff=0)
    assert len(df) == 5
