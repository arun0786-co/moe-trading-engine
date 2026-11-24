"""
Robust OHLCV data downloader for the MoE Trading Engine.

Downloads daily price data from yfinance with exponential-backoff retries,
validates data integrity against the NSE market calendar, supports
incremental updates, and persists raw CSVs for reproducibility.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NSE Market Holiday Calendar (2024–2026) + weekend detection
# ---------------------------------------------------------------------------
# Source: NSE circulars.  Dates are *trading holidays only* (excludes weekends).
# fmt: off
_NSE_HOLIDAYS_RAW: list[str] = [
    # ---- 2024 ----
    "2024-01-22",  # Special Holiday
    "2024-01-26",  # Republic Day
    "2024-03-08",  # Maha Shivaratri
    "2024-03-25",  # Holi
    "2024-03-29",  # Good Friday
    "2024-04-11",  # Id-ul-Fitr (Eid)
    "2024-04-14",  # Dr. Ambedkar Jayanti
    "2024-04-17",  # Ram Navami
    "2024-04-21",  # Mahavir Jayanti
    "2024-05-01",  # Maharashtra Day
    "2024-05-23",  # Buddha Purnima
    "2024-06-17",  # Eid-ul-Adha (Bakri Eid)
    "2024-07-17",  # Muharram
    "2024-08-15",  # Independence Day
    "2024-09-16",  # Milad-un-Nabi
    "2024-10-02",  # Mahatma Gandhi Jayanti
    "2024-10-12",  # Dussehra (Saturday — traded; kept for completeness)
    "2024-11-01",  # Diwali (Laxmi Puja)
    "2024-11-15",  # Guru Nanak Jayanti
    "2024-12-25",  # Christmas
    # ---- 2025 ----
    "2025-01-26",  # Republic Day (Sunday — observed Monday if applicable)
    "2025-02-26",  # Maha Shivaratri
    "2025-03-14",  # Holi
    "2025-03-31",  # Id-ul-Fitr (Eid)
    "2025-04-10",  # Mahavir Jayanti
    "2025-04-14",  # Dr. Ambedkar Jayanti
    "2025-04-18",  # Good Friday
    "2025-05-01",  # Maharashtra Day
    "2025-05-12",  # Buddha Purnima
    "2025-06-07",  # Eid-ul-Adha (Bakri Eid)
    "2025-07-06",  # Muharram
    "2025-08-15",  # Independence Day
    "2025-08-27",  # Janmashtami
    "2025-09-05",  # Milad-un-Nabi
    "2025-10-02",  # Mahatma Gandhi Jayanti / Dussehra
    "2025-10-20",  # Diwali (Laxmi Puja)
    "2025-10-21",  # Diwali Balipratipada
    "2025-11-05",  # Guru Nanak Jayanti (Prakash Utsav)
    "2025-12-25",  # Christmas
    # ---- 2026 ----
    "2026-01-26",  # Republic Day
    "2026-02-17",  # Maha Shivaratri
    "2026-03-03",  # Holi
    "2026-03-20",  # Id-ul-Fitr (Eid)
    "2026-03-30",  # Mahavir Jayanti
    "2026-04-03",  # Good Friday
    "2026-04-14",  # Dr. Ambedkar Jayanti
    "2026-05-01",  # Maharashtra Day / Buddha Purnima
    "2026-05-28",  # Eid-ul-Adha (Bakri Eid)
    "2026-06-25",  # Muharram
    "2026-08-15",  # Independence Day
    "2026-08-16",  # Janmashtami
    "2026-08-25",  # Milad-un-Nabi
    "2026-10-02",  # Mahatma Gandhi Jayanti
    "2026-10-09",  # Diwali (Laxmi Puja)
    "2026-10-20",  # Dussehra
    "2026-10-26",  # Guru Nanak Jayanti
    "2026-12-25",  # Christmas
]
# fmt: on

NSE_HOLIDAYS: frozenset[date] = frozenset(
    datetime.strptime(d, "%Y-%m-%d").date() for d in _NSE_HOLIDAYS_RAW
)


def is_nse_trading_day(d: date) -> bool:
    """Return True if *d* is a valid NSE trading session."""
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return d not in NSE_HOLIDAYS


def expected_trading_days(start: date, end: date) -> list[date]:
    """Return a sorted list of expected NSE trading dates in [start, end]."""
    days: list[date] = []
    current = start
    while current <= end:
        if is_nse_trading_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days


# ---------------------------------------------------------------------------
# Core downloader
# ---------------------------------------------------------------------------

_DEFAULT_RAW_DIR = Path("data/raw")
_MAX_RETRIES = 3
_INITIAL_BACKOFF_S = 2.0


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a MultiIndex column header produced by yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance ≥0.2.28 returns ('Close', 'RELIANCE.NS') style tuples
        df.columns = [
            col[0] if isinstance(col, tuple) else col for col in df.columns
        ]
    # Normalise to title-case so downstream code can rely on Open/High/Low/Close/Volume
    rename_map: dict[str, str] = {}
    for col in df.columns:
        canonical = col.strip().title()
        if canonical in {"Open", "High", "Low", "Close", "Volume", "Adj Close"}:
            rename_map[col] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _validate_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate and clean an OHLCV DataFrame.

    Checks
    ------
    1. No NaN values in OHLC prices.
    2. Volume > 0 for every row.
    3. No suspiciously large date gaps beyond market holidays + weekends.

    Returns the validated DataFrame (rows with zero volume are logged and
    forward-filled for Volume only; NaN-price rows are dropped).
    """
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"[{ticker}] Missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    initial_len = len(df)

    # --- NaN prices ---
    price_cols = ["Open", "High", "Low", "Close"]
    nan_mask = df[price_cols].isna().any(axis=1)
    n_nan = int(nan_mask.sum())
    if n_nan:
        logger.warning(
            "[%s] Dropping %d rows with NaN prices (%.2f%%)",
            ticker,
            n_nan,
            100.0 * n_nan / initial_len,
        )
        df = df.loc[~nan_mask].copy()

    # --- Zero / negative volume ---
    zero_vol_mask = df["Volume"] <= 0
    n_zero_vol = int(zero_vol_mask.sum())
    if n_zero_vol:
        logger.warning(
            "[%s] %d rows with volume ≤ 0 — forward-filling volume.",
            ticker,
            n_zero_vol,
        )
        df.loc[zero_vol_mask, "Volume"] = np.nan
        df["Volume"] = df["Volume"].ffill().bfill()

    # --- Date-gap check ---
    if len(df) >= 2:
        idx = pd.DatetimeIndex(df.index)
        gaps = idx[1:] - idx[:-1]
        for i, gap in enumerate(gaps):
            gap_days = gap.days
            if gap_days <= 1:
                continue
            start_gap = idx[i].date() + timedelta(days=1)
            end_gap = idx[i + 1].date() - timedelta(days=1)
            expected = expected_trading_days(start_gap, end_gap)
            if expected:
                logger.info(
                    "[%s] Gap of %d calendar days (%s → %s) with %d "
                    "expected trading days unaccounted for.",
                    ticker,
                    gap_days,
                    idx[i].date(),
                    idx[i + 1].date(),
                    len(expected),
                )

    logger.info(
        "[%s] Validation passed — %d rows retained (dropped %d).",
        ticker,
        len(df),
        initial_len - len(df),
    )
    return df


def _csv_path(
    ticker: str,
    start: str,
    end: str,
    raw_dir: Path,
) -> Path:
    """Canonical CSV path for a ticker/date-range pair."""
    safe_ticker = ticker.replace("^", "IDX_").replace(".", "_")
    return raw_dir / f"{safe_ticker}_{start}_{end}.csv"


def _load_existing(csv_path: Path) -> pd.DataFrame | None:
    """Load a previously saved CSV (if it exists)."""
    if not csv_path.exists():
        return None
    logger.info("Loading cached data from %s", csv_path)
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = _flatten_columns(df)
    return df


def download_ticker(
    ticker: str,
    start: str = "2010-01-01",
    end: str | None = None,
    raw_dir: str | Path = _DEFAULT_RAW_DIR,
    max_retries: int = _MAX_RETRIES,
    initial_backoff: float = _INITIAL_BACKOFF_S,
) -> pd.DataFrame:
    """Download daily OHLCV data for a single ticker from yfinance.

    Parameters
    ----------
    ticker:
        yfinance ticker symbol, e.g. ``"RELIANCE.NS"`` or ``"^NSEI"``.
    start / end:
        ISO date strings.  *end* defaults to today.
    raw_dir:
        Directory for raw CSV persistence.
    max_retries:
        Number of retry attempts on transient failures.
    initial_backoff:
        Initial backoff in seconds (doubled each retry).

    Returns
    -------
    pd.DataFrame
        Validated OHLCV DataFrame indexed by ``DatetimeIndex``.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if end is None:
        end = date.today().isoformat()

    csv_path = _csv_path(ticker, start, end, raw_dir)

    # ------ Incremental update logic ------
    existing_df = _load_existing(csv_path)
    fetch_start = start
    if existing_df is not None and len(existing_df) > 0:
        last_date: date = pd.Timestamp(existing_df.index[-1]).date()
        desired_end: date = datetime.strptime(end, "%Y-%m-%d").date()
        if last_date >= desired_end:
            logger.info(
                "[%s] Data already up-to-date (last=%s, desired_end=%s).",
                ticker,
                last_date,
                desired_end,
            )
            return existing_df
        # Fetch from the day after the last available date
        fetch_start = (last_date + timedelta(days=1)).isoformat()
        logger.info(
            "[%s] Incremental update: fetching %s → %s",
            ticker,
            fetch_start,
            end,
        )

    # ------ Download with retry + exponential backoff ------
    backoff = initial_backoff
    last_error: Exception | None = None
    new_df: pd.DataFrame | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "[%s] yfinance download attempt %d/%d  (%s → %s)",
                ticker,
                attempt,
                max_retries,
                fetch_start,
                end,
            )
            raw = yf.download(
                ticker,
                start=fetch_start,
                end=end,
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if raw is None or raw.empty:
                raise ValueError(f"yfinance returned empty data for {ticker}")

            new_df = _flatten_columns(raw)
            break  # success

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "[%s] Attempt %d failed: %s — retrying in %.1fs",
                ticker,
                attempt,
                exc,
                backoff,
            )
            time.sleep(backoff)
            backoff *= 2.0

    if new_df is None and existing_df is not None and len(existing_df) > 0:
        # the gap is often a weekend or holiday with no bars at all
        logger.warning("[%s] Incremental update got nothing (%s); using cached data.", ticker, last_error)
        return existing_df

    if new_df is None:
        raise RuntimeError(
            f"[{ticker}] All {max_retries} download attempts failed. "
            f"Last error: {last_error}"
        )

    # ------ Merge with existing data ------
    if existing_df is not None and len(existing_df) > 0:
        combined = pd.concat([existing_df, new_df])
        # Drop duplicates keeping the *latest* download (more recent data wins)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        new_df = combined

    # ------ Validate ------
    new_df = _validate_ohlcv(new_df, ticker)

    # ------ Persist ------
    new_df.to_csv(csv_path)
    logger.info("[%s] Saved %d rows → %s", ticker, len(new_df), csv_path)

    return new_df


# ---------------------------------------------------------------------------
# Batch downloader
# ---------------------------------------------------------------------------

def download_multiple(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
    raw_dir: str | Path = _DEFAULT_RAW_DIR,
    max_retries: int = _MAX_RETRIES,
    initial_backoff: float = _INITIAL_BACKOFF_S,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV data for multiple tickers sequentially.

    Parameters
    ----------
    tickers:
        List of yfinance ticker symbols.
    start / end / raw_dir / max_retries / initial_backoff:
        Forwarded to :func:`download_ticker`.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping ``{ticker: validated_ohlcv_df}``.  Tickers that fail all
        retries are logged and omitted from the result.
    """
    results: dict[str, pd.DataFrame] = {}

    for i, ticker in enumerate(tickers, 1):
        logger.info(
            "=== Downloading [%d/%d] %s ===", i, len(tickers), ticker
        )
        try:
            df = download_ticker(
                ticker,
                start=start,
                end=end,
                raw_dir=raw_dir,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
            )
            results[ticker] = df
            logger.info(
                "[%s] ✓ %d rows  (%s → %s)",
                ticker,
                len(df),
                df.index[0].date(),
                df.index[-1].date(),
            )
        except Exception:  # noqa: BLE001
            logger.exception("[%s] ✗ Download failed — skipping.", ticker)

    logger.info(
        "Batch download complete: %d/%d tickers succeeded.",
        len(results),
        len(tickers),
    )
    return results
