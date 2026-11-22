"""
Comprehensive feature engineering pipeline for the MoE Trading Engine.

Computes 30+ technical, statistical, and calendar features with **zero
look-ahead bias**: every feature at row *t* uses only data available at
or before *t*.

Feature groups
--------------
1. Price returns (4)           2. Volume (3)
3. Trend (4)                   4. Momentum (4)
5. Volatility (5)              6. Regime (2)
7. Volume profile (3)          8. Calendar (3)
9. Sentiment placeholder (1)

Router features (9) — separate accessor for the gating network.
"""

from __future__ import annotations

import calendar as cal_mod
import logging
from datetime import date

import numpy as np
import pandas as pd
import ta  # Technical Analysis library

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════════

def _safe_div(
    numerator: pd.Series,
    denominator: pd.Series,
    fill: float = 0.0,
) -> pd.Series:
    """Element-wise division that replaces inf / NaN results with *fill*."""
    result = numerator / denominator.replace(0, np.nan)
    return result.fillna(fill).replace([np.inf, -np.inf], fill)


def _hurst_rs(series: np.ndarray) -> float:
    """Compute the Hurst exponent via rescaled-range (R/S) analysis.

    Parameters
    ----------
    series : np.ndarray
        1-D array of *log-returns* (or any stationary series).
        Minimum length ≈ 20; returns ``np.nan`` for shorter inputs.

    Returns
    -------
    float
        Estimated Hurst exponent in (0, 1).
        H ≈ 0.5 → random walk, H > 0.5 → trending, H < 0.5 → mean-reverting.
    """
    n = len(series)
    if n < 20:
        return np.nan

    # Use sub-series lengths that are powers of 2 (common R/S approach)
    max_k = int(np.floor(np.log2(n)))
    sizes: list[int] = [2 ** k for k in range(2, max_k + 1) if 2 ** k <= n]
    if len(sizes) < 2:
        return np.nan

    rs_values: list[float] = []
    for size in sizes:
        n_blocks = n // size
        if n_blocks == 0:
            continue
        rs_block: list[float] = []
        for i in range(n_blocks):
            block = series[i * size : (i + 1) * size]
            mean_block = np.mean(block)
            deviations = block - mean_block
            cumulative = np.cumsum(deviations)
            r = np.max(cumulative) - np.min(cumulative)
            s = np.std(block, ddof=1)
            if s > 1e-12:
                rs_block.append(r / s)
        if rs_block:
            rs_values.append(np.mean(rs_block))
        else:
            rs_values.append(np.nan)

    # Drop NaN entries
    valid = [
        (np.log(sizes[i]), np.log(rs_values[i]))
        for i in range(len(sizes))
        if np.isfinite(rs_values[i])
    ]
    if len(valid) < 2:
        return np.nan

    log_n = np.array([v[0] for v in valid])
    log_rs = np.array([v[1] for v in valid])

    # OLS fit:  log(R/S) = H * log(n) + c
    slope, _ = np.polyfit(log_n, log_rs, 1)
    return float(np.clip(slope, 0.0, 1.0))


def _rolling_hurst(
    returns: pd.Series,
    window: int = 100,
) -> pd.Series:
    """Rolling Hurst exponent computed with the R/S method.

    Uses an expanding window until *window* samples are available, then
    a fixed rolling window.
    """
    out = pd.Series(np.nan, index=returns.index, dtype=np.float64)
    arr = returns.values.astype(np.float64)
    for i in range(window, len(arr) + 1):
        block = arr[i - window : i]
        if np.any(np.isnan(block)):
            continue
        out.iloc[i - 1] = _hurst_rs(block)
    return out


def _rolling_autocorr(
    returns: pd.Series,
    lag: int,
    window: int = 60,
) -> pd.Series:
    """Rolling autocorrelation of *returns* at a given *lag*."""
    return returns.rolling(window=window, min_periods=window).apply(
        lambda x: pd.Series(x).autocorr(lag=lag),
        raw=False,
    )


def _last_thursday(year: int, month: int) -> date:
    """Return the date of the last Thursday in the given month (NSE F&O expiry)."""
    # calendar.monthrange -> (weekday_of_first, num_days)
    _, num_days = cal_mod.monthrange(year, month)
    d = date(year, month, num_days)
    # Move backwards to Thursday (weekday 3)
    offset = (d.weekday() - 3) % 7
    return d - pd.Timedelta(days=offset)


def _is_expiry_week(dt: pd.Timestamp) -> bool:
    """Return True if *dt* falls in the calendar week of the last Thursday."""
    lt = _last_thursday(dt.year, dt.month)
    # ISO week number comparison
    return dt.isocalendar()[1] == lt.isocalendar()[1] and dt.year == lt.year


# ═══════════════════════════════════════════════════════════════════════════
# Feature names (must align with config/settings.yaml)
# ═══════════════════════════════════════════════════════════════════════════

EXPERT_FEATURE_NAMES: list[str] = [
    # Price returns
    "ret_open", "ret_high", "ret_low", "ret_close",
    # Volume
    "log_volume", "volume_zscore", "volume_ratio_20",
    # Trend
    "sma_10_cross", "sma_20_cross", "sma_50_cross", "ema_macd_signal",
    # Momentum
    "rsi_14", "macd_histogram", "stoch_k", "stoch_d",
    # Volatility
    "atr_14", "bb_width", "bb_pctb", "rolling_std_10", "rolling_std_30",
    # Volume profile
    "obv_change", "ad_line", "cmf_20",
    # Calendar
    "day_of_week_sin", "day_of_week_cos", "is_expiry_week",
    # Sentiment (placeholder)
    "finbert_score",
]

ROUTER_FEATURE_NAMES: list[str] = [
    "rolling_vol_10",
    "rolling_vol_30",
    "trend_velocity",
    "adx_14",
    "hurst_exponent",
    "return_autocorr_1",
    "return_autocorr_5",
    "volume_regime",
    "vol_ratio_short_long",
]


# ═══════════════════════════════════════════════════════════════════════════
# Main feature-engineering class
# ═══════════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """Compute all technical / statistical / calendar features.

    All computations are strictly causal — no future data is used.

    Usage
    -----
    >>> fe = FeatureEngineer()
    >>> df = fe.compute_all_features(raw_ohlcv_df)
    >>> router_df = fe.compute_router_features(df)
    """

    # ---- Group 1: Price Returns (4) ------------------------------------
    @staticmethod
    def _price_returns(df: pd.DataFrame) -> pd.DataFrame:
        for col, feat in [
            ("Open", "ret_open"),
            ("High", "ret_high"),
            ("Low", "ret_low"),
            ("Close", "ret_close"),
        ]:
            df[feat] = df[col].pct_change()
        return df

    # ---- Group 2: Volume (3) -------------------------------------------
    @staticmethod
    def _volume_features(df: pd.DataFrame) -> pd.DataFrame:
        vol = df["Volume"].astype(np.float64)

        # Log volume (log1p to handle volume=0 edge case)
        df["log_volume"] = np.log1p(vol)

        # Expanding z-score (no look-ahead)
        expanding_mean = vol.expanding(min_periods=2).mean()
        expanding_std = vol.expanding(min_periods=2).std()
        df["volume_zscore"] = _safe_div(vol - expanding_mean, expanding_std)

        # Volume ratio vs 20-day rolling average
        vol_ma20 = vol.rolling(window=20, min_periods=1).mean()
        df["volume_ratio_20"] = _safe_div(vol, vol_ma20, fill=1.0)

        return df

    # ---- Group 3: Trend (4) --------------------------------------------
    @staticmethod
    def _trend_features(df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]

        # SMA crossover signals: (Price / SMA) - 1
        for period, name in [(10, "sma_10_cross"), (20, "sma_20_cross"), (50, "sma_50_cross")]:
            sma = close.rolling(window=period, min_periods=period).mean()
            df[name] = _safe_div(close - sma, sma)

        # MACD signal line value (EMA12 - EMA26 signal via `ta`)
        macd_indicator = ta.trend.MACD(
            close=close, window_slow=26, window_fast=12, window_sign=9
        )
        df["ema_macd_signal"] = macd_indicator.macd_signal()

        return df

    # ---- Group 4: Momentum (4) -----------------------------------------
    @staticmethod
    def _momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # RSI(14)
        df["rsi_14"] = ta.momentum.RSIIndicator(
            close=close, window=14
        ).rsi()

        # MACD histogram
        macd_indicator = ta.trend.MACD(
            close=close, window_slow=26, window_fast=12, window_sign=9
        )
        df["macd_histogram"] = macd_indicator.macd_diff()

        # Stochastic %K, %D
        stoch = ta.momentum.StochasticOscillator(
            high=high, low=low, close=close, window=14, smooth_window=3
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        return df

    # ---- Group 5: Volatility (5) ---------------------------------------
    @staticmethod
    def _volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # ATR(14)
        atr = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        )
        df["atr_14"] = atr.average_true_range()

        # Bollinger Bands (20, 2)
        bb = ta.volatility.BollingerBands(
            close=close, window=20, window_dev=2
        )
        bb_high = bb.bollinger_hband()
        bb_low = bb.bollinger_lband()
        bb_mid = bb.bollinger_mavg()

        # Band width: (upper - lower) / middle
        df["bb_width"] = _safe_div(bb_high - bb_low, bb_mid)

        # %B: (price - lower) / (upper - lower)
        df["bb_pctb"] = _safe_div(close - bb_low, bb_high - bb_low, fill=0.5)

        # Rolling standard deviation of returns
        ret = close.pct_change()
        df["rolling_std_10"] = ret.rolling(window=10, min_periods=5).std()
        df["rolling_std_30"] = ret.rolling(window=30, min_periods=15).std()

        return df

    # ---- Group 6: Regime (2) -------------------------------------------
    @staticmethod
    def _regime_features(df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # ADX(14)
        adx = ta.trend.ADXIndicator(
            high=high, low=low, close=close, window=14
        )
        df["adx_14"] = adx.adx()

        # Hurst exponent (rolling 100-day R/S)
        log_ret = np.log(close / close.shift(1))
        df["hurst_exponent"] = _rolling_hurst(log_ret, window=100)

        return df

    # ---- Group 7: Volume Profile (3) -----------------------------------
    @staticmethod
    def _volume_profile_features(df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"].astype(np.float64)

        # OBV change rate (percentage change of OBV)
        obv = ta.volume.OnBalanceVolumeIndicator(
            close=close, volume=volume
        ).on_balance_volume()
        # OBV can be zero early; use safe division
        df["obv_change"] = _safe_div(obv.diff(), obv.abs().replace(0, 1))

        # Accumulation/Distribution line (raw)
        ad = ta.volume.AccDistIndexIndicator(
            high=high, low=low, close=close, volume=volume
        ).acc_dist_index()
        df["ad_line"] = ad

        # Chaikin Money Flow (20)
        cmf = ta.volume.ChaikinMoneyFlowIndicator(
            high=high, low=low, close=close, volume=volume, window=20
        ).chaikin_money_flow()
        df["cmf_20"] = cmf

        return df

    # ---- Group 8: Calendar (3) -----------------------------------------
    @staticmethod
    def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        idx = pd.DatetimeIndex(df.index)
        day_of_week = idx.dayofweek.astype(np.float64)  # Mon=0 .. Fri=4

        # Cyclical encoding (period = 5 trading days)
        df["day_of_week_sin"] = np.sin(2.0 * np.pi * day_of_week / 5.0)
        df["day_of_week_cos"] = np.cos(2.0 * np.pi * day_of_week / 5.0)

        # Expiry-week flag (last Thursday of month)
        df["is_expiry_week"] = [
            1.0 if _is_expiry_week(ts) else 0.0 for ts in idx
        ]

        return df

    # ---- Group 9: Sentiment placeholder (1) ----------------------------
    @staticmethod
    def _sentiment_placeholder(df: pd.DataFrame) -> pd.DataFrame:
        """Add a zeroed sentiment column to be filled by the sentiment pipeline."""
        if "finbert_score" not in df.columns:
            df["finbert_score"] = 0.0
        return df

    # ════════════════════════════════════════════════════════════════════
    # Public API
    # ════════════════════════════════════════════════════════════════════

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all expert + router feature columns to *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV data with columns ``Open, High, Low, Close, Volume``
            and a ``DatetimeIndex``.

        Returns
        -------
        pd.DataFrame
            The input DataFrame augmented with ~30 feature columns.
            Leading rows with insufficient history are ``NaN`` and should
            be dropped before model consumption.
        """
        logger.info(
            "Computing features on %d rows (%s → %s)",
            len(df),
            df.index[0].date() if len(df) else "?",
            df.index[-1].date() if len(df) else "?",
        )

        df = df.copy()

        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.DatetimeIndex(df.index)

        # Expert features (groups 1–9)
        df = self._price_returns(df)
        df = self._volume_features(df)
        df = self._trend_features(df)
        df = self._momentum_features(df)
        df = self._volatility_features(df)
        df = self._regime_features(df)
        df = self._volume_profile_features(df)
        df = self._calendar_features(df)
        df = self._sentiment_placeholder(df)

        # Router features (some overlap with expert, but computed here
        # for completeness so they exist in the same DataFrame)
        df = self.compute_router_features(df)

        n_features = len(EXPERT_FEATURE_NAMES) + len(ROUTER_FEATURE_NAMES)
        logger.info(
            "Feature engineering complete — %d feature columns added.",
            n_features,
        )
        return df

    def compute_router_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute / ensure router-specific features exist in *df*.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that *already* has the expert features computed
            (or at least the ``Close`` and ``Volume`` columns).

        Returns
        -------
        pd.DataFrame
            Same DataFrame with router columns added / verified.
        """
        close = df["Close"]
        ret = close.pct_change()
        volume = df["Volume"].astype(np.float64)

        # --- rolling_vol_10 / rolling_vol_30 ---
        if "rolling_vol_10" not in df.columns:
            df["rolling_vol_10"] = ret.rolling(window=10, min_periods=5).std()
        if "rolling_vol_30" not in df.columns:
            df["rolling_vol_30"] = ret.rolling(window=30, min_periods=15).std()

        # --- trend_velocity: slope of SMA(20) over last 10 days ---
        if "trend_velocity" not in df.columns:
            sma20 = close.rolling(window=20, min_periods=20).mean()
            df["trend_velocity"] = sma20.pct_change(periods=10)

        # --- adx_14 (may already exist from regime) ---
        if "adx_14" not in df.columns:
            adx = ta.trend.ADXIndicator(
                high=df["High"], low=df["Low"], close=close, window=14
            )
            df["adx_14"] = adx.adx()

        # --- hurst_exponent (may already exist from regime) ---
        if "hurst_exponent" not in df.columns:
            log_ret = np.log(close / close.shift(1))
            df["hurst_exponent"] = _rolling_hurst(log_ret, window=100)

        # --- return_autocorr_1 / return_autocorr_5 ---
        if "return_autocorr_1" not in df.columns:
            df["return_autocorr_1"] = _rolling_autocorr(ret, lag=1, window=60)
        if "return_autocorr_5" not in df.columns:
            df["return_autocorr_5"] = _rolling_autocorr(ret, lag=5, window=60)

        # --- volume_regime: current volume / 60-day avg volume ---
        if "volume_regime" not in df.columns:
            vol_ma60 = volume.rolling(window=60, min_periods=30).mean()
            df["volume_regime"] = _safe_div(volume, vol_ma60, fill=1.0)

        # --- vol_ratio_short_long: rolling_std_10 / rolling_std_30 ---
        if "vol_ratio_short_long" not in df.columns:
            df["vol_ratio_short_long"] = _safe_div(
                df["rolling_std_10"], df["rolling_std_30"], fill=1.0
            )

        return df

    # ════════════════════════════════════════════════════════════════════
    # Normalisation (no look-ahead)
    # ════════════════════════════════════════════════════════════════════

    @staticmethod
    def normalize_features(
        df: pd.DataFrame,
        train_mask: pd.Series | np.ndarray,
        feature_cols: list[str] | None = None,
        clip_value: float = 5.0,
    ) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
        """Z-score normalise *feature_cols* using **only** training-period stats.

        Parameters
        ----------
        df : pd.DataFrame
            Full DataFrame (train + val + test).
        train_mask : array-like of bool
            Boolean mask of the same length as *df*; ``True`` for rows in the
            training period.
        feature_cols : list[str] | None
            Columns to normalise.  Defaults to all expert + router features.
        clip_value : float
            Clip normalised values to [-clip, +clip] to limit outliers.

        Returns
        -------
        (normalised_df, stats_dict)
            ``stats_dict`` maps each column to ``(mean, std)`` computed on
            the training set, allowing re-application at inference time.
        """
        if feature_cols is None:
            feature_cols = [
                c for c in EXPERT_FEATURE_NAMES + ROUTER_FEATURE_NAMES
                if c in df.columns
            ]

        df = df.copy()
        train_mask = np.asarray(train_mask, dtype=bool)
        stats: dict[str, tuple[float, float]] = {}

        for col in feature_cols:
            if col not in df.columns:
                logger.warning(
                    "Normalise: column '%s' not found — skipping.", col
                )
                continue

            train_vals = df.loc[train_mask, col].dropna()
            mu = float(train_vals.mean()) if len(train_vals) else 0.0
            sigma = float(train_vals.std()) if len(train_vals) else 1.0
            if sigma < 1e-12:
                sigma = 1.0  # prevent division by zero
            stats[col] = (mu, sigma)
            df[col] = ((df[col] - mu) / sigma).clip(-clip_value, clip_value)

        logger.info(
            "Normalised %d features using %d training rows.",
            len(stats),
            int(train_mask.sum()),
        )
        return df, stats
