"""Performance metrics calculator for backtesting results.

Provides annualized risk-adjusted returns, drawdown analysis, trade-level
statistics, and comprehensive report generation with benchmark comparison.

All return-series inputs are expected as *daily* simple returns (e.g. 0.01
for a +1 % day).  The Indian risk-free rate default is 6.5 % (91-day T-bill
proxy).  Trading days per year = 252 (NSE calendar average).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Final, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TRADING_DAYS_PER_YEAR: Final[int] = 252
_DEFAULT_RISK_FREE_RATE: Final[float] = 0.065  # 6.5 % — Indian T-bill proxy


# ---------------------------------------------------------------------------
# Trade typing helper
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class TradeRecord:
    """Canonical representation of a completed trade.

    Attributes are intentionally kept as a plain dict-compatible set so
    that the metrics code can accept either ``TradeRecord`` instances or
    raw ``dict`` objects.
    """

    entry_date: date | str
    exit_date: date | str
    entry_price: float
    exit_price: float
    quantity: int
    direction: str  # "long" | "short"
    pnl: float
    costs: float


# ---------------------------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------------------------
class PerformanceMetrics:
    """Static-method collection of backtesting performance metrics.

    Every method is a ``@staticmethod`` so that they can be called without
    instantiation, keeping the API ergonomic for both interactive analysis
    and automated report generation.
    """

    # ------------------------------------------------------------------ #
    # Risk-adjusted return metrics
    # ------------------------------------------------------------------ #

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = _DEFAULT_RISK_FREE_RATE,
    ) -> float:
        """Annualized Sharpe ratio.

        .. math::
            SR = \\frac{\\bar{r} - r_f / 252}{\\sigma_r} \\times \\sqrt{252}

        Parameters
        ----------
        returns : np.ndarray
            Array of daily simple returns.
        risk_free_rate : float
            Annual risk-free rate (default 0.065 for India).

        Returns
        -------
        float
            Annualized Sharpe ratio.  Returns 0.0 when standard deviation
            is zero or the array is empty.
        """
        if len(returns) < 2:
            return 0.0

        daily_rf = risk_free_rate / _TRADING_DAYS_PER_YEAR
        excess = returns - daily_rf
        std = float(np.std(excess, ddof=1))
        if std == 0.0:
            return 0.0

        mean_excess = float(np.mean(excess))
        sharpe = (mean_excess / std) * math.sqrt(_TRADING_DAYS_PER_YEAR)
        return round(sharpe, 4)

    @staticmethod
    def sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = _DEFAULT_RISK_FREE_RATE,
    ) -> float:
        """Annualized Sortino ratio (downside-deviation denominator).

        Parameters
        ----------
        returns : np.ndarray
            Daily simple returns.
        risk_free_rate : float
            Annual risk-free rate.

        Returns
        -------
        float
            Annualized Sortino ratio.  Returns 0.0 when there is no
            downside deviation or the array is empty.
        """
        if len(returns) < 2:
            return 0.0

        daily_rf = risk_free_rate / _TRADING_DAYS_PER_YEAR
        excess = returns - daily_rf
        downside = excess[excess < 0.0]

        if len(downside) == 0:
            return 0.0  # no downside => undefined; return 0 conservatively

        downside_std = float(np.std(downside, ddof=1))
        if downside_std == 0.0:
            return 0.0

        mean_excess = float(np.mean(excess))
        sortino = (mean_excess / downside_std) * math.sqrt(_TRADING_DAYS_PER_YEAR)
        return round(sortino, 4)

    @staticmethod
    def calmar_ratio(returns: np.ndarray) -> float:
        """Calmar ratio — annualized return / max drawdown.

        Parameters
        ----------
        returns : np.ndarray
            Daily simple returns.

        Returns
        -------
        float
            Calmar ratio.  Returns 0.0 when max drawdown is zero or array
            has fewer than 2 elements.
        """
        if len(returns) < 2:
            return 0.0

        equity = np.cumprod(1.0 + returns)
        num_years = len(returns) / _TRADING_DAYS_PER_YEAR
        if num_years <= 0.0:
            return 0.0

        cagr = PerformanceMetrics.cagr(equity, num_years)
        max_dd, _, _ = PerformanceMetrics.max_drawdown(equity)
        if max_dd == 0.0:
            return 0.0

        calmar = cagr / abs(max_dd)
        return round(calmar, 4)

    # ------------------------------------------------------------------ #
    # Drawdown analysis
    # ------------------------------------------------------------------ #

    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> tuple[float, int, int]:
        """Maximum peak-to-trough drawdown.

        Parameters
        ----------
        equity_curve : np.ndarray
            Absolute equity values (not returns).  Must be > 0 everywhere.

        Returns
        -------
        tuple[float, int, int]
            ``(max_dd_pct, peak_idx, trough_idx)`` where ``max_dd_pct``
            is a *negative* fraction (e.g. -0.12 for a 12 % drawdown).
            Returns ``(0.0, 0, 0)`` for empty / single-element arrays.
        """
        if len(equity_curve) < 2:
            return (0.0, 0, 0)

        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max

        trough_idx = int(np.argmin(drawdowns))
        peak_idx = int(np.argmax(equity_curve[:trough_idx + 1]))

        max_dd_pct = float(drawdowns[trough_idx])
        return (round(max_dd_pct, 6), peak_idx, trough_idx)

    @staticmethod
    def drawdown_series(equity_curve: np.ndarray) -> np.ndarray:
        """Full drawdown time series (all values ≤ 0).

        Parameters
        ----------
        equity_curve : np.ndarray
            Absolute equity values.

        Returns
        -------
        np.ndarray
            Drawdown fraction at every point in time.
        """
        if len(equity_curve) == 0:
            return np.array([], dtype=np.float64)

        running_max = np.maximum.accumulate(equity_curve)
        return (equity_curve - running_max) / np.where(
            running_max == 0.0, 1.0, running_max
        )

    # ------------------------------------------------------------------ #
    # Trade-level statistics
    # ------------------------------------------------------------------ #

    @staticmethod
    def win_rate(trades: Sequence[dict[str, Any]]) -> float:
        """Fraction of trades with positive PnL.

        Parameters
        ----------
        trades : list[dict]
            Each dict must contain a ``"pnl"`` key.

        Returns
        -------
        float
            Win rate in [0, 1].  Returns 0.0 for empty trade list.
        """
        if not trades:
            return 0.0

        winners = sum(1 for t in trades if t["pnl"] > 0.0)
        return round(winners / len(trades), 4)

    @staticmethod
    def profit_factor(trades: Sequence[dict[str, Any]]) -> float:
        """Ratio of gross profits to gross losses.

        Parameters
        ----------
        trades : list[dict]
            Each dict must contain a ``"pnl"`` key.

        Returns
        -------
        float
            Profit factor (≥ 0).  Returns ``float('inf')`` if there are
            no losing trades, 0.0 if there are no winning trades, and
            0.0 for empty lists.
        """
        if not trades:
            return 0.0

        gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0.0)
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0.0))

        if gross_loss == 0.0:
            return float("inf") if gross_profit > 0.0 else 0.0

        return round(gross_profit / gross_loss, 4)

    @staticmethod
    def average_win_loss_ratio(trades: Sequence[dict[str, Any]]) -> float:
        """Average winning trade / average losing trade (absolute values).

        Returns
        -------
        float
            Reward-to-risk ratio.  Returns 0.0 when either side is empty.
        """
        winners = [t["pnl"] for t in trades if t["pnl"] > 0.0]
        losers = [t["pnl"] for t in trades if t["pnl"] < 0.0]

        if not winners or not losers:
            return 0.0

        avg_win = sum(winners) / len(winners)
        avg_loss = abs(sum(losers) / len(losers))
        if avg_loss == 0.0:
            return 0.0

        return round(avg_win / avg_loss, 4)

    @staticmethod
    def expectancy(trades: Sequence[dict[str, Any]]) -> float:
        """Per-trade expectancy (expected value of PnL per trade).

        .. math::
            E = (win\\_rate \\times avg\\_win) - (loss\\_rate \\times avg\\_loss)

        Returns
        -------
        float
            Expected PnL per trade in ₹.  Returns 0.0 for empty lists.
        """
        if not trades:
            return 0.0
        return round(sum(t["pnl"] for t in trades) / len(trades), 2)

    @staticmethod
    def max_consecutive_losses(trades: Sequence[dict[str, Any]]) -> int:
        """Longest streak of consecutive losing trades.

        Returns
        -------
        int
            Length of the longest losing streak.  Returns 0 for empty lists.
        """
        if not trades:
            return 0

        max_streak = 0
        current_streak = 0
        for t in trades:
            if t["pnl"] < 0.0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    # ------------------------------------------------------------------ #
    # Growth metrics
    # ------------------------------------------------------------------ #

    @staticmethod
    def cagr(equity_curve: np.ndarray, num_years: float) -> float:
        """Compound Annual Growth Rate.

        Parameters
        ----------
        equity_curve : np.ndarray
            Absolute equity values.  ``equity_curve[0]`` is the starting
            capital.
        num_years : float
            Duration in years.

        Returns
        -------
        float
            CAGR as a fraction (e.g. 0.15 for 15 %).  Returns 0.0 if
            the curve is degenerate.
        """
        if len(equity_curve) < 2 or num_years <= 0.0:
            return 0.0

        start = float(equity_curve[0])
        end = float(equity_curve[-1])
        if start <= 0.0:
            return 0.0

        cagr_val = (end / start) ** (1.0 / num_years) - 1.0
        return round(cagr_val, 6)

    @staticmethod
    def total_return(equity_curve: np.ndarray) -> float:
        """Simple total return over the entire equity curve.

        Returns
        -------
        float
            Total return as a fraction.  Returns 0.0 for degenerate input.
        """
        if len(equity_curve) < 2:
            return 0.0

        start = float(equity_curve[0])
        if start <= 0.0:
            return 0.0

        return round((float(equity_curve[-1]) / start) - 1.0, 6)

    # ------------------------------------------------------------------ #
    # Comprehensive report
    # ------------------------------------------------------------------ #

    @staticmethod
    def generate_report(
        trades: Sequence[dict[str, Any]],
        equity_curve: np.ndarray,
        benchmark_curve: np.ndarray | None = None,
        risk_free_rate: float = _DEFAULT_RISK_FREE_RATE,
    ) -> dict[str, Any]:
        """Generate a comprehensive performance report.

        Includes all risk-adjusted metrics, trade-level statistics, and
        (optionally) a comparison against a buy-and-hold benchmark.

        Parameters
        ----------
        trades : list[dict]
            Completed trades, each with keys: ``entry_date``,
            ``exit_date``, ``entry_price``, ``exit_price``, ``quantity``,
            ``direction``, ``pnl``, ``costs``.
        equity_curve : np.ndarray
            Strategy equity curve (absolute values).
        benchmark_curve : np.ndarray | None
            Buy-and-hold benchmark equity curve for comparison.
        risk_free_rate : float
            Annual risk-free rate for Sharpe / Sortino.

        Returns
        -------
        dict[str, Any]
            Nested dictionary with all computed metrics.
        """
        pm = PerformanceMetrics

        # -- Derive daily returns from equity curve ----------------------
        if len(equity_curve) < 2:
            daily_returns = np.array([], dtype=np.float64)
        else:
            daily_returns = np.diff(equity_curve) / equity_curve[:-1]
            # Guard against NaN / Inf from zero-valued equity
            daily_returns = np.nan_to_num(daily_returns, nan=0.0, posinf=0.0, neginf=0.0)

        num_days = len(equity_curve)
        num_years = num_days / _TRADING_DAYS_PER_YEAR if num_days > 0 else 0.0

        # -- Max drawdown ------------------------------------------------
        max_dd, peak_idx, trough_idx = pm.max_drawdown(equity_curve)

        # -- Build report ------------------------------------------------
        report: dict[str, Any] = {
            "summary": {
                "total_trades": len(trades),
                "total_return": pm.total_return(equity_curve),
                "cagr": pm.cagr(equity_curve, num_years),
                "num_trading_days": num_days,
                "num_years": round(num_years, 2),
            },
            "risk_adjusted": {
                "sharpe_ratio": pm.sharpe_ratio(daily_returns, risk_free_rate),
                "sortino_ratio": pm.sortino_ratio(daily_returns, risk_free_rate),
                "calmar_ratio": pm.calmar_ratio(daily_returns),
            },
            "drawdown": {
                "max_drawdown_pct": max_dd,
                "peak_idx": peak_idx,
                "trough_idx": trough_idx,
            },
            "trade_statistics": {
                "win_rate": pm.win_rate(trades),
                "profit_factor": pm.profit_factor(trades),
                "avg_win_loss_ratio": pm.average_win_loss_ratio(trades),
                "expectancy_per_trade": pm.expectancy(trades),
                "max_consecutive_losses": pm.max_consecutive_losses(trades),
                "total_costs": round(
                    sum(t.get("costs", 0.0) for t in trades), 2
                ),
                "total_gross_pnl": round(
                    sum(t.get("pnl", 0.0) for t in trades), 2
                ),
            },
        }

        # -- Benchmark comparison ----------------------------------------
        if benchmark_curve is not None and len(benchmark_curve) >= 2:
            bm_returns = np.diff(benchmark_curve) / benchmark_curve[:-1]
            bm_returns = np.nan_to_num(bm_returns, nan=0.0, posinf=0.0, neginf=0.0)
            bm_num_years = len(benchmark_curve) / _TRADING_DAYS_PER_YEAR

            bm_max_dd, bm_peak, bm_trough = pm.max_drawdown(benchmark_curve)

            report["benchmark"] = {
                "total_return": pm.total_return(benchmark_curve),
                "cagr": pm.cagr(benchmark_curve, bm_num_years),
                "sharpe_ratio": pm.sharpe_ratio(bm_returns, risk_free_rate),
                "max_drawdown_pct": bm_max_dd,
            }

            # Alpha & beta (simple OLS)
            if len(daily_returns) == len(bm_returns) and len(daily_returns) > 1:
                cov_matrix = np.cov(daily_returns, bm_returns)
                bm_var = cov_matrix[1, 1]
                beta = float(cov_matrix[0, 1] / bm_var) if bm_var != 0.0 else 0.0
                alpha_daily = float(np.mean(daily_returns)) - beta * float(
                    np.mean(bm_returns)
                )
                alpha_annual = alpha_daily * _TRADING_DAYS_PER_YEAR

                report["benchmark"]["beta"] = round(beta, 4)
                report["benchmark"]["alpha_annual"] = round(alpha_annual, 6)
                report["benchmark"]["information_ratio"] = round(
                    pm.sharpe_ratio(daily_returns - bm_returns, risk_free_rate=0.0),
                    4,
                )

        logger.info(
            "Report generated: %d trades, CAGR=%.2f%%, Sharpe=%.2f, MaxDD=%.2f%%",
            len(trades),
            report["summary"]["cagr"] * 100,
            report["risk_adjusted"]["sharpe_ratio"],
            report["drawdown"]["max_drawdown_pct"] * 100,
        )

        return report
