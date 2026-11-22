"""Walk-forward backtesting engine for the MoE swing-trading system.

Simulates realistic trading on Indian equity markets with:
    - Walk-forward cross-validation (train on fold N, test on fold N+1)
    - Sequential day-by-day signal generation (no look-ahead bias)
    - Full transaction cost modelling (IndianMarketCosts)
    - Position sizing via PositionSizer
    - Stop-loss management via StopLossManager
    - Portfolio-level circuit breaker protection
    - Expert utilization tracking for MoE diagnostic

All monetary values in INR (₹).  Position sizes are integer shares.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Final, Protocol, Sequence

import numpy as np
import pandas as pd

from engine.backtesting.costs import IndianMarketCosts
from engine.backtesting.metrics import PerformanceMetrics
from engine.risk.circuit_breaker import CircuitBreaker
from engine.risk.position_sizer import PositionSizer
from engine.risk.stop_loss import StopLossManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols — loose coupling to the model layer
# ---------------------------------------------------------------------------
class TradingModel(Protocol):
    """Minimal interface that a model must satisfy for the backtester.

    The MoE engine (or any replacement) simply needs to implement
    ``train`` and ``predict``.
    """

    def train(self, train_data: pd.DataFrame) -> None:
        """Train / fine-tune on the given data."""
        ...

    def predict(self, row: pd.Series) -> dict[str, Any]:
        """Generate a trading signal for a single bar.

        Must return a dict with at least:
            - ``"signal"``: ``"buy"`` | ``"sell"`` | ``"hold"``
            - ``"confidence"``: float in [0, 1]
            - ``"expert_weights"``: dict[str, float] (optional, for MoE)
        """
        ...


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Trade:
    """Record of a completed (closed) trade."""

    entry_date: date | str
    exit_date: date | str
    symbol: str
    entry_price: float
    exit_price: float
    quantity: int
    direction: str  # "long" | "short"
    pnl: float
    costs: float
    exit_reason: str  # "signal", "stop_loss", "trailing_stop", "time_stop"

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for the metrics module."""
        return {
            "entry_date": str(self.entry_date),
            "exit_date": str(self.exit_date),
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "direction": self.direction,
            "pnl": self.pnl,
            "costs": self.costs,
            "exit_reason": self.exit_reason,
        }


@dataclass(slots=True)
class Position:
    """An open (live) position being tracked by the backtester."""

    symbol: str
    direction: str
    entry_date: date | str
    entry_price: float
    quantity: int
    stop_price: float
    costs_in: float  # transaction costs paid on entry

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_date": str(self.entry_date),
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "stop_price": self.stop_price,
        }


@dataclass
class FoldResult:
    """Results for a single walk-forward fold."""

    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    trades: list[Trade] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    daily_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    expert_weights_history: list[dict[str, float]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Aggregate results across all walk-forward folds."""

    fold_results: list[FoldResult] = field(default_factory=list)
    aggregate_metrics: dict[str, Any] = field(default_factory=dict)
    all_trades: list[Trade] = field(default_factory=list)
    full_equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    benchmark_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    expert_utilization: dict[str, list[float]] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default backtesting configuration
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG: Final[dict[str, Any]] = {
    "initial_capital": 1_000_000.0,  # ₹10 lakh
    "symbol": "NIFTY50",
    "price_col": "close",
    "high_col": "high",
    "low_col": "low",
    "atr_col": "atr",
    "date_col": "date",
    "allow_short": False,  # Shorting delivery is complex in India
    "max_open_positions": 5,
    "slippage_pct": 0.0005,  # 0.05 % slippage per side
}


# ---------------------------------------------------------------------------
# Walk-Forward Backtester
# ---------------------------------------------------------------------------
class WalkForwardBacktester:
    """Walk-forward backtesting engine for MoE swing-trading strategies.

    Processes data fold-by-fold: trains on the training split, then
    iterates through the test split day-by-day, generating signals,
    managing positions, and tracking equity.

    Parameters
    ----------
    model : TradingModel
        The trading model (MoE engine or similar) implementing ``train``
        and ``predict``.
    cost_model : IndianMarketCosts
        Indian market transaction cost calculator.
    risk_manager : StopLossManager
        Stop-loss strategy manager.
    position_sizer : PositionSizer
        Position sizing calculator.
    circuit_breaker : CircuitBreaker
        Portfolio-level circuit breaker.
    config : dict[str, Any] | None
        Backtesting configuration overrides.
    """

    __slots__ = (
        "_model",
        "_cost_model",
        "_risk_manager",
        "_position_sizer",
        "_circuit_breaker",
        "_config",
        "_capital",
        "_positions",
        "_peak_value",
    )

    def __init__(
        self,
        model: TradingModel,
        cost_model: IndianMarketCosts | None = None,
        risk_manager: StopLossManager | None = None,
        position_sizer: PositionSizer | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._cost_model = cost_model or IndianMarketCosts()
        self._risk_manager = risk_manager or StopLossManager()
        self._position_sizer = position_sizer or PositionSizer()
        self._circuit_breaker = circuit_breaker or CircuitBreaker()
        self._config = {**_DEFAULT_CONFIG, **(config or {})}

        # Mutable simulation state (reset per run)
        self._capital: float = 0.0
        self._positions: list[Position] = []
        self._peak_value: float = 0.0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, folds: list[dict[str, Any]]) -> BacktestResult:
        """Execute the full walk-forward backtest.

        Parameters
        ----------
        folds : list[dict]
            Each fold dict must contain:
                - ``"train"`` : ``pd.DataFrame`` — training data.
                - ``"test"``  : ``pd.DataFrame`` — out-of-sample test data.
            Both DataFrames must contain columns matching the config
            (``price_col``, ``date_col``, ``atr_col``, etc.).

        Returns
        -------
        BacktestResult
            Aggregate results across all folds.
        """
        if not folds:
            logger.warning("No folds provided — returning empty result.")
            return BacktestResult(config=self._config)

        # -- Reset simulation state -------------------------------------
        self._capital = float(self._config["initial_capital"])
        self._positions = []
        self._peak_value = self._capital

        result = BacktestResult(config=copy.deepcopy(self._config))
        all_equity_points: list[float] = [self._capital]
        all_benchmark_points: list[float] = []

        for fold_idx, fold in enumerate(folds):
            logger.info(
                "=== Fold %d / %d ===",
                fold_idx + 1,
                len(folds),
            )
            fold_result = self._run_fold(fold_idx, fold)
            result.fold_results.append(fold_result)
            result.all_trades.extend(fold_result.trades)

            # Concatenate equity curves (skip duplicate initial point)
            if len(fold_result.equity_curve) > 1:
                all_equity_points.extend(fold_result.equity_curve[1:].tolist())

            # Aggregate expert weights
            for ew in fold_result.expert_weights_history:
                for expert, weight in ew.items():
                    result.expert_utilization.setdefault(expert, []).append(weight)

        # -- Build aggregate results ------------------------------------
        result.full_equity_curve = np.array(all_equity_points, dtype=np.float64)

        # Build benchmark (buy-and-hold on the test data of all folds)
        if all_benchmark_points:
            result.benchmark_curve = np.array(all_benchmark_points, dtype=np.float64)

        # Generate aggregate report
        trade_dicts = [t.as_dict() for t in result.all_trades]
        result.aggregate_metrics = PerformanceMetrics.generate_report(
            trades=trade_dicts,
            equity_curve=result.full_equity_curve,
            benchmark_curve=result.benchmark_curve if len(result.benchmark_curve) > 0 else None,
        )

        self._log_summary(result)
        return result

    # ------------------------------------------------------------------
    # Per-fold execution
    # ------------------------------------------------------------------

    def _run_fold(
        self,
        fold_idx: int,
        fold: dict[str, Any],
    ) -> FoldResult:
        """Train on fold training data, then simulate on test data."""
        train_df: pd.DataFrame = fold["train"]
        test_df: pd.DataFrame = fold["test"]
        date_col = self._config["date_col"]

        fold_result = FoldResult(
            fold_idx=fold_idx,
            train_start=str(train_df[date_col].iloc[0]) if len(train_df) > 0 else "",
            train_end=str(train_df[date_col].iloc[-1]) if len(train_df) > 0 else "",
            test_start=str(test_df[date_col].iloc[0]) if len(test_df) > 0 else "",
            test_end=str(test_df[date_col].iloc[-1]) if len(test_df) > 0 else "",
        )

        # -- 1. Train ---------------------------------------------------
        logger.info(
            "Training on %d bars (%s → %s)",
            len(train_df),
            fold_result.train_start,
            fold_result.train_end,
        )
        self._model.train(train_df)

        # -- 2. Simulate test period day by day --------------------------
        equity_curve: list[float] = [self._portfolio_value(self._get_price(test_df.iloc[0]))]

        for i in range(len(test_df)):
            row = test_df.iloc[i]
            current_price = self._get_price(row)
            current_date = self._get_date(row)
            atr = self._get_atr(row)

            # Reset daily circuit-breaker counters
            self._circuit_breaker.reset_daily()

            # -- Check existing position stops --------------------------
            closed_trades = self._apply_stops(current_price, current_date, atr)
            fold_result.trades.extend(closed_trades)

            # -- Generate signal ----------------------------------------
            prediction = self._model.predict(row)
            signal = prediction.get("signal", "hold")
            confidence = prediction.get("confidence", 0.0)
            expert_weights = prediction.get("expert_weights", {})
            fold_result.expert_weights_history.append(expert_weights)

            # -- Process signal (open / close positions) ----------------
            new_trades = self._simulate_day(
                current_date, signal, confidence, current_price, atr
            )
            fold_result.trades.extend(new_trades)

            # -- Update daily PnL & circuit breaker ---------------------
            portfolio_value = self._portfolio_value(current_price)
            daily_pnl = portfolio_value - equity_curve[-1]
            self._peak_value = max(self._peak_value, portfolio_value)

            # Determine if last trade was a win or loss
            trade_result: str | None = None
            if new_trades or closed_trades:
                last_trade = (new_trades or closed_trades)[-1]
                trade_result = "win" if last_trade.pnl > 0 else "loss"

            self._circuit_breaker.update(
                daily_pnl=daily_pnl,
                portfolio_value=portfolio_value,
                peak_value=self._peak_value,
                trade_result=trade_result,
            )

            equity_curve.append(portfolio_value)

        fold_result.equity_curve = np.array(equity_curve, dtype=np.float64)

        # -- Per-fold metrics -------------------------------------------
        trade_dicts = [t.as_dict() for t in fold_result.trades]
        fold_result.metrics = PerformanceMetrics.generate_report(
            trades=trade_dicts,
            equity_curve=fold_result.equity_curve,
        )

        logger.info(
            "Fold %d complete: %d trades, return=%.2f%%, Sharpe=%.2f",
            fold_idx,
            len(fold_result.trades),
            fold_result.metrics.get("summary", {}).get("total_return", 0.0) * 100,
            fold_result.metrics.get("risk_adjusted", {}).get("sharpe_ratio", 0.0),
        )
        return fold_result

    # ------------------------------------------------------------------
    # Day simulation
    # ------------------------------------------------------------------

    def _simulate_day(
        self,
        current_date: date | str,
        signal: str,
        confidence: float,
        current_price: float,
        atr: float,
    ) -> list[Trade]:
        """Simulate one trading day: process signal, open/close positions.

        Parameters
        ----------
        current_date : date | str
            The current bar's date.
        signal : str
            Model signal: ``"buy"``, ``"sell"``, or ``"hold"``.
        confidence : float
            Model confidence [0, 1].
        current_price : float
            Closing price for the bar.
        atr : float
            Current ATR value.

        Returns
        -------
        list[Trade]
            Trades closed on this day (if any).
        """
        trades: list[Trade] = []
        symbol = self._config["symbol"]

        # -- Check circuit breaker --------------------------------------
        can_trade, cb_reason = self._circuit_breaker.can_trade()
        cb_scale = self._circuit_breaker.get_position_scale()

        # -- Close existing positions on opposite signal ----------------
        if signal == "sell" and self._has_long_position(symbol):
            trade = self._close_position(symbol, "long", current_price, current_date, "signal")
            if trade is not None:
                trades.append(trade)

        elif signal == "buy" and self._has_short_position(symbol):
            trade = self._close_position(symbol, "short", current_price, current_date, "signal")
            if trade is not None:
                trades.append(trade)

        # -- Open new position if signal and circuit breaker permit -----
        if not can_trade and cb_scale == 0.0:
            logger.debug(
                "Circuit breaker blocking new trades: %s", cb_reason
            )
            return trades

        max_positions = self._config["max_open_positions"]
        if len(self._positions) >= max_positions:
            return trades

        if signal == "buy" and not self._has_long_position(symbol):
            trade = self._open_position(
                symbol, "long", current_price, current_date,
                confidence, atr, cb_scale,
            )
            if trade is not None:
                trades.append(trade)

        elif (
            signal == "sell"
            and self._config.get("allow_short", False)
            and not self._has_short_position(symbol)
        ):
            trade = self._open_position(
                symbol, "short", current_price, current_date,
                confidence, atr, cb_scale,
            )
            if trade is not None:
                trades.append(trade)

        return trades

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _open_position(
        self,
        symbol: str,
        direction: str,
        price: float,
        current_date: date | str,
        confidence: float,
        atr: float,
        cb_scale: float,
    ) -> Trade | None:
        """Open a new position, applying slippage and sizing."""
        # -- Apply slippage ---------------------------------------------
        slippage = price * self._config["slippage_pct"]
        entry_price = price + slippage if direction == "long" else price - slippage

        # -- Calculate stop price ---------------------------------------
        stop_price = self._risk_manager.calculate_initial_stop(
            entry_price, direction, atr=atr if atr > 0 else None,
        )

        # -- Size the position ------------------------------------------
        quantity = self._position_sizer.calculate_size(
            capital=self._capital,
            entry_price=entry_price,
            stop_price=stop_price,
            confidence=confidence,
            atr=atr if atr > 0 else None,
        )

        # Apply circuit-breaker scaling
        if cb_scale < 1.0:
            quantity = int(quantity * cb_scale)

        if quantity <= 0:
            logger.debug(
                "Position size is 0 for %s %s at ₹%.2f — skipping.",
                direction,
                symbol,
                entry_price,
            )
            return None

        # -- Calculate entry costs --------------------------------------
        buy_costs = self._cost_model.calculate_buy_costs(entry_price, quantity)
        total_entry_cost = entry_price * quantity + buy_costs.total

        # -- Check we have enough capital -------------------------------
        if total_entry_cost > self._capital:
            # Reduce quantity to fit capital
            affordable = int((self._capital * 0.99) / entry_price)  # 1% buffer
            quantity = max(0, affordable)
            if quantity <= 0:
                logger.debug(
                    "Insufficient capital (₹%.0f) for %s %s at ₹%.2f.",
                    self._capital,
                    direction,
                    symbol,
                    entry_price,
                )
                return None
            buy_costs = self._cost_model.calculate_buy_costs(entry_price, quantity)
            total_entry_cost = entry_price * quantity + buy_costs.total

        # -- Deduct capital and open position ---------------------------
        self._capital -= total_entry_cost

        position = Position(
            symbol=symbol,
            direction=direction,
            entry_date=current_date,
            entry_price=entry_price,
            quantity=quantity,
            stop_price=stop_price,
            costs_in=buy_costs.total,
        )
        self._positions.append(position)

        logger.info(
            "OPEN %s %s: %d shares @ ₹%.2f (stop=₹%.2f, cost=₹%.2f)",
            direction.upper(),
            symbol,
            quantity,
            entry_price,
            stop_price,
            buy_costs.total,
        )
        return None  # No completed trade yet; will be a Trade on close

    def _close_position(
        self,
        symbol: str,
        direction: str,
        current_price: float,
        current_date: date | str,
        exit_reason: str,
    ) -> Trade | None:
        """Close an existing position, calculate PnL, return Trade."""
        pos = self._find_position(symbol, direction)
        if pos is None:
            return None

        # -- Apply slippage on exit -------------------------------------
        slippage = current_price * self._config["slippage_pct"]
        exit_price = current_price - slippage if direction == "long" else current_price + slippage

        # -- Calculate sell costs ---------------------------------------
        sell_costs = self._cost_model.calculate_sell_costs(exit_price, pos.quantity)
        total_costs = pos.costs_in + sell_costs.total

        # -- Gross and net PnL ------------------------------------------
        if direction == "long":
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:  # short
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity

        net_pnl = gross_pnl - total_costs

        # -- Return capital + proceeds ----------------------------------
        proceeds = exit_price * pos.quantity - sell_costs.total
        self._capital += proceeds

        # -- Remove from open positions ---------------------------------
        self._positions.remove(pos)

        trade = Trade(
            entry_date=pos.entry_date,
            exit_date=current_date,
            symbol=symbol,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            direction=direction,
            pnl=round(net_pnl, 2),
            costs=round(total_costs, 2),
            exit_reason=exit_reason,
        )

        logger.info(
            "CLOSE %s %s: %d shares @ ₹%.2f → ₹%.2f, PnL=₹%.2f (%s)",
            direction.upper(),
            symbol,
            pos.quantity,
            pos.entry_price,
            exit_price,
            net_pnl,
            exit_reason,
        )
        return trade

    def _apply_stops(
        self,
        current_price: float,
        current_date: date | str,
        atr: float,
    ) -> list[Trade]:
        """Check stop conditions on all open positions.

        Parameters
        ----------
        current_price : float
            Current bar's closing price.
        current_date : date | str
            Current bar's date.
        atr : float
            Current ATR.

        Returns
        -------
        list[Trade]
            Trades closed by stop-loss, trailing stop, or time stop.
        """
        trades: list[Trade] = []

        # Iterate over a copy because _close_position mutates the list
        for pos in list(self._positions):
            stop_signal = self._risk_manager.check_stops(
                position=pos.as_dict(),
                current_price=current_price,
                current_date=current_date,
                atr=atr,
            )

            if stop_signal != "hold":
                trade = self._close_position(
                    pos.symbol,
                    pos.direction,
                    current_price,
                    current_date,
                    exit_reason=stop_signal,
                )
                if trade is not None:
                    trades.append(trade)

                    # Update trailing stop back to position if it survived
                    # (already handled in check_stops mutating the dict)

        return trades

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _portfolio_value(self, current_price: float) -> float:
        """Total portfolio value = cash + mark-to-market positions."""
        position_value = 0.0
        for pos in self._positions:
            if pos.direction == "long":
                position_value += current_price * pos.quantity
            else:  # short: value = 2 * entry_value - current_value
                position_value += (2 * pos.entry_price - current_price) * pos.quantity
        return self._capital + position_value

    def _get_price(self, row: pd.Series) -> float:
        """Extract price from a DataFrame row."""
        col = self._config["price_col"]
        return float(row[col]) if col in row.index else 0.0

    def _get_date(self, row: pd.Series) -> date | str:
        """Extract date from a DataFrame row."""
        col = self._config["date_col"]
        val = row[col] if col in row.index else ""
        if isinstance(val, pd.Timestamp):
            return val.date()
        return val

    def _get_atr(self, row: pd.Series) -> float:
        """Extract ATR from a DataFrame row, with fallback."""
        col = self._config["atr_col"]
        if col in row.index:
            val = float(row[col])
            return val if val > 0 else 0.0
        return 0.0

    def _has_long_position(self, symbol: str) -> bool:
        return any(
            p.symbol == symbol and p.direction == "long"
            for p in self._positions
        )

    def _has_short_position(self, symbol: str) -> bool:
        return any(
            p.symbol == symbol and p.direction == "short"
            for p in self._positions
        )

    def _find_position(self, symbol: str, direction: str) -> Position | None:
        for p in self._positions:
            if p.symbol == symbol and p.direction == direction:
                return p
        return None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_summary(self, result: BacktestResult) -> None:
        """Log a high-level summary of the backtest."""
        metrics = result.aggregate_metrics
        summary = metrics.get("summary", {})
        risk = metrics.get("risk_adjusted", {})
        dd = metrics.get("drawdown", {})
        trade_stats = metrics.get("trade_statistics", {})

        logger.info("=" * 60)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 60)
        logger.info("Total trades:     %d", summary.get("total_trades", 0))
        logger.info("Total return:     %.2f%%", summary.get("total_return", 0) * 100)
        logger.info("CAGR:             %.2f%%", summary.get("cagr", 0) * 100)
        logger.info("Sharpe ratio:     %.2f", risk.get("sharpe_ratio", 0))
        logger.info("Sortino ratio:    %.2f", risk.get("sortino_ratio", 0))
        logger.info("Max drawdown:     %.2f%%", dd.get("max_drawdown_pct", 0) * 100)
        logger.info("Win rate:         %.1f%%", trade_stats.get("win_rate", 0) * 100)
        logger.info("Profit factor:    %.2f", trade_stats.get("profit_factor", 0))
        logger.info("Total costs:      ₹%.2f", trade_stats.get("total_costs", 0))
        logger.info("=" * 60)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"WalkForwardBacktester("
            f"model={type(self._model).__name__}, "
            f"capital=₹{self._config['initial_capital']:,.0f}, "
            f"max_positions={self._config['max_open_positions']})"
        )
