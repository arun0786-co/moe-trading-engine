"""Portfolio-level circuit breaker for risk containment.

Implements multiple layers of protection that progressively reduce
exposure or halt trading entirely when losses escalate:

    1. **Daily loss limit** — if a single day's loss exceeds X% of
       portfolio value, halt trading for the remainder of the day.
    2. **Weekly drawdown limit** — if cumulative losses in the rolling
       week exceed Y%, scale position sizes to 50 %.
    3. **Max drawdown halt** — if peak-to-trough drawdown exceeds Z%,
       halt *all* trading until manually resumed.
    4. **Consecutive loss limit** — if N consecutive trades lose money,
       pause new entries until a winning trade occurs.

All thresholds are configurable.  Monetary values in INR (₹).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_DAILY_LOSS_LIMIT: Final[float] = 0.02  # 2 % of portfolio
_DEFAULT_WEEKLY_DD_LIMIT: Final[float] = 0.05  # 5 %
_DEFAULT_MAX_DD_HALT: Final[float] = 0.15  # 15 %
_DEFAULT_CONSEC_LOSS_LIMIT: Final[int] = 5


class BreakerState(Enum):
    """Trading permission states for the circuit breaker."""

    ACTIVE = auto()  # Normal operation — full position sizing
    DAILY_HALT = auto()  # Halted for the rest of the trading day
    WEEKLY_WARNING = auto()  # Scaled down (50 %) due to weekly DD
    CONSEC_LOSS_PAUSE = auto()  # Paused due to consecutive losses
    MAX_DD_HALT = auto()  # Full halt — requires manual resume


@dataclass(slots=True)
class BreakerStatus:
    """Snapshot of current circuit-breaker state.

    Returned by :meth:`CircuitBreaker.status` for monitoring dashboards.
    """

    state: BreakerState
    can_trade: bool
    position_scale: float
    reason: str
    daily_pnl: float
    weekly_pnl: float
    current_drawdown: float
    consecutive_losses: int


class CircuitBreaker:
    """Multi-layered portfolio circuit breaker.

    Parameters
    ----------
    daily_loss_limit : float
        Maximum single-day loss as a fraction of portfolio value before
        halting for the day (default 0.02 = 2 %).
    weekly_dd_limit : float
        Maximum cumulative weekly loss as a fraction of portfolio value
        before scaling positions to 50 % (default 0.05 = 5 %).
    max_dd_halt : float
        Maximum peak-to-trough drawdown as a fraction before halting
        all trading.  Requires :meth:`manual_resume` to restart
        (default 0.15 = 15 %).
    consec_loss_limit : int
        Number of consecutive losing trades before pausing new entries
        (default 5).
    """

    __slots__ = (
        "_daily_loss_limit",
        "_weekly_dd_limit",
        "_max_dd_halt",
        "_consec_loss_limit",
        "_state",
        "_daily_pnl",
        "_weekly_daily_pnls",
        "_peak_value",
        "_consecutive_losses",
        "_halt_reason",
    )

    def __init__(
        self,
        daily_loss_limit: float = _DEFAULT_DAILY_LOSS_LIMIT,
        weekly_dd_limit: float = _DEFAULT_WEEKLY_DD_LIMIT,
        max_dd_halt: float = _DEFAULT_MAX_DD_HALT,
        consec_loss_limit: int = _DEFAULT_CONSEC_LOSS_LIMIT,
    ) -> None:
        if daily_loss_limit < 0.0:
            raise ValueError(
                f"daily_loss_limit must be ≥ 0, got {daily_loss_limit}"
            )
        if weekly_dd_limit < 0.0:
            raise ValueError(
                f"weekly_dd_limit must be ≥ 0, got {weekly_dd_limit}"
            )
        if max_dd_halt < 0.0:
            raise ValueError(f"max_dd_halt must be ≥ 0, got {max_dd_halt}")
        if consec_loss_limit < 1:
            raise ValueError(
                f"consec_loss_limit must be ≥ 1, got {consec_loss_limit}"
            )

        self._daily_loss_limit = daily_loss_limit
        self._weekly_dd_limit = weekly_dd_limit
        self._max_dd_halt = max_dd_halt
        self._consec_loss_limit = consec_loss_limit

        # Internal state
        self._state: BreakerState = BreakerState.ACTIVE
        self._daily_pnl: float = 0.0
        # Rolling 5-day (one trading week) history of daily PnLs
        self._weekly_daily_pnls: deque[float] = deque(maxlen=5)
        self._peak_value: float = 0.0
        self._consecutive_losses: int = 0
        self._halt_reason: str = ""

    # ------------------------------------------------------------------
    # State update methods
    # ------------------------------------------------------------------

    def update(
        self,
        daily_pnl: float,
        portfolio_value: float,
        peak_value: float,
        trade_result: str | None = None,
    ) -> None:
        """Update circuit-breaker state after a trading day or trade.

        Should be called at end-of-day with the day's aggregate PnL,
        and optionally after each individual trade with ``trade_result``.

        Parameters
        ----------
        daily_pnl : float
            Aggregate PnL for the current day in ₹ (negative for losses).
        portfolio_value : float
            Current total portfolio value in ₹.
        peak_value : float
            Historical peak portfolio value in ₹.
        trade_result : str | None
            ``"win"`` or ``"loss"`` for per-trade consecutive-loss tracking.
            Pass ``None`` when updating only daily PnL without a new trade.
        """
        # -- Permanent halt is sticky until manual resume ----------------
        if self._state == BreakerState.MAX_DD_HALT:
            return

        self._daily_pnl = daily_pnl
        self._peak_value = max(self._peak_value, peak_value)

        # -- Per-trade consecutive-loss tracking -------------------------
        if trade_result is not None:
            if trade_result == "loss":
                self._consecutive_losses += 1
            elif trade_result == "win":
                self._consecutive_losses = 0

        # -- 1. Max drawdown halt (most severe) --------------------------
        if self._peak_value > 0.0:
            current_dd = (self._peak_value - portfolio_value) / self._peak_value
            if current_dd >= self._max_dd_halt:
                self._state = BreakerState.MAX_DD_HALT
                self._halt_reason = (
                    f"Max drawdown halt: {current_dd:.1%} drawdown "
                    f"exceeds {self._max_dd_halt:.0%} limit. "
                    f"Peak=₹{self._peak_value:,.0f}, "
                    f"Current=₹{portfolio_value:,.0f}"
                )
                logger.critical(self._halt_reason)
                return

        # -- 2. Consecutive-loss pause -----------------------------------
        if self._consecutive_losses >= self._consec_loss_limit:
            self._state = BreakerState.CONSEC_LOSS_PAUSE
            self._halt_reason = (
                f"Consecutive loss pause: {self._consecutive_losses} "
                f"consecutive losing trades (limit={self._consec_loss_limit})"
            )
            logger.warning(self._halt_reason)
            return

        # -- 3. Daily loss limit -----------------------------------------
        if portfolio_value > 0.0:
            daily_loss_pct = abs(min(0.0, daily_pnl)) / portfolio_value
            if daily_loss_pct >= self._daily_loss_limit:
                self._state = BreakerState.DAILY_HALT
                self._halt_reason = (
                    f"Daily loss halt: {daily_loss_pct:.2%} loss "
                    f"exceeds {self._daily_loss_limit:.0%} daily limit"
                )
                logger.warning(self._halt_reason)
                return

        # -- 4. Weekly drawdown warning ----------------------------------
        weekly_pnl = sum(self._weekly_daily_pnls) + daily_pnl
        if portfolio_value > 0.0:
            weekly_loss_pct = abs(min(0.0, weekly_pnl)) / portfolio_value
            if weekly_loss_pct >= self._weekly_dd_limit:
                self._state = BreakerState.WEEKLY_WARNING
                self._halt_reason = (
                    f"Weekly drawdown warning: {weekly_loss_pct:.2%} "
                    f"weekly loss exceeds {self._weekly_dd_limit:.0%} limit "
                    f"— positions scaled to 50%"
                )
                logger.warning(self._halt_reason)
                return

        # -- All clear ---------------------------------------------------
        self._state = BreakerState.ACTIVE
        self._halt_reason = ""

    # ------------------------------------------------------------------
    # Trading permission queries
    # ------------------------------------------------------------------

    def can_trade(self) -> tuple[bool, str]:
        """Check whether new trades are permitted.

        Returns
        -------
        tuple[bool, str]
            ``(allowed, reason)`` — ``reason`` explains which breaker
            tripped (empty string when trading is allowed).
        """
        if self._state == BreakerState.ACTIVE:
            return (True, "")

        if self._state == BreakerState.WEEKLY_WARNING:
            # Trading allowed but at reduced size
            return (True, self._halt_reason)

        # All other states: halted
        return (False, self._halt_reason)

    def get_position_scale(self) -> float:
        """Position-size scaling factor based on current breaker state.

        Returns
        -------
        float
            - ``1.0`` — normal operation.
            - ``0.5`` — weekly drawdown warning (reduce exposure).
            - ``0.0`` — all halted states (no new trades).
        """
        if self._state == BreakerState.ACTIVE:
            return 1.0
        if self._state == BreakerState.WEEKLY_WARNING:
            return 0.5
        return 0.0

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def reset_daily(self) -> None:
        """Reset daily counters at the start of a new trading day.

        Archives the previous day's PnL into the weekly rolling window
        and resets the daily accumulator.  Clears ``DAILY_HALT`` state
        but *not* ``WEEKLY_WARNING``, ``CONSEC_LOSS_PAUSE``, or
        ``MAX_DD_HALT``.
        """
        # Archive previous day's PnL into the rolling week
        self._weekly_daily_pnls.append(self._daily_pnl)
        self._daily_pnl = 0.0

        # Clear daily halt — other states persist across days
        if self._state == BreakerState.DAILY_HALT:
            self._state = BreakerState.ACTIVE
            self._halt_reason = ""
            logger.info("Daily halt cleared — trading resumed for new day.")

        # Re-evaluate consecutive loss pause — if a win was registered,
        # the counter was already reset in update(); if still paused,
        # keep paused.
        if (
            self._state == BreakerState.CONSEC_LOSS_PAUSE
            and self._consecutive_losses < self._consec_loss_limit
        ):
            self._state = BreakerState.ACTIVE
            self._halt_reason = ""
            logger.info("Consecutive-loss pause cleared after winning trade.")

    def manual_resume(self) -> None:
        """Manually resume trading after a max-drawdown halt.

        This is intentionally an *explicit* action to prevent automated
        systems from blindly resuming after catastrophic losses.  The
        operator should review the situation before calling this.
        """
        if self._state != BreakerState.MAX_DD_HALT:
            logger.info(
                "manual_resume called but state is %s — no action needed.",
                self._state.name,
            )
            return

        logger.warning(
            "MANUAL RESUME: Clearing max-drawdown halt. "
            "Previous reason: %s",
            self._halt_reason,
        )
        self._state = BreakerState.ACTIVE
        self._halt_reason = ""
        self._consecutive_losses = 0

    # ------------------------------------------------------------------
    # Status / monitoring
    # ------------------------------------------------------------------

    @property
    def state(self) -> BreakerState:
        """Current breaker state enum."""
        return self._state

    @property
    def consecutive_losses(self) -> int:
        """Current consecutive losing trade count."""
        return self._consecutive_losses

    def status(self, portfolio_value: float = 0.0) -> BreakerStatus:
        """Full status snapshot for monitoring dashboards.

        Parameters
        ----------
        portfolio_value : float
            Current portfolio value for drawdown calculation.

        Returns
        -------
        BreakerStatus
            Dataclass with all current state information.
        """
        weekly_pnl = sum(self._weekly_daily_pnls) + self._daily_pnl

        current_dd = 0.0
        if self._peak_value > 0.0 and portfolio_value > 0.0:
            current_dd = (self._peak_value - portfolio_value) / self._peak_value

        allowed, reason = self.can_trade()

        return BreakerStatus(
            state=self._state,
            can_trade=allowed,
            position_scale=self.get_position_scale(),
            reason=reason,
            daily_pnl=self._daily_pnl,
            weekly_pnl=weekly_pnl,
            current_drawdown=round(current_dd, 6),
            consecutive_losses=self._consecutive_losses,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CircuitBreaker("
            f"state={self._state.name}, "
            f"daily_limit={self._daily_loss_limit:.0%}, "
            f"weekly_limit={self._weekly_dd_limit:.0%}, "
            f"max_dd={self._max_dd_halt:.0%}, "
            f"consec_limit={self._consec_loss_limit}, "
            f"consec_losses={self._consecutive_losses})"
        )
