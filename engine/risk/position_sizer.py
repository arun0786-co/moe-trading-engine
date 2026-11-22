"""Position sizing strategies for Indian equity swing trading.

Provides three sizing methods:
    1. **Fixed-fractional** — risk a fixed % of capital per trade.
    2. **Volatility-adjusted** — inverse ATR sizing: allocate more
       capital to lower-volatility setups.
    3. **Kelly criterion** — optimal f* based on historical win rate
       and reward-to-risk ratio.

All methods:
    - Cap position size at ``max_position_pct`` of available capital.
    - Scale by model confidence when above ``min_confidence``.
    - Return integer share counts (India does not allow fractional shares).
    - Return 0 shares when confidence is below ``min_confidence``.
"""

from __future__ import annotations

import logging
import math
from typing import Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_METHOD: Final[str] = "volatility_adjusted"
_DEFAULT_MAX_RISK_PER_TRADE: Final[float] = 0.02  # 2 % of capital
_DEFAULT_MAX_POSITION_PCT: Final[float] = 0.25  # 25 % of capital
_DEFAULT_MIN_CONFIDENCE: Final[float] = 0.55  # minimum model confidence
_VALID_METHODS: Final[frozenset[str]] = frozenset(
    {"fixed_fractional", "volatility_adjusted", "kelly"}
)


class PositionSizer:
    """Determine the number of shares to trade for a given setup.

    Parameters
    ----------
    method : str
        Sizing algorithm — one of ``"fixed_fractional"``,
        ``"volatility_adjusted"``, or ``"kelly"``.
    max_risk_per_trade : float
        Maximum fraction of capital risked per trade (default 2 %).
    max_position_pct : float
        Hard ceiling on any single position as a fraction of capital
        (default 25 %).
    min_confidence : float
        Minimum model confidence to take a trade at all.  Below this
        threshold the sizer returns 0 shares.
    kelly_win_rate : float
        Historical win rate for Kelly calculation (updated externally).
    kelly_avg_win_loss : float
        Historical average-win / average-loss ratio for Kelly.
    kelly_fraction : float
        Fraction of full Kelly to use (half-Kelly = 0.5 is a common
        conservative choice).
    """

    __slots__ = (
        "_method",
        "_max_risk_per_trade",
        "_max_position_pct",
        "_min_confidence",
        "_kelly_win_rate",
        "_kelly_avg_win_loss",
        "_kelly_fraction",
    )

    def __init__(
        self,
        method: str = _DEFAULT_METHOD,
        max_risk_per_trade: float = _DEFAULT_MAX_RISK_PER_TRADE,
        max_position_pct: float = _DEFAULT_MAX_POSITION_PCT,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        kelly_win_rate: float = 0.55,
        kelly_avg_win_loss: float = 1.5,
        kelly_fraction: float = 0.5,
    ) -> None:
        if method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown sizing method '{method}'. "
                f"Choose from {sorted(_VALID_METHODS)}."
            )
        self._method = method
        self._max_risk_per_trade = max_risk_per_trade
        self._max_position_pct = max_position_pct
        self._min_confidence = min_confidence
        self._kelly_win_rate = kelly_win_rate
        self._kelly_avg_win_loss = kelly_avg_win_loss
        self._kelly_fraction = kelly_fraction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_size(
        self,
        capital: float,
        entry_price: float,
        stop_price: float,
        confidence: float,
        atr: float | None = None,
    ) -> int:
        """Calculate the number of shares to buy/short.

        Parameters
        ----------
        capital : float
            Current available capital in ₹.
        entry_price : float
            Intended entry price per share in ₹.
        stop_price : float
            Initial stop-loss price per share in ₹.
        confidence : float
            Model prediction confidence in [0, 1].
        atr : float | None
            Current Average True Range for the instrument (required for
            ``volatility_adjusted`` method).

        Returns
        -------
        int
            Number of shares (always ≥ 0).  Returns 0 if the trade
            should be skipped (low confidence, zero capital, etc.).

        Raises
        ------
        ValueError
            If ``entry_price`` is non-positive or ``stop_price`` equals
            ``entry_price`` for risk-based sizing.
        """
        # -- Gate: confidence check --------------------------------------
        if confidence < self._min_confidence:
            logger.debug(
                "Confidence %.3f below threshold %.3f — skipping trade.",
                confidence,
                self._min_confidence,
            )
            return 0

        # -- Gate: degenerate inputs -------------------------------------
        if capital <= 0.0:
            logger.warning("Zero or negative capital (₹%.2f) — cannot size.", capital)
            return 0

        if entry_price <= 0.0:
            raise ValueError(f"Entry price must be positive, got ₹{entry_price}")

        # -- Dispatch to sizing method -----------------------------------
        if self._method == "fixed_fractional":
            raw_shares = self._fixed_fractional(capital, entry_price, stop_price)
        elif self._method == "volatility_adjusted":
            raw_shares = self._volatility_adjusted(capital, entry_price, atr)
        elif self._method == "kelly":
            raw_shares = self._kelly(capital, entry_price, stop_price)
        else:
            # Should never happen due to __init__ validation
            raw_shares = 0.0

        # -- Apply confidence scaling ------------------------------------
        # Linear scaling: confidence of 0.55 (min) → scale 0, 1.0 → scale 1.0
        confidence_range = 1.0 - self._min_confidence
        if confidence_range > 0.0:
            confidence_scale = (confidence - self._min_confidence) / confidence_range
        else:
            confidence_scale = 1.0
        scaled_shares = raw_shares * confidence_scale

        # -- Apply max-position cap --------------------------------------
        max_shares_by_cap = (capital * self._max_position_pct) / entry_price
        final_shares = min(scaled_shares, max_shares_by_cap)

        # -- Integer floor (can't buy fractional shares in India) --------
        result = max(0, int(math.floor(final_shares)))

        logger.debug(
            "PositionSizer [%s]: capital=₹%.0f entry=₹%.2f stop=₹%.2f "
            "conf=%.3f raw=%.1f scaled=%.1f capped=%.1f → %d shares",
            self._method,
            capital,
            entry_price,
            stop_price,
            confidence,
            raw_shares,
            scaled_shares,
            final_shares,
            result,
        )
        return result

    # ------------------------------------------------------------------
    # Kelly parameter update
    # ------------------------------------------------------------------

    def update_kelly_params(
        self,
        win_rate: float,
        avg_win_loss_ratio: float,
    ) -> None:
        """Update Kelly criterion parameters from recent trade history.

        Should be called periodically (e.g. after each walk-forward fold)
        to keep the sizing adaptive.

        Parameters
        ----------
        win_rate : float
            Historical win rate in [0, 1].
        avg_win_loss_ratio : float
            Average winning PnL / average losing PnL (both positive).
        """
        self._kelly_win_rate = max(0.0, min(1.0, win_rate))
        self._kelly_avg_win_loss = max(0.0, avg_win_loss_ratio)

        logger.info(
            "Kelly params updated: win_rate=%.3f, avg_win/loss=%.3f, "
            "half-kelly f*=%.4f",
            self._kelly_win_rate,
            self._kelly_avg_win_loss,
            self._kelly_f_star(),
        )

    # ------------------------------------------------------------------
    # Private sizing implementations
    # ------------------------------------------------------------------

    def _fixed_fractional(
        self,
        capital: float,
        entry_price: float,
        stop_price: float,
    ) -> float:
        """Fixed-fractional: risk X% of capital.

        shares = (capital × risk%) / |entry - stop|
        """
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share == 0.0:
            logger.warning(
                "Entry (₹%.2f) == Stop (₹%.2f) — risk/share is zero.",
                entry_price,
                stop_price,
            )
            return 0.0

        dollar_risk = capital * self._max_risk_per_trade
        return dollar_risk / risk_per_share

    def _volatility_adjusted(
        self,
        capital: float,
        entry_price: float,
        atr: float | None,
    ) -> float:
        """Volatility-adjusted: inverse ATR sizing.

        shares = (capital × risk%) / ATR

        Lower ATR ⟹ larger position (calmer stock, same ₹ risk).
        """
        if atr is None or atr <= 0.0:
            logger.warning(
                "ATR is None or non-positive (%.4f) — falling back to "
                "max-risk / entry_price.",
                atr if atr is not None else 0.0,
            )
            # Fallback: treat ATR as a conservative 2 % of entry price
            atr = entry_price * 0.02

        dollar_risk = capital * self._max_risk_per_trade
        return dollar_risk / atr

    def _kelly_f_star(self) -> float:
        """Compute the (fractional) Kelly fraction f*.

        f* = kelly_fraction × (p × b − q) / b

        where p = win_rate, q = 1 − p, b = avg_win / avg_loss.
        """
        p = self._kelly_win_rate
        q = 1.0 - p
        b = self._kelly_avg_win_loss

        if b <= 0.0:
            return 0.0

        full_kelly = (p * b - q) / b
        # Clamp to [0, 1] — negative Kelly means don't trade
        full_kelly = max(0.0, min(1.0, full_kelly))
        return full_kelly * self._kelly_fraction

    def _kelly(
        self,
        capital: float,
        entry_price: float,
        stop_price: float,
    ) -> float:
        """Kelly criterion sizing.

        Position value = capital × f*, then convert to shares.
        """
        f_star = self._kelly_f_star()
        if f_star <= 0.0:
            logger.debug("Kelly f* ≤ 0 (negative edge) — no trade.")
            return 0.0

        position_value = capital * f_star
        return position_value / entry_price

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def method(self) -> str:
        """Currently active sizing method name."""
        return self._method

    @property
    def min_confidence(self) -> float:
        """Minimum model confidence required to take a trade."""
        return self._min_confidence

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PositionSizer(method='{self._method}', "
            f"max_risk={self._max_risk_per_trade:.1%}, "
            f"max_pos={self._max_position_pct:.0%}, "
            f"min_conf={self._min_confidence:.2f})"
        )
