"""Stop-loss management for Indian equity swing trading.

Provides multiple stop-loss strategies:
    1. **Fixed-percentage** — stop at a fixed % below entry (long) or
       above entry (short).
    2. **ATR-based** — stop at ``entry ± atr_multiplier × ATR``, adapting
       to current volatility.
    3. **Trailing ATR** — a ratcheting stop that follows price in the
       profitable direction by ``trailing_atr_mult × ATR``.

Additionally enforces a **time stop** — positions held beyond
``max_holding_days`` are closed to free capital and avoid drift.

All prices are in INR (₹).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_METHOD: Final[str] = "atr"
_DEFAULT_FIXED_PCT: Final[float] = 0.02  # 2 %
_DEFAULT_ATR_MULTIPLIER: Final[float] = 2.0
_DEFAULT_TRAILING_ATR_MULT: Final[float] = 1.5
_DEFAULT_MAX_HOLDING_DAYS: Final[int] = 15
_VALID_METHODS: Final[frozenset[str]] = frozenset({"fixed", "atr"})


class StopLossManager:
    """Manages initial, trailing, and time-based stop losses.

    Parameters
    ----------
    method : str
        Initial stop method — ``"fixed"`` or ``"atr"``.
    fixed_pct : float
        Fixed stop-loss distance as a fraction of entry price (default
        0.02 = 2 %).
    atr_multiplier : float
        Multiplier applied to ATR for the initial stop distance
        (default 2.0).
    trailing_atr_mult : float
        Multiplier applied to ATR for the trailing stop distance
        (default 1.5).
    max_holding_days : int
        Maximum number of calendar days a position may be held before a
        time stop fires (default 15).
    """

    __slots__ = (
        "_method",
        "_fixed_pct",
        "_atr_multiplier",
        "_trailing_atr_mult",
        "_max_holding_days",
    )

    def __init__(
        self,
        method: str = _DEFAULT_METHOD,
        fixed_pct: float = _DEFAULT_FIXED_PCT,
        atr_multiplier: float = _DEFAULT_ATR_MULTIPLIER,
        trailing_atr_mult: float = _DEFAULT_TRAILING_ATR_MULT,
        max_holding_days: int = _DEFAULT_MAX_HOLDING_DAYS,
    ) -> None:
        if method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown stop method '{method}'. Choose from {sorted(_VALID_METHODS)}."
            )
        if fixed_pct < 0.0:
            raise ValueError(f"fixed_pct must be ≥ 0, got {fixed_pct}")
        if atr_multiplier < 0.0:
            raise ValueError(f"atr_multiplier must be ≥ 0, got {atr_multiplier}")
        if trailing_atr_mult < 0.0:
            raise ValueError(
                f"trailing_atr_mult must be ≥ 0, got {trailing_atr_mult}"
            )
        if max_holding_days < 1:
            raise ValueError(
                f"max_holding_days must be ≥ 1, got {max_holding_days}"
            )

        self._method = method
        self._fixed_pct = fixed_pct
        self._atr_multiplier = atr_multiplier
        self._trailing_atr_mult = trailing_atr_mult
        self._max_holding_days = max_holding_days

    # ------------------------------------------------------------------
    # Initial stop
    # ------------------------------------------------------------------

    def calculate_initial_stop(
        self,
        entry_price: float,
        direction: str,
        atr: float | None = None,
    ) -> float:
        """Calculate the initial stop-loss price level.

        Parameters
        ----------
        entry_price : float
            Entry price per share in ₹.
        direction : str
            ``"long"`` or ``"short"``.
        atr : float | None
            Current Average True Range.  Required when ``method="atr"``.

        Returns
        -------
        float
            Stop-loss price level in ₹.

        Raises
        ------
        ValueError
            If ``direction`` is invalid or ATR is missing when required.
        """
        self._validate_direction(direction)

        if entry_price <= 0.0:
            raise ValueError(f"Entry price must be positive, got ₹{entry_price}")

        if self._method == "fixed":
            distance = entry_price * self._fixed_pct
        elif self._method == "atr":
            if atr is None or atr <= 0.0:
                raise ValueError(
                    f"ATR must be positive for ATR-based stops, got {atr}"
                )
            distance = atr * self._atr_multiplier
        else:
            distance = entry_price * self._fixed_pct  # fallback

        if direction == "long":
            stop = entry_price - distance
            # Floor: stop cannot be negative
            stop = max(0.01, stop)
        else:  # short
            stop = entry_price + distance

        logger.debug(
            "Initial stop: entry=₹%.2f dir=%s method=%s distance=₹%.2f → stop=₹%.2f",
            entry_price,
            direction,
            self._method,
            distance,
            stop,
        )
        return round(stop, 2)

    # ------------------------------------------------------------------
    # Trailing stop update
    # ------------------------------------------------------------------

    def update_trailing_stop(
        self,
        current_price: float,
        current_stop: float,
        direction: str,
        atr: float,
    ) -> float:
        """Update the trailing stop, only ratcheting in the profitable direction.

        For **long** positions the stop can only move *up*; for **short**
        positions it can only move *down*.

        Parameters
        ----------
        current_price : float
            Latest market price in ₹.
        current_stop : float
            Existing stop-loss level in ₹.
        direction : str
            ``"long"`` or ``"short"``.
        atr : float
            Current Average True Range.

        Returns
        -------
        float
            Updated stop-loss price.  Guaranteed to be at least as
            favourable as ``current_stop``.
        """
        self._validate_direction(direction)

        if atr <= 0.0:
            logger.warning("ATR non-positive (%.4f) — keeping current stop.", atr)
            return current_stop

        trailing_distance = atr * self._trailing_atr_mult

        if direction == "long":
            candidate_stop = current_price - trailing_distance
            # Only ratchet up
            new_stop = max(current_stop, candidate_stop)
            # Floor: cannot be negative
            new_stop = max(0.01, new_stop)
        else:  # short
            candidate_stop = current_price + trailing_distance
            # Only ratchet down
            new_stop = min(current_stop, candidate_stop)

        if new_stop != current_stop:
            logger.debug(
                "Trailing stop updated: dir=%s price=₹%.2f "
                "old_stop=₹%.2f → new_stop=₹%.2f (ATR=%.2f, mult=%.1f)",
                direction,
                current_price,
                current_stop,
                new_stop,
                atr,
                self._trailing_atr_mult,
            )

        return round(new_stop, 2)

    # ------------------------------------------------------------------
    # Stop check — the main per-bar evaluation
    # ------------------------------------------------------------------

    def check_stops(
        self,
        position: dict[str, Any],
        current_price: float,
        current_date: date | str,
        atr: float,
    ) -> str:
        """Evaluate all stop conditions for an open position.

        Checks in priority order:
            1. **Stop loss** — price breached initial/trailing stop.
            2. **Trailing stop** — ratcheted stop breached.
            3. **Time stop** — position held beyond ``max_holding_days``.

        Parameters
        ----------
        position : dict
            Must contain keys: ``"direction"`` (``"long"`` / ``"short"``),
            ``"stop_price"`` (current stop level), ``"entry_date"``
            (``date`` or ISO string).
        current_price : float
            Latest market price in ₹.
        current_date : date | str
            Current trading date.
        atr : float
            Current ATR for trailing-stop update.

        Returns
        -------
        str
            One of ``"hold"``, ``"stop_loss"``, ``"trailing_stop"``,
            or ``"time_stop"``.
        """
        direction: str = position["direction"]
        stop_price: float = position["stop_price"]

        # -- Parse dates ------------------------------------------------
        entry_date = self._parse_date(position["entry_date"])
        eval_date = self._parse_date(current_date)

        # -- 1. Hard stop-loss check ------------------------------------
        if direction == "long" and current_price <= stop_price:
            logger.info(
                "STOP LOSS hit (long): price=₹%.2f ≤ stop=₹%.2f",
                current_price,
                stop_price,
            )
            return "stop_loss"

        if direction == "short" and current_price >= stop_price:
            logger.info(
                "STOP LOSS hit (short): price=₹%.2f ≥ stop=₹%.2f",
                current_price,
                stop_price,
            )
            return "stop_loss"

        # -- 2. Update trailing stop and re-check -----------------------
        if atr > 0.0:
            new_stop = self.update_trailing_stop(
                current_price, stop_price, direction, atr
            )

            # Check if the *new* trailing stop was breached on this bar
            # (can happen with large intraday swings when using close price)
            trailing_hit = (
                (direction == "long" and current_price <= new_stop and new_stop > stop_price)
                or (direction == "short" and current_price >= new_stop and new_stop < stop_price)
            )

            if trailing_hit:
                logger.info(
                    "TRAILING STOP hit: dir=%s price=₹%.2f trailing_stop=₹%.2f",
                    direction,
                    current_price,
                    new_stop,
                )
                return "trailing_stop"

            # Persist updated stop back to position
            position["stop_price"] = new_stop

        # -- 3. Time stop -----------------------------------------------
        days_held = (eval_date - entry_date).days
        if days_held >= self._max_holding_days:
            logger.info(
                "TIME STOP: held %d days (limit=%d)",
                days_held,
                self._max_holding_days,
            )
            return "time_stop"

        return "hold"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_direction(direction: str) -> None:
        """Raise ``ValueError`` for invalid trade directions."""
        if direction not in ("long", "short"):
            raise ValueError(
                f"Direction must be 'long' or 'short', got '{direction}'"
            )

    @staticmethod
    def _parse_date(d: date | str) -> date:
        """Coerce a ``date`` or ISO-format string to ``datetime.date``."""
        if isinstance(d, date):
            return d
        return date.fromisoformat(str(d))

    # ------------------------------------------------------------------
    # Properties & repr
    # ------------------------------------------------------------------

    @property
    def method(self) -> str:
        """Active initial-stop method name."""
        return self._method

    @property
    def max_holding_days(self) -> int:
        """Maximum holding period before a time stop fires."""
        return self._max_holding_days

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"StopLossManager(method='{self._method}', "
            f"fixed_pct={self._fixed_pct:.1%}, "
            f"atr_mult={self._atr_multiplier}, "
            f"trail_mult={self._trailing_atr_mult}, "
            f"max_days={self._max_holding_days})"
        )
