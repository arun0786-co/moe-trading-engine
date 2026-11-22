"""Indian market transaction cost model for equity delivery trades.

Implements the complete fee structure for NSE/BSE equity delivery
trades, accurate as of FY 2024-25. All monetary values are in INR (₹).

Fee schedule (equity delivery):
    - Brokerage: min(₹20, 0.03% of turnover) per side
    - STT: 0.025% on sell side only
    - Exchange transaction charges: 0.00345% per side
    - GST: 18% on (brokerage + exchange charges)
    - SEBI turnover fee: 0.0001% per side
    - Stamp duty: 0.015% on buy side only
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — Indian market fee rates (equity delivery, NSE)
# ---------------------------------------------------------------------------
_DEFAULT_BROKERAGE_FLAT_CAP: Final[float] = 20.0  # ₹20 per order
_DEFAULT_BROKERAGE_PCT: Final[float] = 0.0003  # 0.03%
_DEFAULT_STT_SELL_PCT: Final[float] = 0.00025  # 0.025% sell side
_DEFAULT_EXCHANGE_CHARGE_PCT: Final[float] = 0.0000345  # 0.00345%
_DEFAULT_GST_PCT: Final[float] = 0.18  # 18% on brokerage + exchange
_DEFAULT_SEBI_FEE_PCT: Final[float] = 0.000001  # 0.0001%
_DEFAULT_STAMP_DUTY_BUY_PCT: Final[float] = 0.00015  # 0.015% buy side


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Itemized breakdown of transaction costs for one leg of a trade."""

    turnover: float
    brokerage: float
    stt: float
    exchange_charges: float
    gst: float
    sebi_fee: float
    stamp_duty: float
    total: float

    def as_dict(self) -> dict[str, float]:
        """Return the breakdown as a plain dictionary."""
        return {
            "turnover": self.turnover,
            "brokerage": self.brokerage,
            "stt": self.stt,
            "exchange_charges": self.exchange_charges,
            "gst": self.gst,
            "sebi_fee": self.sebi_fee,
            "stamp_duty": self.stamp_duty,
            "total": self.total,
        }


class IndianMarketCosts:
    """Transaction cost calculator for Indian equity delivery trades.

    Encapsulates the full fee schedule applied by brokers, exchanges,
    and regulators on the NSE (or BSE).  All rates are configurable but
    default to the standard FY 2024-25 schedule for equity delivery.

    Parameters
    ----------
    brokerage_flat : float
        Maximum flat brokerage per order in ₹.  Default ₹20.
    brokerage_pct : float
        Brokerage as a fraction of turnover.  Default 0.0003 (0.03%).
    stt_sell_pct : float
        Securities Transaction Tax on *sell* side for delivery.
        Default 0.00025 (0.025%).
    exchange_charge_pct : float
        NSE equity delivery transaction charge.
        Default 0.0000345 (0.00345%).
    gst_pct : float
        GST rate on (brokerage + exchange charges).
        Default 0.18 (18%).
    sebi_fee_pct : float
        SEBI turnover fee.  Default 0.000001 (0.0001%).
    stamp_duty_buy_pct : float
        Stamp duty on *buy* side.
        Default 0.00015 (0.015%).
    """

    __slots__ = (
        "_brokerage_flat",
        "_brokerage_pct",
        "_stt_sell_pct",
        "_exchange_charge_pct",
        "_gst_pct",
        "_sebi_fee_pct",
        "_stamp_duty_buy_pct",
    )

    def __init__(
        self,
        brokerage_flat: float = _DEFAULT_BROKERAGE_FLAT_CAP,
        brokerage_pct: float = _DEFAULT_BROKERAGE_PCT,
        stt_sell_pct: float = _DEFAULT_STT_SELL_PCT,
        exchange_charge_pct: float = _DEFAULT_EXCHANGE_CHARGE_PCT,
        gst_pct: float = _DEFAULT_GST_PCT,
        sebi_fee_pct: float = _DEFAULT_SEBI_FEE_PCT,
        stamp_duty_buy_pct: float = _DEFAULT_STAMP_DUTY_BUY_PCT,
    ) -> None:
        self._brokerage_flat = brokerage_flat
        self._brokerage_pct = brokerage_pct
        self._stt_sell_pct = stt_sell_pct
        self._exchange_charge_pct = exchange_charge_pct
        self._gst_pct = gst_pct
        self._sebi_fee_pct = sebi_fee_pct
        self._stamp_duty_buy_pct = stamp_duty_buy_pct

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_inputs(self, price: float, quantity: int) -> None:
        """Raise ``ValueError`` for nonsensical inputs."""
        if price < 0.0:
            raise ValueError(f"Price must be non-negative, got {price}")
        if quantity < 0:
            raise ValueError(f"Quantity must be non-negative, got {quantity}")

    def _brokerage(self, turnover: float) -> float:
        """Brokerage = min(flat cap, pct of turnover)."""
        return min(self._brokerage_flat, turnover * self._brokerage_pct)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_buy_costs(self, price: float, quantity: int) -> CostBreakdown:
        """Calculate itemized transaction costs for a **buy** order.

        Buy-side charges include stamp duty but **not** STT (which is
        levied on the sell side for delivery trades).

        Parameters
        ----------
        price : float
            Per-share buy price in ₹.
        quantity : int
            Number of shares purchased.

        Returns
        -------
        CostBreakdown
            Frozen dataclass with every fee component and the total.
        """
        self._validate_inputs(price, quantity)

        turnover = price * quantity
        if turnover == 0.0:
            return CostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        brokerage = self._brokerage(turnover)
        stt = 0.0  # STT is sell-side only for delivery
        exchange_charges = turnover * self._exchange_charge_pct
        gst = (brokerage + exchange_charges) * self._gst_pct
        sebi_fee = turnover * self._sebi_fee_pct
        stamp_duty = turnover * self._stamp_duty_buy_pct

        total = brokerage + stt + exchange_charges + gst + sebi_fee + stamp_duty

        logger.debug(
            "BUY  cost: turnover=₹%.2f  total_cost=₹%.4f  (%.4f%%)",
            turnover,
            total,
            (total / turnover) * 100 if turnover else 0.0,
        )

        return CostBreakdown(
            turnover=turnover,
            brokerage=round(brokerage, 4),
            stt=round(stt, 4),
            exchange_charges=round(exchange_charges, 4),
            gst=round(gst, 4),
            sebi_fee=round(sebi_fee, 4),
            stamp_duty=round(stamp_duty, 4),
            total=round(total, 4),
        )

    def calculate_sell_costs(self, price: float, quantity: int) -> CostBreakdown:
        """Calculate itemized transaction costs for a **sell** order.

        Sell-side charges include STT but **not** stamp duty (which is
        levied on the buy side).

        Parameters
        ----------
        price : float
            Per-share sell price in ₹.
        quantity : int
            Number of shares sold.

        Returns
        -------
        CostBreakdown
            Frozen dataclass with every fee component and the total.
        """
        self._validate_inputs(price, quantity)

        turnover = price * quantity
        if turnover == 0.0:
            return CostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        brokerage = self._brokerage(turnover)
        stt = turnover * self._stt_sell_pct
        exchange_charges = turnover * self._exchange_charge_pct
        gst = (brokerage + exchange_charges) * self._gst_pct
        sebi_fee = turnover * self._sebi_fee_pct
        stamp_duty = 0.0  # Stamp duty is buy-side only

        total = brokerage + stt + exchange_charges + gst + sebi_fee + stamp_duty

        logger.debug(
            "SELL cost: turnover=₹%.2f  total_cost=₹%.4f  (%.4f%%)",
            turnover,
            total,
            (total / turnover) * 100 if turnover else 0.0,
        )

        return CostBreakdown(
            turnover=turnover,
            brokerage=round(brokerage, 4),
            stt=round(stt, 4),
            exchange_charges=round(exchange_charges, 4),
            gst=round(gst, 4),
            sebi_fee=round(sebi_fee, 4),
            stamp_duty=round(stamp_duty, 4),
            total=round(total, 4),
        )

    def round_trip_cost(
        self,
        buy_price: float,
        sell_price: float,
        quantity: int,
    ) -> float:
        """Total transaction cost for a complete buy → sell round trip.

        Parameters
        ----------
        buy_price : float
            Per-share entry price in ₹.
        sell_price : float
            Per-share exit price in ₹.
        quantity : int
            Number of shares traded.

        Returns
        -------
        float
            Total cost in ₹ (buy costs + sell costs).
        """
        buy_costs = self.calculate_buy_costs(buy_price, quantity)
        sell_costs = self.calculate_sell_costs(sell_price, quantity)
        total = buy_costs.total + sell_costs.total

        logger.debug(
            "Round trip cost: buy=₹%.4f  sell=₹%.4f  total=₹%.4f",
            buy_costs.total,
            sell_costs.total,
            total,
        )
        return round(total, 4)

    def cost_as_pct(
        self,
        buy_price: float,
        sell_price: float,
        quantity: int,
    ) -> float:
        """Round-trip cost expressed as a percentage of the *buy* turnover.

        This is the effective drag on trade profitability, making it easy
        to compare against expected alpha.

        Parameters
        ----------
        buy_price : float
            Per-share entry price in ₹.
        sell_price : float
            Per-share exit price in ₹.
        quantity : int
            Number of shares traded.

        Returns
        -------
        float
            Cost as a percentage (e.g. 0.15 means 0.15%).  Returns 0.0
            when buy turnover is zero.
        """
        buy_turnover = buy_price * quantity
        if buy_turnover == 0.0:
            return 0.0

        rt_cost = self.round_trip_cost(buy_price, sell_price, quantity)
        pct = (rt_cost / buy_turnover) * 100.0
        return round(pct, 6)

    # ------------------------------------------------------------------
    # Convenience / repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"IndianMarketCosts("
            f"brokerage_flat=₹{self._brokerage_flat}, "
            f"brokerage_pct={self._brokerage_pct:.4%}, "
            f"stt_sell={self._stt_sell_pct:.4%}, "
            f"exchange={self._exchange_charge_pct:.5%}, "
            f"gst={self._gst_pct:.0%}, "
            f"sebi={self._sebi_fee_pct:.5%}, "
            f"stamp_duty_buy={self._stamp_duty_buy_pct:.4%})"
        )
