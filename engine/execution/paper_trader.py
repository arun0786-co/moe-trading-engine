"""
Paper trading execution engine.

Simulates order execution against historical or live market data
without placing real orders. Used for validation before going live.
Has the same interface as a real broker API so it can be swapped seamlessly.
"""

import logging
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Represents a trading order."""
    order_id: int
    ticker: str
    direction: str          # 'BUY' or 'SELL'
    order_type: str         # 'MARKET', 'LIMIT', 'SL'
    quantity: int
    price: float            # Limit/SL price (0 for market)
    status: str = "PENDING" # PENDING, FILLED, CANCELLED, REJECTED
    fill_price: float = 0.0
    fill_time: str = ""
    slippage: float = 0.0
    created_at: str = ""


@dataclass
class Position:
    """Represents an open position."""
    ticker: str
    direction: str          # 'LONG' or 'SHORT'
    quantity: int
    entry_price: float
    entry_date: str
    current_price: float = 0.0
    stop_price: float = 0.0
    unrealized_pnl: float = 0.0
    trade_id: int = 0       # Reference to trade logger


@dataclass
class Portfolio:
    """Tracks the current portfolio state."""
    cash: float
    starting_capital: float
    positions: dict[str, Position] = field(default_factory=dict)
    peak_equity: float = 0.0
    
    @property
    def positions_value(self) -> float:
        """Total value of all open positions."""
        return sum(
            p.quantity * p.current_price for p in self.positions.values()
        )
    
    @property
    def equity(self) -> float:
        """Total portfolio value (cash + positions)."""
        return self.cash + self.positions_value
    
    @property
    def drawdown(self) -> float:
        """Current drawdown from peak equity."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity


class PaperTrader:
    """Simulated execution engine for paper trading.
    
    Provides the same interface as a real broker API but executes
    against provided price data with realistic slippage simulation.
    """

    def __init__(
        self,
        starting_capital: float = 100_000.0,
        slippage_pct: float = 0.0005,
        random_seed: int = 42,
    ):
        """
        Args:
            starting_capital: Initial cash in INR
            slippage_pct: Random slippage percentage per trade (default 0.05%)
            random_seed: For reproducible slippage simulation
        """
        self.portfolio = Portfolio(
            cash=starting_capital,
            starting_capital=starting_capital,
            peak_equity=starting_capital,
        )
        self.slippage_pct = slippage_pct
        self.rng = np.random.RandomState(random_seed)

        self.order_counter = 0
        self.order_history: list[Order] = []

        logger.info(f"PaperTrader initialized with ₹{starting_capital:,.2f}")

    def place_order(
        self,
        ticker: str,
        direction: str,
        quantity: int,
        order_type: str = "MARKET",
        price: float = 0.0,
        current_market_price: float = 0.0,
    ) -> Order:
        """Place a simulated order.
        
        Args:
            ticker: Stock/index ticker
            direction: 'BUY' or 'SELL'
            quantity: Number of shares
            order_type: 'MARKET', 'LIMIT', or 'SL'
            price: Limit or stop-loss price (ignored for MARKET)
            current_market_price: Current market price for fill simulation
        
        Returns:
            Order object with fill details
        """
        self.order_counter += 1
        order = Order(
            order_id=self.order_counter,
            ticker=ticker,
            direction=direction,
            order_type=order_type,
            quantity=quantity,
            price=price,
            created_at=datetime.now().isoformat(),
        )

        if order_type == "MARKET":
            # Simulate market order fill with slippage
            slippage = self.rng.uniform(0, self.slippage_pct)
            if direction == "BUY":
                fill_price = current_market_price * (1 + slippage)
            else:
                fill_price = current_market_price * (1 - slippage)

            order.fill_price = round(fill_price, 2)
            order.slippage = abs(fill_price - current_market_price)
            order.status = "FILLED"
            order.fill_time = datetime.now().isoformat()

            # Update portfolio
            self._execute_fill(order)
        elif order_type == "LIMIT":
            # Limit orders: fill only if market price is favorable
            if direction == "BUY" and current_market_price <= price:
                order.fill_price = price
                order.status = "FILLED"
                self._execute_fill(order)
            elif direction == "SELL" and current_market_price >= price:
                order.fill_price = price
                order.status = "FILLED"
                self._execute_fill(order)
            else:
                order.status = "PENDING"
        elif order_type == "SL":
            # Stop-loss orders
            order.status = "PENDING"

        self.order_history.append(order)
        
        if order.status == "FILLED":
            logger.info(
                f"Order FILLED: {direction} {quantity} {ticker} "
                f"@ ₹{order.fill_price:.2f} (slippage: ₹{order.slippage:.2f})"
            )

        return order

    def _execute_fill(self, order: Order):
        """Update portfolio state after a fill."""
        trade_value = order.fill_price * order.quantity

        if order.direction == "BUY":
            if trade_value > self.portfolio.cash:
                order.status = "REJECTED"
                logger.warning(
                    f"Order REJECTED: Insufficient cash "
                    f"(need ₹{trade_value:,.2f}, have ₹{self.portfolio.cash:,.2f})"
                )
                return

            self.portfolio.cash -= trade_value

            if order.ticker in self.portfolio.positions:
                # Average up existing position
                pos = self.portfolio.positions[order.ticker]
                total_qty = pos.quantity + order.quantity
                avg_price = (
                    (pos.entry_price * pos.quantity + order.fill_price * order.quantity)
                    / total_qty
                )
                pos.quantity = total_qty
                pos.entry_price = avg_price
            else:
                self.portfolio.positions[order.ticker] = Position(
                    ticker=order.ticker,
                    direction="LONG",
                    quantity=order.quantity,
                    entry_price=order.fill_price,
                    entry_date=datetime.now().strftime("%Y-%m-%d"),
                    current_price=order.fill_price,
                )

        elif order.direction == "SELL":
            if order.ticker not in self.portfolio.positions:
                order.status = "REJECTED"
                logger.warning(f"Order REJECTED: No position in {order.ticker}")
                return

            pos = self.portfolio.positions[order.ticker]
            if order.quantity > pos.quantity:
                order.status = "REJECTED"
                logger.warning(
                    f"Order REJECTED: Trying to sell {order.quantity} "
                    f"but only hold {pos.quantity}"
                )
                return

            self.portfolio.cash += trade_value

            if order.quantity == pos.quantity:
                del self.portfolio.positions[order.ticker]
            else:
                pos.quantity -= order.quantity

    def update_prices(self, price_dict: dict[str, float]):
        """Update current prices for all positions.
        
        Args:
            price_dict: {ticker: current_price}
        """
        for ticker, price in price_dict.items():
            if ticker in self.portfolio.positions:
                pos = self.portfolio.positions[ticker]
                pos.current_price = price
                if pos.direction == "LONG":
                    pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
                else:
                    pos.unrealized_pnl = (pos.entry_price - price) * pos.quantity

        # Update peak equity
        current_equity = self.portfolio.equity
        if current_equity > self.portfolio.peak_equity:
            self.portfolio.peak_equity = current_equity

    def check_stop_losses(
        self, price_dict: dict[str, float]
    ) -> list[tuple[str, float]]:
        """Check and trigger stop losses for all positions.
        
        Args:
            price_dict: Current prices {ticker: price}
        
        Returns:
            List of (ticker, stop_price) for triggered stops
        """
        triggered = []
        for ticker, pos in list(self.portfolio.positions.items()):
            if ticker in price_dict and pos.stop_price > 0:
                current = price_dict[ticker]
                if pos.direction == "LONG" and current <= pos.stop_price:
                    triggered.append((ticker, pos.stop_price))
                elif pos.direction == "SHORT" and current >= pos.stop_price:
                    triggered.append((ticker, pos.stop_price))
        return triggered

    def get_portfolio_summary(self) -> dict:
        """Get current portfolio state summary."""
        return {
            "equity": self.portfolio.equity,
            "cash": self.portfolio.cash,
            "positions_value": self.portfolio.positions_value,
            "num_positions": len(self.portfolio.positions),
            "drawdown": self.portfolio.drawdown,
            "peak_equity": self.portfolio.peak_equity,
            "total_return_pct": (
                (self.portfolio.equity / self.portfolio.starting_capital - 1) * 100
            ),
            "positions": {
                ticker: {
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "entry_date": pos.entry_date,
                }
                for ticker, pos in self.portfolio.positions.items()
            },
        }

    def reset(self):
        """Reset the paper trader to initial state."""
        self.portfolio = Portfolio(
            cash=self.portfolio.starting_capital,
            starting_capital=self.portfolio.starting_capital,
            peak_equity=self.portfolio.starting_capital,
        )
        self.order_counter = 0
        self.order_history.clear()
        logger.info("PaperTrader reset to initial state")
