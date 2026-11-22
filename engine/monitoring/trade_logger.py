"""
SQLite-based trade journal for the MoE Trading Engine.

Logs every trade, signal, and daily portfolio state to a local database
for analysis, auditing, and performance tracking.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class TradeLogger:
    """Persistent trade journal backed by SQLite.
    
    Records:
    - Individual trades (entry, exit, PnL, costs, expert used)
    - Daily signals (model predictions before execution)
    - Daily portfolio snapshots (equity, drawdown, positions)
    """

    def __init__(self, db_path: str = "logs/trades.db"):
        """
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"TradeLogger initialized: {self.db_path}")

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_date TEXT NOT NULL,
                    exit_date TEXT,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity INTEGER NOT NULL,
                    pnl REAL,
                    costs REAL,
                    net_pnl REAL,
                    stop_price REAL,
                    exit_reason TEXT,
                    expert_used INTEGER,
                    model_confidence REAL,
                    status TEXT DEFAULT 'open',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    direction_prob REAL NOT NULL,
                    predicted_return REAL,
                    expert_selected INTEGER,
                    gate_probs TEXT,
                    action TEXT,
                    confidence REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    equity REAL NOT NULL,
                    cash REAL NOT NULL,
                    positions_value REAL,
                    num_open_positions INTEGER,
                    daily_pnl REAL,
                    daily_return REAL,
                    peak_equity REAL,
                    drawdown REAL,
                    cumulative_return REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS model_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    update_type TEXT NOT NULL,
                    pre_accuracy REAL,
                    post_accuracy REAL,
                    rolled_back INTEGER DEFAULT 0,
                    ewc_penalty REAL,
                    buffer_size INTEGER,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(entry_date);
                CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(date);
                CREATE INDEX IF NOT EXISTS idx_portfolio_date ON portfolio_snapshots(date);
            """)

    @contextmanager
    def _connect(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def log_trade_open(
        self,
        ticker: str,
        direction: str,
        entry_date: str,
        entry_price: float,
        quantity: int,
        stop_price: float,
        expert_used: int,
        model_confidence: float,
    ) -> int:
        """Log a new trade entry.
        
        Returns:
            Trade ID for later updates
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO trades 
                   (ticker, direction, entry_date, entry_price, quantity, 
                    stop_price, expert_used, model_confidence, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
                (ticker, direction, entry_date, entry_price, quantity,
                 stop_price, expert_used, model_confidence),
            )
            trade_id = cursor.lastrowid
            logger.info(f"Trade opened: {direction} {quantity} {ticker} @ ₹{entry_price:.2f} (ID: {trade_id})")
            return trade_id

    def log_trade_close(
        self,
        trade_id: int,
        exit_date: str,
        exit_price: float,
        pnl: float,
        costs: float,
        exit_reason: str,
    ):
        """Log a trade exit."""
        net_pnl = pnl - costs
        with self._connect() as conn:
            conn.execute(
                """UPDATE trades 
                   SET exit_date=?, exit_price=?, pnl=?, costs=?, net_pnl=?,
                       exit_reason=?, status='closed'
                   WHERE id=?""",
                (exit_date, exit_price, pnl, costs, net_pnl, exit_reason, trade_id),
            )
            logger.info(f"Trade closed (ID: {trade_id}): PnL ₹{pnl:.2f}, Costs ₹{costs:.2f}, Net ₹{net_pnl:.2f} ({exit_reason})")

    def log_signal(
        self,
        date: str,
        ticker: str,
        direction_prob: float,
        predicted_return: float,
        expert_selected: int,
        gate_probs: list[float],
        action: str,
        confidence: float,
    ):
        """Log a model prediction signal."""
        import json
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO signals 
                   (date, ticker, direction_prob, predicted_return, expert_selected,
                    gate_probs, action, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (date, ticker, direction_prob, predicted_return, expert_selected,
                 json.dumps(gate_probs), action, confidence),
            )

    def log_portfolio_snapshot(
        self,
        date: str,
        equity: float,
        cash: float,
        positions_value: float,
        num_open_positions: int,
        daily_pnl: float,
        peak_equity: float,
    ):
        """Log daily portfolio state."""
        daily_return = daily_pnl / max(equity - daily_pnl, 1.0)
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        cumulative_return = (equity / 100000.0 - 1.0)  # vs starting capital

        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO portfolio_snapshots
                   (date, equity, cash, positions_value, num_open_positions,
                    daily_pnl, daily_return, peak_equity, drawdown, cumulative_return)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (date, equity, cash, positions_value, num_open_positions,
                 daily_pnl, daily_return, peak_equity, drawdown, cumulative_return),
            )

    def log_model_update(
        self,
        date: str,
        update_type: str,
        pre_accuracy: float,
        post_accuracy: float,
        rolled_back: bool,
        ewc_penalty: float = 0.0,
        buffer_size: int = 0,
        notes: str = "",
    ):
        """Log a model update event."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO model_updates
                   (date, update_type, pre_accuracy, post_accuracy, rolled_back,
                    ewc_penalty, buffer_size, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (date, update_type, pre_accuracy, post_accuracy,
                 1 if rolled_back else 0, ewc_penalty, buffer_size, notes),
            )

    def get_recent_trades(self, n: int = 20) -> list[dict]:
        """Get the N most recent trades."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_open_positions(self) -> list[dict]:
        """Get all currently open positions."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status='open'"
            ).fetchall()
            return [dict(row) for row in rows]

    def get_equity_curve(self) -> list[dict]:
        """Get the full equity curve history."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT date, equity, drawdown, cumulative_return "
                "FROM portfolio_snapshots ORDER BY date"
            ).fetchall()
            return [dict(row) for row in rows]

    def get_trade_summary(self) -> dict:
        """Get aggregate trade statistics."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN net_pnl <= 0 THEN 1 ELSE 0 END) as losing_trades,
                    AVG(net_pnl) as avg_pnl,
                    SUM(net_pnl) as total_pnl,
                    SUM(costs) as total_costs,
                    MAX(net_pnl) as best_trade,
                    MIN(net_pnl) as worst_trade,
                    AVG(CASE WHEN net_pnl > 0 THEN net_pnl END) as avg_win,
                    AVG(CASE WHEN net_pnl <= 0 THEN net_pnl END) as avg_loss
                FROM trades WHERE status='closed'
            """).fetchone()
            return dict(row) if row else {}
