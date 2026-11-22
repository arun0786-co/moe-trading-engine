"""
Backtesting entrypoint for the MoE Trading Engine.

Runs walk-forward backtesting with full transaction costs and risk management,
then generates a comprehensive performance report.

Usage:
    python scripts/backtest.py --config config/settings.yaml
    python scripts/backtest.py --config config/settings.yaml --ticker RELIANCE.NS
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.data.downloader import DataDownloader
from engine.data.features import FeatureEngineer
from engine.data.dataset import build_dataloaders
from engine.models.moe import MoETradingEngine
from engine.training.losses import CombinedTradingLoss
from engine.training.trainer import Trainer, TrainConfig
from engine.backtesting.backtester import WalkForwardBacktester
from engine.backtesting.costs import IndianMarketCosts
from engine.backtesting.metrics import PerformanceMetrics
from engine.risk.position_sizer import PositionSizer
from engine.risk.stop_loss import StopLossManager
from engine.risk.circuit_breaker import CircuitBreaker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Backtest the MoE Trading Engine")
    parser.add_argument("--config", type=str, default="config/settings.yaml")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None, help="Path to pre-trained model (skip training)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    logger.info("=" * 60)
    logger.info("MoE TRADING ENGINE — WALK-FORWARD BACKTEST")
    logger.info("=" * 60)

    # Data
    ticker = args.ticker or config["data"]["assets"][0]
    downloader = DataDownloader()
    df = downloader.download(
        ticker=ticker,
        start_date=config["data"]["start_date"],
        end_date=config["data"].get("end_date"),
    )

    # Features
    feature_eng = FeatureEngineer()
    df_features = feature_eng.compute_all_features(df)

    # Build folds
    wf_config = config["training"]["walk_forward"]
    folds = build_dataloaders(
        df=df_features,
        expert_feature_names=feature_eng.get_expert_feature_names(),
        router_feature_names=feature_eng.get_router_feature_names(),
        target_col="target_return",
        lookback=config["data"]["lookback_window"],
        batch_size=config["training"]["batch_size"],
        train_years=wf_config["train_years"],
        val_years=wf_config["val_years"],
        test_years=wf_config["test_years"],
        step_years=wf_config["step_years"],
    )

    # Model
    expert_dim = len(feature_eng.get_expert_feature_names())
    router_dim = len(feature_eng.get_router_feature_names())

    model = MoETradingEngine(
        expert_input_dim=expert_dim,
        router_input_dim=router_dim,
        hidden_dim=config["model"]["expert_hidden_dim"],
        num_experts=config["model"]["num_experts"],
        router_hidden_dim=config["model"]["router_hidden_dim"],
        ode_config=config["model"].get("ode", {}),
        hyper_lstm_config=config["model"].get("hyper_lstm", {}),
        ssm_config=config["model"].get("ssm", {}),
        router_config=config["model"].get("router_config", {}),
    )

    # Cost model
    cost_config = config["costs"]
    cost_model = IndianMarketCosts(
        brokerage_flat=cost_config["brokerage_per_trade"],
        brokerage_pct=cost_config["brokerage_pct"],
        stt_sell_pct=cost_config["stt_sell_pct"],
        exchange_charges_pct=cost_config["exchange_charges_pct"],
        gst_pct=cost_config["gst_pct"],
        sebi_fee_pct=cost_config["sebi_fee_pct"],
        stamp_duty_buy_pct=cost_config["stamp_duty_buy_pct"],
    )

    # Risk management
    risk_config = config["risk"]
    position_sizer = PositionSizer(
        method=risk_config["method"],
        max_risk_per_trade=risk_config["max_risk_per_trade"],
        max_position_pct=risk_config["max_position_pct"],
        min_confidence=risk_config["min_confidence"],
    )

    sl_config = risk_config["stop_loss"]
    stop_loss_mgr = StopLossManager(
        method=sl_config["method"],
        fixed_pct=sl_config["fixed_pct"],
        atr_multiplier=sl_config["atr_multiplier"],
        trailing_atr_mult=sl_config["trailing_atr_multiplier"],
        max_holding_days=risk_config["max_holding_days"],
    )

    cb_config = risk_config["circuit_breaker"]
    circuit_breaker = CircuitBreaker(
        daily_loss_limit=cb_config["daily_loss_limit"],
        weekly_dd_limit=cb_config["weekly_drawdown_limit"],
        max_dd_halt=cb_config["max_drawdown_halt"],
        consec_loss_limit=cb_config["consecutive_loss_limit"],
    )

    # Loss & trainer
    loss_fn = CombinedTradingLoss(
        direction_weight=config["training"]["direction_loss_weight"],
        magnitude_weight=config["training"]["magnitude_loss_weight"],
        balance_weight=config["training"]["balance_loss_weight"],
        num_experts=config["model"]["num_experts"],
    )

    train_config = TrainConfig(
        batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        max_epochs=config["training"]["max_epochs"],
        patience=config["training"]["patience"],
        gradient_clip=config["training"]["gradient_clip"],
        use_amp=config["training"]["use_amp"],
        use_wandb=False,  # No wandb for backtesting
    )

    # Run backtest
    backtester = WalkForwardBacktester(
        model=model,
        loss_fn=loss_fn,
        train_config=train_config,
        cost_model=cost_model,
        position_sizer=position_sizer,
        stop_loss_mgr=stop_loss_mgr,
        circuit_breaker=circuit_breaker,
        starting_capital=risk_config["starting_capital"],
        device=device,
    )

    results = backtester.run(folds, df_features, ticker)

    # Print report
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)

    report = results.get("aggregate", {})
    for key, value in report.items():
        if isinstance(value, float):
            if "pct" in key or "return" in key or "rate" in key or "ratio" in key or "drawdown" in key:
                logger.info(f"  {key}: {value:.2%}")
            else:
                logger.info(f"  {key}: ₹{value:,.2f}")
        else:
            logger.info(f"  {key}: {value}")

    # Profitability gates
    logger.info("\n" + "-" * 40)
    logger.info("PROFITABILITY GATES")
    logger.info("-" * 40)

    gates = {
        "Sharpe > 1.0": report.get("sharpe_ratio", 0) > 1.0,
        "Max DD < 20%": report.get("max_drawdown", 1.0) < 0.20,
        "Win Rate > 50%": report.get("win_rate", 0) > 0.50,
        "Profit Factor > 1.3": report.get("profit_factor", 0) > 1.3,
        "Profitable after costs": report.get("total_net_pnl", 0) > 0,
    }

    all_passed = True
    for gate_name, passed in gates.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {status}: {gate_name}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("\n  🎯 ALL GATES PASSED — System is ready for paper trading!")
    else:
        logger.info("\n  ⚠️  Some gates failed — Review model before live deployment.")


if __name__ == "__main__":
    main()
