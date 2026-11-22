"""
Automated daily pipeline for the MoE Trading Engine.

Runs after market close each day to:
1. Download today's data
2. Compute features and sentiment
3. Run online learning update
4. Generate tomorrow's signal
5. Log everything

Usage:
    python scripts/daily_pipeline.py --config config/settings.yaml
    
    # Or run as a scheduled cron job (add to crontab):
    # 30 16 * * 1-5 cd /path/to/project && python scripts/daily_pipeline.py
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.data.downloader import DataDownloader
from engine.data.features import FeatureEngineer
from engine.data.sentiment import SentimentPipeline
from engine.data.dataset import TradingDataset
from engine.models.moe import MoETradingEngine
from engine.training.losses import CombinedTradingLoss
from engine.training.online_learner import OnlineLearner, OnlineLearningConfig
from engine.execution.paper_trader import PaperTrader
from engine.risk.position_sizer import PositionSizer
from engine.risk.stop_loss import StopLossManager
from engine.risk.circuit_breaker import CircuitBreaker
from engine.monitoring.trade_logger import TradeLogger


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/pipeline_{datetime.now().strftime('%Y%m%d')}.log"),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="MoE Daily Pipeline")
    parser.add_argument("--config", type=str, default="config/settings.yaml")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Don't update model or place orders")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ticker = args.ticker or config["data"]["assets"][0]
    today = datetime.now().strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info(f"MoE DAILY PIPELINE — {today}")
    logger.info(f"Ticker: {ticker} | Device: {device}")
    logger.info("=" * 60)

    # Check if market was open today (skip weekends)
    day_of_week = datetime.now().weekday()
    if day_of_week >= 5:
        logger.info("Weekend — market closed. Skipping pipeline.")
        return

    # === Step 1: Download today's data ===
    logger.info("\n[Step 1/6] Downloading latest data...")
    downloader = DataDownloader()
    df = downloader.download(
        ticker=ticker,
        start_date=config["data"]["start_date"],
        end_date=None,  # Up to today
    )
    logger.info(f"  Data: {len(df)} trading days, latest: {df.index[-1]}")

    # === Step 2: Compute sentiment ===
    logger.info("\n[Step 2/6] Computing sentiment...")
    sentiment = SentimentPipeline(cache_dir=config["data"]["sentiment_dir"])
    today_sentiment = sentiment.get_daily_sentiment(ticker=ticker, date=today)
    logger.info(f"  Sentiment score: {today_sentiment:.4f}")

    # === Step 3: Feature engineering ===
    logger.info("\n[Step 3/6] Engineering features...")
    feature_eng = FeatureEngineer()
    df_features = feature_eng.compute_all_features(df)

    # Inject today's sentiment
    if "finbert_score" in df_features.columns:
        df_features.loc[df_features.index[-1], "finbert_score"] = today_sentiment

    logger.info(f"  Features: {df_features.shape[1]} columns")

    # === Step 4: Load model & online update ===
    logger.info("\n[Step 4/6] Loading model and running online update...")
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

    loss_fn = CombinedTradingLoss(
        num_experts=config["model"]["num_experts"],
    )

    ol_state_path = Path("models") / "online_learner_state.pt"
    ol_config = OnlineLearningConfig(
        replay_buffer_days=config["online_learning"]["replay_buffer_days"],
        mini_batch_steps=config["online_learning"]["mini_batch_steps"],
        online_lr=config["online_learning"]["online_lr"],
        ewc_lambda=config["online_learning"]["ewc_lambda"],
    )
    online_learner = OnlineLearner(
        model=model, loss_fn=loss_fn, config=ol_config, device=device
    )

    if ol_state_path.exists():
        online_learner.load_state(ol_state_path)
        logger.info("  Loaded existing online learner state")
    else:
        model_path = Path("models") / "moe_engine_final.pt"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            logger.info("  Loaded pre-trained model weights")
        else:
            logger.error("  No model found! Run scripts/train.py first.")
            return

    # Prepare today's data for replay buffer
    lookback = config["data"]["lookback_window"]
    expert_features = feature_eng.get_expert_feature_names()
    router_features = feature_eng.get_router_feature_names()

    if len(df_features) >= lookback + 1:
        recent = df_features.iloc[-(lookback + 1):]
        expert_data = torch.tensor(
            recent[expert_features].iloc[:lookback].values, dtype=torch.float32
        ).unsqueeze(0)  # [1, T, F]
        router_data = torch.tensor(
            recent[router_features].iloc[:lookback].values, dtype=torch.float32
        ).unsqueeze(0)
        yesterday_return = torch.tensor(
            [[recent["target_return"].iloc[-2]]], dtype=torch.float32
        ) if "target_return" in recent.columns else torch.tensor([[0.0]])

        # Add to replay buffer and update
        online_learner.add_daily_data(expert_data, router_data, yesterday_return)

        if not args.dry_run:
            update_result = online_learner.update()
            logger.info(f"  Online update: {update_result}")

            # Save updated state
            online_learner.save_state(ol_state_path)
        else:
            logger.info("  [DRY RUN] Skipping model update")

    # === Step 5: Generate tomorrow's signal ===
    logger.info("\n[Step 5/6] Generating prediction...")
    model.eval()
    with torch.no_grad():
        # Use latest lookback window for prediction
        latest_expert = torch.tensor(
            df_features[expert_features].iloc[-lookback:].values, dtype=torch.float32
        ).unsqueeze(0).to(device)
        latest_router = torch.tensor(
            df_features[router_features].iloc[-lookback:].values, dtype=torch.float32
        ).unsqueeze(0).to(device)

        output = model(latest_expert, latest_router)

        direction_prob = output["direction_prob"].cpu().item()
        magnitude = output["magnitude"].cpu().item()
        gate_probs = output["gate_probs"].cpu().numpy().flatten().tolist()
        selected_expert = int(torch.argmax(output["gate_probs"], dim=-1).item())

    expert_names = ["ODE (Volatility)", "Hyper-LSTM (Regime)", "SSM (Trend)"]
    signal = "BULLISH" if direction_prob > 0.5 else "BEARISH"
    confidence = abs(direction_prob - 0.5) * 2  # 0 to 1

    logger.info(f"  Signal: {signal}")
    logger.info(f"  Direction Probability: {direction_prob:.4f}")
    logger.info(f"  Predicted Return: {magnitude:.4f}%")
    logger.info(f"  Confidence: {confidence:.2%}")
    logger.info(f"  Expert Selected: {expert_names[selected_expert]}")
    logger.info(f"  Gate Probs: {[f'{p:.3f}' for p in gate_probs]}")

    # === Step 6: Log results ===
    logger.info("\n[Step 6/6] Logging results...")
    trade_logger = TradeLogger(db_path=config["monitoring"]["trade_log_db"])

    # Determine action
    risk_config = config["risk"]
    action = "HOLD"
    if confidence >= risk_config["min_confidence"] - 0.5:
        action = "BUY" if signal == "BULLISH" else "SELL"

    trade_logger.log_signal(
        date=today,
        ticker=ticker,
        direction_prob=direction_prob,
        predicted_return=magnitude,
        expert_selected=selected_expert,
        gate_probs=gate_probs,
        action=action,
        confidence=confidence,
    )

    logger.info(f"\n  Action: {action}")
    logger.info("=" * 60)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
