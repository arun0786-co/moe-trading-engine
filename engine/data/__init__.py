# Data ingestion, feature engineering, and dataset management
"""
engine.data — Data pipeline for the MoE Trading Engine.

Public API
----------
- ``download_ticker`` / ``download_multiple`` — OHLCV data acquisition
- ``FeatureEngineer`` — Technical / statistical feature computation
- ``TradingDataset`` / ``WalkForwardSplitter`` — PyTorch dataset + CV
- ``build_dataloaders`` — End-to-end pipeline wiring
"""

from engine.data.dataset import (
    TradingDataset,
    WalkForwardSplitter,
    build_dataloaders,
)
from engine.data.downloader import (
    download_multiple,
    download_ticker,
)
from engine.data.features import (
    EXPERT_FEATURE_NAMES,
    ROUTER_FEATURE_NAMES,
    FeatureEngineer,
)

__all__ = [
    "download_ticker",
    "download_multiple",
    "FeatureEngineer",
    "EXPERT_FEATURE_NAMES",
    "ROUTER_FEATURE_NAMES",
    "TradingDataset",
    "WalkForwardSplitter",
    "build_dataloaders",
]
