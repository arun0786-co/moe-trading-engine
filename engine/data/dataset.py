"""
PyTorch Dataset and walk-forward cross-validation splitter for the MoE Engine.

Classes
-------
TradingDataset
    Sliding-window time-series dataset producing ``[B, T, F]`` tensors for
    expert and router inputs plus direction / return targets.

WalkForwardSplitter
    Generates expanding-window train / validation / test folds with
    normalization stats computed strictly on the training partition.

build_dataloaders
    End-to-end convenience function:  download → features → normalize →
    split → ``DataLoader`` objects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from engine.data.downloader import download_ticker
from engine.data.features import (
    EXPERT_FEATURE_NAMES,
    ROUTER_FEATURE_NAMES,
    FeatureEngineer,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# TradingDataset
# ═══════════════════════════════════════════════════════════════════════════

class TradingDataset(Dataset):
    """Sliding-window dataset for time-series classification / regression.

    Each sample *i* is a window ``[i, i + lookback)`` producing:

    * ``expert_input``      — ``[T, F_expert]``  float32
    * ``router_input``      — ``[T, F_router]``  float32
    * ``target_direction``  — scalar ``{0, 1}`` (down / up) at ``i + lookback + horizon - 1``
    * ``target_return``     — scalar float32 (forward return)
    * ``date_index``        — integer position in the original DataFrame (for attribution)

    Parameters
    ----------
    expert_features : np.ndarray
        Shape ``[N, F_expert]``.
    router_features : np.ndarray
        Shape ``[N, F_router]``.
    targets_return : np.ndarray
        Shape ``[N]`` — forward returns.
    dates : np.ndarray
        Integer indices (or timestamps cast to int64) for bookkeeping.
    lookback : int
        Number of time-steps per sample window.
    horizon : int
        How many days ahead the target sits (default 1).
    """

    def __init__(
        self,
        expert_features: np.ndarray,
        router_features: np.ndarray,
        targets_return: np.ndarray,
        dates: np.ndarray,
        lookback: int = 30,
        horizon: int = 1,
    ) -> None:
        super().__init__()

        if len(expert_features) != len(router_features):
            raise ValueError(
                "expert_features and router_features must have the same "
                f"length, got {len(expert_features)} vs {len(router_features)}."
            )
        if len(expert_features) != len(targets_return):
            raise ValueError(
                "Feature arrays and targets must have the same length, "
                f"got {len(expert_features)} vs {len(targets_return)}."
            )

        self.expert_features = expert_features.astype(np.float32)
        self.router_features = router_features.astype(np.float32)
        self.targets_return = targets_return.astype(np.float32)
        self.dates = dates
        self.lookback = lookback
        self.horizon = horizon

        # Valid start indices: a window [i, i+lookback) must exist and
        # the target at i + lookback + horizon - 1 must be in-bounds.
        self._max_start = len(expert_features) - lookback - horizon
        if self._max_start < 0:
            raise ValueError(
                f"Dataset too short ({len(expert_features)} rows) for "
                f"lookback={lookback} + horizon={horizon}."
            )

        logger.info(
            "TradingDataset created — %d samples, lookback=%d, horizon=%d, "
            "expert_dim=%d, router_dim=%d",
            len(self),
            lookback,
            horizon,
            expert_features.shape[1],
            router_features.shape[1],
        )

    def __len__(self) -> int:
        return self._max_start + 1

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx > self._max_start:
            raise IndexError(
                f"Index {idx} out of range [0, {self._max_start}]"
            )

        # Window of features
        expert_window = self.expert_features[idx : idx + self.lookback]
        router_window = self.router_features[idx : idx + self.lookback]

        # targets_return[i] is already the return from day i to day i + horizon
        # (pct_change shifted back by horizon), so the target belongs to the
        # last day of the window. adding horizon again here skipped a day.
        target_idx = idx + self.lookback - 1
        fwd_return = self.targets_return[target_idx]
        direction = 1.0 if fwd_return >= 0 else 0.0

        return {
            "expert_input": torch.from_numpy(expert_window),        # [T, F_e]
            "router_input": torch.from_numpy(router_window),        # [T, F_r]
            "target_direction": torch.tensor(direction, dtype=torch.float32),
            "target_return": torch.tensor(fwd_return, dtype=torch.float32),
            "date_index": torch.tensor(
                self.dates[target_idx], dtype=torch.long
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════
# WalkForwardSplitter
# ═══════════════════════════════════════════════════════════════════════════

class WalkForwardSplitter:
    """Walk-forward (expanding-window) cross-validation splitter.

    Generates folds like::

        Fold 1:  Train [2010-2015]  Val [2015-2016]  Test [2016-2017]
        Fold 2:  Train [2010-2016]  Val [2016-2017]  Test [2017-2018]
        ...

    Normalization statistics are computed **only** on each fold's training
    partition to prevent look-ahead bias.

    Parameters
    ----------
    train_years : int
        Minimum years for the first training window.
    val_years : int
        Validation window length.
    test_years : int
        Test window length.
    step_years : int
        How many years to advance between folds.
    """

    def __init__(
        self,
        train_years: int = 5,
        val_years: int = 1,
        test_years: int = 1,
        step_years: int = 1,
    ) -> None:
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
        self.step_years = step_years

    # ── Fold generation ──────────────────────────────────────────────────

    def _year_cutoff(self, base_year: int, delta: int) -> pd.Timestamp:
        """Return ``pd.Timestamp`` for Jan 1 of ``base_year + delta``."""
        return pd.Timestamp(year=base_year + delta, month=1, day=1)

    def generate_folds(
        self,
        df: pd.DataFrame,
    ) -> list[dict[str, pd.DataFrame]]:
        """Split *df* into walk-forward folds.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered DataFrame with a ``DatetimeIndex``.

        Returns
        -------
        list[dict]
            Each element is ``{"train": df, "val": df, "test": df}``.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.DatetimeIndex(df.index)

        start_year: int = df.index[0].year
        end_year: int = df.index[-1].year

        folds: list[dict[str, pd.DataFrame]] = []
        fold_idx = 0

        while True:
            train_end_year = start_year + self.train_years + fold_idx * self.step_years
            val_end_year = train_end_year + self.val_years
            test_end_year = val_end_year + self.test_years

            train_start = self._year_cutoff(start_year, 0)
            train_end = self._year_cutoff(train_end_year, 0)
            val_end = self._year_cutoff(val_end_year, 0)
            test_end = self._year_cutoff(test_end_year, 0)

            # Check we have enough data for the test window
            if train_end_year > end_year:
                break

            train_df = df.loc[
                (df.index >= train_start) & (df.index < train_end)
            ]
            val_df = df.loc[
                (df.index >= train_end) & (df.index < val_end)
            ]
            test_df = df.loc[
                (df.index >= val_end) & (df.index < test_end)
            ]

            # Skip fold if val or test are empty (not enough data)
            if len(val_df) == 0 or len(test_df) == 0:
                logger.info(
                    "Fold %d skipped — val (%d) or test (%d) empty.",
                    fold_idx,
                    len(val_df),
                    len(test_df),
                )
                fold_idx += 1
                continue

            logger.info(
                "Fold %d: train=%d [%s→%s], val=%d [%s→%s], test=%d [%s→%s]",
                fold_idx,
                len(train_df),
                train_df.index[0].date(),
                train_df.index[-1].date(),
                len(val_df),
                val_df.index[0].date(),
                val_df.index[-1].date(),
                len(test_df),
                test_df.index[0].date(),
                test_df.index[-1].date(),
            )

            folds.append({
                "train": train_df,
                "val": val_df,
                "test": test_df,
            })

            fold_idx += 1

        if not folds:
            raise ValueError(
                f"No valid folds generated from data spanning "
                f"{start_year}–{end_year} with train_years={self.train_years}, "
                f"val_years={self.val_years}, test_years={self.test_years}."
            )

        logger.info("Generated %d walk-forward folds.", len(folds))
        return folds

    # ── DataLoaders per fold ─────────────────────────────────────────────

    def build_fold_loaders(
        self,
        fold: dict[str, pd.DataFrame],
        expert_cols: list[str],
        router_cols: list[str],
        lookback: int = 30,
        horizon: int = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        clip_value: float = 5.0,
    ) -> dict[str, DataLoader]:
        """Build normalised ``DataLoader`` objects for a single fold.

        Normalization stats are computed from the **train** split only and
        applied to all three splits.

        Parameters
        ----------
        fold : dict
            ``{"train": df, "val": df, "test": df}``.
        expert_cols / router_cols : list[str]
            Column names to extract.
        lookback / horizon : int
            Window parameters forwarded to :class:`TradingDataset`.
        batch_size : int
            Batch size for training loader. Val/test use the same size.
        num_workers : int
            DataLoader workers.
        clip_value : float
            Normalised values clipped to ±clip_value.

        Returns
        -------
        dict[str, DataLoader]
            Keys: ``"train"``, ``"val"``, ``"test"``.
        """
        train_df = fold["train"]
        val_df = fold["val"]
        test_df = fold["test"]

        # ── Compute per-column (mean, std) on training set ──
        feature_cols = [
            c for c in expert_cols + router_cols if c in train_df.columns
        ]
        stats: dict[str, tuple[float, float]] = {}
        for col in feature_cols:
            vals = train_df[col].dropna()
            mu = float(vals.mean()) if len(vals) else 0.0
            sigma = float(vals.std()) if len(vals) else 1.0
            if sigma < 1e-12:
                sigma = 1.0
            stats[col] = (mu, sigma)

        def _normalise_split(
            split_df: pd.DataFrame,
        ) -> pd.DataFrame:
            split_df = split_df.copy()
            for col, (mu, sigma) in stats.items():
                if col in split_df.columns:
                    split_df[col] = (
                        (split_df[col] - mu) / sigma
                    ).clip(-clip_value, clip_value)
            return split_df

        norm_train = _normalise_split(train_df)
        norm_val = _normalise_split(val_df)
        norm_test = _normalise_split(test_df)

        # ── Compute forward returns for targets (causal: shift(-horizon)) ──
        def _prepare_arrays(
            split_df: pd.DataFrame,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            # Forward return of the *original* (un-normalised) close
            close_col = "Close"
            # We need raw close to compute forward return; grab from
            # the un-normalised fold DataFrames that share the same index.
            raw_close = split_df[close_col] if close_col in split_df.columns else None

            if raw_close is not None:
                fwd_return = raw_close.pct_change().shift(-horizon).fillna(0.0).values
            else:
                fwd_return = np.zeros(len(split_df), dtype=np.float32)

            # Resolve available columns (skip missing ones with zeros)
            expert_arr = _extract_features(split_df, expert_cols)
            router_arr = _extract_features(split_df, router_cols)
            dates_arr = np.arange(len(split_df), dtype=np.int64)

            return expert_arr, router_arr, fwd_return, dates_arr

        def _make_loader(
            split_df: pd.DataFrame,
            shuffle: bool,
        ) -> DataLoader:
            expert_arr, router_arr, fwd_ret, dates = _prepare_arrays(split_df)

            # Need enough rows for at least one sample
            min_rows = lookback + horizon
            if len(split_df) < min_rows:
                logger.warning(
                    "Split has %d rows but needs ≥%d — returning empty loader.",
                    len(split_df),
                    min_rows,
                )
                # Return a loader over a tiny dummy dataset
                expert_arr = np.zeros(
                    (min_rows, expert_arr.shape[1]), dtype=np.float32
                )
                router_arr = np.zeros(
                    (min_rows, router_arr.shape[1]), dtype=np.float32
                )
                fwd_ret = np.zeros(min_rows, dtype=np.float32)
                dates = np.arange(min_rows, dtype=np.int64)

            ds = TradingDataset(
                expert_features=expert_arr,
                router_features=router_arr,
                targets_return=fwd_ret,
                dates=dates,
                lookback=lookback,
                horizon=horizon,
            )
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
            )

        return {
            "train": _make_loader(norm_train, shuffle=True),
            "val": _make_loader(norm_val, shuffle=False),
            "test": _make_loader(norm_test, shuffle=False),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_features(
    df: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    """Extract *columns* from *df* as a float32 NumPy array.

    Missing columns are filled with zeros and logged.
    """
    arrays: list[np.ndarray] = []
    for col in columns:
        if col in df.columns:
            arr = df[col].values.astype(np.float32)
        else:
            logger.debug("Column '%s' missing — filling with zeros.", col)
            arr = np.zeros(len(df), dtype=np.float32)
        # Replace any remaining NaN with 0 (edges of rolling windows)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arrays.append(arr)
    return np.column_stack(arrays)


# ═══════════════════════════════════════════════════════════════════════════
# End-to-end builder
# ═══════════════════════════════════════════════════════════════════════════

def build_dataloaders(config: dict[str, Any]) -> list[dict[str, DataLoader]]:
    """End-to-end pipeline: download → features → normalize → split → DataLoaders.

    Parameters
    ----------
    config : dict
        Nested configuration dictionary matching ``config/settings.yaml``.
        Required keys (with defaults)::

            data.assets            -> list of ticker strings
            data.start_date        -> "2010-01-01"
            data.end_date          -> None (today)
            data.raw_dir           -> "data/raw"
            data.lookback_window   -> 30
            data.forecast_horizon  -> 1
            features.expert_features  -> nested dict of feature name lists
            features.router_features  -> list of router feature names
            training.batch_size    -> 64
            training.walk_forward.train_years  -> 5
            training.walk_forward.val_years    -> 1
            training.walk_forward.test_years   -> 1
            training.walk_forward.step_years   -> 1

    Returns
    -------
    list[dict[str, DataLoader]]
        One dict per walk-forward fold with keys ``"train"``, ``"val"``,
        ``"test"``.
    """
    # ── Parse config ──────────────────────────────────────────────────
    data_cfg = config.get("data", {})
    feat_cfg = config.get("features", {})
    train_cfg = config.get("training", {})
    wf_cfg = train_cfg.get("walk_forward", {})

    tickers: list[str] = data_cfg.get("assets", ["^NSEI"])
    start_date: str = data_cfg.get("start_date", "2010-01-01")
    end_date: str | None = data_cfg.get("end_date", None)
    raw_dir: str = data_cfg.get("raw_dir", "data/raw")
    lookback: int = int(data_cfg.get("lookback_window", 30))
    horizon: int = int(data_cfg.get("forecast_horizon", 1))
    batch_size: int = int(train_cfg.get("batch_size", 64))

    # Flatten expert feature names from nested YAML structure
    expert_features_cfg = feat_cfg.get("expert_features", {})
    expert_cols: list[str] = []
    if isinstance(expert_features_cfg, dict):
        for group_list in expert_features_cfg.values():
            if isinstance(group_list, list):
                expert_cols.extend(group_list)
    else:
        expert_cols = list(expert_features_cfg)
    # Fallback to the module-level names
    if not expert_cols:
        expert_cols = list(EXPERT_FEATURE_NAMES)

    router_cols: list[str] = feat_cfg.get(
        "router_features", list(ROUTER_FEATURE_NAMES)
    )

    train_years: int = int(wf_cfg.get("train_years", 5))
    val_years: int = int(wf_cfg.get("val_years", 1))
    test_years: int = int(wf_cfg.get("test_years", 1))
    step_years: int = int(wf_cfg.get("step_years", 1))

    # ── Step 1: Download ──────────────────────────────────────────────
    logger.info("=== Step 1/4: Data Download ===")
    # Use the first ticker (primary asset); extend later for multi-asset
    ticker = tickers[0]
    raw_df = download_ticker(
        ticker=ticker,
        start=start_date,
        end=end_date,
        raw_dir=raw_dir,
    )

    # ── Step 2: Feature Engineering ───────────────────────────────────
    logger.info("=== Step 2/4: Feature Engineering ===")
    fe = FeatureEngineer()
    feat_df = fe.compute_all_features(raw_df)

    # Drop rows that don't have enough history for all rolling windows
    min_warmup = max(lookback, 100)  # 100 for Hurst, lookback for windows
    if len(feat_df) > min_warmup:
        feat_df = feat_df.iloc[min_warmup:].copy()
        logger.info(
            "Dropped first %d warmup rows — %d rows remaining.",
            min_warmup,
            len(feat_df),
        )

    # ── Step 3: Walk-Forward Splits ───────────────────────────────────
    logger.info("=== Step 3/4: Walk-Forward Splitting ===")
    splitter = WalkForwardSplitter(
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        step_years=step_years,
    )
    folds = splitter.generate_folds(feat_df)

    # ── Step 4: Build DataLoaders per fold ────────────────────────────
    logger.info("=== Step 4/4: Building DataLoaders ===")
    all_loaders: list[dict[str, DataLoader]] = []
    for i, fold in enumerate(folds):
        logger.info("Building loaders for fold %d/%d …", i + 1, len(folds))
        loaders = splitter.build_fold_loaders(
            fold=fold,
            expert_cols=expert_cols,
            router_cols=router_cols,
            lookback=lookback,
            horizon=horizon,
            batch_size=batch_size,
        )
        all_loaders.append(loaders)
        logger.info(
            "Fold %d — train batches=%d, val batches=%d, test batches=%d",
            i,
            len(loaders["train"]),
            len(loaders["val"]),
            len(loaders["test"]),
        )

    logger.info(
        "Pipeline complete — %d folds × 3 splits ready.", len(all_loaders)
    )
    return all_loaders
