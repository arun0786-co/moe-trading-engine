"""
Training loop for the MoE Trading Engine.

Features:
- Walk-forward training with per-fold model checkpointing
- AdamW optimizer with cosine annealing LR schedule
- Gradient clipping for stable training
- Early stopping on validation Sharpe ratio (not MSE)
- Mixed precision training (AMP) for GPU acceleration
- Comprehensive WandB logging
- Model checkpointing with best-model tracking
"""

import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration."""
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    gradient_clip: float = 1.0

    direction_loss_weight: float = 0.6
    magnitude_loss_weight: float = 0.3
    balance_loss_weight: float = 0.1

    use_amp: bool = True
    scheduler: str = "cosine"
    warmup_epochs: int = 5

    checkpoint_dir: str = "models"
    wandb_project: str = "moe-trading-engine"
    use_wandb: bool = False


@dataclass
class EpochMetrics:
    """Metrics collected during one epoch."""
    loss_total: float = 0.0
    loss_direction: float = 0.0
    loss_magnitude: float = 0.0
    loss_balance: float = 0.0
    direction_accuracy: float = 0.0
    num_batches: int = 0
    expert_utilization: dict = field(default_factory=dict)


class Trainer:
    """Training loop for the MoE Trading Engine.
    
    Handles the full training lifecycle including:
    - Multi-fold walk-forward training
    - Gradient accumulation and clipping
    - Learning rate scheduling with warmup
    - Early stopping on validation directional accuracy
    - Model checkpointing
    - WandB experiment tracking
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: TrainConfig,
        device: torch.device | None = None,
    ):
        """
        Args:
            model: The MoETradingEngine model
            loss_fn: CombinedTradingLoss instance
            config: Training configuration
            device: Torch device (auto-detected if None)
        """
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Mixed precision only pays off (and only works with the scaler) on cuda
        self.amp = config.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler(self.device.type, enabled=self.amp)

        # Scheduler (set up per fold)
        self.scheduler = None

        # Early stopping state
        self.best_val_metric = -float('inf')
        self.patience_counter = 0
        self.best_epoch = 0

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # WandB
        self._wandb_run = None
        if config.use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=config.wandb_project,
                    config=vars(config),
                    reinit=True,
                )
            except ImportError:
                logger.warning("wandb not installed, skipping experiment tracking")

        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def _setup_scheduler(self, num_epochs: int):
        """Set up learning rate scheduler for a training fold."""
        if self.config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs - self.config.warmup_epochs,
                eta_min=self.config.learning_rate * 0.01,
            )
        elif self.config.scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.5
            )
        else:
            self.scheduler = None

    def _warmup_lr(self, epoch: int):
        """Linear warmup for the first few epochs."""
        if epoch < self.config.warmup_epochs:
            warmup_factor = (epoch + 1) / self.config.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * warmup_factor

    def train_fold(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fold_idx: int = 0,
    ) -> dict:
        """Train the model on one walk-forward fold.
        
        Args:
            train_loader: Training data DataLoader
            val_loader: Validation data DataLoader
            fold_idx: Fold index for checkpointing
        
        Returns:
            Dictionary with training history and best metrics
        """
        logger.info(f"{'='*60}")
        logger.info(f"Training Fold {fold_idx + 1}")
        logger.info(f"{'='*60}")

        self._setup_scheduler(self.config.max_epochs)
        self.best_val_metric = -float('inf')
        self.patience_counter = 0
        self.best_epoch = 0

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "learning_rate": [],
        }

        for epoch in range(self.config.max_epochs):
            # Warmup
            self._warmup_lr(epoch)

            # Train
            train_metrics = self._train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self._validate_epoch(val_loader)

            # Learning rate step (after warmup)
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler and epoch >= self.config.warmup_epochs:
                self.scheduler.step()

            # Record history
            history["train_loss"].append(train_metrics.loss_total)
            history["val_loss"].append(val_metrics.loss_total)
            history["train_accuracy"].append(train_metrics.direction_accuracy)
            history["val_accuracy"].append(val_metrics.direction_accuracy)
            history["learning_rate"].append(current_lr)

            # Log
            logger.info(
                f"Epoch {epoch+1:3d}/{self.config.max_epochs} | "
                f"Train Loss: {train_metrics.loss_total:.4f} | "
                f"Val Loss: {val_metrics.loss_total:.4f} | "
                f"Train Acc: {train_metrics.direction_accuracy:.2%} | "
                f"Val Acc: {val_metrics.direction_accuracy:.2%} | "
                f"LR: {current_lr:.2e}"
            )

            # WandB logging
            if self._wandb_run:
                import wandb
                log_dict = {
                    f"fold_{fold_idx}/train_loss": train_metrics.loss_total,
                    f"fold_{fold_idx}/val_loss": val_metrics.loss_total,
                    f"fold_{fold_idx}/train_dir_acc": train_metrics.direction_accuracy,
                    f"fold_{fold_idx}/val_dir_acc": val_metrics.direction_accuracy,
                    f"fold_{fold_idx}/lr": current_lr,
                    f"fold_{fold_idx}/loss_direction": train_metrics.loss_direction,
                    f"fold_{fold_idx}/loss_magnitude": train_metrics.loss_magnitude,
                    f"fold_{fold_idx}/loss_balance": train_metrics.loss_balance,
                }
                if train_metrics.expert_utilization:
                    for k, v in train_metrics.expert_utilization.items():
                        log_dict[f"fold_{fold_idx}/expert_{k}_usage"] = v
                wandb.log(log_dict, step=epoch)

            # Early stopping on validation accuracy
            val_metric = val_metrics.direction_accuracy
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(fold_idx, epoch, val_metric, is_best=True)
                logger.info(f"  ↑ New best validation accuracy: {val_metric:.2%}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {self.config.patience} epochs)")
                    break

        # Load best model
        best_path = self.checkpoint_dir / f"fold_{fold_idx}_best.pt"
        if best_path.exists():
            self._load_checkpoint(best_path)
            logger.info(f"Loaded best model from epoch {self.best_epoch + 1}")

        result = {
            "fold_idx": fold_idx,
            "best_epoch": self.best_epoch,
            "best_val_accuracy": self.best_val_metric,
            "total_epochs": len(history["train_loss"]),
            "history": history,
        }

        return result

    def _train_epoch(self, loader: DataLoader, epoch: int) -> EpochMetrics:
        """Run one training epoch."""
        self.model.train()
        metrics = EpochMetrics()

        for batch in loader:
            expert_input = batch["expert_input"].to(self.device)
            router_input = batch["router_input"].to(self.device)
            target_return = batch["target_return"].to(self.device)

            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type, enabled=self.amp):
                output = self.model(expert_input, router_input)
                loss_dict = self.loss_fn(
                    output["direction_prob"],
                    output["magnitude"],
                    target_return,
                    output["gate_probs"],
                )

            # Backward with gradient scaling
            self.scaler.scale(loss_dict["total"]).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate metrics
            metrics.loss_total += loss_dict["total"].item()
            metrics.loss_direction += loss_dict["direction"].item()
            metrics.loss_magnitude += loss_dict["magnitude"].item()
            metrics.loss_balance += loss_dict["balance"].item()

            # Direction accuracy
            pred_direction = (output["direction_prob"] > 0.5).float()
            true_direction = (target_return > 0).float()
            correct = (pred_direction == true_direction).float().sum().item()
            metrics.direction_accuracy += correct / target_return.shape[0]
            metrics.num_batches += 1

            # Expert utilization
            if "gate_probs" in output:
                assignments = torch.argmax(output["gate_probs"], dim=-1)
                for i in range(output["gate_probs"].shape[-1]):
                    key = f"expert_{i}"
                    count = (assignments == i).float().mean().item()
                    metrics.expert_utilization[key] = (
                        metrics.expert_utilization.get(key, 0) + count
                    )

        # Average
        if metrics.num_batches > 0:
            metrics.loss_total /= metrics.num_batches
            metrics.loss_direction /= metrics.num_batches
            metrics.loss_magnitude /= metrics.num_batches
            metrics.loss_balance /= metrics.num_batches
            metrics.direction_accuracy /= metrics.num_batches
            for key in metrics.expert_utilization:
                metrics.expert_utilization[key] /= metrics.num_batches

        return metrics

    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader) -> EpochMetrics:
        """Run one validation epoch."""
        self.model.eval()
        metrics = EpochMetrics()

        for batch in loader:
            expert_input = batch["expert_input"].to(self.device)
            router_input = batch["router_input"].to(self.device)
            target_return = batch["target_return"].to(self.device)

            with autocast(device_type=self.device.type, enabled=self.amp):
                output = self.model(expert_input, router_input)
                loss_dict = self.loss_fn(
                    output["direction_prob"],
                    output["magnitude"],
                    target_return,
                    output["gate_probs"],
                )

            metrics.loss_total += loss_dict["total"].item()
            metrics.loss_direction += loss_dict["direction"].item()
            metrics.loss_magnitude += loss_dict["magnitude"].item()
            metrics.loss_balance += loss_dict["balance"].item()

            pred_direction = (output["direction_prob"] > 0.5).float()
            true_direction = (target_return > 0).float()
            correct = (pred_direction == true_direction).float().sum().item()
            metrics.direction_accuracy += correct / target_return.shape[0]
            metrics.num_batches += 1

        if metrics.num_batches > 0:
            metrics.loss_total /= metrics.num_batches
            metrics.loss_direction /= metrics.num_batches
            metrics.loss_magnitude /= metrics.num_batches
            metrics.loss_balance /= metrics.num_batches
            metrics.direction_accuracy /= metrics.num_batches

        return metrics

    def _save_checkpoint(
        self, fold_idx: int, epoch: int, metric: float, is_best: bool = False
    ):
        """Save a model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "metric": metric,
            "fold_idx": fold_idx,
        }

        if is_best:
            path = self.checkpoint_dir / f"fold_{fold_idx}_best.pt"
        else:
            path = self.checkpoint_dir / f"fold_{fold_idx}_epoch_{epoch}.pt"

        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved: {path}")

    def _load_checkpoint(self, path: Path):
        """Load a model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")

    def train_walk_forward(
        self, folds: list[dict[str, DataLoader]]
    ) -> list[dict]:
        """Train across all walk-forward folds.
        
        Args:
            folds: List of dicts, each with 'train' and 'val' DataLoader keys
        
        Returns:
            List of per-fold training results
        """
        all_results = []
        
        for fold_idx, fold_data in enumerate(folds):
            # Reset optimizer for each fold
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            self.scaler = GradScaler(self.device.type, enabled=self.amp)

            result = self.train_fold(
                train_loader=fold_data["train"],
                val_loader=fold_data["val"],
                fold_idx=fold_idx,
            )
            all_results.append(result)

            logger.info(
                f"Fold {fold_idx + 1} complete: "
                f"Best Val Accuracy = {result['best_val_accuracy']:.2%} "
                f"at epoch {result['best_epoch'] + 1}"
            )

        # Summary
        avg_accuracy = np.mean([r["best_val_accuracy"] for r in all_results])
        logger.info(f"\n{'='*60}")
        logger.info(f"Walk-Forward Training Complete")
        logger.info(f"Average Val Accuracy across {len(folds)} folds: {avg_accuracy:.2%}")
        logger.info(f"{'='*60}")

        return all_results

    def predict(
        self,
        expert_input: torch.Tensor,
        router_input: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Generate predictions for new data.
        
        Args:
            expert_input: Expert features [B, T, F_expert]
            router_input: Router features [B, T, F_router]
        
        Returns:
            Dictionary with 'direction_prob', 'magnitude', 'expected_return',
            'gate_probs', 'selected_expert'
        """
        self.model.eval()
        with torch.no_grad():
            expert_input = expert_input.to(self.device)
            router_input = router_input.to(self.device)
            output = self.model(expert_input, router_input)

            # Add expected return
            direction_signal = (output["direction_prob"] - 0.5) * 2.0
            output["expected_return"] = direction_signal * torch.abs(output["magnitude"])

            # Add selected expert index
            output["selected_expert"] = torch.argmax(output["gate_probs"], dim=-1)

        return {k: v.cpu() for k, v in output.items()}
