"""
Online / Continual Learning module for the MoE Trading Engine.

Implements daily model updates with safeguards against catastrophic forgetting:
- Replay buffer: stores recent data for experience replay
- Elastic Weight Consolidation (EWC): protects important weights
- Validation gate: only deploys updated model if it improves performance
- Automatic rollback: reverts to previous weights on degradation

This runs daily after market close to adapt the model to changing market conditions.
"""

import copy
import logging
from pathlib import Path
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class OnlineLearningConfig:
    """Configuration for online/continual learning."""
    replay_buffer_days: int = 90
    mini_batch_steps: int = 5
    online_lr: float = 1e-4
    ewc_lambda: float = 5000.0
    validation_window_days: int = 30
    batch_size: int = 32


class ReplayBuffer:
    """Fixed-size buffer storing recent training samples for experience replay.
    
    Prevents catastrophic forgetting by mixing new data with recent history
    during online updates.
    """

    def __init__(self, max_days: int = 90):
        """
        Args:
            max_days: Maximum number of days of data to retain
        """
        self.max_days = max_days
        self.expert_inputs: deque[torch.Tensor] = deque(maxlen=max_days)
        self.router_inputs: deque[torch.Tensor] = deque(maxlen=max_days)
        self.targets: deque[torch.Tensor] = deque(maxlen=max_days)

    def add(
        self,
        expert_input: torch.Tensor,
        router_input: torch.Tensor,
        target: torch.Tensor,
    ):
        """Add a day's data to the buffer.
        
        Args:
            expert_input: Expert features [1, T, F_expert] (single sample)
            router_input: Router features [1, T, F_router]
            target: Target return [1, 1]
        """
        self.expert_inputs.append(expert_input.cpu())
        self.router_inputs.append(router_input.cpu())
        self.targets.append(target.cpu())

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor] | None:
        """Sample a random mini-batch from the buffer.
        
        Args:
            batch_size: Number of samples to draw
        
        Returns:
            Dictionary with 'expert_input', 'router_input', 'target_return'
            or None if buffer is empty
        """
        if len(self.expert_inputs) == 0:
            return None

        n = len(self.expert_inputs)
        actual_size = min(batch_size, n)
        indices = np.random.choice(n, size=actual_size, replace=False)

        expert_batch = torch.cat([self.expert_inputs[i] for i in indices], dim=0)
        router_batch = torch.cat([self.router_inputs[i] for i in indices], dim=0)
        target_batch = torch.cat([self.targets[i] for i in indices], dim=0)

        return {
            "expert_input": expert_batch,
            "router_input": router_batch,
            "target_return": target_batch,
        }

    def get_all(self) -> dict[str, torch.Tensor] | None:
        """Get all data in the buffer as a single batch."""
        if len(self.expert_inputs) == 0:
            return None

        return {
            "expert_input": torch.cat(list(self.expert_inputs), dim=0),
            "router_input": torch.cat(list(self.router_inputs), dim=0),
            "target_return": torch.cat(list(self.targets), dim=0),
        }

    def __len__(self) -> int:
        return len(self.expert_inputs)


class EWC:
    """Elastic Weight Consolidation.
    
    Protects weights that were important for past tasks by adding a quadratic
    penalty to the loss function when those weights change.
    
    L_ewc = lambda/2 * sum_i F_i * (theta_i - theta_i*)^2
    
    where F_i is the Fisher information (approximated by squared gradients)
    and theta_i* are the optimal parameters from the previous task.
    
    Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in
    neural networks", PNAS 2017.
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 5000.0):
        """
        Args:
            model: The neural network model
            ewc_lambda: Regularization strength (higher = more conservative updates)
        """
        self.ewc_lambda = ewc_lambda
        self.params: dict[str, torch.Tensor] = {}
        self.fisher: dict[str, torch.Tensor] = {}

    def compute_fisher(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        num_samples: int = 200,
    ):
        """Compute the Fisher Information Matrix (diagonal approximation).
        
        Should be called after training on a dataset to record which weights
        are important for that data distribution.
        
        Args:
            model: Trained model
            data_loader: DataLoader for the training data
            loss_fn: Loss function
            device: Torch device
            num_samples: Number of samples to estimate Fisher
        """
        model.eval()

        # Store current optimal parameters
        self.params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # Initialize Fisher
        self.fisher = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        count = 0
        for batch in data_loader:
            if count >= num_samples:
                break

            expert_input = batch["expert_input"].to(device)
            router_input = batch["router_input"].to(device)
            target = batch["target_return"].to(device)

            model.zero_grad()
            output = model(expert_input, router_input)
            loss_dict = loss_fn(
                output["direction_prob"],
                output["magnitude"],
                target,
                output["gate_probs"],
            )
            loss_dict["total"].backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.data ** 2

            count += expert_input.shape[0]

        # Normalize
        for name in self.fisher:
            self.fisher[name] /= max(count, 1)

        logger.info(f"Fisher Information computed from {count} samples")

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute the EWC penalty for the current model state.
        
        Args:
            model: Current model
        
        Returns:
            Scalar EWC penalty loss
        """
        if not self.params:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self.params and param.requires_grad:
                fisher = self.fisher[name].to(param.device)
                optimal = self.params[name].to(param.device)
                loss += (fisher * (param - optimal) ** 2).sum()

        return self.ewc_lambda / 2.0 * loss


class OnlineLearner:
    """Manages daily online model updates with safety mechanisms.
    
    Workflow:
    1. Add today's data to replay buffer
    2. Sample mini-batches from replay buffer
    3. Update model with EWC regularization
    4. Validate updated model against recent performance
    5. Deploy if improved, rollback if degraded
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: OnlineLearningConfig,
        device: torch.device | None = None,
    ):
        """
        Args:
            model: Pre-trained MoE model
            loss_fn: Combined loss function
            config: Online learning configuration
            device: Torch device
        """
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.replay_buffer = ReplayBuffer(max_days=config.replay_buffer_days)
        self.ewc = EWC(model, ewc_lambda=config.ewc_lambda)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.online_lr,
            weight_decay=1e-5,
        )

        # Store the pre-update model for rollback
        self._backup_state: dict | None = None

        # Performance tracking
        self.update_history: list[dict] = []

        logger.info("OnlineLearner initialized")

    def add_daily_data(
        self,
        expert_input: torch.Tensor,
        router_input: torch.Tensor,
        target: torch.Tensor,
    ):
        """Add today's data point to the replay buffer.
        
        Called daily after market close when we have the actual return
        for today (which was yesterday's prediction target).
        
        Args:
            expert_input: Today's expert features [1, T, F_expert]
            router_input: Today's router features [1, T, F_router]
            target: Today's actual return [1, 1]
        """
        self.replay_buffer.add(expert_input, router_input, target)
        logger.debug(f"Added daily data to replay buffer (size: {len(self.replay_buffer)})")

    def update(self) -> dict:
        """Perform a daily model update.
        
        Returns:
            Dictionary with update metrics:
            - 'updated': bool, whether the model was actually updated
            - 'pre_accuracy': accuracy before update
            - 'post_accuracy': accuracy after update
            - 'rolled_back': bool, whether update was rolled back
            - 'losses': list of losses during update steps
        """
        if len(self.replay_buffer) < self.config.batch_size:
            logger.info(
                f"Replay buffer too small ({len(self.replay_buffer)}), "
                f"skipping update (need {self.config.batch_size})"
            )
            return {"updated": False, "reason": "insufficient_data"}

        # Backup current model state for potential rollback
        self._backup_state = copy.deepcopy(self.model.state_dict())

        # Compute pre-update validation metric
        pre_accuracy = self._evaluate_recent()

        # Training steps with replay
        self.model.train()
        step_losses = []

        for step in range(self.config.mini_batch_steps):
            batch = self.replay_buffer.sample_batch(self.config.batch_size)
            if batch is None:
                break

            expert_input = batch["expert_input"].to(self.device)
            router_input = batch["router_input"].to(self.device)
            target = batch["target_return"].to(self.device)

            self.optimizer.zero_grad()

            output = self.model(expert_input, router_input)
            loss_dict = self.loss_fn(
                output["direction_prob"],
                output["magnitude"],
                target,
                output["gate_probs"],
            )

            # Add EWC penalty
            ewc_penalty = self.ewc.penalty(self.model)
            total_loss = loss_dict["total"] + ewc_penalty

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            step_losses.append(total_loss.item())

        # Compute post-update validation metric
        post_accuracy = self._evaluate_recent()

        # Validation gate: only keep update if it improves performance
        rolled_back = False
        if post_accuracy < pre_accuracy - 0.02:  # Allow 2% tolerance
            logger.warning(
                f"Online update DEGRADED performance: "
                f"{pre_accuracy:.2%} -> {post_accuracy:.2%}. Rolling back."
            )
            self.model.load_state_dict(self._backup_state)
            rolled_back = True
        else:
            logger.info(
                f"Online update accepted: "
                f"{pre_accuracy:.2%} -> {post_accuracy:.2%}"
            )

        result = {
            "updated": not rolled_back,
            "pre_accuracy": pre_accuracy,
            "post_accuracy": post_accuracy,
            "rolled_back": rolled_back,
            "losses": step_losses,
            "ewc_penalty": ewc_penalty.item() if isinstance(ewc_penalty, torch.Tensor) else 0.0,
            "buffer_size": len(self.replay_buffer),
        }
        self.update_history.append(result)

        return result

    @torch.no_grad()
    def _evaluate_recent(self) -> float:
        """Evaluate model on recent replay buffer data.
        
        Returns:
            Directional accuracy on recent data
        """
        self.model.eval()
        data = self.replay_buffer.get_all()
        if data is None:
            return 0.5

        # Use last N days for validation
        n = min(self.config.validation_window_days, len(self.replay_buffer))
        expert_input = data["expert_input"][-n:].to(self.device)
        router_input = data["router_input"][-n:].to(self.device)
        target = data["target_return"][-n:].to(self.device)

        output = self.model(expert_input, router_input)

        pred_direction = (output["direction_prob"] > 0.5).float()
        true_direction = (target > 0).float()
        accuracy = (pred_direction == true_direction).float().mean().item()

        return accuracy

    def update_fisher(self, train_loader: DataLoader):
        """Update EWC Fisher information after a major retraining.
        
        Should be called after weekend full retrain or after initial training.
        
        Args:
            train_loader: DataLoader with training data
        """
        self.ewc.compute_fisher(
            self.model, train_loader, self.loss_fn, self.device
        )
        logger.info("EWC Fisher Information updated")

    def save_state(self, path: str | Path):
        """Save the online learner state (model + buffer + EWC).
        
        Args:
            path: Path to save the state
        """
        path = Path(path)
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ewc_params": self.ewc.params,
            "ewc_fisher": self.ewc.fisher,
            "update_history": self.update_history,
            "buffer_expert": list(self.replay_buffer.expert_inputs),
            "buffer_router": list(self.replay_buffer.router_inputs),
            "buffer_targets": list(self.replay_buffer.targets),
        }
        torch.save(state, path)
        logger.info(f"Online learner state saved to {path}")

    def load_state(self, path: str | Path):
        """Load the online learner state.
        
        Args:
            path: Path to load from
        """
        path = Path(path)
        state = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.ewc.params = state["ewc_params"]
        self.ewc.fisher = state["ewc_fisher"]
        self.update_history = state.get("update_history", [])

        # Restore replay buffer
        if "buffer_expert" in state:
            self.replay_buffer = ReplayBuffer(max_days=self.config.replay_buffer_days)
            for e, r, t in zip(
                state["buffer_expert"],
                state["buffer_router"],
                state["buffer_targets"],
            ):
                self.replay_buffer.add(e, r, t)

        logger.info(f"Online learner state loaded from {path}")
