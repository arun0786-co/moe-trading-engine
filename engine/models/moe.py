"""Full Mixture-of-Experts Trading Engine Assembly.

Wires together:
    1. :class:`MasterRouter` — produces gate weights ``[B, 3]``.
    2. Three experts — Liquid ODE, Hyper-LSTM, Selective SSM.
    3. :class:`DualHeadProjection` — maps to direction + magnitude.

Training mode uses **soft routing** (all experts run; weighted sum).
Eval mode uses **hard routing** (only the selected expert runs, for speed).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn

from engine.models.expert_hyper_lstm import HyperLSTMExpert
from engine.models.expert_ode import LiquidODEExpert
from engine.models.expert_ssm import SelectiveSSMExpert
from engine.models.projection import DualHeadProjection, ProjectionOutput
from engine.models.router import MasterRouter

logger = logging.getLogger(__name__)


class MoETradingEngine(nn.Module):
    """Mixture-of-Experts trading engine.

    Args:
        expert_input_dim: Number of input features for the experts (``F``).
        router_input_dim: Number of input features for the router
            (``F_router``).  May differ from ``expert_input_dim`` if the
            router uses a subset of features.
        hidden_dim: Hidden dimensionality shared by all experts
            (default 128).
        router_hidden_dim: Hidden size of the router gating MLP
            (default 64).
        confidence_threshold: Hard-routing threshold for the router
            (default 0.7).
        balance_coeff: Load-balancing loss coefficient (default 0.01).
    """

    NUM_EXPERTS: Final[int] = 3
    EXPERT_NAMES: Final[tuple[str, ...]] = (
        "liquid_ode",
        "hyper_lstm",
        "selective_ssm",
    )

    def __init__(
        self,
        expert_input_dim: int,
        router_input_dim: int,
        hidden_dim: int = 128,
        router_hidden_dim: int = 64,
        confidence_threshold: float = 0.7,
        balance_coeff: float = 0.01,
    ) -> None:
        super().__init__()
        self.expert_input_dim: Final[int] = expert_input_dim
        self.router_input_dim: Final[int] = router_input_dim
        self.hidden_dim: Final[int] = hidden_dim

        # ---- Router ----
        self.router = MasterRouter(
            input_dim=router_input_dim,
            num_experts=self.NUM_EXPERTS,
            hidden_dim=router_hidden_dim,
            confidence_threshold=confidence_threshold,
            balance_coeff=balance_coeff,
        )

        # ---- Experts ----
        self.experts = nn.ModuleList([
            LiquidODEExpert(input_dim=expert_input_dim, hidden_dim=hidden_dim),
            HyperLSTMExpert(input_dim=expert_input_dim, hidden_dim=hidden_dim),
            SelectiveSSMExpert(input_dim=expert_input_dim, hidden_dim=hidden_dim),
        ])

        # ---- Projection ----
        self.projection = DualHeadProjection(hidden_dim=hidden_dim)

        total_params = self.get_param_count()["total"]
        logger.info(
            "MoETradingEngine initialised: expert_input_dim=%d, "
            "router_input_dim=%d, hidden_dim=%d, total_params=%s",
            expert_input_dim,
            router_input_dim,
            hidden_dim,
            f"{total_params:,}",
        )

    # ----- forward -----
    def forward(
        self,
        expert_input: torch.Tensor,
        router_input: torch.Tensor,
    ) -> ProjectionOutput:
        """Run the full MoE forward pass.

        Args:
            expert_input: Input for the experts ``[B, T, F]``.
            router_input: Input for the router ``[B, T, F_router]``.

        Returns:
            :class:`ProjectionOutput` with ``direction_prob``,
            ``magnitude``, and ``expected_return`` (each ``[B, 1]``).
        """
        # 1. Router → gate weights [B, 3]
        gate_weights = self.router(router_input)

        if self.training:
            # Soft routing: run all experts, weighted sum
            expert_outputs = torch.stack(
                [expert(expert_input) for expert in self.experts],
                dim=1,
            )  # [B, 3, H]
            # gate_weights: [B, 3] → [B, 3, 1]
            combined = (gate_weights.unsqueeze(-1) * expert_outputs).sum(
                dim=1,
            )  # [B, H]
        else:
            # Hard routing: only run the selected expert per sample
            combined = self._hard_route(expert_input, gate_weights)

        # 2. Projection → direction + magnitude
        return self.projection(combined)

    # ----- hard routing helper -----
    def _hard_route(
        self,
        expert_input: torch.Tensor,
        gate_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run only the dominant expert(s) during inference.

        When the gate weights are one-hot (hard routing via argmax), only
        the selected expert is executed.  When some samples fall back to
        soft blending, all experts needed by those samples are evaluated.

        Args:
            expert_input: ``[B, T, F]``
            gate_weights: ``[B, num_experts]``

        Returns:
            Combined expert output ``[B, H]``.
        """
        batch_size = expert_input.shape[0]
        device = expert_input.device
        dtype = expert_input.dtype
        combined = torch.zeros(
            batch_size, self.hidden_dim, device=device, dtype=dtype,
        )

        # Determine which experts have non-zero weights
        active_mask = gate_weights > 0  # [B, E]
        active_experts = active_mask.any(dim=0)  # [E]

        # Cache expert outputs to avoid recomputation
        expert_cache: dict[int, torch.Tensor] = {}
        for i in range(self.NUM_EXPERTS):
            if active_experts[i]:
                # Find samples that need this expert
                sample_mask = active_mask[:, i]  # [B]
                if sample_mask.any():
                    expert_out = self.experts[i](
                        expert_input[sample_mask],
                    )  # [n, H]
                    expert_cache[i] = (sample_mask, expert_out)

        # Combine
        for i, (sample_mask, expert_out) in expert_cache.items():
            weights_i = gate_weights[sample_mask, i].unsqueeze(-1)  # [n, 1]
            combined[sample_mask] += weights_i * expert_out

        return combined

    # ----- balance loss pass-through -----
    def get_balance_loss(self) -> torch.Tensor:
        """Return the router's load-balancing auxiliary loss.

        Returns:
            Scalar loss tensor.
        """
        return self.router.get_balance_loss()

    # ----- utilization pass-through -----
    def get_utilization_stats(self) -> dict[str, float]:
        """Return per-expert utilisation statistics.

        Returns:
            Dict mapping expert names to dispatch fractions.
        """
        raw = self.router.get_utilization_stats()
        return {
            self.EXPERT_NAMES[i]: v
            for i, (_, v) in enumerate(sorted(raw.items()))
        }

    # ----- temperature annealing pass-through -----
    def anneal_temperature(self) -> float:
        """Decay the router's Gumbel-Softmax temperature (call per epoch).

        Returns:
            New temperature value.
        """
        return self.router.anneal_temperature()

    # ----- parameter counting -----
    def get_param_count(self) -> dict[str, int]:
        """Count learnable parameters per component.

        Returns:
            Dict with keys ``'router'``, ``'liquid_ode'``,
            ``'hyper_lstm'``, ``'selective_ssm'``, ``'projection'``,
            and ``'total'``.
        """
        counts: dict[str, int] = {}
        counts["router"] = sum(
            p.numel() for p in self.router.parameters() if p.requires_grad
        )
        for i, name in enumerate(self.EXPERT_NAMES):
            counts[name] = sum(
                p.numel()
                for p in self.experts[i].parameters()
                if p.requires_grad
            )
        counts["projection"] = sum(
            p.numel()
            for p in self.projection.parameters()
            if p.requires_grad
        )
        counts["total"] = sum(counts.values())
        return counts

    # ----- checkpointing -----
    def save_checkpoint(self, path: str | Path) -> None:
        """Save model state and metadata to disk.

        Args:
            path: Destination file path (``.pt`` recommended).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "config": {
                "expert_input_dim": self.expert_input_dim,
                "router_input_dim": self.router_input_dim,
                "hidden_dim": self.hidden_dim,
                "num_experts": self.NUM_EXPERTS,
            },
            "param_counts": self.get_param_count(),
        }
        torch.save(payload, path)
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model state from a checkpoint file.

        Args:
            path: Source checkpoint file path.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(payload["model_state_dict"])
        logger.info(
            "Checkpoint loaded from %s (params: %s)",
            path,
            payload.get("param_counts", "unknown"),
        )

    def extra_repr(self) -> str:
        return (
            f"expert_input_dim={self.expert_input_dim}, "
            f"router_input_dim={self.router_input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_experts={self.NUM_EXPERTS}"
        )
