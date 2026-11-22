"""Master Gating Network (Router) for the Mixture-of-Experts engine.

Routes each sample in a batch to one or more experts via learned attention
over time steps followed by a gating MLP.

Key features:
    - **Temporal aggregation**: learned attention weights over *T* timesteps.
    - **Gumbel-Softmax**: differentiable discrete expert selection during
      training with temperature annealing.
    - **Hard routing**: argmax selection at inference when the top expert's
      probability exceeds a confidence threshold; soft blend otherwise.
    - **Load-balancing loss**: encourages even utilisation across experts.
    - **Utilization tracking**: exponential moving average of per-expert
      dispatch fractions.
"""

from __future__ import annotations

import logging
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger(__name__)


class MasterRouter(nn.Module):
    """Gating network that produces expert routing weights.

    Args:
        input_dim: Dimensionality of each router input feature vector
            (``F_router``).
        num_experts: Number of experts to route over.
        hidden_dim: Hidden size of the gating MLP (default 64).
        confidence_threshold: During eval, if max gate probability is above
            this value, use hard (argmax) routing (default 0.7).
        balance_coeff: Coefficient *α* for the load-balancing loss
            (default 0.01).
        initial_temperature: Starting Gumbel-Softmax temperature (default 1.0).
        min_temperature: Minimum temperature after annealing (default 0.1).
        temperature_decay: Amount to subtract from temperature per
            ``anneal_temperature()`` call (default 0.01).
        ema_decay: Decay factor for the utilisation running average
            (default 0.99).
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int = 3,
        hidden_dim: int = 64,
        confidence_threshold: float = 0.7,
        balance_coeff: float = 0.01,
        initial_temperature: float = 1.0,
        min_temperature: float = 0.1,
        temperature_decay: float = 0.01,
        ema_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.input_dim: Final[int] = input_dim
        self.num_experts: Final[int] = num_experts
        self.hidden_dim: Final[int] = hidden_dim
        self.confidence_threshold: Final[float] = confidence_threshold
        self.balance_coeff: Final[float] = balance_coeff
        self.min_temperature: Final[float] = min_temperature
        self.temperature_decay: Final[float] = temperature_decay
        self.ema_decay: Final[float] = ema_decay

        # ---- Temporal attention ----
        self.W_attn = nn.Linear(input_dim, input_dim)
        self.v_attn = nn.Parameter(torch.empty(input_dim))

        # ---- Gating MLP ----
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

        # ---- Non-learnable state (buffers) ----
        self.register_buffer(
            "temperature",
            torch.tensor(initial_temperature, dtype=torch.float32),
        )
        self.register_buffer(
            "utilization",
            torch.zeros(num_experts, dtype=torch.float32),
        )

        # Cache for the most recent forward pass (used by get_balance_loss)
        self._last_gate_probs: torch.Tensor | None = None
        self._last_gate_hard: torch.Tensor | None = None

        self._init_weights()

        logger.info(
            "MasterRouter initialised: input_dim=%d, num_experts=%d, "
            "hidden_dim=%d, confidence_threshold=%.2f",
            input_dim,
            num_experts,
            hidden_dim,
            confidence_threshold,
        )

    # ----- weight init -----
    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.W_attn.weight)
        nn.init.zeros_(self.W_attn.bias)
        nn.init.uniform_(self.v_attn, -0.1, 0.1)
        for module in self.gate_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    # ----- temporal attention -----
    def _temporal_aggregate(self, x: torch.Tensor) -> torch.Tensor:
        """Learned attention pooling over the time dimension.

        ``score_t = v^T @ tanh(W_attn @ r_t + b)``
        ``alpha = softmax(scores)``
        ``output = Σ alpha_t · r_t``

        Args:
            x: ``[B, T, F_router]``

        Returns:
            ``[B, F_router]``
        """
        # W_attn applied to every timestep
        energy = torch.tanh(self.W_attn(x))  # [B, T, F]
        scores = (energy * self.v_attn).sum(dim=-1)  # [B, T]
        alpha = F.softmax(scores, dim=-1)  # [B, T]
        aggregated = (alpha.unsqueeze(-1) * x).sum(dim=1)  # [B, F]
        return aggregated

    # ----- forward -----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute expert gate weights.

        Args:
            x: Router input ``[B, T, F_router]``.

        Returns:
            Gate weights ``[B, num_experts]``.  Values sum to 1 along dim=-1.
        """
        # 1. Temporal aggregation → [B, F_router]
        agg = self._temporal_aggregate(x)

        # 2. Gate logits
        logits = self.gate_mlp(agg)  # [B, num_experts]

        if self.training:
            # Gumbel-Softmax for differentiable sampling
            gate_weights = F.gumbel_softmax(
                logits,
                tau=self.temperature.item(),
                hard=False,
                dim=-1,
            )
        else:
            # Soft probabilities
            probs = F.softmax(logits, dim=-1)  # [B, num_experts]
            max_probs, max_idx = probs.max(dim=-1)  # [B]

            # Hard routing where confident, soft blend otherwise
            hard_mask = (max_probs > self.confidence_threshold).unsqueeze(-1)  # [B, 1]
            one_hot = F.one_hot(
                max_idx, num_classes=self.num_experts,
            ).float()  # [B, E]
            gate_weights = torch.where(hard_mask, one_hot, probs)

        # Cache for auxiliary losses
        self._last_gate_probs = F.softmax(logits, dim=-1).detach()
        self._last_gate_hard = gate_weights.detach()

        # Update utilization EMA
        with torch.no_grad():
            batch_fracs = gate_weights.detach().mean(dim=0)  # [E]
            self.utilization.mul_(self.ema_decay).add_(
                batch_fracs * (1.0 - self.ema_decay),
            )

        return gate_weights

    # ----- auxiliary losses -----
    def get_balance_loss(self) -> torch.Tensor:
        """Compute the load-balancing auxiliary loss.

        ``L_balance = α · N · Σ_i (f_i · P_i)``

        where:
            - ``f_i``: fraction of the batch routed to expert *i*
            - ``P_i``: mean routing probability for expert *i*

        Returns:
            Scalar loss tensor (detached gate probs to avoid double
            back-prop through the Gumbel sampling).
        """
        if self._last_gate_probs is None or self._last_gate_hard is None:
            return torch.tensor(0.0, device=self.utilization.device)

        f = self._last_gate_hard.mean(dim=0)  # [E]
        P = self._last_gate_probs.mean(dim=0)  # [E]

        loss = self.balance_coeff * self.num_experts * (f * P).sum()
        return loss

    # ----- utilization stats -----
    def get_utilization_stats(self) -> dict[str, float]:
        """Return per-expert utilisation as a plain dict.

        Returns:
            Mapping from ``"expert_{i}"`` to its running-average dispatch
            fraction.
        """
        return {
            f"expert_{i}": float(self.utilization[i].item())
            for i in range(self.num_experts)
        }

    # ----- temperature annealing -----
    def anneal_temperature(self) -> float:
        """Decay the Gumbel-Softmax temperature by one step.

        Call once per epoch.

        Returns:
            The new temperature value.
        """
        new_temp = max(
            self.temperature.item() - self.temperature_decay,
            self.min_temperature,
        )
        self.temperature.fill_(new_temp)
        logger.debug("Router temperature annealed to %.4f", new_temp)
        return new_temp

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, num_experts={self.num_experts}, "
            f"hidden_dim={self.hidden_dim}, "
            f"confidence_threshold={self.confidence_threshold}"
        )
