"""Dual-Head Output Projection.

Maps the expert hidden-state vector ``[B, H]`` to two outputs:

    - **Direction head**: probability that price goes up (sigmoid output).
    - **Magnitude head**: predicted percentage return (unbounded).
    - **Combined signal**: ``expected_return = (dir_prob − 0.5) × 2 × |mag|``.
"""

from __future__ import annotations

import logging
from typing import Final, NamedTuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ProjectionOutput(NamedTuple):
    """Container for dual-head projection outputs.

    Attributes:
        direction_prob: Probability of price going up ``[B, 1]`` ∈ (0, 1).
        magnitude: Predicted percentage return ``[B, 1]`` (unbounded).
        expected_return: Combined signal ``[B, 1]``.
    """

    direction_prob: torch.Tensor
    magnitude: torch.Tensor
    expected_return: torch.Tensor


class DualHeadProjection(nn.Module):
    """Dual-head output projection for trading signal generation.

    Architecture per head::

        Linear(H → 64) → ReLU → Dropout(0.3) → Linear(64 → 1)

    The direction head appends a ``Sigmoid`` activation.

    Args:
        hidden_dim: Dimensionality of the input expert vector (``H``).
        intermediate_dim: Width of the intermediate layer (default 64).
        dropout: Dropout probability (default 0.3).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        intermediate_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_dim: Final[int] = hidden_dim
        self.intermediate_dim: Final[int] = intermediate_dim

        # Direction head: P(price up)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, 1),
            nn.Sigmoid(),
        )

        # Magnitude head: predicted % return
        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, 1),
        )

        self._init_weights()

        logger.info(
            "DualHeadProjection initialised: hidden_dim=%d, "
            "intermediate_dim=%d, dropout=%.2f",
            hidden_dim,
            intermediate_dim,
            dropout,
        )

    # ----- weight init -----
    def _init_weights(self) -> None:
        """Xavier uniform for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ----- forward -----
    def forward(self, x: torch.Tensor) -> ProjectionOutput:
        """Compute direction, magnitude, and combined trading signal.

        Args:
            x: Expert output tensor ``[B, H]``.

        Returns:
            :class:`ProjectionOutput` named tuple containing
            ``direction_prob``, ``magnitude``, and ``expected_return``,
            each of shape ``[B, 1]``.
        """
        direction_prob = self.direction_head(x)   # [B, 1] ∈ (0, 1)
        magnitude = self.magnitude_head(x)        # [B, 1]

        # Combined signal: (dir - 0.5) * 2 * |mag|
        expected_return = (direction_prob - 0.5) * 2.0 * magnitude.abs()

        return ProjectionOutput(
            direction_prob=direction_prob,
            magnitude=magnitude,
            expected_return=expected_return,
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"intermediate_dim={self.intermediate_dim}"
        )
