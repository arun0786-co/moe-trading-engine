"""Hyper-LSTM Expert with Dynamic Weight Generation.

A primary LSTM whose gate weights are dynamically modulated at every time step
by a smaller *hyper-network* LSTM.  At each step *t*:

    1.  The hyper-LSTM processes ``[h_{t-1}, x_t]`` → ``hyper_hidden``.
    2.  ``hyper_hidden`` is projected to row-scale ``d_row ∈ ℝ^H`` and
        column-scale ``d_col ∈ ℝ^{F+H}`` vectors.
    3.  The unified base weight matrix ``W_base ∈ ℝ^{4H × (F+H)}`` is
        element-wise scaled:  ``W_t(i,j) = W_base(i,j) * d_row(i) * d_col(j)``.
    4.  The primary LSTM gates are computed using ``W_t``.

Additional regularisation:
    - **Layer normalisation** on the cell state before the output gate.
    - **Zoneout** (prob=0.05): stochastically preserves previous hidden/cell.
"""

from __future__ import annotations

import logging
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger(__name__)


class HyperLSTMExpert(nn.Module):
    """Hyper-LSTM expert for sequential trading features.

    Args:
        input_dim: Number of input features per time step (``F``).
        hidden_dim: Hidden dimension of the primary LSTM (default 128).
        hyper_hidden_dim: Hidden dimension of the hyper-network LSTM
            (default 32).
        zoneout_prob: Probability of keeping the *previous* hidden/cell
            state (default 0.05).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        hyper_hidden_dim: int = 32,
        zoneout_prob: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_dim: Final[int] = input_dim
        self.hidden_dim: Final[int] = hidden_dim
        self.hyper_hidden_dim: Final[int] = hyper_hidden_dim
        self.zoneout_prob: Final[float] = zoneout_prob

        concat_dim: int = input_dim + hidden_dim  # F + H

        # ---------- Hyper-network LSTM ----------
        # Input to hyper-LSTM: [h_{t-1}, x_t] → concat_dim
        self.hyper_lstm_cell = nn.LSTMCell(concat_dim, hyper_hidden_dim)

        # Projections from hyper-hidden → row / column scaling vectors
        # d_row must broadcast over the 4 gate blocks, so output is 4*H
        self.proj_d_row = nn.Linear(hyper_hidden_dim, 4 * hidden_dim)
        self.proj_d_col = nn.Linear(hyper_hidden_dim, concat_dim)

        # ---------- Primary LSTM ----------
        # Unified weight matrix for all 4 gates: [4H, F+H]
        self.W_base = nn.Parameter(torch.empty(4 * hidden_dim, concat_dim))
        self.b_gates = nn.Parameter(torch.zeros(4 * hidden_dim))

        # Layer normalisation on cell state
        self.cell_ln = nn.LayerNorm(hidden_dim)

        self._init_weights()

        logger.info(
            "HyperLSTMExpert initialised: input_dim=%d, hidden_dim=%d, "
            "hyper_hidden_dim=%d, zoneout=%.2f",
            input_dim,
            hidden_dim,
            hyper_hidden_dim,
            zoneout_prob,
        )

    # ----- weight init -----
    def _init_weights(self) -> None:
        """Xavier for linear/projections, orthogonal for recurrent weights."""
        nn.init.xavier_uniform_(self.W_base)

        # Hyper LSTM cell – orthogonal for recurrent, xavier for input
        nn.init.xavier_uniform_(self.hyper_lstm_cell.weight_ih)
        nn.init.orthogonal_(self.hyper_lstm_cell.weight_hh)
        nn.init.zeros_(self.hyper_lstm_cell.bias_ih)
        nn.init.zeros_(self.hyper_lstm_cell.bias_hh)

        for proj in (self.proj_d_row, self.proj_d_col):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.ones_(proj.bias)  # init scaling near 1.0

    # ----- zoneout helper -----
    def _zoneout(
        self,
        new: torch.Tensor,
        prev: torch.Tensor,
    ) -> torch.Tensor:
        """Apply zoneout: with probability ``p`` keep ``prev``.

        Args:
            new: Newly computed state ``[B, H]``.
            prev: Previous state ``[B, H]``.

        Returns:
            Regularised state ``[B, H]``.
        """
        if self.training and self.zoneout_prob > 0.0:
            mask = torch.bernoulli(
                torch.full_like(new, 1.0 - self.zoneout_prob),
            )
            return mask * new + (1.0 - mask) * prev
        return new

    # ----- forward -----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a full input sequence through the Hyper-LSTM.

        Args:
            x: Input tensor of shape ``[B, T, F]``.

        Returns:
            Final hidden state of shape ``[B, H]``.
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Primary LSTM states
        h = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

        # Hyper-network LSTM states
        h_hyper = torch.zeros(
            batch_size, self.hyper_hidden_dim, device=device, dtype=dtype,
        )
        c_hyper = torch.zeros(
            batch_size, self.hyper_hidden_dim, device=device, dtype=dtype,
        )

        for t_step in range(seq_len):
            x_t = x[:, t_step, :]  # [B, F]

            # 1. Hyper-LSTM step ───────────────────────────────────────
            hyper_input = torch.cat([h, x_t], dim=-1)  # [B, F+H]
            h_hyper, c_hyper = self.hyper_lstm_cell(hyper_input, (h_hyper, c_hyper))

            # 2. Compute scaling vectors ───────────────────────────────
            d_row = self.proj_d_row(h_hyper)  # [B, 4H]
            d_col = self.proj_d_col(h_hyper)  # [B, F+H]

            # 3. Scale base weights: W_t(i,j) = W_base(i,j) * d_row(i) * d_col(j)
            # W_base: [4H, F+H]
            # d_row:  [B, 4H] -> [B, 4H, 1]
            # d_col:  [B, F+H] -> [B, 1, F+H]
            scale = d_row.unsqueeze(2) * d_col.unsqueeze(1)  # [B, 4H, F+H]
            W_t = self.W_base.unsqueeze(0) * scale  # [B, 4H, F+H]

            # 4. Primary LSTM gates ────────────────────────────────────
            concat_input = torch.cat([x_t, h], dim=-1)  # [B, F+H]
            # Batched matmul: [B, 4H, F+H] @ [B, F+H, 1] -> [B, 4H, 1]
            gates = torch.bmm(W_t, concat_input.unsqueeze(2)).squeeze(2)  # [B, 4H]
            gates = gates + self.b_gates  # broadcast bias

            # Split into 4 gates
            H = self.hidden_dim
            f_gate = torch.sigmoid(gates[:, 0 * H : 1 * H])       # forget
            i_gate = torch.sigmoid(gates[:, 1 * H : 2 * H])       # input
            g_gate = torch.tanh(gates[:, 2 * H : 3 * H])          # candidate
            o_gate = torch.sigmoid(gates[:, 3 * H : 4 * H])       # output

            # 5. Cell & hidden update ──────────────────────────────────
            c_new = f_gate * c + i_gate * g_gate
            c_new = self.cell_ln(c_new)
            h_new = o_gate * torch.tanh(c_new)

            # 6. Zoneout ──────────────────────────────────────────────
            h = self._zoneout(h_new, h)
            c = self._zoneout(c_new, c)

        return h  # [B, H]

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"hyper_hidden_dim={self.hyper_hidden_dim}, "
            f"zoneout_prob={self.zoneout_prob}"
        )
