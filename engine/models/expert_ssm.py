"""Selective State Space Model Expert (Mamba-inspired).

Implements a selective SSM with input-dependent dynamics:

    Continuous:  h'(t) = A(x) h(t) + B(x) x(t),   y(t) = C(x) h(t)
    Discrete (ZOH):
        Ā = exp(Δ · A)
        B̄ = (Ā - I) · A⁻¹ · B         (diagonal A ⇒ element-wise)
        h_t = Ā h_{t-1} + B̄ x_t
        y_t = C h_t

    Δ (step size) is input-dependent: Δ = softplus(Linear(x_t))
    A is diagonal (stored as a vector) for efficiency.

The linear recurrence runs as a plain sequential loop over time (T = 30).
"""

from __future__ import annotations

import logging
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scan for the linear recurrence h_t = a_t * h_{t-1} + b_t
# ---------------------------------------------------------------------------
# A parallel (Blelloch) scan was started here but never finished; the old
# function threw its partial result away and ran the sequential scan anyway.
# With T = 30 the sequential loop is cheap, so it is the only version now.

def _sequential_scan(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Sequential linear-recurrence scan (reference implementation).

    Args:
        a: ``[B, T, N]`` multiplicative coefficients.
        b: ``[B, T, N]`` additive coefficients.

    Returns:
        ``[B, T, N]`` hidden states.
    """
    B_batch, _, N = a.shape
    h = torch.zeros(B_batch, N, device=a.device, dtype=a.dtype)
    outputs: list[torch.Tensor] = []
    # unbind once: indexing a[:, t] in the loop makes backward build a full
    # size zero tensor for every step
    for a_t, b_t in zip(a.unbind(1), b.unbind(1)):
        h = a_t * h + b_t
        outputs.append(h)
    return torch.stack(outputs, dim=1)  # [B, T, N]


# ---------------------------------------------------------------------------
# Selective SSM Expert
# ---------------------------------------------------------------------------

class SelectiveSSMExpert(nn.Module):
    """Selective State Space Model expert (Mamba-inspired).

    Args:
        input_dim: Number of input features per time step (``F``).
        hidden_dim: Output (and working) dimension (default 128).
        state_dim: Internal SSM state dimension ``N`` (default 64).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        state_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim: Final[int] = input_dim
        self.hidden_dim: Final[int] = hidden_dim
        self.state_dim: Final[int] = state_dim

        # Input projection to working dimension D = hidden_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)

        # ---- Input-dependent parameter projections ----
        # A is a learnable *log-space* diagonal (negative real part)
        # Shape: [D, N]  (one diagonal A per feature channel)
        self.A_log = nn.Parameter(torch.empty(hidden_dim, state_dim))

        # B projection: x → B  ([B_batch, T, D] → [B_batch, T, N])
        self.proj_B = nn.Linear(hidden_dim, state_dim, bias=False)
        # C projection: x → C  ([B_batch, T, D] → [B_batch, T, N])
        self.proj_C = nn.Linear(hidden_dim, state_dim, bias=False)

        # Δ (step size) projection: x → Δ  ([B_batch, T, D] → [B_batch, T, D])
        self.proj_delta = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Selective input gate on B and C
        self.gate_B = nn.Linear(hidden_dim, state_dim, bias=True)
        self.gate_C = nn.Linear(hidden_dim, state_dim, bias=True)

        # Output projection back from D to hidden_dim (identity if D == hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Layer norm before output
        self.norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

        logger.info(
            "SelectiveSSMExpert initialised: input_dim=%d, hidden_dim=%d, "
            "state_dim=%d",
            input_dim,
            hidden_dim,
            state_dim,
        )

    # ----- weight init -----
    def _init_weights(self) -> None:
        """Initialise parameters following S4/Mamba conventions."""
        # A_log initialised so that A = -exp(A_log) spans a range of time scales
        # Using the S4 HiPPO-inspired initialisation: A_n = -(n+1)
        with torch.no_grad():
            arange = torch.arange(1, self.state_dim + 1, dtype=torch.float32)
            self.A_log.copy_(
                arange.log().unsqueeze(0).expand(self.hidden_dim, -1),
            )

        for module in (
            self.in_proj,
            self.proj_B,
            self.proj_C,
            self.proj_delta,
            self.gate_B,
            self.gate_C,
            self.out_proj,
        ):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ----- discretize -----
    @staticmethod
    def _discretize(
        A_diag: torch.Tensor,
        B: torch.Tensor,
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-Order Hold discretization for diagonal A.

        Args:
            A_diag: Continuous diagonal A ``[D, N]`` (negative real).
            B: Input-dependent B ``[B_batch, T, N]``.
            delta: Step size ``[B_batch, T, D]``.

        Returns:
            A_bar: ``[B_batch, T, D, N]``
            B_bar: ``[B_batch, T, D, N]``
        """
        # delta: [B, T, D] → [B, T, D, 1]
        delta = delta.unsqueeze(-1)
        # A_diag: [D, N] → [1, 1, D, N]
        A = A_diag.unsqueeze(0).unsqueeze(0)

        # Ā = exp(Δ · A)
        dA = delta * A  # [B, T, D, N]
        A_bar = torch.exp(dA)

        # B̄ = (Ā - I) · A⁻¹ · B  =  (exp(ΔA) - 1) / A · B
        # For numerical stability when A → 0: use expm1
        # (exp(ΔA) - 1) / A = Δ · expm1(ΔA) / (ΔA)  — but simpler:
        # direct:  (A_bar - 1) / A  (A is never exactly 0 due to init)
        B_expand = B.unsqueeze(2).expand_as(A_bar)  # [B, T, D, N]
        # Safe division: A is always negative and non-zero
        B_bar = (A_bar - 1.0) * B_expand / (A + 1e-8)

        return A_bar, B_bar

    # ----- forward -----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a sequence through the selective SSM.

        Args:
            x: Input tensor ``[B, T, F]``.

        Returns:
            Final hidden state ``[B, H]``.
        """
        B_batch, T, _ = x.shape

        # Project to working dimension D
        u = self.in_proj(x)  # [B, T, D]

        # Continuous A (negative real eigenvalues)
        A = -torch.exp(self.A_log)  # [D, N]

        # Input-dependent B, C, Δ
        B = self.proj_B(u)  # [B, T, N]
        C = self.proj_C(u)  # [B, T, N]
        delta = F.softplus(self.proj_delta(u))  # [B, T, D]

        # Selective gating on B and C
        B = B * torch.sigmoid(self.gate_B(u))  # [B, T, N]
        C = C * torch.sigmoid(self.gate_C(u))  # [B, T, N]

        # Discretize
        A_bar, B_bar = self._discretize(A, B, delta)  # [B, T, D, N]

        # Run the linear recurrence per feature channel via scan
        # Reshape for scan: merge D and N → run N independent scans per D
        # For each (d, n): h_t = A_bar[d,n] * h_{t-1} + B_bar[d,n]
        # Then y_t = Σ_n C[n] * h_t[n]  for each d

        D = self.hidden_dim
        N = self.state_dim

        # Reshape: [B, T, D, N] → [B*D, T, N]
        A_flat = A_bar.permute(0, 2, 1, 3).reshape(B_batch * D, T, N)
        B_flat = B_bar.permute(0, 2, 1, 3).reshape(B_batch * D, T, N)

        # Parallel scan
        h_all = _sequential_scan(A_flat, B_flat)  # [B*D, T, N]

        # Reshape back: [B, D, T, N]
        h_all = h_all.reshape(B_batch, D, T, N)

        # Output: y_t = Σ_n C_t[n] * h_t[d, n]
        # C: [B, T, N] → [B, 1, T, N];  h_all: [B, D, T, N]
        C_expand = C.unsqueeze(1)  # [B, 1, T, N]
        y = (h_all * C_expand).sum(dim=-1)  # [B, D, T]
        y = y.permute(0, 2, 1)  # [B, T, D]

        # Take last time step and project
        y_last = y[:, -1, :]  # [B, D]
        output = self.out_proj(self.norm(y_last))  # [B, H]

        return output

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"state_dim={self.state_dim}"
        )
