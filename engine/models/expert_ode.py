"""Liquid Time-Constant Neural ODE Expert.

Implements a continuous-time Liquid Neural Network as an ODE system:
    dh/dt = -[w_leak + f(x, θ)] ⊙ h + f(x, θ) ⊙ h_tilde

where f(x, θ) is a data-dependent time constant produced by a 2-layer MLP,
and h_tilde = tanh(W_h @ h + W_x @ x + b) is the target steady state.

Uses torchdiffeq's Dormand-Prince adaptive-step solver (dopri5), with plain
backpropagation through the solver steps.
"""

from __future__ import annotations

import logging
from typing import Final

import torch
import torch.nn as nn
# plain odeint, not odeint_adjoint: the adjoint pass re-runs the dynamics after
# the time loop is over, when set_input holds the last step's x, so its
# gradients were wrong. the state is small, so the memory saving is not needed
from torchdiffeq import odeint  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ODE function that defines the Liquid Time-Constant dynamics
# ---------------------------------------------------------------------------

class _LiquidODEFunc(nn.Module):
    """ODE right-hand-side for the Liquid Time-Constant network.

    At each ODE solver evaluation, the external input ``x`` is held constant
    (zero-order hold) while the hidden state ``h`` evolves continuously.

    Dynamics::

        f_val   = sigmoid(MLP(x))              # data-dependent time constant
        h_tilde = tanh(W_h @ h + W_x @ x + b) # target steady state
        dh/dt   = -(w_leak + f_val) * h + f_val * h_tilde

    Args:
        input_dim: Dimensionality of the input features.
        hidden_dim: Dimensionality of the hidden state.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim: Final[int] = hidden_dim
        self.input_dim: Final[int] = input_dim

        # Data-dependent time-constant network: 2-layer MLP
        self.tau_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),  # bound f(x, θ) ∈ (0, 1)
        )

        # Steady-state target parameters
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_x = nn.Linear(input_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        # Learnable leak rate (positive via softplus during forward)
        self.w_leak_raw = nn.Parameter(torch.zeros(hidden_dim))

        # External input, set before each ODE integration interval
        self._x: torch.Tensor | None = None

        self._init_weights()

    # ----- weight init -----
    def _init_weights(self) -> None:
        """Xavier uniform for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ----- external input setter -----
    def set_input(self, x: torch.Tensor) -> None:
        """Register the current time-step input for the ODE solver.

        Args:
            x: Input tensor of shape ``[B, F]``.
        """
        self._x = x
        # x is constant over the step: compute its terms once, not per solver call
        self._f_val = self.tau_net(x)
        self._x_drive = self.W_x(x) + self.bias

    # ----- ODE rhs -----
    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:  # noqa: ARG002 – t required by API
        """Compute dh/dt.

        Args:
            t: Current ODE time (unused – autonomous system).
            h: Hidden state ``[B, H]``.

        Returns:
            dh/dt of shape ``[B, H]``.
        """
        assert self._x is not None, "Call set_input(x) before ODE integration."

        f_val = self._f_val                                     # [B, H]
        h_tilde = torch.tanh(self.W_h(h) + self._x_drive)       # [B, H]
        w_leak = nn.functional.softplus(self.w_leak_raw)        # [H]

        dhdt = -(w_leak + f_val) * h + f_val * h_tilde          # [B, H]
        return dhdt


# ---------------------------------------------------------------------------
# Public expert module
# ---------------------------------------------------------------------------

class LiquidODEExpert(nn.Module):
    """Continuous-time Liquid Neural ODE expert for sequential data.

    Processes an input sequence ``[B, T, F]`` by:
        1. Iterating over time steps.
        2. At each step, holding the input constant and integrating the ODE
           from ``t=0`` to ``t=1`` using the dopri5 solver.
        3. Returning the final hidden state ``[B, H]``.

    Args:
        input_dim: Number of input features per time step (``F``).
        hidden_dim: Dimensionality of the hidden state (default 128).
        solver: ODE solver name (default ``'dopri5'``).
        rtol: Relative tolerance for the solver.
        atol: Absolute tolerance for the solver.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        solver: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ) -> None:
        super().__init__()
        self.input_dim: Final[int] = input_dim
        self.hidden_dim: Final[int] = hidden_dim
        self.solver: Final[str] = solver
        self.rtol: Final[float] = rtol
        self.atol: Final[float] = atol

        self.ode_func = _LiquidODEFunc(input_dim, hidden_dim)

        # Integration time span: [0, 1] per step
        self.register_buffer(
            "integration_time",
            torch.tensor([0.0, 1.0], dtype=torch.float32),
        )

        logger.info(
            "LiquidODEExpert initialised: input_dim=%d, hidden_dim=%d, solver=%s",
            input_dim,
            hidden_dim,
            solver,
        )

    # ----- forward -----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the Liquid ODE over a full input sequence.

        Args:
            x: Input tensor of shape ``[B, T, F]``.

        Returns:
            Final hidden state of shape ``[B, H]``.
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        h = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=x.dtype)

        integration_time = self.integration_time.to(dtype=x.dtype)

        for t_step in range(seq_len):
            x_t = x[:, t_step, :]                       # [B, F]
            self.ode_func.set_input(x_t)

            # Integrate from t=0 to t=1 (returns states at both endpoints)
            h_trajectory = odeint(
                self.ode_func,
                h,
                integration_time,
                method=self.solver,
                rtol=self.rtol,
                atol=self.atol,
            )  # [2, B, H]

            h = h_trajectory[-1]  # take state at t=1

        return h  # [B, H]

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"solver={self.solver!r}, rtol={self.rtol}, atol={self.atol}"
        )
