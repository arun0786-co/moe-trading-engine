"""
Custom loss functions for the MoE Trading Engine.

Includes:
- Directional loss (binary cross-entropy for up/down classification)
- Asymmetric magnitude loss (penalizes wrong-direction predictions heavily)
- Router load balancing loss (prevents expert collapse)
- Combined weighted loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class DirectionalLoss(nn.Module):
    """Binary cross-entropy loss for direction prediction (up/down).
    
    Predicts whether tomorrow's return will be positive or negative.
    Uses label smoothing to prevent overconfident predictions.
    """

    def __init__(self, label_smoothing: float = 0.05):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, pred_prob: torch.Tensor, target_return: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_prob: Predicted probability of price going up [B, 1], values in (0, 1)
            target_return: Actual returns [B, 1], positive = up, negative = down
        
        Returns:
            Scalar loss value
        """
        # Convert returns to binary labels: 1 if return > 0, else 0
        target_direction = (target_return > 0).float()

        # Apply label smoothing
        if self.label_smoothing > 0:
            target_direction = target_direction * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        loss = F.binary_cross_entropy(pred_prob, target_direction, reduction='mean')
        return loss


class AsymmetricMagnitudeLoss(nn.Module):
    """Asymmetric MSE loss for return magnitude prediction.
    
    Penalizes wrong-direction predictions more heavily than magnitude errors
    in the correct direction. This encourages the model to get the direction
    right first, magnitude second.
    """

    def __init__(self, wrong_direction_weight: float = 3.0):
        """
        Args:
            wrong_direction_weight: Multiplier for loss when predicted direction
                                   disagrees with actual direction.
        """
        super().__init__()
        self.wrong_direction_weight = wrong_direction_weight

    def forward(self, pred_magnitude: torch.Tensor, target_return: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_magnitude: Predicted return magnitude [B, 1]
            target_return: Actual returns [B, 1]
        
        Returns:
            Scalar loss value
        """
        # Base MSE
        sq_error = (pred_magnitude - target_return) ** 2

        # Check if prediction direction matches target direction
        pred_sign = torch.sign(pred_magnitude)
        target_sign = torch.sign(target_return)
        direction_mismatch = (pred_sign != target_sign).float()

        # Apply higher weight to wrong-direction errors
        weights = 1.0 + direction_mismatch * (self.wrong_direction_weight - 1.0)
        weighted_loss = (weights * sq_error).mean()

        return weighted_loss


class RouterBalanceLoss(nn.Module):
    """Load balancing loss for the MoE router.
    
    Prevents mode collapse by penalizing the router when it under-utilizes
    any expert. Based on the Switch Transformer auxiliary loss.
    
    L_balance = alpha * N * sum(f_i * P_i) for i in 1..N
    
    where:
        N = number of experts
        f_i = fraction of tokens routed to expert i (from hard routing)
        P_i = mean routing probability for expert i (from soft routing)
    
    This loss is minimized when all experts are used equally (uniform routing).
    """

    def __init__(self, num_experts: int = 3, alpha: float = 0.1):
        """
        Args:
            num_experts: Number of expert models
            alpha: Scaling factor for the balance loss
        """
        super().__init__()
        self.num_experts = num_experts
        self.alpha = alpha

    def forward(self, gate_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate_probs: Router output probabilities [B, num_experts]
        
        Returns:
            Scalar balance loss
        """
        # f_i: fraction routed to each expert (based on hard assignment)
        expert_assignments = torch.argmax(gate_probs, dim=-1)  # [B]
        f = torch.zeros(self.num_experts, device=gate_probs.device)
        for i in range(self.num_experts):
            f[i] = (expert_assignments == i).float().mean()

        # P_i: mean routing probability for each expert
        P = gate_probs.mean(dim=0)  # [num_experts]

        # Balance loss
        loss = self.alpha * self.num_experts * (f * P).sum()

        return loss


class CombinedTradingLoss(nn.Module):
    """Combined loss function for the MoE Trading Engine.
    
    Weighted sum of:
    1. Directional loss (get the direction right)
    2. Magnitude loss (get the size of the move right) 
    3. Router balance loss (keep all experts active)
    
    The direction loss is weighted most heavily because in trading,
    getting the direction right is more important than predicting
    exact return magnitudes.
    """

    def __init__(
        self,
        direction_weight: float = 0.6,
        magnitude_weight: float = 0.3,
        balance_weight: float = 0.1,
        num_experts: int = 3,
        label_smoothing: float = 0.05,
        wrong_direction_penalty: float = 3.0,
        balance_alpha: float = 0.1,
    ):
        """
        Args:
            direction_weight: Weight for directional (classification) loss
            magnitude_weight: Weight for magnitude (regression) loss
            balance_weight: Weight for router load balancing loss
            num_experts: Number of expert models
            label_smoothing: Label smoothing for direction loss
            wrong_direction_penalty: Penalty multiplier for wrong-direction magnitude errors
            balance_alpha: Alpha scaling for balance loss
        """
        super().__init__()
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        self.balance_weight = balance_weight

        self.direction_loss = DirectionalLoss(label_smoothing=label_smoothing)
        self.magnitude_loss = AsymmetricMagnitudeLoss(wrong_direction_weight=wrong_direction_penalty)
        self.balance_loss = RouterBalanceLoss(num_experts=num_experts, alpha=balance_alpha)

    def forward(
        self,
        pred_direction: torch.Tensor,
        pred_magnitude: torch.Tensor,
        target_return: torch.Tensor,
        gate_probs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the combined loss.
        
        Args:
            pred_direction: Predicted up probability [B, 1]
            pred_magnitude: Predicted return magnitude [B, 1]
            target_return: Actual returns [B, 1]
            gate_probs: Router probabilities [B, num_experts]
        
        Returns:
            Dictionary with 'total', 'direction', 'magnitude', 'balance' losses
        """
        l_dir = self.direction_loss(pred_direction, target_return)
        l_mag = self.magnitude_loss(pred_magnitude, target_return)
        l_bal = self.balance_loss(gate_probs)

        total = (
            self.direction_weight * l_dir
            + self.magnitude_weight * l_mag
            + self.balance_weight * l_bal
        )

        return {
            "total": total,
            "direction": l_dir.detach(),
            "magnitude": l_mag.detach(),
            "balance": l_bal.detach(),
        }
