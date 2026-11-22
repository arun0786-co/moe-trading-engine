import torch

from engine.models.moe import MoETradingEngine
from engine.training.losses import CombinedTradingLoss


def test_outputs_are_flat_and_loss_is_finite():
    torch.manual_seed(0)
    m = MoETradingEngine(expert_input_dim=6, router_input_dim=3, hidden_dim=16, router_hidden_dim=8)
    x, r = torch.randn(4, 10, 6), torch.randn(4, 10, 3)
    o = m(x, r)
    assert o["direction_prob"].shape == (4,)
    assert o["magnitude"].shape == (4,)
    assert o["gate_probs"].shape == (4, 3)
    loss = CombinedTradingLoss()(o["direction_prob"], o["magnitude"], torch.randn(4) * 0.01, o["gate_probs"])
    assert torch.isfinite(loss["total"])
    loss["total"].backward()
    # the router gets a gradient through the balance loss and the gates
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.router.parameters())


def test_column_shaped_predictions_give_the_same_loss():
    # a [B, 1] prediction against [B] targets used to broadcast to [B, B]
    torch.manual_seed(0)
    loss = CombinedTradingLoss()
    prob = torch.rand(8) * 0.8 + 0.1
    mag = torch.randn(8) * 0.01
    tgt = torch.randn(8) * 0.01
    gates = torch.softmax(torch.randn(8, 3), -1)
    flat = loss(prob, mag, tgt, gates)["total"]
    col = loss(prob[:, None], mag[:, None], tgt, gates)["total"]
    torch.testing.assert_close(flat, col)
