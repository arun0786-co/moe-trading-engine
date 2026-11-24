import torch

from engine.models.expert_ssm import _sequential_scan


def test_scan_matches_closed_form():
    # h_t = a * h_{t-1} + 1 with constant a gives h_t = (1 - a^(t+1)) / (1 - a).
    # T = 30 on purpose: the old scan padded to 32 and broke the decay terms
    a = torch.full((2, 30, 4), 0.9, dtype=torch.double)
    b = torch.ones_like(a)
    t = torch.arange(30, dtype=torch.double).view(1, 30, 1)
    want = (1 - 0.9 ** (t + 1)) / (1 - 0.9)
    assert torch.allclose(_sequential_scan(a, b), want.expand_as(a))


def test_scan_leaves_inputs_alone():
    a, b = torch.rand(2, 30, 4), torch.randn(2, 30, 4)
    a0, b0 = a.clone(), b.clone()
    _sequential_scan(a, b)
    assert torch.equal(a, a0) and torch.equal(b, b0)
