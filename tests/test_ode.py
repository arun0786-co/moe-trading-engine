import torch

from engine.models.expert_ode import LiquidODEExpert


def test_grads_match_finite_differences():
    # the old adjoint version re-ran the dynamics with the last step's input
    # during backward and got most of the gradient wrong
    torch.manual_seed(0)
    e = LiquidODEExpert(5, 16, rtol=1e-10, atol=1e-10).double()
    x = torch.randn(4, 6, 5, dtype=torch.double)
    params = list(e.parameters())
    v = [torch.randn_like(p) for p in params]

    e(x).square().sum().backward()
    analytic = sum((p.grad * d).sum() for p, d in zip(params, v))

    def loss_at(step):
        with torch.no_grad():
            for p, d in zip(params, v):
                p.add_(step * d)
            out = e(x).square().sum()
            for p, d in zip(params, v):
                p.sub_(step * d)
        return out

    eps = 1e-5
    numeric = (loss_at(eps) - loss_at(-eps)) / (2 * eps)
    assert abs(float(analytic - numeric)) < 1e-4 * abs(float(numeric))
