"""
Tests for the reverse (sampling) step. The central check is independent of
the implementation's own formulas: with an oracle noise prediction (the true
eps), one reverse step from z_t ~ q(z_t | x0) must give z_s ~ q(z_s | x0) =
N(sqrt(ab_s) x0, 1 - ab_s). This holds for ancestral sampling and for DDIM
with any noise level sigma (Song et al. 2021).
"""
import pytest

torch = pytest.importorskip("torch")

from diffusion import EquivariantMoleculeDiffusion, reverse_step  # noqa: E402
from utils import NUM_ATOM_TYPES  # noqa: E402

SAMPLERS = [("ancestral", None), ("ddim", 0.0), ("ddim", 0.2), ("ddim", 1.0)]


@pytest.mark.parametrize("sampler,ddim_noise", SAMPLERS)
@pytest.mark.parametrize("ab_t,ab_s", [(0.3, 0.5), (0.05, 0.9), (0.7, 0.75)])
def test_oracle_step_preserves_marginals(sampler, ddim_noise, ab_t, ab_s):
    g = torch.Generator().manual_seed(0)
    n = 400_000
    x0 = torch.tensor(1.7)
    eps = torch.randn(n, generator=g, dtype=torch.float64)
    z_t = ab_t ** 0.5 * x0 + (1 - ab_t) ** 0.5 * eps
    noise = torch.randn(n, generator=g, dtype=torch.float64)
    z_s = reverse_step(z_t, eps, torch.tensor(ab_t), torch.tensor(ab_s), noise, sampler, ddim_noise)
    assert abs(z_s.mean().item() - ab_s ** 0.5 * x0.item()) < 5e-3
    assert abs(z_s.var().item() - (1 - ab_s)) < 5e-3


@pytest.mark.parametrize("sampler,ddim_noise", SAMPLERS)
def test_last_step_returns_x0_prediction(sampler, ddim_noise):
    """At s = 0 (ab_s = 1) every sampler must return the clean prediction, with no noise."""
    g = torch.Generator().manual_seed(1)
    z_t, eps, noise = (torch.randn(50, generator=g) for _ in range(3))
    ab_t = torch.tensor(0.98)
    x0_pred = (z_t - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
    z_s = reverse_step(z_t, eps, ab_t, torch.tensor(1.0), noise, sampler, ddim_noise)
    assert torch.allclose(z_s, x0_pred, atol=1e-5)


def test_ancestral_matches_ddpm_posterior():
    """Ancestral step = sample from q(z_s | z_t, x0) with x0 = x0_pred:
    mean sqrt(ab_s) b/(1-ab_t) x0 + sqrt(a)(1-ab_s)/(1-ab_t) z_t, var b(1-ab_s)/(1-ab_t),
    with a = ab_t/ab_s, b = 1 - a (Ho et al. 2020, eq. 7)."""
    ab_t, ab_s = torch.tensor(0.4), torch.tensor(0.6)
    a = ab_t / ab_s; b = 1 - a
    z_t, eps = torch.tensor([0.8]), torch.tensor([-0.3])
    x0 = (z_t - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
    mean = ab_s.sqrt() * b / (1 - ab_t) * x0 + a.sqrt() * (1 - ab_s) / (1 - ab_t) * z_t
    std = (b * (1 - ab_s) / (1 - ab_t)).sqrt()
    for noise in (torch.tensor([0.0]), torch.tensor([1.0])):
        z_s = reverse_step(z_t, eps, ab_t, ab_s, noise, "ancestral", None)
        assert torch.allclose(z_s, mean + std * noise, atol=1e-6)


def test_default_sampler_is_unchanged():
    """sampler='ddim', ddim_noise=0.2 must reproduce the original update exactly."""
    g = torch.Generator().manual_seed(2)
    z_t, eps, noise = (torch.randn(20, generator=g) for _ in range(3))
    ab_t, ab_s = torch.tensor(0.3), torch.tensor(0.45)
    x0 = (z_t - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
    sigma = (1 - ab_s).sqrt() * 0.2
    old = ab_s.sqrt() * x0 + (1 - ab_s - sigma ** 2).clamp(min=0).sqrt() * eps + sigma * noise
    assert torch.allclose(reverse_step(z_t, eps, ab_t, ab_s, noise, "ddim", 0.2), old, atol=1e-6)


@pytest.mark.parametrize("sampler,ddim_noise", SAMPLERS)
def test_sample_runs_for_every_sampler(sampler, ddim_noise):
    torch.manual_seed(0)
    model = EquivariantMoleculeDiffusion(hidden_dim=16, n_layers=2).eval()
    node_mask = torch.zeros(3, 9, 1)
    for b, n in enumerate([5, 7, 9]):
        node_mask[b, :n] = 1.0
    x, h = model.sample(node_mask, n_steps=10, sampler=sampler, ddim_noise=ddim_noise)
    assert torch.isfinite(x).all() and torch.isfinite(h).all()
    pad = node_mask.squeeze(-1) == 0
    assert torch.allclose(x[pad], torch.zeros_like(x[pad]))
    com = (x * node_mask).sum(1) / node_mask.sum(1)
    assert torch.allclose(com, torch.zeros_like(com), atol=1e-5)


def test_unknown_sampler_is_rejected():
    with pytest.raises(ValueError):
        reverse_step(torch.zeros(1), torch.zeros(1), torch.tensor(0.5), torch.tensor(0.6),
                     torch.zeros(1), "euler", None)
