"""
Requires torch (pip install -r requirements.txt). Not runnable in this
sandbox (no network to install torch) — run in your own environment.
"""
import pytest

torch = pytest.importorskip("torch")

from diffusion import EquivariantMoleculeDiffusion, angle_consistency_penalty  # noqa: E402
from utils import NUM_ATOM_TYPES, cosine_alpha_bar, remove_mean_with_mask  # noqa: E402


def toy_batch(B=4, N=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(B, N, 3, generator=g)
    h0 = torch.zeros(B, N, NUM_ATOM_TYPES)
    types = torch.randint(0, NUM_ATOM_TYPES, (B, N), generator=g)
    n_atoms = torch.randint(3, N + 1, (B,), generator=g)
    node_mask = torch.zeros(B, N, 1)
    for b, n in enumerate(n_atoms):
        node_mask[b, :n] = 1.0
        h0[b, torch.arange(n), types[b, :n]] = 1.0
    x0 = x0 * node_mask
    return x0, h0, node_mask


def test_alpha_bar_bounds_and_monotonic():
    t = torch.linspace(0, 1, 50)
    ab = cosine_alpha_bar(t)
    assert torch.isclose(ab[0], torch.tensor(1.0), atol=1e-3)
    assert ab[-1] < 0.01
    assert (ab[1:] <= ab[:-1] + 1e-6).all(), "alpha_bar should be non-increasing in t"


def test_q_sample_matches_target_variance():
    """At a fixed t, z_t's deviation from sqrt(alpha_bar) x0 should have
    (population) variance close to (1 - alpha_bar) — sanity-checks the
    noising formula, in the large-sample limit.
    """
    torch.manual_seed(0)
    model = EquivariantMoleculeDiffusion(hidden_dim=8, n_layers=1)
    x0, h0, node_mask = toy_batch(B=64, N=10)
    t = torch.full((64,), 0.5)
    x_t, h_t, eps_x, eps_h = model.q_sample(x0, h0, t, node_mask)
    ab = cosine_alpha_bar(t)[0].item()
    # reconstruct implied noise and check its empirical std matches sqrt(1-ab)
    implied_eps_h = (h_t - (ab ** 0.5) * h0) / max((1 - ab) ** 0.5, 1e-6)
    mask = node_mask.expand_as(h0).bool()
    assert abs(implied_eps_h[mask].std().item() - 1.0) < 0.25


def test_zero_com_maintained_through_forward_process():
    torch.manual_seed(0)
    model = EquivariantMoleculeDiffusion(hidden_dim=8, n_layers=1)
    x0, h0, node_mask = toy_batch()
    t = torch.rand(x0.shape[0])
    x_t, *_ = model.q_sample(x0, h0, t, node_mask)
    com = (x_t * node_mask).sum(dim=1) / node_mask.sum(dim=1).clamp(min=1.0)
    assert torch.allclose(com, torch.zeros_like(com), atol=1e-5)


def test_loss_is_finite_and_positive():
    torch.manual_seed(0)
    model = EquivariantMoleculeDiffusion(hidden_dim=16, n_layers=2)
    x0, h0, node_mask = toy_batch()
    loss, parts = model.loss(x0, h0, node_mask)
    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert "loss_x" in parts and "loss_h" in parts


def test_loss_decreases_with_a_few_gradient_steps():
    """Overfitting a tiny model to a tiny fixed batch should reduce the
    loss — a basic 'the training loop actually learns something' check.
    """
    torch.manual_seed(0)
    model = EquivariantMoleculeDiffusion(hidden_dim=32, n_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x0, h0, node_mask = toy_batch(B=8, N=6)

    losses = []
    for _ in range(30):
        loss, _ = model.loss(x0, h0, node_mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5


def test_angle_penalty_high_for_60_degrees_low_for_180():
    """Direct sanity check of angle_consistency_penalty's core claim: a
    center atom with two neighbors exactly 60 degrees apart (the
    cyclopropane-like angle diagnose_scale.py flagged as overrepresented
    in generated molecules) should score near the penalty's max (1.0,
    since cos(60deg)=0.5 is exactly the Gaussian bump's center), while a
    180-degree (straight-line) arrangement should score near zero.
    """
    # 3 atoms: atom 0 at origin, two unit-distance neighbors at 60 degrees
    # apart from each other (as seen from atom 0).
    x0_60 = torch.tensor([[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.8660254, 0.0],  # 60 degrees from (1,0,0)
    ]])
    node_mask = torch.ones(1, 3, 1)
    penalty_60 = angle_consistency_penalty(x0_60, node_mask, k_neighbors=4)
    assert penalty_60.item() > 0.9  # cos(60deg)=0.5 is exactly the bump center

    x0_180 = torch.tensor([[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],  # 180 degrees from (1,0,0)
    ]])
    penalty_180 = angle_consistency_penalty(x0_180, node_mask, k_neighbors=4)
    assert penalty_180.item() < 1e-3


def test_angle_penalty_ignores_padded_atoms():
    """A padded (node_mask=0) atom placed exactly at the 60-degree spot
    must not contribute to the penalty — only real atoms should count.
    """
    x0 = torch.tensor([[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.5, 0.8660254, 0.0],  # would score high if not masked out
    ]])
    node_mask = torch.tensor([[[1.0], [1.0], [1.0], [0.0]]])
    penalty = angle_consistency_penalty(x0, node_mask, k_neighbors=4)
    assert penalty.item() < 1e-3


def test_loss_angle_opt_in_and_backward_compatible():
    torch.manual_seed(0)
    x0, h0, node_mask = toy_batch()

    model_off = EquivariantMoleculeDiffusion(hidden_dim=16, n_layers=2)  # angle_loss_weight=0.0 default
    _, parts_off = model_off.loss(x0, h0, node_mask)
    assert "loss_angle" not in parts_off

    model_on = EquivariantMoleculeDiffusion(hidden_dim=16, n_layers=2, angle_loss_weight=0.5)
    loss_on, parts_on = model_on.loss(x0, h0, node_mask)
    assert torch.isfinite(loss_on)
    assert "loss_angle" in parts_on
    assert parts_on["loss_angle"] >= 0.0


def test_sample_shapes_and_no_nans():
    torch.manual_seed(0)
    model = EquivariantMoleculeDiffusion(hidden_dim=16, n_layers=2, timesteps=1000)
    model.eval()
    node_mask = torch.zeros(3, 9, 1)
    for b, n in enumerate([5, 7, 9]):
        node_mask[b, :n] = 1.0
    x_t, h_t = model.sample(node_mask, n_steps=10)  # few steps for a fast test
    assert x_t.shape == (3, 9, 3)
    assert h_t.shape == (3, 9, NUM_ATOM_TYPES)
    assert torch.isfinite(x_t).all()
    assert torch.isfinite(h_t).all()
    # padded atoms should stay at zero
    pad = (node_mask.squeeze(-1) == 0)
    assert torch.allclose(x_t[pad], torch.zeros_like(x_t[pad]), atol=1e-5)
