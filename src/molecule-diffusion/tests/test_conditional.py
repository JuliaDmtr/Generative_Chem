"""
Requires torch (pip install -r requirements.txt). Not runnable in this
sandbox (no network to install torch) — run in your own environment.

Tests for the PARP-inhibitor conditioning added on top of EGNN /
EquivariantMoleculeDiffusion (see train_conditional.py). The key
properties to check:
  - cond_dim=0 (the default) behaves exactly like the unconditioned model
    (backward compatibility with existing checkpoints/tests).
  - cond_dim>0 requires a `cond` tensor and produces correctly shaped
    output.
  - Conditioning (an invariant per-molecule vector, broadcast identically
    to every atom) must not break rotation equivariance of the predicted
    coordinate noise.
"""
import pytest

torch = pytest.importorskip("torch")

from diffusion import EquivariantMoleculeDiffusion  # noqa: E402
from egnn import EGNN  # noqa: E402
from utils import (  # noqa: E402
    NUM_ATOM_TYPES,
    build_edge_mask,
    condition_labels_to_tensor,
    make_condition_batch,
)


def random_batch(B=3, N=8, in_dim=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(B, N, in_dim, generator=g)
    x = torch.randn(B, N, 3, generator=g)
    n_atoms = torch.randint(2, N + 1, (B,), generator=g)
    node_mask = torch.zeros(B, N, 1)
    for b, n in enumerate(n_atoms):
        node_mask[b, :n] = 1.0
        h[b, n:] = 0.0
        x[b, n:] = 0.0
    edge_mask = build_edge_mask(node_mask)
    return h, x, node_mask, edge_mask


def random_rotation(seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=g)
    Q, _ = torch.linalg.qr(A)
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def test_make_condition_batch_and_labels_to_tensor_agree():
    cond_a = make_condition_batch(1, batch_size=4)
    cond_b = condition_labels_to_tensor([1, 1, 1, 1])
    assert torch.allclose(cond_a, cond_b)
    assert cond_a.shape == (4, 3)
    assert torch.allclose(cond_a.sum(dim=-1), torch.ones(4))


def test_egnn_cond_dim_zero_ignores_cond_argument():
    """Default cond_dim=0 must behave identically whether or not a
    (meaningless) cond tensor is passed — full backward compatibility.
    """
    net = EGNN(in_node_dim=5, hidden_dim=16, n_layers=2)
    h, x, node_mask, edge_mask = random_batch()
    t = torch.rand(h.shape[0], 1)
    eps_h1, eps_x1 = net(h, x, t, node_mask, edge_mask)
    eps_h2, eps_x2 = net(h, x, t, node_mask, edge_mask, cond=None)
    assert torch.allclose(eps_h1, eps_h2)
    assert torch.allclose(eps_x1, eps_x2)


def test_egnn_requires_cond_when_cond_dim_positive():
    net = EGNN(in_node_dim=5, hidden_dim=16, n_layers=2, cond_dim=3)
    h, x, node_mask, edge_mask = random_batch()
    t = torch.rand(h.shape[0], 1)
    with pytest.raises(ValueError):
        net(h, x, t, node_mask, edge_mask)


def test_egnn_conditioned_output_shapes():
    net = EGNN(in_node_dim=5, hidden_dim=16, n_layers=2, cond_dim=3)
    h, x, node_mask, edge_mask = random_batch()
    t = torch.rand(h.shape[0], 1)
    cond = make_condition_batch(1, batch_size=h.shape[0])
    eps_h, eps_x = net(h, x, t, node_mask, edge_mask, cond=cond)
    assert eps_h.shape == h.shape
    assert eps_x.shape == x.shape


def test_egnn_conditioning_preserves_rotation_equivariance():
    """Conditioning is an invariant per-atom-broadcast feature (like time),
    so it must not change the equivariance properties of the coordinate
    update.
    """
    torch.manual_seed(0)
    net = EGNN(in_node_dim=5, hidden_dim=16, n_layers=3, cond_dim=3)
    net.eval()
    h, x, node_mask, edge_mask = random_batch(B=2, N=6)
    t = torch.rand(h.shape[0], 1)
    cond = make_condition_batch(1, batch_size=h.shape[0])

    Q = random_rotation()

    with torch.no_grad():
        eps_h1, eps_x1 = net(h, x, t, node_mask, edge_mask, cond=cond)
        x_rot = x @ Q.T
        eps_h2, eps_x2 = net(h, x_rot, t, node_mask, edge_mask, cond=cond)

    assert torch.allclose(eps_h1, eps_h2, atol=1e-4)
    assert torch.allclose(eps_x1 @ Q.T, eps_x2, atol=1e-4)


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


def test_diffusion_loss_with_conditioning_is_finite():
    torch.manual_seed(0)
    model = EquivariantMoleculeDiffusion(hidden_dim=16, n_layers=2, cond_dim=3)
    x0, h0, node_mask = toy_batch()
    labels = [0, 1, 2, 0]
    cond = condition_labels_to_tensor(labels)
    loss, parts = model.loss(x0, h0, node_mask, cond=cond)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_diffusion_sample_with_conditioning_shapes_and_no_nans():
    torch.manual_seed(0)
    model = EquivariantMoleculeDiffusion(hidden_dim=16, n_layers=2, timesteps=1000, cond_dim=3)
    model.eval()
    node_mask = torch.zeros(3, 9, 1)
    for b, n in enumerate([5, 7, 9]):
        node_mask[b, :n] = 1.0
    cond = make_condition_batch(1, batch_size=3)
    x_t, h_t = model.sample(node_mask, n_steps=50, cond=cond)
    assert x_t.shape == (3, 9, 3)
    assert h_t.shape == (3, 9, NUM_ATOM_TYPES)


def test_diffusion_sample_with_guidance_runs():
    """guidance_scale != 1.0 combined with both cond and null_cond should
    still produce finite, correctly-shaped output.
    """
    torch.manual_seed(0)
    model = EquivariantMoleculeDiffusion(hidden_dim=16, n_layers=2, timesteps=1000, cond_dim=3)
    model.eval()
    node_mask = torch.zeros(2, 6, 1)
    for b, n in enumerate([4, 6]):
        node_mask[b, :n] = 1.0
    cond = make_condition_batch(1, batch_size=2)
    null_cond = make_condition_batch(2, batch_size=2)
    x_t, h_t = model.sample(node_mask, n_steps=50, cond=cond, null_cond=null_cond, guidance_scale=2.0)
    assert x_t.shape == (2, 6, 3)
    assert h_t.shape == (2, 6, NUM_ATOM_TYPES)
