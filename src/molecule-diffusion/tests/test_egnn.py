"""
Requires torch (pip install -r requirements.txt). Not runnable in this
sandbox (no network to install torch) — run in your own environment.
"""
import pytest

torch = pytest.importorskip("torch")

from egnn import EGNN  # noqa: E402
from utils import build_edge_mask  # noqa: E402


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


def test_output_shapes():
    net = EGNN(in_node_dim=5, hidden_dim=16, n_layers=2)
    h, x, node_mask, edge_mask = random_batch()
    t = torch.rand(h.shape[0], 1)
    eps_h, eps_x = net(h, x, t, node_mask, edge_mask)
    assert eps_h.shape == h.shape
    assert eps_x.shape == x.shape


def test_padding_produces_zero_output():
    """Padded (masked-out) atoms should get exactly zero predicted noise."""
    net = EGNN(in_node_dim=5, hidden_dim=16, n_layers=2)
    h, x, node_mask, edge_mask = random_batch()
    t = torch.rand(h.shape[0], 1)
    eps_h, eps_x = net(h, x, t, node_mask, edge_mask)
    pad = (node_mask.squeeze(-1) == 0)
    assert torch.allclose(eps_h[pad], torch.zeros_like(eps_h[pad]))
    assert torch.allclose(eps_x[pad], torch.zeros_like(eps_x[pad]))


def test_rotation_equivariance():
    """Rotating the input coordinates should rotate the predicted eps_x by
    the same rotation, and leave eps_h (invariant features) unchanged.
    This is the defining property of the architecture and the reason the
    model can share statistical strength across all molecule orientations.
    """
    torch.manual_seed(0)
    net = EGNN(in_node_dim=5, hidden_dim=16, n_layers=3)
    net.eval()
    h, x, node_mask, edge_mask = random_batch(B=2, N=6)
    t = torch.rand(h.shape[0], 1)

    Q = random_rotation()

    with torch.no_grad():
        eps_h1, eps_x1 = net(h, x, t, node_mask, edge_mask)
        x_rot = x @ Q.T
        eps_h2, eps_x2 = net(h, x_rot, t, node_mask, edge_mask)

    assert torch.allclose(eps_h1, eps_h2, atol=1e-4), "invariant features changed under rotation"
    assert torch.allclose(eps_x1 @ Q.T, eps_x2, atol=1e-4), "coordinate output did not rotate equivariantly"


def test_translation_invariance_of_relative_update():
    """Translating the input should translate eps_x identically (i.e. the
    *displacement* x_out - x_in is translation invariant), since all
    internal computation uses only relative vectors x_i - x_j.
    """
    torch.manual_seed(0)
    net = EGNN(in_node_dim=5, hidden_dim=16, n_layers=3)
    net.eval()
    h, x, node_mask, edge_mask = random_batch(B=2, N=6)
    t = torch.rand(h.shape[0], 1)
    shift = torch.randn(1, 1, 3)

    with torch.no_grad():
        _, eps_x1 = net(h, x, t, node_mask, edge_mask)
        _, eps_x2 = net(h, x + shift * node_mask, t, node_mask, edge_mask)

    assert torch.allclose(eps_x1, eps_x2, atol=1e-4)
