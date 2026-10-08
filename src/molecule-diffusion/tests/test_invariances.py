"""
Invariance tests that must hold for ANY parameter values, not just at
initialization.

Why parameters are randomized: at init, coord_mlp's last layer has zero bias
and ~0.001-scale weights, so coord_mlp(0) ~ 0 and a padding leak through
masked edges is invisible. A trained network has coord_mlp(0) != 0 (the v6
checkpoint had |coord_mlp(0)| up to 0.15 per layer), so every test here first
randomizes all parameters to look like a trained network.

Requires torch (pip install -r requirements.txt).
"""
import pytest

torch = pytest.importorskip("torch")

from diffusion import EquivariantMoleculeDiffusion  # noqa: E402
from egnn import EGNN  # noqa: E402
from utils import H_SCALE, NUM_ATOM_TYPES, build_edge_mask, cosine_alpha_bar, remove_mean_with_mask  # noqa: E402

IN_DIM = 5
ATOL = 1e-6


def randomized_egnn(seed=0, n_layers=3, hidden_dim=16):
    torch.manual_seed(seed)
    net = EGNN(in_node_dim=IN_DIM, hidden_dim=hidden_dim, n_layers=n_layers).double().eval()
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    return net


def molecule(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(1, n, IN_DIM, generator=g, dtype=torch.float64)
    x = torch.randn(1, n, 3, generator=g, dtype=torch.float64)
    x = x - x.mean(dim=1, keepdim=True)
    return h, x


def pad(h, x, N, garbage_seed=None):
    """Pad one molecule to N slots. Padded slots are zero, or random garbage
    if garbage_seed is given (their content must not matter either)."""
    _, n, _ = h.shape
    hp = torch.zeros(1, N, h.shape[-1], dtype=h.dtype)
    xp = torch.zeros(1, N, 3, dtype=x.dtype)
    hp[:, :n], xp[:, :n] = h, x
    if garbage_seed is not None and N > n:
        g = torch.Generator().manual_seed(garbage_seed)
        hp[:, n:] = torch.randn(1, N - n, h.shape[-1], generator=g, dtype=h.dtype) * 5
        xp[:, n:] = torch.randn(1, N - n, 3, generator=g, dtype=x.dtype) * 5
    node_mask = torch.zeros(1, N, 1, dtype=h.dtype)
    node_mask[:, :n] = 1.0
    return hp, xp, node_mask, build_edge_mask(node_mask)


def run(net, h, x, node_mask, edge_mask, t=0.3):
    with torch.no_grad():
        return net(h, x, torch.full((h.shape[0], 1), t, dtype=h.dtype), node_mask, edge_mask)


@pytest.mark.parametrize("N", [10, 30])
def test_padding_amount_does_not_change_output(N):
    """The same molecule padded to different sizes must get identical
    predictions on its real atoms."""
    net = randomized_egnn()
    h, x = molecule(n=6)
    n = h.shape[1]
    eh_ref, ex_ref = run(net, *pad(h, x, n))
    eh, ex = run(net, *pad(h, x, N))
    assert torch.allclose(ex[:, :n], ex_ref, atol=ATOL), "eps_x depends on the amount of padding"
    assert torch.allclose(eh[:, :n], eh_ref, atol=ATOL), "eps_h depends on the amount of padding"


def test_padded_slot_content_does_not_change_output():
    """Whatever sits in the padded slots (zeros or garbage) must not leak
    into the real atoms' predictions."""
    net = randomized_egnn()
    h, x = molecule(n=6)
    eh0, ex0 = run(net, *pad(h, x, 12))
    eh1, ex1 = run(net, *pad(h, x, 12, garbage_seed=1))
    assert torch.allclose(ex1[:, :6], ex0[:, :6], atol=ATOL), "padded coordinates leak into eps_x"
    assert torch.allclose(eh1[:, :6], eh0[:, :6], atol=ATOL), "padded features leak into eps_h"


def test_padded_coordinates_stay_zero_through_layers():
    """Padded atoms must not drift away from the origin inside the network
    (drifting 'ghost atoms' are a second leak channel in later layers)."""
    net = randomized_egnn()
    h, x = molecule(n=6)
    hp, xp, node_mask, edge_mask = pad(h, x, 12)
    t = torch.full((1, 1, 1), 0.3, dtype=hp.dtype).expand(1, 12, 1)
    with torch.no_grad():
        h_i = net.embed_in(torch.cat([hp, t], dim=-1)) * node_mask
        x_i = xp
        for k, layer in enumerate(net.layers):
            h_i, x_i = layer(h_i, x_i, node_mask, edge_mask)
            assert torch.allclose(x_i[:, 6:], torch.zeros_like(x_i[:, 6:]), atol=ATOL), \
                f"padded coordinates drifted after layer {k}"


def test_permutation_equivariance():
    """Reordering the real atoms must reorder the outputs the same way."""
    net = randomized_egnn()
    h, x = molecule(n=7)
    perm = torch.randperm(7, generator=torch.Generator().manual_seed(3))
    eh, ex = run(net, *pad(h, x, 7))
    eh_p, ex_p = run(net, *pad(h[:, perm], x[:, perm], 7))
    assert torch.allclose(ex_p, ex[:, perm], atol=ATOL)
    assert torch.allclose(eh_p, eh[:, perm], atol=ATOL)


def test_eps_x_has_zero_center_of_mass():
    """The network's coordinate output must live on the zero-CoM subspace,
    in training as well as sampling (the target eps_x does)."""
    net = randomized_egnn()
    h, x = molecule(n=6)
    hp, xp, node_mask, edge_mask = pad(h, x, 10)
    _, ex = run(net, hp, xp, node_mask, edge_mask)
    com = (ex * node_mask).sum(dim=1) / node_mask.sum(dim=1)
    assert torch.allclose(com, torch.zeros_like(com), atol=ATOL), f"eps_x CoM = {com}"


def test_coordinate_update_is_bounded_for_large_inputs():
    """Per-layer coordinate updates must stay bounded as the molecule gets
    larger (EDM: (x_i - x_j)/(d_ij + 1) and a tanh-bounded weight). An
    unbounded update grows linearly with scale and lets small errors run
    away during sampling."""
    net = randomized_egnn()
    h, x = molecule(n=6)
    _, ex_small = run(net, *pad(h, x * 10, 6))
    _, ex_large = run(net, *pad(h, x * 1000, 6))
    ratio = ex_large.abs().max() / ex_small.abs().max()
    assert ratio < 2.0, f"coordinate update grew {ratio:.1f}x when the molecule was scaled 100x"


def test_h_scale_applies_to_the_diffused_data():
    """EDM's feature scaling works by shrinking h0 *before* diffusion, which
    lowers the signal-to-noise ratio of atom types at every t. Scaling only
    the network input is absorbed by the first linear layer and does nothing.
    Checks: the noised h_t carries signal sqrt(ab) * H_SCALE * h0, and the
    network receives exactly that h_t."""
    torch.manual_seed(0)
    model = EquivariantMoleculeDiffusion(hidden_dim=16, n_layers=2)
    B, n = 2, 5
    x0 = remove_mean_with_mask(torch.randn(B, n, 3), torch.ones(B, n, 1))
    h0 = torch.zeros(B, n, NUM_ATOM_TYPES)
    h0[..., 1] = 1.0
    node_mask = torch.ones(B, n, 1)

    captured = {}
    orig_q_sample, orig_forward = model.q_sample, model.net.forward

    def q_sample_spy(x0_, h0_, t_, node_mask_):
        out = orig_q_sample(x0_, h0_, t_, node_mask_)
        captured["t"], captured["h_t"], captured["eps_h"] = t_, out[1], out[3]
        return out

    def forward_spy(h, *args, **kwargs):
        captured["net_h"] = h
        return orig_forward(h, *args, **kwargs)

    model.q_sample, model.net.forward = q_sample_spy, forward_spy
    model.loss(x0, h0, node_mask)

    ab = cosine_alpha_bar(captured["t"]).view(B, 1, 1)
    signal = captured["h_t"] - (1 - ab).sqrt() * captured["eps_h"]
    assert torch.allclose(signal, ab.sqrt() * H_SCALE * h0, atol=1e-5), \
        "h0 is not scaled by H_SCALE before diffusion"
    assert torch.allclose(captured["net_h"], captured["h_t"], atol=1e-6), \
        "network input h differs from the diffused h_t (extra input-only scaling?)"


def test_old_checkpoints_are_refused():
    """Checkpoints trained before the padding fix compute something else and
    must not load silently."""
    from utils import ARCH_VERSION, check_arch_version
    check_arch_version({"arch_version": ARCH_VERSION})
    with pytest.raises(ValueError):
        check_arch_version({})  # pre-versioning checkpoint
