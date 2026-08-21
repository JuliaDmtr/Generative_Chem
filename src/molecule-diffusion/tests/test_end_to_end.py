"""
Requires torch (pip install -r requirements.txt). Not runnable in this
sandbox (no network to install torch) — run in your own environment.

This exercises the full pipeline end to end on the synthetic dataset:
dataset -> collate -> a few training steps -> checkpoint -> generate.py's
sampling path -> a valid .xyz file. It's intentionally tiny (small model,
few molecules, few diffusion steps) so it runs in seconds and is meant to
catch integration bugs, not to produce chemically meaningful molecules.
"""
import functools
import os

import pytest

torch = pytest.importorskip("torch")

from dataset import SyntheticMoleculeDataset, collate_molecules  # noqa: E402
from diffusion import EquivariantMoleculeDiffusion  # noqa: E402
from generate import molecules_from_output, write_xyz  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def test_train_then_generate_smoke(tmp_path):
    torch.manual_seed(0)
    dataset = SyntheticMoleculeDataset(n_molecules=32, min_atoms=4, max_atoms=8, seed=1)
    max_atoms = max(dataset[i]["n_nodes"] for i in range(len(dataset)))
    loader = DataLoader(dataset, batch_size=8, shuffle=True,
                         collate_fn=functools.partial(collate_molecules, max_atoms=max_atoms))

    model = EquivariantMoleculeDiffusion(hidden_dim=16, n_layers=2, timesteps=100)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(3):  # a few epochs is enough for a smoke test
        for x0, h0, node_mask in loader:
            loss, _ = model.loss(x0, h0, node_mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
    assert torch.isfinite(loss)

    # --- generate ---
    model.eval()
    node_mask = torch.zeros(4, max_atoms, 1)
    for b, n in enumerate([4, 5, 6, 7]):
        node_mask[b, :n] = 1.0
    x_t, h_t = model.sample(node_mask, n_steps=10)
    molecules = molecules_from_output(x_t, h_t, node_mask)

    assert len(molecules) == 4
    out_dir = tmp_path / "samples"
    out_dir.mkdir()
    for i, (symbols, coords) in enumerate(molecules):
        assert len(symbols) == coords.shape[0]
        path = out_dir / f"mol_{i}.xyz"
        write_xyz(str(path), symbols, coords)
        assert path.exists()
        lines = path.read_text().splitlines()
        assert int(lines[0]) == len(symbols)
        assert len(lines) == len(symbols) + 2
