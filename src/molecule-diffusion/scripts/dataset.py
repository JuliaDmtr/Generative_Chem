"""
Dataset utilities.

Two sources are supported:

1. `QM9Dataset` — loads real molecules from an .xyz-per-molecule directory
   or a preprocessed .npz file (positions, atom types, n_nodes per
   molecule). QM9 itself isn't bundled here (no network in this
   environment) — point `root` at a local copy, e.g. downloaded via
   `torch_geometric.datasets.QM9` or the raw QM9 .xyz dump. See
   references/approach.md for exact expected format.

2. `SyntheticMoleculeDataset` — procedurally generates small point clouds
   with plausible bond-length statistics. Not real chemistry, but shaped
   exactly like real batches, so you can smoke-test the whole model /
   training / generation pipeline (including this skill's test suite)
   with zero external data or downloads.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import ATOM2IDX, NUM_ATOM_TYPES


class SyntheticMoleculeDataset(Dataset):
    """Generates random small "molecules": n_atoms atoms placed with
    roughly bond-length-scale spacing (~1.0-1.6 A steps from a random
    walk), with atom types drawn from ATOM_VOCAB with realistic-ish
    frequency (mostly C/H, few N/O/F) — enough structure for the pipeline
    tests to be meaningful without needing a real dataset.
    """

    def __init__(self, n_molecules: int = 512, min_atoms: int = 4, max_atoms: int = 19,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.samples = []
        # Frequency-plausible weights over ATOM_VOCAB (H,C,N,O,F,S,Cl,Br,I,P,B,Si
        # by default) — mostly H/C, common heteroatoms next, rare ones last.
        # Recomputed from ATOM_VOCAB's length so this never desyncs if the
        # vocab is extended/shrunk; only the *shape* of the distribution
        # (front-loaded on H/C) is hand-tuned, not exact per-element values.
        base_weights = [0.45, 0.35, 0.08, 0.10, 0.02]  # H, C, N, O, F
        if NUM_ATOM_TYPES > len(base_weights):
            extra = [0.10 / (NUM_ATOM_TYPES - len(base_weights))] * (NUM_ATOM_TYPES - len(base_weights))
            base_weights = base_weights + extra
        else:
            base_weights = base_weights[:NUM_ATOM_TYPES]
        type_probs = np.array(base_weights)
        type_probs = type_probs / type_probs.sum()
        for _ in range(n_molecules):
            n = int(rng.integers(min_atoms, max_atoms + 1))
            steps = rng.normal(scale=1.3, size=(n, 3))
            pos = np.cumsum(steps, axis=0)
            pos -= pos.mean(axis=0, keepdims=True)  # zero CoM, like real preprocessing
            types = rng.choice(NUM_ATOM_TYPES, size=n, p=type_probs)
            self.samples.append((pos.astype(np.float32), types.astype(np.int64)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pos, types = self.samples[idx]
        return {"pos": pos, "types": types, "n_nodes": len(types)}


class QM9Dataset(Dataset):
    """Loads molecules from a preprocessed .npz with arrays:
        positions: object array of (n_i, 3) float32
        atom_types: object array of (n_i,) int64, indices into ATOM_VOCAB
    Build this once from raw QM9 .xyz files with a short preprocessing
    script (left as a to-do — format documented in references/approach.md
    so you can adapt it to whatever QM9 mirror/loader you have locally).
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.positions = data["positions"]
        self.atom_types = data["atom_types"]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        pos = self.positions[idx].astype(np.float32)
        types = self.atom_types[idx].astype(np.int64)
        return {"pos": pos, "types": types, "n_nodes": len(types)}


def collate_molecules(batch, max_atoms: int | None = None):
    """Pads a list of {'pos', 'types', 'n_nodes'} dicts into dense tensors
    (x0, h0, node_mask) ready for EquivariantMoleculeDiffusion.loss(...).
    """
    n_max = max_atoms or max(b["n_nodes"] for b in batch)
    B = len(batch)
    x0 = torch.zeros(B, n_max, 3)
    h0 = torch.zeros(B, n_max, NUM_ATOM_TYPES)
    node_mask = torch.zeros(B, n_max, 1)

    for i, b in enumerate(batch):
        n = b["n_nodes"]
        x0[i, :n] = torch.from_numpy(b["pos"])
        h0[i, torch.arange(n), torch.from_numpy(b["types"])] = 1.0
        node_mask[i, :n] = 1.0

    return x0, h0, node_mask


def sample_molecule_sizes(dataset: Dataset, n: int, seed: int = 0) -> list[int]:
    """Sample molecule sizes from the empirical size distribution of a
    dataset — used at generation time so generated molecules have a
    realistic size spread rather than a single fixed atom count.
    """
    rng = np.random.default_rng(seed)
    sizes = [dataset[i]["n_nodes"] for i in range(len(dataset))]
    return list(rng.choice(sizes, size=n, replace=True))
