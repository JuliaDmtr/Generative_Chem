"""The periodic training diagnostics must use the heavy-atom checks on
heavy-atom-only data: the with-H RDKit pipeline reports ~0% there regardless
of molecule quality, which would make checkpoint selection meaningless."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from train_conditional import run_validity_diagnostics  # noqa: E402
from utils import ATOM2IDX, COND_POSITIVE  # noqa: E402


class FakeModel:
    """Stands in for the diffusion model: returns the same real heavy-atom
    molecule for every requested sample, so the diagnostics' only job is to
    judge known-good molecules."""

    def __init__(self, symbols, coords):
        self.symbols, self.coords = symbols, coords

    def sample(self, node_mask, cond=None, n_steps=None, device=None):
        B, N, _ = node_mask.shape
        n = len(self.symbols)
        assert N >= n
        x = torch.zeros(B, N, 3)
        h = torch.zeros(B, N, len(ATOM2IDX))
        for b in range(B):
            x[b, :n] = torch.tensor(self.coords, dtype=torch.float32)
            for i, sym in enumerate(self.symbols):
                h[b, i, ATOM2IDX[sym]] = 1.0
        return x, h


@pytest.fixture
def real_heavy_molecule():
    mol = Chem.AddHs(Chem.MolFromSmiles("O=C(O)/C=C/c1cccc2c(=O)[nH]ccc12"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 3
    assert AllChem.EmbedMolecule(mol, params) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    pos = mol.GetConformer().GetPositions()
    keep = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
    return [mol.GetAtomWithIdx(i).GetSymbol() for i in keep], pos[keep]


def run(model, n, **kw):
    return run_validity_diagnostics(model, COND_POSITIVE, np.array([n]), n_samples=4,
                                    n_steps=1, device=torch.device("cpu"),
                                    rng=np.random.default_rng(0), **kw)


def test_heavy_atom_mode_accepts_real_molecules(real_heavy_molecule):
    symbols, coords = real_heavy_molecule
    validity, single = run(FakeModel(symbols, coords), len(symbols), no_hydrogens=True)
    assert validity == 1.0 and single == 1.0


def test_with_hydrogen_mode_would_reject_the_same_molecules(real_heavy_molecule):
    """Documents why the flag is needed: the default pipeline scores these
    perfectly good heavy-atom molecules far below the heavy-atom checks."""
    symbols, coords = real_heavy_molecule
    validity, single = run(FakeModel(symbols, coords), len(symbols), no_hydrogens=False)
    assert validity < 1.0


def test_heavy_atom_mode_rejects_fragments(real_heavy_molecule):
    symbols, coords = real_heavy_molecule
    far = np.vstack([coords, coords + np.array([40.0, 0.0, 0.0])])
    validity, single = run(FakeModel(symbols * 2, far), len(symbols) * 2, no_hydrogens=True)
    assert single == 0.0, "two separate pieces must not count as one molecule"
