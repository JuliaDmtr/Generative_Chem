"""Bond-order recovery from heavy-atom coordinates must reproduce known
molecules: aromatic rings must stay aromatic, saturated rings saturated,
carbonyls double, and the result must not depend on rotation or atom order."""
import numpy as np
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("networkx")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from bond_orders import assign_bond_orders, smiles_from_heavy_coords  # noqa: E402


def heavy_conformer(smiles: str, seed: int = 0xC0FFEE):
    """A real 3D conformer with hydrogens stripped, as the data pipeline makes."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, params) == 0, smiles
    AllChem.MMFFOptimizeMolecule(mol)
    pos = mol.GetConformer().GetPositions()
    keep = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
    return [mol.GetAtomWithIdx(i).GetSymbol() for i in keep], pos[keep]


def canonical(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize("smiles", [
    "c1ccccc1",                 # benzene: aromatic 6-ring
    "C1CCCCC1",                 # cyclohexane: saturated 6-ring, must NOT aromatize
    "c1ccncc1",                 # pyridine: aromatic N
    "CC(C)=O",                  # acetone: C=O must be double
    "c1cc[nH]c1",               # pyrrole: aromatic 5-ring, N keeps its H
    "c1ccoc1",                  # furan: aromatic 5-ring with O
    "CC(=O)Nc1ccccc1",          # paracetamol-like: amide C-N stays single
    "O=C(O)c1ccccc1",           # benzoic acid: carboxyl
])
def test_round_trips_known_molecules(smiles):
    got = smiles_from_heavy_coords(*heavy_conformer(smiles))
    assert got == canonical(smiles), f"{smiles!r} -> {got!r}"


def test_saturated_and_aromatic_rings_are_distinguished():
    """The whole difficulty: both are 6-rings of carbon, differing only in
    geometry (flat vs puckered, 1.39 A vs 1.54 A bonds)."""
    benzene = smiles_from_heavy_coords(*heavy_conformer("c1ccccc1"))
    cyclohexane = smiles_from_heavy_coords(*heavy_conformer("C1CCCCC1"))
    assert benzene == canonical("c1ccccc1")
    assert cyclohexane == canonical("C1CCCCC1")
    assert benzene != cyclohexane


def test_result_is_invariant_to_rotation_and_atom_order():
    symbols, coords = heavy_conformer("O=C(O)c1ccccc1")
    base = smiles_from_heavy_coords(symbols, coords)
    assert base is not None

    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1          # keep it a rotation, not a reflection (stereochemistry)
    assert smiles_from_heavy_coords(symbols, coords @ q.T + 7.0) == base

    perm = rng.permutation(len(symbols))
    assert smiles_from_heavy_coords([symbols[i] for i in perm], coords[perm]) == base


def test_nonsense_geometry_is_rejected_not_guessed():
    """Atoms crammed together must not silently produce a molecule."""
    coords = np.array([[0.0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.3, 0.3, 0.3]])
    assert assign_bond_orders(["C"] * 5, coords) is None


def test_returns_none_rather_than_raising_on_empty_input():
    assert assign_bond_orders([], np.zeros((0, 3))) is None


# --- ring-flatness tolerance -------------------------------------------------
# The planarity test that decides whether a 5/6-ring is aromatic used a literal
# 0.05 A. These tests pin it to a named constant and check that the constant
# genuinely drives the computation, so the tolerance can be calibrated against
# real molecules instead of guessed (scripts/calibrate_bond_orders.py
# --flatness-tol, scripts/diagnose_aromaticity.py).

def _puckered_ring(thickness, n=5, bond=1.40):
    """n-membered carbon ring with a given RMS out-of-plane thickness (angstrom)."""
    import numpy as np
    r = bond / (2.0 * np.sin(np.pi / n))
    ang = 2.0 * np.pi * np.arange(n) / n
    coords = np.stack([r * np.cos(ang), r * np.sin(ang), np.zeros(n)], axis=1)
    z = np.array([(-1.0) ** i for i in range(n)])
    z -= z.mean()
    coords[:, 2] = z
    centred = coords - coords.mean(axis=0)
    got = np.linalg.svd(centred, compute_uv=False)[-1] / np.sqrt(n)
    coords[:, 2] *= thickness / got
    return ["C"] * n, coords


def _n_double_bonds(symbols, coords):
    from rdkit import Chem
    from bond_orders import assign_bond_orders
    mol = assign_bond_orders(symbols, coords)
    assert mol is not None
    return sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)


def test_ring_flatness_tolerance_is_a_named_constant():
    import bond_orders
    assert bond_orders.RING_FLATNESS_TOL == 0.05


def test_ring_flatness_tolerance_drives_the_computation():
    """A ring puckered by 0.07 A is saturated at the default tolerance and
    aromatic at a looser one. If this passes with the tolerance ignored, the
    constant is decoration."""
    import numpy as np
    import bond_orders
    from bond_orders import ring_thickness_of

    symbols, coords = _puckered_ring(0.07)
    assert 0.05 < ring_thickness_of(coords, range(len(symbols))) < 0.12

    assert _n_double_bonds(symbols, coords) == 0        # default 0.05: too puckered
    old = bond_orders.RING_FLATNESS_TOL
    try:
        bond_orders.RING_FLATNESS_TOL = 0.12
        assert _n_double_bonds(symbols, coords) == 2    # looser: aromatic-style ring
    finally:
        bond_orders.RING_FLATNESS_TOL = old
    assert _n_double_bonds(symbols, coords) == 0        # and restored
