"""Bond-order assignment must respect each element's valence.

Geometry alone is not enough: an amide nitrogen is planar with three
neighbours (sp2), but it cannot take a double bond — it already has three
bonds and nitrogen's limit is three. The same holds for ether oxygen. Before
this was enforced, the matching handed such atoms a double bond and RDKit
rejected the whole molecule with "Explicit valence greater than permitted",
which was the single largest cause of reconstruction failure on generated
molecules (roughly half of all plausible skeletons).
"""
import numpy as np
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("networkx")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from bond_orders import assign_bond_orders, pi_capacity_limit, smiles_from_heavy_coords  # noqa: E402


def heavy_conformer(smiles: str, seed: int = 11):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, params) == 0, smiles
    AllChem.MMFFOptimizeMolecule(mol)
    pos = mol.GetConformer().GetPositions()
    keep = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
    return [mol.GetAtomWithIdx(i).GetSymbol() for i in keep], pos[keep]


@pytest.mark.parametrize("symbol,degree,expected", [
    ("N", 3, 0),   # amide / aniline N: planar but full
    ("N", 2, 1),   # imine or pyridine N: room for one
    ("N", 1, 2),   # nitrile N
    ("O", 2, 0),   # ether or ester O: full
    ("O", 1, 1),   # carbonyl O
    ("C", 4, 0),
    ("C", 3, 1),
    ("F", 1, 0),   # halogens never take a second bond
])
def test_valence_headroom(symbol, degree, expected):
    assert pi_capacity_limit(symbol, degree) == expected


@pytest.mark.parametrize("smiles", [
    "CC(=O)Nc1ccccc1",          # acetanilide: planar amide N with 3 neighbours
    "COc1ccccc1",               # anisole: ether O with 2 neighbours
    "O=C(N(C)C)c1ccccc1",       # tertiary amide
    "c1ccc2c(c1)OCCO2",         # benzodioxane: two ether oxygens in a ring
    "CN1CCN(C)CC1",             # piperazine: two N with 3 neighbours each
])
def test_molecules_with_full_valence_atoms_reconstruct(smiles):
    got = smiles_from_heavy_coords(*heavy_conformer(smiles))
    expected = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    assert got == expected, f"{smiles} -> {got}"


def test_amide_nitrogen_keeps_single_bonds():
    mol = assign_bond_orders(*heavy_conformer("CC(=O)Nc1ccccc1"))
    assert mol is not None
    n = [a for a in mol.GetAtoms() if a.GetSymbol() == "N"][0]
    assert all(b.GetBondType() == Chem.BondType.SINGLE for b in n.GetBonds()), \
        "amide N must not receive a double bond"
