"""Heavy-atom checks must accept real molecular geometry and reject the
specific failure modes we care about (fragments, impossible valences)."""
import numpy as np
import pytest

pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from heavy_atom_eval import heavy_atom_checks, heavy_bond_set, heavy_mol_from_coords  # noqa: E402


def heavy_conformer(smiles: str, seed: int = 0xf00d):
    """Embed a real 3D conformer (with H, as in the data pipeline), then keep
    heavy atoms only — the same operation strip_hydrogens.py performs."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, params) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    pos = mol.GetConformer().GetPositions()
    keep = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
    return [mol.GetAtomWithIdx(i).GetSymbol() for i in keep], pos[keep]


@pytest.mark.parametrize("smiles", [
    "c1ccccc1",                                    # benzene
    "O=C(O)/C=C/c1cccc2c(=O)[nH]ccc12",            # the memorization-test target
    "O=C(c1cc(Cc2n[nH]c(=O)c3ccccc23)ccc1F)N1CCn2c(nnc2-c2ccc(CN3CCSCC3)o2)C1",  # a curated PARP inhibitor
])
def test_real_molecules_pass(smiles):
    res = heavy_atom_checks(*heavy_conformer(smiles))
    assert res["plausible"], res


def test_recovers_the_reference_bond_graph():
    """Perception on heavy atoms only must reproduce the heavy-heavy bonds of
    the molecule it came from (this is what makes the check trustworthy)."""
    smiles = "O=C(O)/C=C/c1cccc2c(=O)[nH]ccc12"
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 7
    assert AllChem.EmbedMolecule(mol, params) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    pos = mol.GetConformer().GetPositions()
    keep = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
    remap = {old: new for new, old in enumerate(keep)}
    truth = {frozenset((remap[b.GetBeginAtomIdx()], remap[b.GetEndAtomIdx()]))
             for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in remap and b.GetEndAtomIdx() in remap}
    got = heavy_bond_set(heavy_mol_from_coords([mol.GetAtomWithIdx(i).GetSymbol() for i in keep], pos[keep]))
    assert got == truth


def test_two_distant_pieces_are_not_connected():
    symbols, coords = heavy_conformer("c1ccccc1")
    far = np.vstack([coords, coords + np.array([30.0, 0.0, 0.0])])
    res = heavy_atom_checks(symbols * 2, far)
    assert res["n_fragments"] == 2 and not res["connected"] and not res["plausible"]
    assert res["valence_ok"], "two intact benzene rings have valid valences"


def test_overcrowded_atom_fails_valence():
    """An oxygen surrounded by five carbons at bonding distance cannot exist."""
    coords = np.array([[0.0, 0.0, 0.0], [1.4, 0, 0], [-1.4, 0, 0],
                       [0, 1.4, 0], [0, -1.4, 0], [0, 0, 1.4]])
    res = heavy_atom_checks(["O", "C", "C", "C", "C", "C"], coords)
    assert not res["valence_ok"] and not res["plausible"]
    assert "O" in res["bad_valence_atoms"]


def test_atoms_far_apart_give_no_bonds():
    coords = np.array([[0.0, 0, 0], [10.0, 0, 0], [20.0, 0, 0]])
    res = heavy_atom_checks(["C", "C", "C"], coords)
    assert res["perceived"] and res["n_fragments"] == 3 and not res["plausible"]


def test_skeleton_smiles_is_rotation_and_order_invariant():
    """The same molecule, rotated and with its atoms reordered, must give the
    same skeleton SMILES — otherwise it cannot be used to compare molecules."""
    from heavy_atom_eval import skeleton_smiles
    symbols, coords = heavy_conformer("O=C(O)/C=C/c1cccc2c(=O)[nH]ccc12")
    base = skeleton_smiles(symbols, coords)
    assert base is not None

    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated = coords @ q.T + np.array([5.0, -2.0, 1.0])
    assert skeleton_smiles(symbols, rotated) == base

    perm = rng.permutation(len(symbols))
    assert skeleton_smiles([symbols[i] for i in perm], coords[perm]) == base


def test_skeleton_smiles_distinguishes_different_molecules():
    from heavy_atom_eval import skeleton_smiles
    a = skeleton_smiles(*heavy_conformer("c1ccccc1"))          # benzene ring
    b = skeleton_smiles(*heavy_conformer("CCCCCC"))            # hexane chain
    assert a is not None and b is not None and a != b
