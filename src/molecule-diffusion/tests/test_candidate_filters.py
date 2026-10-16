"""The candidate filters must reject chemically impossible motifs (all of
these were produced by the model and passed the first version of the filters)
while keeping ordinary medicinal chemistry."""
import pytest

pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

from generate_candidates import structural_alerts  # noqa: E402


@pytest.mark.parametrize("smiles,expected", [
    ("O=C(c1ccc(F)cc1)C[C@H]1CC#CCC1", "triple bond in a small ring"),  # cyclohexyne: impossible
    ("CN(F)C", "N-halogen bond"),
    ("C=C=C", "allene"),
    ("CCOOCC", "peroxide"),
    ("CNNNC", "3+ aliphatic heteroatoms in a row"),
])
def test_rejects_implausible_motifs(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    assert expected in structural_alerts(mol), f"{smiles} should be flagged as {expected}"


@pytest.mark.parametrize("smiles", [
    "NC(=O)c1cc(=O)[nH]cc1-c1nc2ccccc2c(=O)[nH]1",  # our own best candidate
    "O=C1N=N/C(=C\\c2cc3ccccc3cn2)c2ccccc21",       # phthalazinone (olaparib core)
    "c1nnn[nH]1",                                   # tetrazole: 4 aromatic N in a row, common in drugs
    "c1nnc[nH]1",                                   # triazole
    "CC(=O)NNC(C)=O",                               # hydrazide: 2 N in a row is fine
    "N#Cc1ccccc1",                                  # nitrile: triple bond NOT in a ring
    "C1CCC#CCCC1",                                  # cyclooctyne: strained but real, 8-ring is allowed
    "CC(=O)N1CCC(CNc2nc3cc(OCC4CCN(C(C)=O)CC4)cc(F)c3c(=O)[nH]2)CC1",  # a real PARP inhibitor
])
def test_keeps_reasonable_chemistry(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    assert structural_alerts(mol) == [], f"{smiles} wrongly flagged: {structural_alerts(mol)}"
