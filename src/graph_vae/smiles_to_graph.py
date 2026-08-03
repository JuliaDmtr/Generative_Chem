"""
smiles_to_graph.py

Converts a SMILES string into a PyTorch Geometric graph Data object,
suitable as input to a GNN encoder for the molecule VAE.

Atom features (one-hot / numeric, concatenated):
    - atom type one-hot over a fixed vocabulary of common elements
    - degree (number of bonded neighbors)
    - formal charge
    - aromaticity flag

Edge features:
    - edge_index: [2, num_edges] connectivity (undirected -> stored both directions)
    - edge_attr: one-hot bond type (single, double, triple, aromatic)
"""

import torch
from torch_geometric.data import Data
from rdkit import Chem


# Fixed vocabulary of atom types we expect to see in ChEMBL-like drug molecules.
# Anything outside this list falls into the last "other" slot.
ATOM_TYPES = ["C", "N", "O", "F", "S", "Cl", "Br", "I", "P", "B", "Si", "other"]
ATOM_TYPE_TO_IDX = {atom: i for i, atom in enumerate(ATOM_TYPES)}
MAX_ATOMS = 60  

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_TYPE_TO_IDX = {bond: i for i, bond in enumerate(BOND_TYPES)}


def one_hot(idx: int, length: int) -> list:
    """Simple one-hot encoding helper."""
    vec = [0.0] * length
    vec[idx] = 1.0
    return vec


def atom_features(atom: Chem.Atom) -> list:
    """Build the feature vector for a single atom."""
    symbol = atom.GetSymbol()
    atom_idx = ATOM_TYPE_TO_IDX.get(symbol, ATOM_TYPE_TO_IDX["other"])
    type_one_hot = one_hot(atom_idx, len(ATOM_TYPES))

    degree = atom.GetDegree()
    charge = atom.GetFormalCharge()
    is_aromatic = 1.0 if atom.GetIsAromatic() else 0.0

    return type_one_hot + [float(degree), float(charge), is_aromatic]


def smiles_to_graph(smiles: str) -> Data | None:
    """
    Convert a SMILES string into a PyG Data object.

    Returns None if the SMILES is invalid or cannot be parsed by RDKit,
    so the caller (e.g. Dataset.__getitem__) can skip it.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    if mol.GetNumAtoms() > MAX_ATOMS:
        return None  # skip molecules that exceed the max atom limit

    # --- Canonical atom ordering ---
    # RDKit assigns atom indices in the order atoms appear in the SMILES string,
    # which is NOT a structural property of the molecule - the same molecule
    # written as a different (but equivalent) SMILES can get a totally different
    # atom order. Since our decoder predicts a fixed "slot 1, slot 2, ..." order,
    # we need atom order to depend only on molecular structure, not on how the
    # SMILES happened to be written. CanonicalRankAtoms gives exactly that:
    # a structure-based canonical rank for every atom.
    canonical_ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    # canonical_ranks[old_idx] = new_canonical_position
    # build the permutation that reorders atoms 0..n-1 into canonical order
    old_to_new = {old_idx: new_idx for old_idx, new_idx in enumerate(canonical_ranks)}

    # --- Node features, built in canonical order ---
    atoms_in_order = sorted(mol.GetAtoms(), key=lambda a: old_to_new[a.GetIdx()])
    node_feats = [atom_features(atom) for atom in atoms_in_order]
    if len(node_feats) == 0:
        return None
    x = torch.tensor(node_feats, dtype=torch.float)

    # --- Edges (undirected: add both directions) ---
    edge_indices = []
    edge_feats = []
    for bond in mol.GetBonds():
        # remap RDKit's original parse-order indices to canonical positions
        i = old_to_new[bond.GetBeginAtomIdx()]
        j = old_to_new[bond.GetEndAtomIdx()]
        bond_type = bond.GetBondType()
        bond_idx = BOND_TYPE_TO_IDX.get(bond_type, 0)  # default to SINGLE if unseen
        bond_one_hot = one_hot(bond_idx, len(BOND_TYPES))

        # add both (i -> j) and (j -> i) since PyG expects directed edge_index
        # for undirected graphs
        edge_indices.append([i, j])
        edge_feats.append(bond_one_hot)
        edge_indices.append([j, i])
        edge_feats.append(bond_one_hot)

    if len(edge_indices) == 0:
        # single-atom molecule edge case: no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(BOND_TYPES)), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles  # keep original string around for debugging/inspection
    return data


if __name__ == "__main__":
    # Quick manual test - run this file directly to sanity check the conversion.
    test_smiles = [
        "CCO",                      # ethanol
        "c1ccccc1",                 # benzene
        "CC(=O)Oc1ccccc1C(=O)O",    # aspirin
        "not_a_real_smiles!!!",     # should return None
    ]
    for smi in test_smiles:
        g = smiles_to_graph(smi)
        if g is None:
            print(f"{smi!r} -> INVALID (skipped)")
        else:
            print(f"{smi!r} -> x.shape={tuple(g.x.shape)}, "
                  f"edge_index.shape={tuple(g.edge_index.shape)}, "
                  f"edge_attr.shape={tuple(g.edge_attr.shape)}")