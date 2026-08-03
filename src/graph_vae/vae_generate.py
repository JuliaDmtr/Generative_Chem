
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from rdkit import Chem
from graph_vae.vae import GraphVAE, MAX_ATOMS, NUM_ATOM_TYPES, NUM_BOND_TYPES
from graph_vae.smiles_to_graph import ATOM_TYPES, BOND_TYPES
from torch_geometric.data import Data

CHECKPOINT_PATH = "vae_checkpoint.pt"
LATENT_DIM = 128


def graph_decoder(atom_logits, bond_logits):
    """
    Convert the decoder's dense output into a PyG graph object.
    This is the inverse of build_dense_targets() in vae_loss.py.

    Args:
        atom_logits: [1, max_atoms, num_atom_types+1] (logits for each atom slot)
        bond_logits: [1, max_atoms, max_atoms, num_bond_types] (logits for each atom pair)  
    Returns:
        A PyG graph object representing the molecule.
    """
    atom_probs = torch.softmax(atom_logits, dim=-1)  # [1, max_atoms, num_atom_types+1]
    bond_probs = torch.softmax(bond_logits, dim=-1)  # [1, max_atoms, max_atoms, num_bond_types]

    # Get the predicted atom types (class index) for each atom slot
    atom_types = torch.argmax(atom_probs, dim=-1).squeeze(0)  # [max_atoms]

    # Get the predicted bond types (class index) for each atom pair
    bond_types = torch.argmax(bond_probs, dim=-1).squeeze(0)  # [max_atoms, max_atoms]

  

    # Filter out padding atoms (where atom type is the padding index)
    valid_atom_mask = atom_types != NUM_ATOM_TYPES  # padding index is NUM_ATOM_TYPES
    valid_atom_indices = torch.nonzero(valid_atom_mask).squeeze(-1)

    # Create edge_index and edge_attr for valid bonds
    edge_index_list = []
    edge_attr_list = []

    #for i in valid_atom_indices:
    #    print (f"Atom {i.item()} type: {atom_types[i].item()}")
    #    print (f"Atom {i.item()} type: {ATOM_TYPES[atom_types[i].item()]}")

    for i in valid_atom_indices:
        for j in valid_atom_indices:
            if i < j:  # avoid duplicates and self-loops
                bond_type = bond_types[i, j].item()
                if bond_type != NUM_BOND_TYPES -1:  # skip no-bond class
                    edge_index_list.append([i.item(), j.item()])
                    edge_attr_list.append(bond_type)
                    #print (f"Bond {i.item()}-{j.item()} type: {BOND_TYPES[bond_type]}")

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.long)

    graph_data = Data(x=atom_types[valid_atom_mask].unsqueeze(-1), 
                      edge_index=edge_index,
                      edge_attr=edge_attr)

    return graph_data


def graph_to_molecule(graph_data):
    """Convert a decoded PyG graph into an RDKit molecule."""
    atom_types = graph_data.x.squeeze(-1).tolist()
    edge_index = graph_data.edge_index
    edge_attr = graph_data.edge_attr.tolist()

    mol = Chem.RWMol()
    atom_lookup = {}

    for atom_idx, atom_type in enumerate(atom_types):
        atom_symbol = ATOM_TYPES[atom_type]
        if atom_symbol == "other":
            print("Warning: 'other' atom type encountered. Skipping this atom.")
            atom_symbol = "C"
        atom = Chem.Atom(atom_symbol)
        mol.AddAtom(atom)
        atom_lookup[atom_idx] = atom_idx

    for edge_idx, bond_type in zip(range(edge_index.size(1)), edge_attr):
        i = edge_index[0, edge_idx].item()
        j = edge_index[1, edge_idx].item()
        if i == j:
            continue
        bond_type_obj = BOND_TYPES[bond_type]
        if bond_type_obj == Chem.rdchem.BondType.SINGLE:
            mol.AddBond(i, j, Chem.rdchem.BondType.SINGLE)
        elif bond_type_obj == Chem.rdchem.BondType.DOUBLE:
            mol.AddBond(i, j, Chem.rdchem.BondType.DOUBLE)
        elif bond_type_obj == Chem.rdchem.BondType.TRIPLE:
            mol.AddBond(i, j, Chem.rdchem.BondType.TRIPLE)
        elif bond_type_obj == Chem.rdchem.BondType.AROMATIC:
            mol.AddBond(i, j, Chem.rdchem.BondType.AROMATIC)

    try:
        return mol.GetMol()
    except Exception:
        return None


def graph_to_smiles(graph_data):
    """Convert a decoded PyG graph into a SMILES string."""
    mol = graph_to_molecule(graph_data)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None
   


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    vae = GraphVAE().to(device)

    state_dict = checkpoint["model_state_dict"]
    decoder_state_dict = {
        k.replace("decoder.", "", 1): v
        for k, v in state_dict.items()
        if k.startswith("decoder.")
    }

    vae.decoder.load_state_dict(decoder_state_dict)
    z = torch.randn(1, LATENT_DIM).to(device)
    atom_logits, bond_logits = vae.decoder(z)
    graph = graph_decoder(atom_logits, bond_logits)
    smiles = graph_to_smiles(graph)
    print("decoded graph:", graph)
    print("decoded smiles:", smiles)

if __name__ == "__main__":
    main()