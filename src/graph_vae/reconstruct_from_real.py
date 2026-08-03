import argparse
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from graph_vae.chembl_graph_dataset import ChemblGraphDataset
from graph_vae.vae import GraphVAE, MAX_ATOMS, NUM_ATOM_TYPES, NUM_BOND_TYPES
from graph_vae.smiles_to_graph import ATOM_TYPES, BOND_TYPES
from graph_vae.train_vae import BATCH_SIZE
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch
from rdkit import Chem

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "vae_checkpoint.pt")
DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Datasets"))
DATASET_PREFIX = DATASET_ROOT + os.sep

def mol_decoder(atom_logits, bond_logits):
    """
    Convert the decoder's dense output into an RDKit molecule.
    This is the inverse of build_dense_targets() in vae_loss.py.

    Args:
        atom_logits: [1, max_atoms, num_atom_types+1] (logits for each atom slot)
        bond_logits: [1, max_atoms, max_atoms, num_bond_types] (logits for each atom pair)
    Returns:
        An RDKit molecule object representing the molecule.
    """
    atom_probs = torch.softmax(atom_logits, dim=-1)
    bond_probs = torch.softmax(bond_logits, dim=-1)

    atom_types = torch.argmax(atom_probs, dim=-1).squeeze(0)
    bond_types = torch.argmax(bond_probs, dim=-1).squeeze(0)

    mol = Chem.RWMol()

    valid_atom_mask = atom_types != NUM_ATOM_TYPES
    valid_atom_indices = torch.nonzero(valid_atom_mask).squeeze(-1).tolist()


    dense_to_compact = {}
    for compact_idx, dense_idx in enumerate(valid_atom_indices):
        atom_symbol = ATOM_TYPES[atom_types[dense_idx]]
        if atom_symbol == "other":
            atom_symbol = "C"
        mol.AddAtom(Chem.Atom(atom_symbol))
        dense_to_compact[dense_idx] = compact_idx

    for i_dense in valid_atom_indices:
        for j_dense in valid_atom_indices:
            if i_dense < j_dense:
                bond_type_idx = bond_types[i_dense, j_dense].item()
                if bond_type_idx != NUM_BOND_TYPES - 1:
                    bond_type_obj = BOND_TYPES[bond_type_idx]
                    mol.AddBond(
                        dense_to_compact[i_dense],
                        dense_to_compact[j_dense],
                        bond_type_obj,
                    )

    try:
        return mol.GetMol()
    except Exception:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct molecules from a trained GraphVAE checkpoint")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of molecules to reconstruct")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for reconstruction")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    vae = GraphVAE().to(device)

    state_dict = checkpoint["model_state_dict"]
    vae.load_state_dict(state_dict)
    vae.eval()

    dataset = ChemblGraphDataset(num_samples=args.num_samples, max_len_of_sample=100,
                                 data_path_prefix=DATASET_PREFIX)

    if len(dataset) == 0:
        raise RuntimeError("No valid molecules were loaded from the dataset")

    # Build a loader that keeps all graphs on the selected device.
    class DeviceDataset(torch.utils.data.Dataset):
        def __init__(self, data_list, device):
            self.data_list = data_list
            self.device = device

        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, idx):
            return self.data_list[idx].to(self.device)

    device_dataset = DeviceDataset(dataset.graphs, device)
    loader = DataLoader(device_dataset, batch_size=args.batch_size, shuffle=False)
    decoded_smiles = []
    total_batches = math.ceil(len(device_dataset) / args.batch_size)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader, 1):
            batch_data = batch_data.to(device)
            mu, logvar = vae.encoder(batch_data.x, batch_data.edge_index, batch_data.batch)
            z = mu
            atom_logits, bond_logits = vae.decoder(z)

            for i in range(batch_data.num_graphs):
                mol = mol_decoder(atom_logits[i:i+1], bond_logits[i:i+1])
                try:
                    decoded_smiles.append(Chem.MolToSmiles(mol) if mol is not None else None)
                except Exception as exc:
                    decoded_smiles.append(None)
                    print(f"Warning: failed to serialize molecule {i} in batch {batch_idx}: {exc}")

            print(f"Processed batch {batch_idx}/{total_batches}")
        
    print(f"Reconstructed {len(decoded_smiles)} molecules")
    for i, smiles in enumerate(decoded_smiles[:10]):
        print(f"{i}: {smiles}")


if __name__ == "__main__":
    main()
