"""
test_vae_pipeline.py

Quick sanity check: converts a few SMILES into graphs, batches them,
runs one forward pass through the VAE, and computes the loss.
Run this locally to catch shape/wiring errors before scaling up.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch_geometric.loader import DataLoader
from smiles_vae.chembl_processor import ChemblProcessor
from graph_vae.smiles_to_graph import smiles_to_graph
from graph_vae.vae import GraphVAE
from graph_vae.vae_loss import vae_loss


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

   #test_smiles = [
   #     "CCO",                      # ethanol
   #     "c1ccccc1",                 # benzene
   #     "CC(=O)Oc1ccccc1C(=O)O",    # aspirin
   #     "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
   # ]
    
    processor = ChemblProcessor("../../Datasets/")
    test_smiles = processor.make_samples(num_samples=1000, max_len_of_sample=100)
    graphs = []
    for smi in test_smiles:
       
        g = smiles_to_graph(smi)
        if g is not None:
            graphs.append(g)
        else:
            print(f"Skipped invalid SMILES: {smi}")

    print(f"Built {len(graphs)} graphs")

    loader = DataLoader(graphs, batch_size=len(graphs), shuffle=False)
    batch = next(iter(loader)).to(device)

    model = GraphVAE().to(device)
    model.train()

    atom_logits, bond_logits, mu, logvar = model(batch.x, batch.edge_index, batch.batch)

    print("atom_logits shape:", atom_logits.shape)
    print("bond_logits shape:", bond_logits.shape)
    print("mu shape:", mu.shape)
    print("logvar shape:", logvar.shape)

    total_loss, atom_loss, bond_loss, kl = vae_loss(atom_logits, bond_logits, mu, logvar, batch)

    print(f"total_loss={total_loss.item():.4f}  atom_loss={atom_loss:.4f}  "
          f"bond_loss={bond_loss:.4f}  kl={kl:.4f}")

    # confirm gradients flow end-to-end
    total_loss.backward()
    print("Backward pass OK - gradients computed successfully")


if __name__ == "__main__":
    main()