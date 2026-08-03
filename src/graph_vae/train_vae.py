"""
train_vae.py

Trains the GraphVAE on a 50K-molecule subset of ChEMBL.
Logs per-epoch loss components and saves a checkpoint after each epoch,
so you can inspect progress and resume/reuse the encoder for diffusion later.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


import torch
from torch_geometric.loader import DataLoader

from graph_vae.chembl_graph_dataset import ChemblGraphDataset
from graph_vae.vae import GraphVAE
from graph_vae.vae_loss import vae_loss

NUM_SAMPLES = 50_000
MAX_LEN = 100          # max SMILES string length to include
BATCH_SIZE = 128
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3
KL_WEIGHT_MAX = 0.02    # target KL weight once fully annealed in (lowered from 0.1 -
                        # even the gentler previous ceiling still let KL crush toward
                        # ~0.004 by epoch 40; testing whether a smaller ceiling reduces
                        # collapse severity)
KL_ANNEAL_EPOCHS = 25   # slower ramp - was 15
CHECKPOINT_PATH = "vae_checkpoint.pt"


def kl_weight_for_epoch(epoch: int) -> float:
    """
    Linear KL annealing: start at ~0 (let reconstruction dominate early),
    ramp up to KL_WEIGHT_MAX by KL_ANNEAL_EPOCHS, then hold steady.
    This prevents posterior collapse, where the encoder gives up and
    outputs the same generic distribution for every molecule.
    """
    if epoch >= KL_ANNEAL_EPOCHS:
        return KL_WEIGHT_MAX
    return KL_WEIGHT_MAX * (epoch / KL_ANNEAL_EPOCHS)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset = ChemblGraphDataset(num_samples=NUM_SAMPLES, 
                                 max_len_of_sample=MAX_LEN,
                                 data_path_prefix="../../Datasets/")
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = GraphVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on {len(dataset)} molecules, {len(loader)} batches/epoch")

    for epoch in range(1, NUM_EPOCHS + 1):
        current_kl_weight = kl_weight_for_epoch(epoch)
        model.train()
        total_loss_sum = 0.0
        atom_loss_sum = 0.0
        bond_loss_sum = 0.0
        kl_sum = 0.0
        num_batches = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            atom_logits, bond_logits, mu, logvar = model(
                batch.x, batch.edge_index, batch.batch
            )
            total_loss, atom_loss, bond_loss, kl = vae_loss(
                atom_logits, bond_logits, mu, logvar, batch, kl_weight=current_kl_weight
            )

            total_loss.backward()
            optimizer.step()

            total_loss_sum += total_loss.item()
            atom_loss_sum += atom_loss
            bond_loss_sum += bond_loss
            kl_sum += kl
            num_batches += 1

        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
            f"kl_w={current_kl_weight:.4f} | "
            f"total={total_loss_sum / num_batches:.4f} | "
            f"atom={atom_loss_sum / num_batches:.4f} | "
            f"bond={bond_loss_sum / num_batches:.4f} | "
            f"kl={kl_sum / num_batches:.4f}"
        )

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, CHECKPOINT_PATH)

    print(f"Training complete. Checkpoint saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()