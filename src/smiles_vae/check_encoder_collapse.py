"""
check_encoder_collapse.py

Diagnostic: encodes 10 different real SMILES and prints their mu vectors,
plus pairwise differences, to check whether the encoder is producing
genuinely distinct mu per molecule, or collapsing to (near-)identical
values regardless of input.
"""

import torch

from smiles_transformer_vae import SmilesTransformerEncoder

CHECKPOINT_PATH = "smiles_vae_checkpoint.pt"
LATENT_DIM = 128
START_CHAR = "$"
END_CHAR = "E"

TEST_SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CCN(CC)CC",
    "c1ccc2ccccc2c1",
    "CC(C)O",
    "CCCCCCCC",
    "OCC(O)CO",
    "c1ccncc1",
]


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    char_to_int = checkpoint["char_to_int"]
    pad_idx = checkpoint["pad_idx"]
    max_len = checkpoint["max_len"]
    vocab_size = len(char_to_int)

    encoder = SmilesTransformerEncoder(
        vocab_size=vocab_size, pad_idx=pad_idx, max_len=max_len, latent_dim=LATENT_DIM
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()

    mus = []
    with torch.no_grad():
        for smi in TEST_SMILES:
            seq = START_CHAR + smi + END_CHAR
            ids = [char_to_int[c] for c in seq]
            ids = ids + [pad_idx] * (max_len - len(ids))
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)

            mu, logvar = encoder(input_ids)
            mus.append(mu.squeeze(0))

    mus = torch.stack(mus)  # [10, latent_dim]

    print("Per-molecule mu norm (magnitude):")
    for smi, mu in zip(TEST_SMILES, mus):
        print(f"  {smi:35s} ||mu|| = {mu.norm().item():.4f}")

    print("\nPairwise L2 distance between mu vectors (0 = identical):")
    for i in range(len(TEST_SMILES)):
        for j in range(i + 1, len(TEST_SMILES)):
            dist = (mus[i] - mus[j]).norm().item()
            print(f"  {TEST_SMILES[i]:25s} vs {TEST_SMILES[j]:25s} -> dist = {dist:.4f}")

    avg_dist = torch.pdist(mus).mean().item()
    print(f"\nAverage pairwise distance across all 10 molecules: {avg_dist:.4f}")
    print("(Near 0 => encoder collapse: mu ignores the input molecule entirely)")


if __name__ == "__main__":
    main()