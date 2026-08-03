"""
check_encoder_gradients.py

Diagnostic: runs ONE real training step (forward + backward, no optimizer
step) on a small batch of genuinely different molecules, then inspects
whether gradients actually reach the encoder's fc_mu layer - and how large
they are. This directly tests "is there a bug blocking the gradient path"
vs "gradient exists but is too small / landscape is flat".
"""

import torch

from smiles_vae.chembl_processor import ChemblProcessor
from smiles_vae.smiles_tensor_dataset import SmilesTensorDataset
from smiles_transformer_vae import SmilesTransformerEncoder, reparameterize
from smiles_transformer_decoder import SmilesTransformerDecoder
from smiles_vae_loss import smiles_vae_loss

CHECKPOINT_PATH = "smiles_vae_checkpoint.pt"
LATENT_DIM = 128


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
    decoder = SmilesTransformerDecoder(
        vocab_size=vocab_size, pad_idx=pad_idx, max_len=max_len, latent_dim=LATENT_DIM
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    encoder.train()  # keep train mode so word dropout / gradients behave as in real training
    decoder.train()

    test_smiles = [
        "CCO",
        "c1ccccc1",
        "CC(=O)Oc1ccccc1C(=O)O",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    ]
    seqs = [f"${smi}E" for smi in test_smiles]
    dataset = SmilesTensorDataset(seqs, char_to_int, pad_idx, max_len)
    batch = torch.stack([dataset[i] for i in range(len(dataset))]).to(device)

    mu, logvar = encoder(batch)
    z = reparameterize(mu, logvar)
    logits = decoder(z, batch)

    total_loss, recon_loss, kl = smiles_vae_loss(logits, batch, mu, logvar, pad_idx, kl_weight=0.1)

    encoder.zero_grad()
    decoder.zero_grad()
    total_loss.backward()

    print(f"recon_loss={recon_loss:.4f}, kl={kl:.4f}\n")

    print("=== Gradient norms per encoder parameter ===")
    for name, param in encoder.named_parameters():
        if param.grad is None:
            print(f"  {name:40s} grad = None (NO GRADIENT REACHED - this would be a bug)")
        else:
            print(f"  {name:40s} grad norm = {param.grad.norm().item():.8f}")

    print("\n=== Specifically fc_mu ===")
    fc_mu_grad_norm = encoder.fc_mu.weight.grad.norm().item()
    print(f"fc_mu.weight grad norm: {fc_mu_grad_norm:.8f}")
    print("(near-zero => landscape is flat / gradient vanished here, not a broken connection)")
    print("(clearly nonzero, e.g. > 1e-4 => gradient IS flowing; mu just isn't moving apart yet)")


if __name__ == "__main__":
    main()
