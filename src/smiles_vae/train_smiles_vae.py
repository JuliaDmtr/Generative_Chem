"""
train_smiles_vae.py

Trains the Transformer-based SMILES VAE (encoder + decoder) on a subset
of ChEMBL, reusing ChemblProcessor for data loading/tokenization.
"""

import torch
from torch.utils.data import DataLoader

from smiles_vae.chembl_processor import ChemblProcessor
from smiles_vae.smiles_tensor_dataset import SmilesTensorDataset
from smiles_transformer_vae import SmilesTransformerEncoder, reparameterize
from smiles_transformer_decoder import SmilesTransformerDecoder
from smiles_vae_loss import smiles_vae_loss

NUM_SAMPLES = 50_000
MAX_SMILE_LEN = 100      # max raw SMILES length passed to make_samples
BATCH_SIZE = 64
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3
LATENT_DIM = 128
KL_WEIGHT_MAX = 0.1     # target KL weight once fully annealed in
KL_ANNEAL_EPOCHS = 15   # gives the encoder a protected head start to spread mu
                        # out before KL pressure appears - free bits (in
                        # smiles_vae_loss.py) then prevents collapse AFTER
                        # annealing completes. Both are needed together.
CHECKPOINT_PATH = "smiles_vae_checkpoint.pt"


def kl_weight_for_epoch(epoch: int) -> float:
    if epoch >= KL_ANNEAL_EPOCHS:
        return KL_WEIGHT_MAX
    return KL_WEIGHT_MAX * (epoch / KL_ANNEAL_EPOCHS)


def evaluate(encoder, decoder, test_loader, pad_idx, device):
    """
    Computes test-set loss with:
      - model.eval() -> disables word dropout (only meant for training)
      - z = mu directly (no reparameterization sampling) -> deterministic,
        reproducible number across runs, not jittered by random eps
      - torch.no_grad() -> no gradient tracking, faster + less memory
    """
    encoder.eval()
    decoder.eval()

    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        for input_ids in test_loader:
            input_ids = input_ids.to(device)

            mu, logvar = encoder(input_ids)
            z = mu  # deterministic: use the mean directly, no sampling noise

            logits = decoder(z, input_ids)
            total_loss, recon_loss, kl = smiles_vae_loss(
                logits, input_ids, mu, logvar, pad_idx, kl_weight=KL_WEIGHT_MAX
            )

            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss
            kl_sum += kl
            num_batches += 1

    return (
        total_loss_sum / num_batches,
        recon_loss_sum / num_batches,
        kl_sum / num_batches,
    )


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Data loading (reusing your existing ChemblProcessor) ---
    processor = ChemblProcessor()
    raw_samples = processor.make_samples(NUM_SAMPLES, MAX_SMILE_LEN)
    train_seqs, test_seqs = processor.prepare_data_for_lstm(raw_samples)

    vocab_size = len(processor.char_to_int)
    pad_idx = processor.pad_idx
    # +2 accounts for the start/end chars added on top of max_smile_length
    max_len = processor.max_smile_length + 2

    print(f"vocab_size={vocab_size}, pad_idx={pad_idx}, max_len={max_len}")
    print(f"train sequences: {len(train_seqs)}, test sequences: {len(test_seqs)}")

    train_dataset = SmilesTensorDataset(train_seqs, processor.char_to_int, pad_idx, max_len)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = SmilesTensorDataset(test_seqs, processor.char_to_int, pad_idx, max_len)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Models ---
    encoder = SmilesTransformerEncoder(
        vocab_size=vocab_size, pad_idx=pad_idx, max_len=max_len, latent_dim=LATENT_DIM
    ).to(device)
    decoder = SmilesTransformerDecoder(
        vocab_size=vocab_size, pad_idx=pad_idx, max_len=max_len, latent_dim=LATENT_DIM
    ).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

    print(f"Training on {len(train_dataset)} molecules, {len(train_loader)} batches/epoch")

    for epoch in range(1, NUM_EPOCHS + 1):
        current_kl_weight = kl_weight_for_epoch(epoch)
        encoder.train()
        decoder.train()

        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_sum = 0.0
        num_batches = 0

        for input_ids in train_loader:
            input_ids = input_ids.to(device)
            optimizer.zero_grad()

            mu, logvar = encoder(input_ids)
            z = reparameterize(mu, logvar)
            logits = decoder(z, input_ids)

            total_loss, recon_loss, kl = smiles_vae_loss(
                logits, input_ids, mu, logvar, pad_idx, kl_weight=current_kl_weight
            )

            total_loss.backward()
            optimizer.step()

            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss
            kl_sum += kl
            num_batches += 1

        train_recon = recon_loss_sum / num_batches
        train_kl = kl_sum / num_batches

        test_total, test_recon, test_kl = evaluate(encoder, decoder, test_loader, pad_idx, device)

        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
            f"kl_w={current_kl_weight:.4f} | "
            f"train_recon={train_recon:.4f} | train_kl={train_kl:.4f} | "
            f"test_recon={test_recon:.4f} | test_kl={test_kl:.4f}"
        )

        torch.save({
            "epoch": epoch,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "char_to_int": processor.char_to_int,
            "int_to_char": processor.int_to_char,
            "pad_idx": pad_idx,
            "max_len": max_len,
        }, CHECKPOINT_PATH)

    print(f"Training complete. Checkpoint saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()