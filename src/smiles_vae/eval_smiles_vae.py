"""
eval_smiles_vae.py

Two sanity checks for the trained Transformer SMILES VAE:

1. RECONSTRUCTION TEST: encode a real molecule -> z -> decode.
   Does the decoded SMILES match (or closely resemble) the original?

2. PRIOR SAMPLING TEST: sample z ~ N(0, I) directly (no real molecule
   involved at all) -> decode. Does this produce a VALID, NOVEL molecule?
   This is the test that actually matters for diffusion: it tells us
   whether the latent space is smooth/organized enough that "random-ish"
   points (which is what diffusion will eventually produce) decode into
   real molecules, not garbage.
"""

import torch
from rdkit import Chem

from smiles_transformer_vae import SmilesTransformerEncoder, reparameterize
from smiles_transformer_decoder import SmilesTransformerDecoder

CHECKPOINT_PATH = "smiles_vae_checkpoint.pt"
LATENT_DIM = 128
START_CHAR = "$"
END_CHAR = "E"


def autoregressive_decode(decoder, z, char_to_int, int_to_char, pad_idx, max_len, device):
    """
    Generates one SMILES string from a fixed latent vector z, one token at a
    time (greedy: always pick the highest-probability next token), starting
    from the start-of-sequence character and stopping at end-of-sequence
    (or max_len).
    """
    decoder.eval()
    batch_size = z.size(0)

    generated = torch.full((batch_size, 1), char_to_int[START_CHAR], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_len - 1):
            logits = decoder(z, generated)           # [batch, cur_len, vocab_size]
            next_token_logits = logits[:, -1, :]      # last position predicts the next token
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # greedy
            generated = torch.cat([generated, next_token], dim=1)

    results = []
    for row in generated:
        chars = []
        for idx in row.tolist():
            ch = int_to_char[idx]
            if ch == END_CHAR:
                break
            if ch == START_CHAR or idx == pad_idx:
                continue
            chars.append(ch)
        results.append("".join(chars))
    return results


def check_validity(smiles_list):
    valid = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        valid.append(mol is not None)
    return valid


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    char_to_int = checkpoint["char_to_int"]
    int_to_char = checkpoint["int_to_char"]
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
    encoder.eval()
    decoder.eval()

    # ---------- TEST 1: Reconstruction ----------
    print("\n=== Reconstruction test ===")
    test_smiles = [
        "CCO",
        "c1ccccc1",
        "CC(=O)Oc1ccccc1C(=O)O",   # aspirin
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    ]

    for smi in test_smiles:
        seq = START_CHAR + smi + END_CHAR
        ids = [char_to_int[c] for c in seq]
        ids = ids + [pad_idx] * (max_len - len(ids))
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            mu, logvar = encoder(input_ids)
            z = mu  # use the mean directly (no sampling noise) for a fair reconstruction test
        print(z)
        decoded = autoregressive_decode(decoder, z, char_to_int, int_to_char, pad_idx, max_len, device)
        is_valid = check_validity(decoded)
        print(f"Original:  {smi}")
        print(f"Decoded:   {decoded[0]}  (valid: {is_valid[0]})")
        print()

    # ---------- TEST 2: Prior sampling (no real molecule involved) ----------
    print("\n=== Prior sampling test (z ~ N(0,I), no encoder involved) ===")
    num_samples = 20
    z_prior = torch.randn(num_samples, LATENT_DIM, device=device)

    decoded_samples = autoregressive_decode(
        decoder, z_prior, char_to_int, int_to_char, pad_idx, max_len, device
    )
    valid_flags = check_validity(decoded_samples)

    for smi, is_valid in zip(decoded_samples, valid_flags):
        print(f"{'VALID  ' if is_valid else 'INVALID'} | {smi}")

    validity_rate = sum(valid_flags) / len(valid_flags)
    print(f"\nValidity rate on random prior samples: {validity_rate:.1%}")

def decoder_test():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    char_to_int = checkpoint["char_to_int"]
    int_to_char = checkpoint["int_to_char"]
    pad_idx = checkpoint["pad_idx"]
    max_len = checkpoint["max_len"]
    vocab_size = len(char_to_int)

    decoder = SmilesTransformerDecoder(
        vocab_size=vocab_size, pad_idx=pad_idx, max_len=max_len, latent_dim=LATENT_DIM
    ).to(device)
    decoder.eval()

    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    z = torch.zeros(1, LATENT_DIM, device=device)
    decoded = autoregressive_decode(decoder, z, char_to_int, int_to_char, pad_idx, max_len, device)
    is_valid = check_validity(decoded)
    print(f"Decoded:   {decoded[0]}  (valid: {is_valid[0]})")


    z = torch.ones(1, LATENT_DIM, device=device)
    decoded = autoregressive_decode(decoder, z, char_to_int, int_to_char, pad_idx, max_len, device)
    is_valid = check_validity(decoded)
    print(f"Decoded:   {decoded[0]}  (valid: {is_valid[0]})")
    


if __name__ == "__main__":
    main()
    #decoder_test()