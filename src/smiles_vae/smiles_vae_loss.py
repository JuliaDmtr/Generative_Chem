"""
smiles_vae_loss.py

Loss for the Transformer-based SMILES VAE:
  reconstruction = token-level cross-entropy (ignoring padding positions)
  + KL divergence (mu, logvar -> N(0, I)), same annealing idea as the graph VAE.
"""

import torch
import torch.nn.functional as F


def smiles_vae_loss(logits, input_ids, mu, logvar, pad_idx, kl_weight=1.0,
                     free_bits=0.05):
    """
    logits:    [batch, seq_len, vocab_size]
    input_ids: [batch, seq_len]
    mu, logvar:[batch, latent_dim]
    pad_idx:   int
    free_bits: minimum KL (in nats) allowed per latent dimension before any
        penalty applies. Below this threshold, that dimension contributes
        zero gradient to the loss - removing the incentive to collapse mu
        all the way to 0 just to minimize KL. Typical values: 0.05-0.2 per
        dimension (NOT per whole latent vector).
    """
    pred_logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]

    recon_loss = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_idx,
        reduction="mean",
    )

    # --- KL divergence with free bits ---
    # per-dimension KL, NOT yet summed across dimensions
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [batch, latent_dim]

    # clamp: any dimension below the free_bits floor contributes no gradient
    # (max()'s gradient is 0 where the constant wins, 1 where kl_per_dim wins)
    kl_per_dim_clamped = torch.clamp(kl_per_dim, min=free_bits)

    kl_div_for_loss = kl_per_dim_clamped.sum(dim=1).mean()   # used in the actual loss
    kl_div_raw = kl_per_dim.sum(dim=1).mean()                # true KL, for logging/monitoring only

    total_loss = recon_loss + kl_weight * kl_div_for_loss
    return total_loss, recon_loss.item(), kl_div_raw.item()