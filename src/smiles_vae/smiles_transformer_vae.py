"""
smiles_transformer_vae.py

Transformer-based VAE encoder for SMILES strings.

Reuses the same embedding/positional-encoding pattern as SmilesTransformerLM,
but:
  - NO causal mask (bidirectional attention - we're compressing, not generating)
  - mean-pools over all non-padding token positions to get one fixed-size vector
  - outputs (mu, logvar) instead of raw hidden states, for the VAE reparameterization trick

This plugs into the same latent-diffusion pipeline we designed for the graph VAE:
Encoder -> (mu, logvar) -> reparameterize -> z -> [diffusion later] -> Decoder -> SMILES
"""

import torch
import torch.nn as nn


class SmilesTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_idx,
        max_len,
        latent_dim=128,
        d_model=256,
        nhead=4,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        # VAE heads: project pooled representation to mu and logvar
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

    def forward(self, input_ids):
        """
        input_ids: [batch, seq_len]
        returns: mu [batch, latent_dim], logvar [batch, latent_dim]
        """
        batch_size, seq_len = input_ids.shape

        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_ids = pos_ids.expand(batch_size, seq_len)

        x = (self.token_emb(input_ids) + self.pos_emb(pos_ids)) * (self.token_emb.embedding_dim ** 0.5)

        # No causal mask here - every token can attend to every other token,
        # since we're building one summary representation, not generating.
        key_padding_mask = input_ids.eq(self.pad_idx)
        hidden = self.transformer(x, src_key_padding_mask=key_padding_mask)  # [batch, seq_len, d_model]

        # Mean-pool over real (non-padding) tokens only
        real_token_mask = (~key_padding_mask).unsqueeze(-1).float()  # [batch, seq_len, 1]
        summed = (hidden * real_token_mask).sum(dim=1)               # [batch, d_model]
        count = real_token_mask.sum(dim=1).clamp(min=1)              # [batch, 1]
        pooled = summed / count                                       # [batch, d_model]

        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar


def reparameterize(mu, logvar):
    """Same reparameterization trick as the graph VAE: z = mu + std * eps."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps