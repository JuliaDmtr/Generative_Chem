"""
smiles_transformer_decoder.py

Decoder: generates a SMILES string autoregressively, conditioned on a
latent vector z.

CHANGED FROM THE PREPEND-ONLY VERSION:
Previously z was injected as a single pseudo-token at position 0, relying
on self-attention to carry that information forward. In practice, the
Transformer decoder was strong enough to ignore z almost entirely
(posterior collapse) - it could get a good loss just by learning general
token statistics, since attending back to one token among many is "optional"
from the model's perspective.

Fix: project z once, then ADD it to every token embedding directly (every
position, every layer input). This removes the "attend-away" escape hatch -
z is now mixed into the representation at every single step, not just
available if the model chooses to look for it.
"""

import torch
import torch.nn as nn


class SmilesTransformerDecoder(nn.Module):
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
        word_dropout_prob=0.3,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.word_dropout_prob = word_dropout_prob

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Projects the continuous latent vector into a d_model-sized vector
        # that gets ADDED to every token embedding (not prepended as one token).
        self.latent_proj = nn.Linear(latent_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.register_buffer("causal_mask", self._build_causal_mask(max_len))

    @staticmethod
    def _build_causal_mask(max_len):
        mask = torch.ones((max_len, max_len), dtype=torch.bool)
        return torch.triu(mask, diagonal=1)

    def forward(self, z, input_ids):
        """
        z: [batch, latent_dim]          - the (denoised, post-diffusion) latent vector
        input_ids: [batch, seq_len]     - target token ids (teacher forcing during training)

        Returns logits of shape [batch, seq_len, vocab_size], standard
        next-token-prediction alignment: logits[:, i] predicts input_ids[:, i+1].
        """
        batch_size, seq_len = input_ids.shape

        max_supported_len = self.causal_mask.size(0)
        if seq_len > max_supported_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds decoder max_len {max_supported_len}."
            )

        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_ids = pos_ids.expand(batch_size, seq_len)

        token_embeds = self.token_emb(input_ids) * (self.d_model ** 0.5)  # [batch, seq_len, d_model]

        # --- Word dropout (training only) ---
        # Randomly "blank" some real (non-padding) token embeddings to zero,
        # forcing the model to rely on z (and position) to fill the gap,
        # rather than always having the true previous character available.
        # This directly targets the shortcut that caused posterior collapse:
        # teacher forcing making next-token prediction "too easy" without z.
        if self.training and self.word_dropout_prob > 0:
            real_token_mask = ~input_ids.eq(self.pad_idx)  # don't drop actual padding
            drop_mask = (torch.rand(input_ids.shape, device=input_ids.device) < self.word_dropout_prob)
            drop_mask = drop_mask & real_token_mask
            token_embeds = token_embeds.masked_fill(drop_mask.unsqueeze(-1), 0.0)

        pos_embeds = self.pos_emb(pos_ids)                                # [batch, seq_len, d_model]

        # Project z once, broadcast-add it to every position/timestep.
        latent_embed = self.latent_proj(z).unsqueeze(1)  # [batch, 1, d_model] -> broadcasts over seq_len

        x = token_embeds + pos_embeds + latent_embed  # z is now part of every single token's input

        key_padding_mask = input_ids.eq(self.pad_idx)
        mask = self.causal_mask[:seq_len, :seq_len]
        hidden = self.transformer(x, mask=mask, src_key_padding_mask=key_padding_mask)

        logits = self.lm_head(hidden)  # [batch, seq_len, vocab_size]
        return logits