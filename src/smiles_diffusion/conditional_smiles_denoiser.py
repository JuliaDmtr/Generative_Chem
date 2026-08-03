import math
import random
import re
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


SMILES_TOKEN_PATTERN = re.compile(r"\[[^\]]+\]|Br|Cl|Si|B|C|N|O|P|S|F|I|b|c|n|o|p|s|\(|\)|=|#|@|@@|\-|\+|/|\\|\d+|\.")


def tokenize_smiles(smiles: str) -> List[str]:
    """Tokenize a SMILES string using a regex-based scheme for atoms, bonds, branches, and ring numbers."""
    tokens = []
    pos = 0
    while pos < len(smiles):
        match = SMILES_TOKEN_PATTERN.match(smiles, pos)
        if match is None:
            pos += 1
            continue
        tokens.append(match.group(0))
        pos = match.end()
    return tokens


def build_smiles_vocab(smiles_list: List[str], special_tokens: List[str] | None = None) -> List[str]:
    """Build a regex-based SMILES vocabulary from a corpus of strings."""
    specials = list(special_tokens or ["<pad>", "<mask>", "<eos>"])
    tokens = set()
    for smiles in smiles_list:
        tokens.update(tokenize_smiles(smiles))
    vocab = specials + sorted(tokens)
    return vocab


def encode_smiles(
    smiles: str,
    vocab: List[str],
    max_len: int,
    pad_token: str = "<pad>",
    mask_token: str = "<mask>",
    append_eos: bool = False,
    eos_token: str = "<eos>",
) -> List[int]:
    """Encode a SMILES string to integer token IDs."""
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    tokens = tokenize_smiles(smiles)
    ids = [token_to_id[token] for token in tokens if token in token_to_id]
    if append_eos and eos_token in token_to_id:
        ids = ids[: max_len - 1] + [token_to_id[eos_token]]
    else:
        ids = ids[:max_len]
    if len(ids) < max_len:
        ids = ids + [token_to_id[pad_token]] * (max_len - len(ids))
    return ids


def decode_token_ids(
    token_ids: List[int],
    vocab: List[str],
    pad_token: str = "<pad>",
    mask_token: str = "<mask>",
    eos_token: str = "<eos>",
) -> str:
    """Decode token IDs back to a SMILES string."""
    id_to_token = {idx: token for idx, token in enumerate(vocab)}
    chars = []
    for token_id in token_ids:
        token = id_to_token.get(token_id, pad_token)
        if token in {pad_token, mask_token, eos_token}:
            continue
        chars.append(token)
    return "".join(chars)


def decode_token_ids_with_specials(token_ids: List[int], vocab: List[str]) -> str:
    """Decode token IDs while preserving special tokens for debugging."""
    id_to_token = {idx: token for idx, token in enumerate(vocab)}
    return " ".join(id_to_token.get(token_id, str(token_id)) for token_id in token_ids)


def decode_logits_to_token_ids(logits: torch.Tensor) -> torch.Tensor:
    """Convert per-position logits into the most likely token ids."""
    return torch.argmax(logits, dim=-1)


def is_smiles_syntax_plausible(smiles: str) -> bool:
    """Return True for strings that are structurally plausible before full RDKit validation."""
    if not smiles:
        return False

    open_parens = 0
    bracket_depth = 0
    escaped = False
    ring_numbers = []
    for ch in smiles:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "(":
            open_parens += 1
        elif ch == ")":
            open_parens -= 1
            if open_parens < 0:
                return False
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
            if bracket_depth < 0:
                return False
        elif ch.isdigit():
            ring_numbers.append(ch)

    if open_parens != 0 or bracket_depth != 0:
        return False

    if "[" in smiles and "]" not in smiles.split("[", 1)[1]:
        return False

    # Reject obvious unbalanced ring numbering such as "C1CC".
    ring_openings = [i for i, ch in enumerate(smiles) if ch.isdigit()]
    if ring_openings:
        if len(ring_openings) % 2 != 0:
            return False

    return True


def filter_candidate_token_ids_by_syntax(
    prefix_ids: List[int],
    candidate_ids: List[int],
    vocab: List[str],
    pad_token: str = "<pad>",
    mask_token: str = "<mask>",
) -> List[int]:
    """Keep only candidate token IDs that preserve a syntactically plausible prefix."""
    if not candidate_ids:
        return []

    filtered = []
    for token_id in candidate_ids:
        candidate_ids_seq = prefix_ids + [token_id]
        decoded = decode_token_ids(candidate_ids_seq, vocab, pad_token=pad_token, mask_token=mask_token)
        if is_smiles_syntax_plausible(decoded):
            filtered.append(token_id)

    if not filtered:
        return [candidate_ids[0]]
    return filtered


def build_generation_positions(seq_len: int, step_idx: int, total_steps: int, min_positions: int = 4) -> List[int]:
    """Return a subset of positions to unmask at a given denoising step."""
    if seq_len <= 0:
        return []
    progress = (step_idx + 1) / max(1, total_steps)
    target_positions = max(min_positions, int(round(seq_len * progress)))
    target_positions = min(seq_len, target_positions)
    return random.sample(range(seq_len), k=target_positions)


def sample_token_ids_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Sample token ids from per-position logits using temperature, top-k, and top-p filtering."""
    if logits.dim() != 3:
        raise ValueError("Expected logits with shape [batch, seq_len, vocab_size]")

    logits = logits.float()
    if temperature is None or temperature <= 0:
        temperature = 1.0

    scaled_logits = logits / temperature
    filtered_logits = scaled_logits

    if top_k is not None and top_k > 0:
        top_k = min(top_k, scaled_logits.size(-1))
        kth_value = torch.topk(scaled_logits, k=top_k, dim=-1).values[..., -1, None]
        filtered_logits = scaled_logits.masked_fill(scaled_logits < kth_value, -torch.inf)

    if top_p is not None and 0 < top_p < 1.0:
        probs = torch.softmax(scaled_logits, dim=-1)
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        filtered_logits = filtered_logits.masked_fill(indices_to_remove, -torch.inf)

    probs = torch.softmax(filtered_logits, dim=-1)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sampled = torch.multinomial(probs.reshape(-1, probs.size(-1)), num_samples=1).reshape(*probs.shape[:-1], 1)
    return sampled.squeeze(-1)


def denoise_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    condition: torch.Tensor,
    mask_idx: int,
    timestep: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    positions: List[int] | None = None,
) -> torch.Tensor:
    """Run one denoising step and replace masked positions with sampled tokens."""
    logits = model(input_ids, condition, timestep=torch.tensor([timestep], device=input_ids.device))
    predicted = sample_token_ids_from_logits(logits, temperature=temperature, top_k=top_k, top_p=top_p)
    output = input_ids.clone()
    if positions is None:
        mask = output == mask_idx
    else:
        mask = torch.zeros_like(output, dtype=torch.bool)
        mask[:, positions] = True
    output[mask] = predicted[mask].to(input_ids.device)
    return output


class ConditionalSmilesDenoiser(nn.Module):
    """A tiny conditional denoising model for SMILES tokens."""

    def __init__(self, vocab_size: int, max_len: int, cond_dim: int = 64, d_model: int = 128, num_timesteps: int = 10):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.timestep_emb = nn.Embedding(num_timesteps, d_model)
        self.cond_proj = nn.Linear(cond_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True),
            num_layers=2,
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, condition: torch.Tensor, timestep: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_emb(input_ids)
        pos_emb = self.pos_emb(pos_ids)
        cond_emb = self.cond_proj(condition)
        if cond_emb.dim() == 2:
            cond_emb = cond_emb.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            cond_emb = cond_emb.reshape(batch_size, seq_len, -1)
        if timestep is not None:
            t_emb = self.timestep_emb(timestep.to(device=input_ids.device))
            if t_emb.dim() == 2:
                t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)
            else:
                t_emb = t_emb.reshape(batch_size, seq_len, -1)
        else:
            t_emb = torch.zeros_like(token_emb)
        x = token_emb + pos_emb + cond_emb + t_emb
        hidden = self.transformer(x)
        return self.lm_head(hidden)


class ConditionalSmilesAutoregModel(nn.Module):
    """A stronger decoder-only transformer for conditional SMILES generation."""

    def __init__(self, vocab_size: int, max_len: int, cond_dim: int = 64, d_model: int = 256, num_layers: int = 4, nhead: int = 8):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.cond_proj = nn.Linear(cond_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, input_ids: torch.Tensor, condition: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_emb(input_ids)
        pos_emb = self.pos_emb(pos_ids)
        cond_emb = self.cond_proj(condition)
        if cond_emb.dim() == 1:
            cond_emb = cond_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, seq_len, -1)
        elif cond_emb.dim() == 2:
            cond_emb = cond_emb.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            cond_emb = cond_emb.reshape(batch_size, seq_len, -1)
        x = self.dropout(token_emb + pos_emb + cond_emb)
        tgt_mask = self._build_causal_mask(seq_len, input_ids.device)
        hidden = self.transformer(tgt=x, memory=cond_emb, tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask)
        return self.lm_head(hidden)


def corrupt_tokens(
    token_ids: List[int],
    mask_idx: int,
    vocab_size: int | None = None,
    corruption_prob: float = 0.15,
    random_prob: float = 0.1,
) -> Tuple[List[int], List[int]]:
    """Corrupt a token sequence by masking or replacing tokens.

    Returns the corrupted sequence and a mask indicating which positions were changed.
    """
    corrupted = list(token_ids)
    mask = [0] * len(token_ids)
    if vocab_size is None:
        vocab_size = max(mask_idx + 1, 1)

    for i, tok in enumerate(token_ids):
        if random.random() < corruption_prob:
            mask[i] = 1
            if random.random() < random_prob:
                corrupted[i] = random.randrange(vocab_size)
            else:
                corrupted[i] = mask_idx

    return corrupted, mask


def diffuse_tokens(
    token_ids: List[int],
    mask_idx: int,
    noise_level: float,
    vocab_size: int,
) -> Tuple[List[int], List[int]]:
    """Apply a discrete diffusion-style corruption schedule.

    Higher noise_level means more positions are replaced by mask or random tokens.
    """
    corrupted = list(token_ids)
    mask = [0] * len(token_ids)
    corrupted_any = False
    for i, tok in enumerate(token_ids):
        if random.random() < noise_level:
            mask[i] = 1
            corrupted_any = True
            if random.random() < 0.5:
                corrupted[i] = mask_idx
            else:
                corrupted[i] = random.randrange(vocab_size)

    if not corrupted_any and noise_level > 0.0:
        idx = random.randrange(len(token_ids))
        mask[idx] = 1
        corrupted[idx] = mask_idx
    return corrupted, mask


def diffuse_tokens_with_timestep(
    token_ids: List[int],
    mask_idx: int,
    vocab_size: int,
    timestep: int,
    num_steps: int = 10,
) -> Tuple[List[int], List[int]]:
    """Corrupt a token sequence according to a discrete timestep.

    timestep=0 leaves the sequence unchanged. Larger timesteps make the sequence noisier.
    """
    if timestep <= 0:
        return list(token_ids), [0] * len(token_ids)

    noise_level = min(0.95, timestep / max(1, num_steps))
    return diffuse_tokens(token_ids, mask_idx=mask_idx, noise_level=noise_level, vocab_size=vocab_size)
