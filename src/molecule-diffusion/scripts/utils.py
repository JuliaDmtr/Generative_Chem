"""Shared utilities for the molecule diffusion model."""

from __future__ import annotations

import copy
import math

import torch


# Default atom vocabulary. QM9 only needs the first five (light organic
# elements); this extended list also covers what actually shows up in
# ChEMBL-scale drug-like molecules. If you change this list, any existing
# checkpoint is invalid (the vocab size is baked into the model's input/
# output dims) — retrain from scratch.
ATOM_VOCAB = ["H", "C", "N", "O", "F", "S", "Cl", "Br", "I", "P", "B", "Si"]
ATOM2IDX = {a: i for i, a in enumerate(ATOM_VOCAB)}
IDX2ATOM = {i: a for a, i in ATOM2IDX.items()}
NUM_ATOM_TYPES = len(ATOM_VOCAB)


# Conditioning labels for the PARP-conditional model (see
# scripts/train_conditional.py). This is a simple 3-way class-conditioning
# scheme rather than textbook classifier-free-guidance dropout: every
# training molecule is tagged as a generic ChEMBL-like negative, a
# PARP-inhibitor-like positive, or an explicit "null"/unconditional class
# used for representative background training and for the null branch of
# guided sampling.
COND_NEGATIVE = 0
COND_POSITIVE = 1
COND_NULL = 2
COND_DIM = 3


def make_condition_batch(label: int, batch_size: int, device=None) -> torch.Tensor:
    """Build a (batch_size, COND_DIM) one-hot condition tensor for a single label."""
    cond = torch.zeros(batch_size, COND_DIM, device=device)
    cond[:, label] = 1.0
    return cond


def condition_labels_to_tensor(labels, device=None) -> torch.Tensor:
    """Convert a list/1D tensor of per-molecule integer condition labels
    (COND_NEGATIVE/COND_POSITIVE/COND_NULL) into a (B, COND_DIM) one-hot tensor.
    """
    labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)
    return torch.nn.functional.one_hot(labels_t, num_classes=COND_DIM).float()


def remove_mean_with_mask(x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Subtract the per-molecule center of mass (over real atoms only).

    This is what makes the model translation-invariant: we only ever train
    on / sample from the zero-CoM subspace, so epsilon predictions for x
    are only meaningful (and only compared against targets) after this
    projection. See references/approach.md for why this matters.
    """
    n_real = node_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1, 1)
    mean = (x * node_mask).sum(dim=1, keepdim=True) / n_real
    return (x - mean) * node_mask


def build_edge_mask(node_mask: torch.Tensor) -> torch.Tensor:
    """(B, N, 1) node_mask -> (B, N, N, 1) edge_mask, excluding self-loops."""
    B, N, _ = node_mask.shape
    em = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)  # (B,N,N,1)
    diag = torch.eye(N, device=node_mask.device, dtype=torch.bool)
    em = em.masked_fill(diag.view(1, N, N, 1), 0.0)
    return em


def cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal 2021), t in [0, 1].

    Returns alpha_bar(t) = cos^2( (t/1 + s) / (1 + s) * pi/2 ), normalized
    so alpha_bar(0) = 1. This gives a variance-preserving forward process:
        z_t = sqrt(alpha_bar(t)) * x_0 + sqrt(1 - alpha_bar(t)) * eps
    """
    f = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
    f0 = math.cos((s / (1 + s)) * math.pi / 2) ** 2
    return (f / f0).clamp(min=1e-5, max=1.0)


class EMA:
    """Exponential moving average of model parameters, used at generation
    time for more stable samples (standard practice for diffusion models).
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for s_p, p in zip(self.shadow.parameters(), model.parameters()):
            s_p.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)
