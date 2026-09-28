"""
Equivariant diffusion process over (atom positions x, atom types h).

Follows Hoogeboom et al. 2022, "Equivariant Diffusion for Molecule
Generation in 3D" (EDM): a single variance-preserving DDPM-style diffusion
jointly over continuous 3D coordinates and continuous relaxations of
categorical atom-type one-hot vectors, with an E(3)-equivariant denoiser
(EGNN, see egnn.py).

Key ideas (details in references/approach.md):
  1. x lives in the zero-center-of-mass subspace (translation invariance).
  2. Noise added to x is likewise zero-CoM, so the whole forward/reverse
     process stays on that subspace -> the model only needs to be
     rotation/reflection equivariant, not translation equivariant, on top.
  3. h (atom types) is treated as continuous data in R^{num_types} via the
     one-hot encoding and diffused the same way; at sampling time we take
     an argmax to recover discrete atom types.
  4. Loss is the standard simple noise-prediction objective (Ho et al. 2020
     DDPM), applied jointly to x and h with a relative weight.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from egnn import EGNN
from utils import H_SCALE, NUM_ATOM_TYPES, build_edge_mask, cosine_alpha_bar, remove_mean_with_mask


def angle_consistency_penalty(x0_pred: torch.Tensor, node_mask: torch.Tensor,
                               k_neighbors: int = 4, cos_center: float = 0.5,
                               sigma: float = 0.075) -> torch.Tensor:
    """Auxiliary training-time penalty for near-60-degree ("equilateral
    triangle" / cyclopropane-like) angles in the model's predicted x0,
    the direct geometry defect diagnose_scale.py's angle diagnostic
    confirmed (generated molecules show a 6-8x excess of angles in the
    50-70 degree band vs. real PARP conformers, at matched bond lengths).

    For each atom, finds its k_neighbors nearest OTHER real atoms (by
    current predicted distance; this selection is NOT differentiable —
    only the angle VALUE for the selected triples carries gradient, the
    standard way to do threshold-like selection inside a differentiable
    loss) and penalizes every pairwise angle among those neighbors that
    sits near cos(60 deg) = 0.5, via a smooth Gaussian bump in COSINE
    space rather than degrees: this avoids arccos's unstable/exploding
    gradient near +-1 and only needs a dot product + norms.

    Restricting to each atom's k_neighbors nearest neighbors (instead of
    every pair within some distance cutoff, as the diagnostic script
    does) keeps this O(N * k^2) instead of O(N^3) per molecule -- cheap
    enough to run every training step even for ChEMBL-sized (~80-atom)
    molecules, and k=4 is already a generous cap on real coordination
    number.

    Returns a (B,) tensor (one value per molecule, normalized by real
    atom count like loss_x/loss_h), NOT yet weighted or reduced.
    """
    B, N, _ = x0_pred.shape
    k = min(k_neighbors, N - 1)
    if k < 2:
        return torch.zeros(B, device=x0_pred.device)

    # vec[:, i, j, :] = x_j - x_i (vector FROM atom i TO atom j)
    vec = x0_pred.unsqueeze(1) - x0_pred.unsqueeze(2)
    dists = vec.norm(dim=-1)  # (B, N, N)

    real = node_mask.view(B, N).bool()
    self_mask = torch.eye(N, device=x0_pred.device, dtype=torch.bool).view(1, N, N)
    invalid_neighbor = self_mask | (~real).view(B, 1, N)
    dists = dists.masked_fill(invalid_neighbor, float("inf"))

    topk_dists, topk_idx = torch.topk(dists, k=k, dim=-1, largest=False)
    neighbor_valid = torch.isfinite(topk_dists)  # (B, N, k)

    idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    neighbor_vecs = torch.gather(vec, dim=2, index=idx_expanded)  # (B, N, k, 3)

    dot = (neighbor_vecs.unsqueeze(3) * neighbor_vecs.unsqueeze(2)).sum(-1)  # (B, N, k, k)
    norms = neighbor_vecs.norm(dim=-1).clamp(min=1e-8)  # (B, N, k)
    cos_angle = dot / (norms.unsqueeze(3) * norms.unsqueeze(2))

    pair_mask = torch.triu(torch.ones(k, k, device=x0_pred.device, dtype=torch.bool), diagonal=1).view(1, 1, k, k)
    valid_pair = pair_mask & neighbor_valid.unsqueeze(3) & neighbor_valid.unsqueeze(2)

    penalty = torch.exp(-((cos_angle - cos_center) ** 2) / (2 * sigma ** 2))
    penalty = penalty * valid_pair.float() * node_mask.view(B, N, 1, 1)

    n_real = node_mask.sum(dim=(1, 2)).clamp(min=1.0)
    return penalty.sum(dim=(1, 2, 3)) / n_real


class EquivariantMoleculeDiffusion(nn.Module):
    def __init__(self, num_atom_types: int = NUM_ATOM_TYPES, hidden_dim: int = 128,
                 n_layers: int = 4, timesteps: int = 1000, h_loss_weight: float = 1.0,
                 cond_dim: int = 0, angle_loss_weight: float = 0.0,
                 angle_k_neighbors: int = 4, angle_sigma: float = 0.075):
        super().__init__()
        self.num_atom_types = num_atom_types
        self.timesteps = timesteps
        self.h_loss_weight = h_loss_weight
        self.cond_dim = cond_dim
        # Auxiliary angle-consistency loss (opt-in, default 0.0 = fully
        # backward compatible with checkpoints/behavior before this was
        # added). See angle_consistency_penalty() above for what it does
        # and why; see references/approach.md for the diagnosis that
        # motivated it.
        self.angle_loss_weight = angle_loss_weight
        self.angle_k_neighbors = angle_k_neighbors
        self.angle_sigma = angle_sigma
        # in_node_dim = one-hot atom type; time (and optionally a condition
        # vector) are appended inside EGNN
        self.net = EGNN(in_node_dim=num_atom_types, hidden_dim=hidden_dim, n_layers=n_layers, cond_dim=cond_dim)

    # ---------------------------------------------------------------- #
    # Forward (noising) process
    # ---------------------------------------------------------------- #
    def q_sample(self, x0, h0, t, node_mask):
        """Sample z_t ~ q(z_t | x0, h0) for continuous timestep t in [0,1].

        Returns (x_t, h_t, eps_x, eps_h) — the noise targets are returned
        too since the training loss regresses the network onto them.
        """
        alpha_bar = cosine_alpha_bar(t).view(-1, 1, 1)          # (B,1,1)
        sqrt_ab = alpha_bar.sqrt()
        sqrt_1mab = (1 - alpha_bar).sqrt()

        eps_x = torch.randn_like(x0)
        eps_x = remove_mean_with_mask(eps_x, node_mask)  # keep noise on zero-CoM subspace
        eps_h = torch.randn_like(h0) * node_mask

        x_t = sqrt_ab * x0 + sqrt_1mab * eps_x
        h_t = sqrt_ab * h0 + sqrt_1mab * eps_h
        x_t = remove_mean_with_mask(x_t, node_mask)
        return x_t, h_t, eps_x, eps_h

    # ---------------------------------------------------------------- #
    # Training loss
    # ---------------------------------------------------------------- #
    def loss(self, x0, h0, node_mask, cond=None):
        """
        x0: (B, N, 3) ground-truth atom coordinates (will be CoM-centered)
        h0: (B, N, num_atom_types) ground-truth one-hot atom types
        node_mask: (B, N, 1)
        cond: (B, cond_dim) optional per-molecule condition vector, required
              iff this model was constructed with cond_dim > 0.
        """
        B = x0.shape[0]
        device = x0.device
        x0 = remove_mean_with_mask(x0, node_mask)

        t = torch.rand(B, device=device)  # continuous t in [0, 1), one per molecule
        x_t, h_t, eps_x, eps_h = self.q_sample(x0, h0, t, node_mask)

        edge_mask = build_edge_mask(node_mask)
        # h_t is scaled down only for what the network sees (see utils.H_SCALE) —
        # eps_h_pred is still compared against the unscaled eps_h target below.
        eps_h_pred, eps_x_pred = self.net(h_t * H_SCALE, x_t, t.view(B, 1), node_mask, edge_mask, cond=cond)


        n_real = node_mask.sum(dim=(1, 2)).clamp(min=1.0)  # atoms per molecule
        # sum over atoms & coord dims, then mean over the batch (normalized
        # by size so molecules of different lengths contribute comparably)
        loss_x = ((eps_x_pred - eps_x) ** 2 * node_mask).sum(dim=(1, 2)) / n_real
        loss_h = ((eps_h_pred - eps_h) ** 2 * node_mask).sum(dim=(1, 2)) / n_real

        parts = {"loss_x": loss_x.mean().item(), "loss_h": loss_h.mean().item()}
        loss = loss_x + self.h_loss_weight * loss_h

        if self.angle_loss_weight > 0:
            # Recover the network's implied denoised coordinates using the
            # same x0-from-eps algebra sample() uses (see its x0_pred line),
            # so the angle penalty acts on what the model actually predicts,
            # not just its noise-prediction error.
            ab_t = cosine_alpha_bar(t).view(B, 1, 1)
            x0_pred = (x_t - (1 - ab_t).sqrt() * eps_x_pred) / ab_t.sqrt().clamp(min=1e-8)
            x0_pred = remove_mean_with_mask(x0_pred, node_mask)
            loss_angle = angle_consistency_penalty(
                x0_pred, node_mask, k_neighbors=self.angle_k_neighbors, sigma=self.angle_sigma,
            )
            parts["loss_angle"] = loss_angle.mean().item()
            loss = loss + self.angle_loss_weight * loss_angle

        return loss.mean(), parts

    # ---------------------------------------------------------------- #
    # Reverse (sampling) process
    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def sample(self, node_mask, n_steps: int | None = None, device=None,
               cond=None, null_cond=None, guidance_scale: float = 1.0):
        """Generate new molecules by ancestral sampling from pure noise.

        node_mask: (B, N, 1) — determines how many atoms each generated
                   molecule has (you choose molecule sizes up front, e.g.
                   sampled from the size distribution of the training set).
        cond: (B, cond_dim) optional condition vector (e.g. a one-hot class
              label), required iff this model was built with cond_dim > 0.
        null_cond: (B, cond_dim) optional "unconditional"/null condition
              vector. If provided together with `cond` and
              `guidance_scale != 1.0`, each step extrapolates between the
              null and conditioned noise predictions:
                  eps = eps_null + guidance_scale * (eps_cond - eps_null)
              (classifier-free-guidance-style). With guidance_scale == 1.0
              (default) or null_cond=None, this reduces to plain
              conditional sampling using `cond` directly.
        Returns (x0_pred, h0_pred): coordinates and one-hot-ish atom types.
        """
        n_steps = n_steps or self.timesteps
        device = device or node_mask.device
        B, N, _ = node_mask.shape
        edge_mask = build_edge_mask(node_mask)
        use_guidance = null_cond is not None and guidance_scale != 1.0

        x_t = remove_mean_with_mask(torch.randn(B, N, 3, device=device), node_mask)
        h_t = torch.randn(B, N, self.num_atom_types, device=device) * node_mask

        ts = torch.linspace(1.0, 1.0 / n_steps, n_steps, device=device)
        for i, t_scalar in enumerate(ts):
            t = t_scalar.expand(B)
            t_next = (ts[i + 1] if i + 1 < n_steps else torch.zeros_like(t_scalar)).expand(B)

            ab_t = cosine_alpha_bar(t).view(B, 1, 1)
            ab_next = cosine_alpha_bar(t_next).view(B, 1, 1)

            # h_t is scaled down only for what the network sees (see utils.H_SCALE);
            # the state h_t itself (used below for h0_pred/next-step update) stays unscaled.
            if use_guidance:
                eps_h_cond, eps_x_cond = self.net(h_t * H_SCALE, x_t, t.view(B, 1), node_mask, edge_mask, cond=cond)
                eps_h_null, eps_x_null = self.net(h_t * H_SCALE, x_t, t.view(B, 1), node_mask, edge_mask, cond=null_cond)
                eps_h_pred = eps_h_null + guidance_scale * (eps_h_cond - eps_h_null)
                eps_x_pred = eps_x_null + guidance_scale * (eps_x_cond - eps_x_null)
            else:
                eps_h_pred, eps_x_pred = self.net(h_t * H_SCALE, x_t, t.view(B, 1), node_mask, edge_mask, cond=cond)
            eps_x_pred = remove_mean_with_mask(eps_x_pred, node_mask)

            # predict x0 from current noisy sample + predicted noise
            x0_pred = (x_t - (1 - ab_t).sqrt() * eps_x_pred) / ab_t.sqrt().clamp(min=1e-8)
            h0_pred = (h_t - (1 - ab_t).sqrt() * eps_h_pred) / ab_t.sqrt().clamp(min=1e-8)
            x0_pred = remove_mean_with_mask(x0_pred, node_mask)

            # DDIM-style deterministic-ish step toward t_next (with a small
            # amount of injected noise, except on the very last step)
            noise_x = remove_mean_with_mask(torch.randn_like(x_t), node_mask)
            noise_h = torch.randn_like(h_t) * node_mask
            sigma = (1 - ab_next).sqrt() * 0.2 if i < n_steps - 1 else 0.0

            x_t = ab_next.sqrt() * x0_pred + (1 - ab_next - sigma ** 2).clamp(min=0).sqrt() * eps_x_pred + sigma * noise_x
            h_t = ab_next.sqrt() * h0_pred + (1 - ab_next - sigma ** 2).clamp(min=0).sqrt() * eps_h_pred + sigma * noise_h
            x_t = remove_mean_with_mask(x_t, node_mask)
            h_t = h_t * node_mask

        return x_t, h_t
